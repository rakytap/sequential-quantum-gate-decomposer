"""Catalogue workers reuse interpreters without changing local synthesis."""

import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pytest

from squander.partitioning import routing


def _probe_message(target, config, topology):
    index = int(target[0, 0].real)
    if config.get("crash") and index == 0:
        os._exit(17)
    if config.get("hang") and index == 0:
        time.sleep(30)
    if config.get("error"):
        return ("error", "controlled worker error")
    directory = config.get("barrier")
    if directory is not None and index < 2:
        directory = Path(directory)
        (directory / str(index)).touch()
        deadline = time.monotonic() + 15
        while not all((directory / str(i)).exists() for i in range(2)):
            if time.monotonic() > deadline:
                return ("error", "synthesis workers did not overlap")
            time.sleep(0.01)
    return ("probe", {
        "index": index, "pid": os.getpid(), "config": config,
        "threads": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    })


def _decode_probe(message):
    if message[0] == "error":
        raise RuntimeError(message[1])
    assert message[0] == "probe"
    return message[1]


def _targets(count):
    return tuple(np.array([[index]], dtype=np.complex128)
                 for index in range(count))


def _probe_config(**overrides):
    return {"routing_synthesis_workers": 2, **overrides}


def _install_probe(monkeypatch):
    monkeypatch.setattr(routing, "_synthesis_worker_message", _probe_message)
    monkeypatch.setattr(routing, "_decode_synthesis_message", _decode_probe)


def test_workers_overlap_reuse_processes_and_preserve_result_order(
    monkeypatch, tmp_path,
):
    _install_probe(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "9")
    result = routing._call_shared_synthesis_batch(
        _targets(8), _probe_config(barrier=str(tmp_path)), ())
    assert [item["index"] for item in result] == list(range(8))
    assert len({item["pid"] for item in result}) == 2
    assert all(set(item["threads"].values()) == {"1"} for item in result)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert "OMP_NUM_THREADS" not in os.environ
    assert os.environ["MKL_NUM_THREADS"] == "9"
    assert not any(child.name == "routing-osr-synthesis"
                   for child in mp.active_children())


def test_worker_payload_excludes_only_whole_circuit_runtime_artifacts(monkeypatch):
    _install_probe(monkeypatch)
    config = _probe_config(random_seed=19, tolerance=1e-14,
                           synthesis_acceptance_tolerance=1e-10,
                           tree_level_max=8, parallel=0)
    config.update({key: lambda: None
                   for key in routing._ROUTING_SYNTHESIS_RUNTIME_KEYS})
    result = routing._call_shared_synthesis_batch(_targets(3), config, ())
    expected = {key: value for key, value in config.items()
                if key not in routing._ROUTING_SYNTHESIS_RUNTIME_KEYS}
    assert all(item["config"] == expected for item in result)


def test_native_crash_is_isolated_and_worker_is_replaced(monkeypatch):
    _install_probe(monkeypatch)
    result = routing._call_shared_synthesis_batch(
        _targets(4), _probe_config(routing_synthesis_workers=1, crash=True), ())
    assert result[0] is None
    assert [item["index"] for item in result[1:]] == [1, 2, 3]
    assert len({item["pid"] for item in result[1:]}) == 1
    assert not any(child.name == "routing-osr-synthesis"
                   for child in mp.active_children())


def test_configured_timeout_does_not_poison_later_tasks(monkeypatch):
    _install_probe(monkeypatch)
    result = routing._call_shared_synthesis_batch(
        _targets(2), _probe_config(routing_synthesis_workers=1, hang=True,
                                  routing_synthesis_timeout_seconds=5), ())
    assert result[0] is None
    assert result[1]["index"] == 1


def test_reported_error_raises_and_reaps_all_children(monkeypatch):
    _install_probe(monkeypatch)
    with pytest.raises(RuntimeError, match="controlled worker error"):
        routing._call_shared_synthesis_batch(
            _targets(4), _probe_config(error=True), ())
    assert not any(child.name == "routing-osr-synthesis"
                   for child in mp.active_children())


def test_memory_guard_failure_reaps_already_started_workers(monkeypatch):
    _install_probe(monkeypatch)
    checks = []

    def guard(config):
        checks.append(True)
        if len(checks) == 3:
            raise routing.RoutingMemoryPressure("controlled reserve")

    monkeypatch.setattr(routing, "_check_routing_memory_pressure", guard)
    with pytest.raises(routing.RoutingMemoryPressure, match="controlled reserve"):
        routing._call_shared_synthesis_batch(_targets(4), _probe_config(), ())
    assert not any(child.name == "routing-osr-synthesis"
                   for child in mp.active_children())


def test_spawn_failure_restores_the_coordinator_environment(monkeypatch):
    class FailingProcess:
        def start(self):
            assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
            raise RuntimeError("controlled spawn failure")

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    with pytest.raises(RuntimeError, match="controlled spawn failure"):
        routing._start_routing_synthesis_process(FailingProcess())
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"


def test_empty_and_mismatched_batches_do_not_start_workers():
    assert routing._call_shared_synthesis_batch((), {}, ()) == ()
    with pytest.raises(ValueError, match="One synthesis config"):
        routing._call_shared_synthesis_batch(_targets(2), {}, (),
                                              target_configs=({},))


@pytest.mark.parametrize("workers", [1, 2])
def test_real_osr_targets_match_direct_solves_and_strict_unitaries(workers):
    from squander.gates.qgd_Circuit import qgd_Circuit

    circuit = qgd_Circuit(3)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(2)
    parameters = np.array([0.123, -0.234, 0.345, 0.456, -0.567, 0.678])
    target = circuit.get_Matrix(parameters, is_f32=False)
    targets = (target, target, np.eye(8, dtype=np.complex128))
    config = {
        "strategy": "TreeSearch", "use_osr": True, "use_graph_search": True,
        "parallel": 0, "random_seed": 20260926, "verbosity": 0,
        "synthesis_acceptance_tolerance": 1e-10,
        "routing_synthesis_workers": workers,
    }
    topology = ((0, 1), (1, 2))
    direct = tuple(routing._synthesize_routing_column(value, config, topology)
                   for value in targets)
    pooled = routing._call_shared_synthesis_batch(targets, config, topology)
    for value, reference, result in zip(targets, direct, pooled):
        assert reference is not None and result is not None
        assert result.circuit.get_Gate_Nums().get("CNOT", 0) == (
            reference.circuit.get_Gate_Nums().get("CNOT", 0))
        for candidate in (reference, result):
            matrix = candidate.circuit.get_Matrix(candidate.parameters, is_f32=False)
            overlap = abs(np.vdot(value, matrix)) / len(value)
            assert max(0.0, 1.0 - overlap ** 2) <= 1e-10
