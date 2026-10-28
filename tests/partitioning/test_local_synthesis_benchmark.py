"""Check benchmark result delivery and completion-aware comparisons."""

from argparse import Namespace

import numpy as np
import pytest

from benchmarks.partitioning import local_synthesis_head_to_head as benchmark


def _large_result_worker(case, method, tolerance, seed, out_queue,
                         osr_config=None, cpu_id=None):
    out_queue.put({"status": "verified", "output_qasm": "x" * 500_000})


def test_large_worker_result_does_not_become_a_timeout(monkeypatch):
    monkeypatch.setattr(benchmark, "solve_worker", _large_result_worker)
    args = Namespace(tolerance=1e-10, seed=1, timeout_seconds=10,
                     memory_mb=1024, threads=1)
    result = benchmark.run_one({}, "osr", args)
    assert result["status"] == "verified"
    assert len(result["output_qasm"]) == 500_000


def test_osr_timeout_against_verified_bqskit_is_a_loss():
    cases = [{"case_id": "timeout", "width": 3, "context": "a2a",
              "dataset": "test", "source_circuit": "case",
              "target_sha256": "target", "graph": [[0, 1], [1, 2]]}]
    rows = [{"case_id": "timeout", "method": "osr", "status": "timeout"},
            {"case_id": "timeout", "method": "bqskit", "status": "verified",
             "seconds": 1.0, "cnots": 4}]
    summary = benchmark.summarize(rows, cases, 1, 1e-10)["3q/a2a"]
    assert summary["paired_verified"] == 0
    assert summary["completion_aware_osr_losses"] == 1
    assert summary["osr_timeouts_against_verified_bqskit"] == 1
    assert benchmark.regression_audit(rows, cases)[0]["kind"] == "osr_timeout_loss"


def test_nonfinite_unitary_cannot_pass_certification():
    target = np.eye(8, dtype=np.complex128)
    candidate = target.copy()
    candidate[0, 0] = np.nan
    assert benchmark.process_infidelity(target, candidate) == float("inf")


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("graph", [[], None])
def test_legacy_all_to_all_graph_certifies_any_coupler(width, graph):
    from qiskit import QuantumCircuit, qasm2
    from qiskit.quantum_info import Operator

    case = {"width": width, "context": "a2a", "graph": graph,
            "case_id": "unchanged-legacy-identity"}
    circuit = QuantumCircuit(width)
    circuit.cx(0, width - 1)
    target = np.asarray(Operator(circuit).data, dtype=np.complex128)
    result = benchmark.certify_output(
        case, target, target, qasm2.dumps(circuit), 1, 1e-10,
    )
    assert result["status"] == "verified"
    assert result["cnots"] == 1
    assert result["topology_verified"]
    assert result["process_infidelity"] < 1e-10
    assert case == {"width": width, "context": "a2a", "graph": graph,
                    "case_id": "unchanged-legacy-identity"}


def test_empty_topology_does_not_become_all_to_all():
    from qiskit import QuantumCircuit, qasm2
    from qiskit.quantum_info import Operator

    case = {"width": 3, "context": "topology", "graph": []}
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    target = np.asarray(Operator(circuit).data, dtype=np.complex128)
    assert benchmark.effective_coupling_graph(case) == []
    with pytest.raises(ValueError, match="off-topology CNOT"):
        benchmark.certify_output(case, target, target, qasm2.dumps(circuit), 1, 1e-10)


def test_explicit_coupling_graph_is_not_expanded_or_mutated():
    case = {"width": 3, "context": "a2a", "graph": [[0, 1], [1, 2]]}
    resolved = benchmark.effective_coupling_graph(case)
    assert resolved == [[0, 1], [1, 2]]
    resolved.append([0, 2])
    assert case["graph"] == [[0, 1], [1, 2]]


def test_missing_topology_graph_is_rejected():
    with pytest.raises(ValueError, match="must specify its coupling graph"):
        benchmark.effective_coupling_graph({"width": 3, "context": "topology"})


def _fake_runtime_context(monkeypatch, ready, failed=False):
    from types import SimpleNamespace

    trace = []
    state = {"alive": False, "locked": False}

    class StartupLock:
        def acquire(self):
            assert not state["locked"]
            state["locked"] = True
            trace.append("acquire")

        def release(self):
            assert state["locked"]
            state["locked"] = False
            trace.append("release")

    class Queue:
        def get(self, timeout):
            trace.append("get")
            if ready:
                assert not state["locked"], "READY must release startup lock"
            return {"status": "failed" if failed else "verified"}

        def close(self):
            pass

        def join_thread(self):
            pass

    class Process:
        pid = 123456

        def __init__(self, target, args):
            assert len(args) == 8
            self.event = args[-1]

        def start(self):
            assert state["locked"]
            trace.append("start")
            state["alive"] = True

        def is_alive(self):
            return state["alive"]

        def join(self, timeout):
            state["alive"] = False
            trace.append("join")

    context = SimpleNamespace(
        Queue=lambda maxsize: Queue(),
        Event=lambda: SimpleNamespace(is_set=lambda: ready),
        Process=Process,
    )
    monkeypatch.setattr(benchmark.mp, "get_context", lambda name: context)
    monkeypatch.setattr(benchmark, "_BQSKIT_STARTUP_LOCK", StartupLock())
    monkeypatch.setattr(
        benchmark.time, "monotonic",
        lambda: trace.append("clock") or 0.0,
    )
    return trace, state


def test_bqskit_only_serializes_startup_and_excludes_scheduling_wait(monkeypatch):
    trace, state = _fake_runtime_context(monkeypatch, ready=True)
    args = Namespace(tolerance=1e-10, seed=1, timeout_seconds=10,
                     memory_mb=1024, threads=1)
    result = benchmark.run_one({}, "bqskit", args)
    assert result["status"] == "verified"
    assert result["bqskit_startup_serialized_until_ready"]
    assert trace.index("acquire") < trace.index("clock") < trace.index("start")
    assert trace.index("start") < trace.index("release") < trace.index("get")
    assert trace.count("release") == 1
    assert not state["locked"]


def test_bqskit_startup_failure_releases_the_scheduling_lock(monkeypatch):
    trace, state = _fake_runtime_context(monkeypatch, ready=False, failed=True)
    args = Namespace(tolerance=1e-10, seed=1, timeout_seconds=10,
                     memory_mb=1024, threads=1)
    result = benchmark.run_one({}, "bqskit", args)
    assert result["status"] == "failed"
    assert trace.index("get") < trace.index("release")
    assert trace.count("release") == 1
    assert not state["locked"]


def test_timeout_signals_only_a_private_worker_process_group(monkeypatch):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(benchmark.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        benchmark.os, "killpg", lambda pid, sig: calls.append((pid, sig))
    )
    process = SimpleNamespace(
        pid=123456, join=lambda timeout: calls.append(("join", timeout)),
    )
    benchmark.kill_process_tree(process)
    assert calls == [(123456, benchmark.signal.SIGKILL), ("join", 2)]


def test_timeout_never_signals_the_controllers_process_group(monkeypatch):
    from types import SimpleNamespace
    import psutil

    calls = []
    monkeypatch.setattr(benchmark.os, "getpgid", lambda pid: 1)
    monkeypatch.setattr(
        benchmark.os, "killpg", lambda pid, sig: calls.append((pid, sig))
    )

    def missing_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", missing_process)
    process = SimpleNamespace(
        pid=123456, join=lambda timeout: calls.append(("join", timeout)),
    )
    benchmark.kill_process_tree(process)
    assert calls == [("join", 2)]

def _native_unitary_target(width):
    from qiskit import QuantumCircuit, qasm2
    from qiskit.quantum_info import Operator

    circuit = QuantumCircuit(width)
    circuit.u(0.41, -0.3, 0.61, 0)
    circuit.cx(0, width - 1)
    circuit.u(0.1, 0.2, -0.13, width - 2)
    matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
    representation = {
        "format": "unitary", "qubits": width,
        "unitary": {"dimension": 1 << width,
                    "real": matrix.real.ravel().tolist(),
                    "imag": matrix.imag.ravel().tolist()},
    }
    return representation, matrix, qasm2.dumps(circuit)


def _audit_target(event, **kwargs):
    return benchmark.audit_event_candidate(
        event, dataset="test", origin=kwargs.pop("origin", "TreeSearch"),
        source_circuit="case", source_file="case.audit.json.gz",
        source_index=0, full_graph=kwargs.pop("full_graph", []), **kwargs)


@pytest.mark.parametrize("width", [3, 4])
def test_native_matrix_audit_preserves_exact_target_basis_and_unknown_source_count(width):
    representation, matrix, qasm = _native_unitary_target(width)
    case = _audit_target({
        "kind": "rewrite", "stage": "all_to_all",
        "component": "squander_partition_synthesis", "before": representation,
        "after": {"gate_counts": {"CNOT": 1}}, "within_tolerance": True,
    })
    assert case["graph"] == benchmark.complete_graph(width)
    assert case["input_cnot"] is None
    assert case["known_feasible_cnot_upper_bound"] == 1
    assert "qasm" not in case
    decoded = benchmark.matrix_from_case(case)
    assert np.array_equal(decoded, matrix)
    assert np.array_equal(decoded, benchmark.matrix_from_qasm(qasm))
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        _audit_representation_unitary)
    assert np.array_equal(decoded, _audit_representation_unitary(representation))


def test_failed_native_synthesis_attempt_is_included_and_can_be_sampled():
    representation, matrix, _ = _native_unitary_target(4)
    case = _audit_target({
        "kind": "synthesis_attempt", "stage": "all_to_all",
        "component": "squander_partition_synthesis", "target": representation,
        "accepted": False, "reported_error": 0.5,
    })
    assert case["source_event_kind"] == "synthesis_attempt"
    assert case["input_cnot"] is None
    assert case["known_feasible_cnot_upper_bound"] is None
    selected = benchmark.choose_cases([case], 1, 1, 1)
    assert len(selected) == 1
    assert selected[0]["target_sha256"] == benchmark.unitary_digest(matrix)


def test_native_matrix_topology_uses_recorded_local_graph_without_locations():
    representation, matrix, qasm = _native_unitary_target(3)
    case = _audit_target({
        "kind": "rewrite", "stage": "topology_optimization",
        "before": representation, "subtopology": [[0, 1], [1, 2]],
    })
    assert case["graph"] == [[0, 1], [1, 2]]
    with pytest.raises(ValueError, match="off-topology CNOT"):
        benchmark.certify_output(case, matrix, matrix, qasm, 1, 1e-10)


def test_missing_local_topology_is_reported_instead_of_guessed():
    representation, _, _ = _native_unitary_target(4)
    exclusions = []
    assert _audit_target({
        "kind": "rewrite", "stage": "topology_optimization",
        "before": representation,
    }, exclusions=exclusions) is None
    assert exclusions == ["missing-local-topology:case.audit.json.gz:0"]


def test_unknown_matrix_producer_basis_is_reported():
    representation, _, _ = _native_unitary_target(4)
    exclusions = []
    assert _audit_target({
        "kind": "rewrite", "stage": "all_to_all",
        "component": "other_compiler", "before": representation,
    }, origin="bqskit", exclusions=exclusions) is None
    assert exclusions == ["unknown-unitary-basis:case.audit.json.gz:0"]


@pytest.mark.parametrize("bad", ["width", "size", "nonfinite", "nonunitary", "basis"])
def test_serialized_target_validation_fails_closed_without_projection(bad):
    representation, _, _ = _native_unitary_target(4)
    case = {"width": 4, "target_unitary": representation["unitary"],
            "target_basis": "little_endian"}
    if bad == "width":
        case["width"] = 3
    elif bad == "size":
        case["target_unitary"]["real"].pop()
    elif bad == "nonfinite":
        case["target_unitary"]["real"][0] = float("nan")
    elif bad == "nonunitary":
        case["target_unitary"]["real"] = [2 * x for x in case["target_unitary"]["real"]]
        case["target_unitary"]["imag"] = [2 * x for x in case["target_unitary"]["imag"]]
    else:
        case["target_basis"] = "unknown"
    with pytest.raises(ValueError):
        benchmark.matrix_from_case(case)


def test_matrix_archive_reader_includes_rewrites_and_failed_attempts(tmp_path):
    import gzip
    import json

    before3, _, _ = _native_unitary_target(3)
    before4, _, _ = _native_unitary_target(4)
    directory = tmp_path / "IBMEagle_results_4qbit_TreeSearch"
    directory.mkdir()
    with gzip.open(directory / "case.audit.json.gz", "wt") as stream:
        json.dump({
            "run": {"input_circuit": {"qubits": 5}},
            "events": [
                {"kind": "rewrite", "stage": "all_to_all", "before": before3},
                {"kind": "synthesis_attempt", "stage": "all_to_all",
                 "target": before4, "accepted": False},
            ],
        }, stream)
    candidates, exclusions = benchmark.archived_candidates(tmp_path, 1, 10, 1)
    assert not exclusions
    assert {(item["width"], item["source_event_kind"]) for item in candidates} == {
        (3, "rewrite"), (4, "synthesis_attempt"),
    }


def test_invalid_serialized_candidates_are_reported_during_sampling():
    before, _, _ = _native_unitary_target(4)
    case = _audit_target({"kind": "rewrite", "stage": "all_to_all", "before": before})
    case["target_unitary"]["real"][0] = float("inf")
    exclusions = []
    assert benchmark.choose_cases([case], 1, 1, 1, exclusions=exclusions) == []
    assert len(exclusions) == 1
    assert exclusions[0].startswith("invalid-target:case.audit.json.gz:0:")


def test_osr_worker_receives_requested_benchmark_seed(monkeypatch):
    from types import SimpleNamespace
    from squander.decomposition import qgd_Wide_Circuit_Optimization as wco_module

    captured, rows, seeds = {}, [], []
    class FakeWCO:
        def __init__(self, config):
            captured.update(config)
            self.config = config
        def DecomposePartition(self, target, config, mini_topology):
            return []
    monkeypatch.setattr(wco_module, "qgd_Wide_Circuit_Optimization", FakeWCO)
    monkeypatch.setattr(benchmark.os, "setsid", lambda: None)
    monkeypatch.setattr(np.random, "seed", seeds.append)
    representation, _, _ = _native_unitary_target(3)
    case = {"width": 3, "context": "a2a", "graph": [],
            "target_unitary": representation["unitary"]}
    benchmark.solve_worker(
        case, "osr", 1e-10, 17, SimpleNamespace(put=rows.append),
        osr_config={"random_seed": 99},
    )
    assert captured["random_seed"] == 17
    assert seeds == [17]
    assert rows[0]["status"] == "no_solution"
