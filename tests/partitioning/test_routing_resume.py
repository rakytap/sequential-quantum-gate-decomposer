"""Scientific invariants of cached benchmark continuation."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

import squander.decomposition.qgd_Wide_Circuit_Optimization as backend
from benchmarks.partitioning.replay_routing import (
    cumulative_timing, execution_identity, recover_interrupted_final_attempt,
    verify_routing_checkpoint, write_json,
)


def representation(angle):
    return {"format": "qasm2", "qubits": 2,
            "qasm": ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                     f'rz({angle}) q[0];\ncx q[0],q[1];\n')}


def test_reuse_requires_exact_parameters_and_both_mappings():
    source = representation("0.3")
    identity = execution_identity(backend, source, (0, 1), (1, 0))
    assert execution_identity(backend, deepcopy(source), (0, 1), (1, 0)) == identity
    assert execution_identity(backend, representation("0.30000000000000004"), (0, 1), (1, 0)) != identity
    assert execution_identity(backend, source, (1, 0), (1, 0)) != identity
    assert execution_identity(backend, source, (0, 1), (0, 1)) != identity


def test_replay_never_relabels_warm_time_as_a_fresh_run():
    original = dict(a2a=5.0, routing=120.0, optimization=30.0, total=155.0)
    assert cumulative_timing(original, 2.0, 0.0) == dict(
        a2a=5.0, routing=122.0, optimization=30.0, total=157.0)
    assert cumulative_timing(original, 2.0, 3.0) == dict(
        a2a=5.0, routing=122.0, optimization=33.0, total=160.0)
    assert original["total"] == 155.0


def test_interrupted_final_attempt_is_charged_once_across_restarts(tmp_path):
    pending = tmp_path / "final-attempt.json"
    write_json(pending, dict(status="running", started_monotonic=10))
    assert sum(a["seconds"] for a in recover_interrupted_final_attempt(tmp_path, now=25)) == 15
    # Simulate interruption between committing history and marking it charged.
    write_json(pending, dict(status="running", started_monotonic=10))
    assert sum(a["seconds"] for a in recover_interrupted_final_attempt(tmp_path, now=35)) == 15
    write_json(pending, dict(status="running", started_monotonic=40))
    assert sum(a["seconds"] for a in recover_interrupted_final_attempt(tmp_path, now=50)) == 25
    write_json(pending, dict(status="completed", seconds=4))
    assert sum(a["seconds"] for a in recover_interrupted_final_attempt(tmp_path, now=99)) == 25


def test_checkpoint_code_drift_requires_an_independent_exact_source_proof():
    verified = []
    checker = SimpleNamespace(_verify_exact_osr_routing_replay=lambda r: verified.append(r))
    archived = dict(input_sha256="exact-source", cnot_count=10)
    route = dict(input_sha256="exact-source", cnot_count=8)
    saved = dict(route=route, implementation_sha256={"old-router.py": "old-version"})
    assert verify_routing_checkpoint(saved, archived, checker) is route
    assert verified == [route]
    assert saved["implementation_sha256"] == {"old-router.py": "old-version"}
    for corrupted in (dict(input_sha256="other-source", cnot_count=8),
                      dict(input_sha256="exact-source", cnot_count=11)):
        with pytest.raises(ValueError):
            verify_routing_checkpoint(dict(saved, route=corrupted), archived, checker)
    def rejected(route):
        raise ValueError("Invalid permutation certificate")
    with pytest.raises(ValueError, match="Invalid permutation certificate"):
        verify_routing_checkpoint(saved, archived, SimpleNamespace(_verify_exact_osr_routing_replay=rejected))


@pytest.mark.parametrize("corrupt", [False, True])
def test_optional_pam_limit_preserves_best_but_corruption_fails(monkeypatch, corrupt):
    from benchmarks.partitioning.replay_routing import cached_portfolio
    from squander.partitioning import routing, ilp, partition

    circuit = SimpleNamespace(get_Qbit_Num=lambda: 3)
    catalog = SimpleNamespace(
        circuit=circuit, parameters=np.empty(0), topology=((0, 1), (1, 2)),
        partitions=(frozenset((0,)),), gate_orders=((0,),),
        alternatives={0: (SimpleNamespace(cnot_count=2),)}, implicit_path_frames=False)
    monkeypatch.setattr(routing, "augment_catalog_with_recombined_covers", lambda c, **kw: c)
    monkeypatch.setattr(ilp, "get_all_partitions", lambda *a: (None, None, None, {0: ()}))
    monkeypatch.setattr(partition, "PartitionCircuit", lambda *a, **kw: (None, None, ((0,),)))
    monkeypatch.setattr(routing, "_minimum_partition_cover_selection", lambda **kw: (0,))
    monkeypatch.setattr(routing, "mapping_aware_catalog_seeds", lambda *a, **kw: ())
    monkeypatch.setattr(routing, "_cover_selection_warm_start", lambda **kw: None)
    monkeypatch.setattr(routing, "construct_exact_routed_circuit", lambda *a: (circuit, np.empty(0)))
    calls = []
    def pam(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(cnot_count=3)
        if corrupt:
            raise ValueError("Corrupt local certificate")
        raise routing.ExactRoutingLimitExceeded("Optional seed budget")
    monkeypatch.setattr(routing, "_precomputed_osr_pam_warm_start", pam)
    archived = dict(cnot_count=4, initial_mapping=(0, 1, 2), final_mapping=(0, 1, 2),
                    selections=[dict(source_gate_indices=[0])])
    if corrupt:
        with pytest.raises(ValueError, match="Corrupt"):
            cached_portfolio(catalog, archived, lambda *a: None)
    else:
        _, result, variants = cached_portfolio(catalog, archived, lambda *a: None)
        assert result.solution.cnot_count == 3
        assert any(v.get("timed_out") for v in variants)


def test_catalogue_export_roundoff_restores_exact_audited_source():
    from qiskit import QuantumCircuit
    from squander import Qiskit_IO
    from squander.partitioning import routing
    from squander.partitioning._routing_catalog_replay import bind_catalog_source

    qiskit = QuantumCircuit(2)
    qiskit.u(0.3, 0.2, 0.1, 0)
    qiskit.cx(0, 1)
    circuit, cached_parameters = Qiskit_IO.convert_Qiskit_to_Squander(qiskit)
    cached_parameters = np.asarray(cached_parameters)
    exact_parameters = cached_parameters.copy()
    exact_parameters[1] += 1e-15
    payload = routing.SynthesizedRoutingPayload(
        circuit=circuit, parameters=cached_parameters, topology=((0, 1),),
        input_assignment=(0, 1), output_assignment=(0, 1),
        source_circuit=circuit, source_parameters=cached_parameters)
    alternative = routing.RoutingAlternative(
        0, (0, 1), (0, 1), (0, 1), 1, payload=payload)
    catalog = routing.LoadedRoutingOSRCatalog(
        circuit, cached_parameters, ((0, 1),), (frozenset((0, 1)),),
        ((0, 1),), {0: (alternative,)}, {})
    starts = tuple(g.get_Parameter_Start_Index() for g in circuit.get_Gates())
    restored = bind_catalog_source(catalog, circuit, exact_parameters)
    selected = restored.alternatives[0][0]
    assert selected.payload.circuit is payload.circuit
    np.testing.assert_array_equal(selected.payload.parameters, cached_parameters)
    np.testing.assert_array_equal(selected.payload.source_parameters, exact_parameters)
    assert selected.payload.source_parameters[1] != payload.source_parameters[1]
    expected = circuit.get_Matrix(exact_parameters, is_f32=False)
    np.testing.assert_array_equal(selected.payload.source_circuit.get_Matrix(
        selected.payload.source_parameters, is_f32=False), expected)
    assert starts == tuple(g.get_Parameter_Start_Index() for g in circuit.get_Gates())
    assert routing._process_infidelity(expected, payload.circuit.get_Matrix(
        payload.parameters, is_f32=False)) < 1e-10
    wrong = exact_parameters.copy()
    wrong[1] += 1e-3
    with pytest.raises(ValueError, match="beyond export roundoff"):
        bind_catalog_source(catalog, circuit, wrong)
