"""Scientific invariants of cached benchmark continuation."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

import squander.decomposition.qgd_Wide_Circuit_Optimization as backend
from benchmarks.partitioning.replay_routing import (
    cumulative_timing, execution_identity, recover_interrupted_final_attempt,
    verify_routing_checkpoint, write_json, final_prefix_boundary,
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


def test_extended_phase_totals_do_not_reuse_old_wall_and_audit_measurements():
    original = dict(a2a=5.0, routing=120.0, optimization=30.0, total=155.0,
                    wall_total=170.0, audit_export=3.0, audit_verification=12.0,
                    audit_total=15.0)
    updated = cumulative_timing(original, 10.0, 40.0)
    assert updated["total"] == 205.0
    assert not {"wall_total", "audit_export", "audit_verification", "audit_total"} & updated.keys()
    assert original["wall_total"] == 170.0
    assert original["audit_total"] == 15.0


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


def test_final_resume_excludes_uncommitted_rewrites_and_truncated_tail(tmp_path):
    import json
    route = dict(kind="exact_osr_routing", event_id="route", output_sha256="raw")
    complete = dict(kind="round", stage="topology_optimization", output_sha256="fitted")
    prefix = "".join(json.dumps(event) + "\n" for event in (route, complete))
    path = tmp_path / "candidate.jsonl"
    path.write_text(prefix + json.dumps(dict(kind="rewrite", accepted=True)) + "\n" + '{"kind":"round"')
    boundary, event, rounds = final_prefix_boundary(path, route)
    assert boundary == len(prefix.encode())
    assert event == complete
    assert rounds == 1
    assert final_prefix_boundary(path, dict(route, event_id="other-route")) is None


def test_final_resume_requires_a_completed_topology_round(tmp_path):
    import json
    route = dict(kind="exact_osr_routing", event_id="route", output_sha256="raw")
    events = [dict(kind="round", stage="all_to_all"), route,
              dict(kind="squander_basis_conversion", stage="topology_optimization")]
    path = tmp_path / "candidate.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert final_prefix_boundary(path, route) is None


def test_final_resume_rejects_corrupt_complete_ledger_lines(tmp_path):
    path = tmp_path / "candidate.jsonl"
    path.write_text('{not-json}\n')
    with pytest.raises(ValueError):
        final_prefix_boundary(path, {})


def _stagewise_source(tmp_path):
    import hashlib
    source = tmp_path / "source"
    source.mkdir()
    (source / "candidate.audit").write_bytes(b"candidate-proof")
    (source / "candidate.qasm").write_bytes(b"candidate-circuit")
    route = dict(kind="exact_osr_routing", event_id="old", input_sha256="a2a", cnot_count=10)
    candidate_route = dict(route, event_id="best-raw", cnot_count=8)
    audit = dict(run=dict(input_circuit_sha256="input"), events=[route])
    candidate = dict(run=dict(input_circuit_sha256="input",
                             output_file_sha256=hashlib.sha256(b"candidate-circuit").hexdigest()),
                     events=[candidate_route])
    old = dict(routed=dict(cnot_equiv=8), rewrite_audit=dict(sha256="canonical-proof"),
               routing_replay=dict(candidate_audit=dict(file="candidate.audit",
                    sha256=hashlib.sha256(b"candidate-proof").hexdigest()),
                    candidate_qasm="candidate.qasm", candidate_final=dict(cnot_equiv=7),
                    candidate_adopted=False, final_seconds=90))
    return source, old, audit, candidate, candidate_route


def test_cached_resume_keeps_best_raw_route_and_its_separate_final_certificate(tmp_path, monkeypatch):
    from benchmarks.partitioning import replay_routing as replay
    source, old, audit, candidate, expected = _stagewise_source(tmp_path)
    verified = []
    checker = SimpleNamespace(load_rewrite_audit=lambda path: candidate,
        _verify_exact_osr_routing_replay=lambda route: verified.append(route))
    route, digest = replay.archived_routing_incumbent(old, audit, source, checker)
    assert route is expected and route["cnot_count"] == 8
    assert verified == [expected]
    assert digest == old["routing_replay"]["candidate_audit"]["sha256"]
    destination = tmp_path / "destination"
    monkeypatch.setattr(replay, "ROOT", destination)
    provenance = dict(final_seconds=0, final_reused=True)
    replay.retain_prior_candidate(old, source, provenance)
    assert (destination / "candidate.audit").read_bytes() == b"candidate-proof"
    assert (destination / "candidate.qasm").read_bytes() == b"candidate-circuit"
    assert provenance["candidate_adopted"] is False
    assert provenance["final_seconds"] == 0
    assert provenance["candidate_final"] == {"cnot_equiv": 7}


@pytest.mark.parametrize("corruption", ["input", "count"])
def test_cached_resume_rejects_a_mismatched_prior_candidate(tmp_path, corruption):
    from benchmarks.partitioning import replay_routing as replay
    source, old, audit, candidate, route = _stagewise_source(tmp_path)
    if corruption == "input":
        candidate["run"]["input_circuit_sha256"] = "other-input"
    else:
        route["cnot_count"] = 9
    checker = SimpleNamespace(load_rewrite_audit=lambda path: candidate)
    with pytest.raises(ValueError):
        replay.archived_routing_incumbent(old, audit, source, checker)


def test_cached_resume_requires_evidence_for_the_recorded_best_raw_count():
    from benchmarks.partitioning import replay_routing as replay
    old = dict(routed=dict(cnot_equiv=8), rewrite_audit=dict(sha256="canonical"))
    audit = dict(events=[dict(kind="exact_osr_routing", cnot_count=10)])
    with pytest.raises(ValueError, match="no matching certificate"):
        replay.archived_routing_incumbent(old, audit, None, None)


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


@pytest.mark.parametrize("rounded_pi", [False, True])
def test_catalogue_export_roundoff_restores_exact_audited_source(rounded_pi):
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
    if rounded_pi:
        cached_parameters[1] = np.pi
        exact_parameters[1] = np.pi - 2.87e-12
    else:
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
