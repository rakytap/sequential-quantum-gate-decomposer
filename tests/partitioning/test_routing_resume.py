"""Scientific invariants of cached benchmark continuation."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

import squander.decomposition.qgd_Wide_Circuit_Optimization as backend
from benchmarks.partitioning.replay_routing import cumulative_timing, execution_identity


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
