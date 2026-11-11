"""Scientific invariants of cached benchmark continuation."""
from copy import deepcopy

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
