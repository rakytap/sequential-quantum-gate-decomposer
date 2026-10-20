"""Resolved PAM copies preserve circuit data and independent mutation."""
import numpy as np
import pytest
from bqskit import Circuit
from bqskit.ir.gates import CNOTGate, HGate, U3Gate
from squander.partitioning._pam_circuit_cache import copy_resolved_pam_circuit

@pytest.mark.parametrize("width", [2, 3, 4])
def test_resolved_pam_copy_preserves_matrix_parameters_and_schedule(width):
    rng = np.random.default_rng(20261009)
    for trial in range(10):
        source = Circuit(width)
        for _ in range(2 + trial % 3):
            for qudit in range(width):
                source.append_gate(U3Gate(), qudit, list(rng.standard_normal(3)))
            for qudit in range(width - 1):
                source.append_gate(CNOTGate(), (qudit, qudit + 1))
        parameters = np.asarray(source.params).copy()
        matrix = np.asarray(source.get_unitary()).copy()
        copied = copy_resolved_pam_circuit(source)
        assert copied.gate_counts == source.gate_counts
        assert copied.num_cycles == source.num_cycles
        assert np.array_equal(np.asarray(copied.params), parameters)
        assert np.array_equal(np.asarray(copied.get_unitary()), matrix)
        assert all(copy is not original for copy, original in zip(copied, source))
        copied.renumber_qudits(tuple(reversed(range(width))))
        copied.set_params(np.asarray(copied.params) + 0.1)
        assert np.array_equal(np.asarray(source.params), parameters)
        assert np.array_equal(np.asarray(source.get_unitary()), matrix)

def test_resolved_pam_copy_keeps_unknown_gate_deepcopy(monkeypatch):
    source = Circuit(2)
    source.append_gate(HGate(), 0)
    source.append_gate(CNOTGate(), (0, 1))
    ordinary = Circuit.copy
    calls = []
    def copy(self):
        calls.append(self)
        return ordinary(self)
    monkeypatch.setattr(Circuit, "copy", copy)
    copied = copy_resolved_pam_circuit(source)
    assert calls == [source]
    assert np.array_equal(np.asarray(copied.get_unitary()), np.asarray(source.get_unitary()))
