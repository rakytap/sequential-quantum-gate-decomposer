import pytest
from squander import Circuit, CNOT, CH, CZ, CRY, H
from squander.partitioning.partition import (
    get_qubits,
    kahn_partition,
    translate_param_order,
    qasm_to_partitioned_circuit
)

"""
CORRECTNESS TESTS
"""

@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_Partition_Empty_Circuit(max_qubits):
    empty_c = Circuit(5)
    top_c, param_order, _  = kahn_partition(empty_c, max_qubits)
    assert len(top_c.get_Gates()) == 1 # NOTE: should be 0
    assert len(param_order) == 0


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_Partition_Single_Gate(max_qubits):
    single_c = Circuit(5)
    single_c.add_CNOT(0, 1)
    top_c, param_order, _  = kahn_partition(single_c, max_qubits)
    assert len(top_c.get_Gates()) == 1
    assert len(param_order) == 1


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_Partition_Total_Gates(max_qubits):
    c = Circuit(5)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CNOT(2, 3)
    top_c, _, _ = kahn_partition(c, max_qubits)
    total_gates = sum(len(p.get_Gates()) for p in top_c.get_Gates())
    assert total_gates == len(c.get_Gates())


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_Partition_Max_Qubit_Constraint(max_qubits):
    c = Circuit(5)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CNOT(2, 3)
    top_c, _, _ = kahn_partition(c, max_qubits)
    for p in top_c.get_Gates():
        qubits = set.union(*(get_qubits(gate) for gate in p.get_Gates()))
        assert len(qubits) <= max_qubits


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_Partition_Max_Qubits_Equals_Total_Qubits(max_qubits):
    c = Circuit(max_qubits)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    top_c, _, _ = kahn_partition(c, max_qubits)
    assert len(top_c.get_Gates()) == 1