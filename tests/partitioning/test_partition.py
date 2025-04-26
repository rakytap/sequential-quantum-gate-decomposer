import pytest
import numpy as np
from squander import Circuit, CNOT, CH, CZ, CRY, H
from squander import utils
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
    
    
    
    
def test_Correctness_of_Partitioned_Circuit():
    filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    
    initial_circuit, initial_parameters = utils.qasm_to_squander_circuit(filename)
    
    max_partition_size = 4
    partitined_circuit, partitioned_parameters = qasm_to_partitioned_circuit( filename, max_partition_size )
    
    
    # generate random initial state on which we test the circuits
    qbit_num = initial_circuit.get_Qbit_Num()
    
    
    matrix_size = 1 << qbit_num
    initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state = initial_state_real + initial_state_imag*1j
    initial_state = initial_state/np.linalg.norm(initial_state)
    


    transformed_state_1 = initial_state.copy()
    transformed_state_2 = initial_state.copy()    
    
    initial_circuit.apply_to( initial_parameters, transformed_state_1 )
    partitined_circuit.apply_to( partitioned_parameters, transformed_state_2)    
    
    diff = np.linalg.norm( transformed_state_1 - transformed_state_2 )
    
    assert( diff < 1e-10 )
    
    
    
    
    

