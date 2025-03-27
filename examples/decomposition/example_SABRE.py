from squander import SABRE
from squander import Qiskit_IO
from squander import utils

from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator

import numpy as np
parameters = np.array([])

# The example shows how to transpile a gicen circuit to a new circuit accounting for 
# the possible connections (i.e. topology or coupling map) between the qubits
# The circuit is transformed by SWAP gate insertions and qubit remapping (i.e. reordering)

# path to the circuit to be transpiled
filename = 'benchmarks/qfast/4q/adder_q4.qasm'

# load the qasm file into a QISKIT circuit
circuit_qiskit = QuantumCircuit.from_qasm_file(filename)

# convert the QISKIT circuit into Squander circuti representation
Squander_initial_circuit, parameters_initial = Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit)

# defining the qubit topology/connectivity for Squander
topology = [(0,1),(0,2),(0,3)]

# transpile the circuit by the Sabre method implemented in Squander
sabre = SABRE(Squander_initial_circuit, topology)
Squander_remapped_circuit, parameters_remapped_circuit, pi, final_pi, swap_count = sabre.map_circuit(parameters_initial)

#Umtx = Operator(circuit_qiskit).to_matrix()

print("INITIAL CIRCUIT:")
print( circuit_qiskit )

print("mapping (q -> Q):", pi)
print("Final mapping:", final_pi)

Qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit( Squander_remapped_circuit, parameters_remapped_circuit )
print("CIRCUIT MAPPED WITH SABRE:")
print( Qiskit_circuit )
print("SABRE SWAP COUNT:", swap_count)

# defining the qubit topology/connectivity for Squander
coupling_map = [[0,1],[0,2],[0,3]]
'''
# transpile the circuit by Qiskit
Qiskit_circuit_mapped = transpile(circuit_qiskit, basis_gates=['cx','h','cz','swap','u3'], coupling_map=coupling_map)

print("CIRCUIT MAPPED WITH QISKIT:")
print( Qiskit_circuit_mapped )
print("QISKIT SWAP COUNT:",  dict(Qiskit_circuit_mapped.count_ops())['swap'])
'''

# test the generated squander circuits
matrix_size = 1 << Squander_initial_circuit.get_Qbit_Num()
unitary_squander_initial = np.eye( matrix_size, dtype=np.complex128 )
Squander_initial_circuit.apply_to( parameters_initial, unitary_squander_initial)


unitary_squander_remapped_circuit = np.eye( 1 << Squander_initial_circuit.get_Qbit_Num(), dtype=np.complex128 )
Squander_remapped_circuit.apply_to( parameters_remapped_circuit, unitary_squander_remapped_circuit)



product_matrix = np.dot(unitary_squander_initial, unitary_squander_remapped_circuit.conj().T)
phase = np.angle(product_matrix[0,0])
product_matrix = product_matrix*np.exp(-1j*phase)


    
product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T

# the error of the decomposition
decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
print('The error of the decomposition is ' + str(decomposition_error))

