from squander import SABRE
from squander import Qiskit_IO
from squander import utils
from squander import Circuit

from qiskit import transpile
from qiskit import QuantumCircuit
import numpy as np
parameters = np.array([])

# The example shows how to transpile a gicen circuit to a new circuit accounting for 
# the possible connections (i.e. topology or coupling map) between the qubits
# The circuit is transformed by SWAP gate insertions and qubit remapping (i.e. reordering)

# path to the circuit to be transpiled
filename = 'examples/partitioning/qasm_samples/heisenberg-16-20.qasm'
N = 16
# load the qasm file into a QISKIT circuit
circuit_qiskit = QuantumCircuit.from_qasm_file(filename)

# convert the QISKIT circuit into Squander circuti representation
Squander_initial_circuit, parameters_initial = Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit)

# defining the qubit topology/connectivity for Squander
topology =     [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15),
    (0, 8),]

# transpile the circuit by the Sabre method implemented in Squander
sabre = SABRE(Squander_initial_circuit, topology,stochastic_selection=True)
Squander_remapped_circuit, parameters_remapped_circuit, pi, final_pi, swap_count = sabre.map_circuit(parameters_initial,30)

#Umtx = Operator(circuit_qiskit).to_matrix()

print("INITIAL CIRCUIT:")
#print( circuit_qiskit )
print("mapping (q -> Q):", pi)
qubits = list(range(N))
pi_map = list(np.array(sabre.get_inverse_pi(final_pi)))
print("Final mapping:", final_pi)
final_circuit = Circuit(N)
final_circuit.add_Permutation(list(pi)) 
final_circuit.add_Circuit(Squander_remapped_circuit)
final_circuit.add_Permutation(list(pi_map))
Qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit( final_circuit.get_Flat_Circuit(), parameters_remapped_circuit )
print("CIRCUIT MAPPED WITH SABRE:")
#print( Qiskit_circuit )
print("SABRE SWAP COUNT:", swap_count)
# defining the qubit topology/connectivity for Squander
coupling_map = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
    [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15],
    [0, 8],
]
# transpile the circuit by Qiskit
Qiskit_circuit_mapped = transpile(circuit_qiskit, coupling_map=coupling_map)

print("CIRCUIT MAPPED WITH QISKIT:")
#print( Qiskit_circuit_mapped )
print("QISKIT SWAP COUNT:",  dict(Qiskit_circuit_mapped.count_ops())['swap'])
num_qubits = final_circuit.get_Qbit_Num() 
matrix_size = 1 << num_qubits 
initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state = initial_state_real + initial_state_imag*1j
initial_state = initial_state/np.linalg.norm(initial_state)
original_state = initial_state.copy()
Squander_initial_circuit.apply_to(parameters_initial,original_state)
SABRE_state = initial_state.copy()
final_circuit.apply_to(parameters_remapped_circuit,SABRE_state)
print(f"ERROR: {1-abs(np.vdot(SABRE_state,original_state))}")