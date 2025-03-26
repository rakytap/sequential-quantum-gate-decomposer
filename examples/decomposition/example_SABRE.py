from squander import SABRE
import numpy as np
import squander
from copy import copy
from squander import Qiskit_IO
parameters = np.array([])
from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
filename = 'benchmarks/qfast/4q/adder_q4.qasm'
circuit_qiskit = QuantumCircuit.from_qasm_file(filename)
initial_circuit,parameters = Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit)
topology = [(0,1),(0,2),(0,3)]
sabre = SABRE(initial_circuit,topology)
map_circuit,new_parameters,pi,final_pi,swap_count = sabre.map_circuit(parameters)
Qiskit_circuit_init = Qiskit_IO.get_Qiskit_Circuit( initial_circuit, parameters )
Umtx = Operator(circuit_qiskit).to_matrix()
print("INITIAL CIRCUIT:")
print( Qiskit_circuit_init )
print("mapping (q -> Q):", pi)
print("Final mapping:", final_pi)
Qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit( map_circuit, new_parameters )
print("CIRCUIT MAPPED WITH SABRE:")
print( Qiskit_circuit )
print("SABRE SWAP COUNT:", swap_count)
coupling_map = [[0,1],[0,2],[0,3]]
Qiskit_circuit_mapped = transpile(Qiskit_circuit_init, basis_gates=['cx','h','cz','swap','u3'],coupling_map=coupling_map)
print("CIRCUIT MAPPED WITH QISKIT:")
print( Qiskit_circuit_mapped )
print("QISKIT SWAP COUNT:",  dict(Qiskit_circuit_mapped.count_ops())['swap'])

