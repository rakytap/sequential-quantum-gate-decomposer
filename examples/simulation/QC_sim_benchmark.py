# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file QC_sim_benchmark.py
## \brief Simple example python code demonstrating how to use state vector simulation with the SQUANDER package and compare the perfromance to QISKIT and Qulacs

from squander import Circuit       


import numpy as np
import random
import scipy.linalg
import time


np.set_printoptions(linewidth=200) 


# number of qubits
qbit_num_min = 4
qbit_num_max = 23

# number of levels
levels = 4

random_initial_state = False

##########################################################################################################
################################ SQUANER #################################################################

execution_times_squander = {}
transformed_states_squander = {}
parameters_squander = {}
initial_state_squander      = {}

for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):

	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	if (random_initial_state ) :
		initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
		initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
		initial_state = initial_state_real + initial_state_imag*1j
		initial_state = initial_state/np.linalg.norm(initial_state)
	else:
		initial_state = np.zeros( (matrix_size), dtype=np.complex128 )
		initial_state[0] = 1.0 + 0j

	initial_state_squander[ qbit_num ] = initial_state.copy()

	# prepare circuit

	circuit_squander = Circuit( qbit_num )

	gates_num = 0
	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_squander.add_U3(target_qbit)
				circuit_squander.add_U3(control_qbit)
				#circuit_squander.add_CNOT( target_qbit=target_qbit, control_qbit=control_qbit )
				circuit_squander.add_CRY( target_qbit=target_qbit, control_qbit=control_qbit )
				gates_num = gates_num + 3

	for target_qbit in range(qbit_num):
		circuit_squander.add_U3(target_qbit)
		gates_num = gates_num + 1
		break		



	num_of_parameters = circuit_squander.get_Parameter_Num()
	#print("The number of free parameters at qubit_num= ", qbit_num, ": ", num_of_parameters )


	parameters = np.random.rand(num_of_parameters)*2*np.pi

	t0 = time.time()
	circuit_squander.apply_to( parameters, initial_state )
	t_SQUANDER = time.time() - t0
	print( "Time elapsed SQUANDER: ", t_SQUANDER, " seconds at qbit_num = ", qbit_num, ' number of gates: ', gates_num )

	execution_times_squander[ qbit_num ] = t_SQUANDER
	transformed_states_squander[ qbit_num ] = np.reshape(initial_state, (initial_state.size,) )
	parameters_squander[ qbit_num ] = parameters


print("SQUANDER execution times [s]:")
print( execution_times_squander )


##########################################################################################################
################################ QISKIT #################################################################

execution_times_qiskit = {}
transformed_states_qiskit = {}


import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit
import qiskit_aer as Aer    
    
if qiskit_version[0] == '1':
    from qiskit import transpile
else :
    from qiskit import execute
    
    


for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):

	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	initial_state = initial_state_squander[ qbit_num ]

	parameters = parameters_squander[ qbit_num ]
	parameter_idx = 0

	# prepare circuit

	# creating Qiskit quantum circuit
	circuit_qiskit = QuantumCircuit(qbit_num)

	if random_initial_state:
		circuit_qiskit.initialize( initial_state )

	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_qiskit.u(parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2], target_qbit )
				parameter_idx = parameter_idx+3
				circuit_qiskit.u(parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2], control_qbit )
				parameter_idx = parameter_idx+3
				#circuit_qiskit.cx( control_qbit, target_qbit )
				circuit_qiskit.cry( parameters[parameter_idx]*2, control_qbit, target_qbit )
				parameter_idx = parameter_idx+1

	for target_qbit in range(qbit_num):
		circuit_qiskit.u(parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2], target_qbit )
		parameter_idx = parameter_idx+3
		break
		




	t0 = time.time()

	# Execute and get the state vector
	if qiskit_version[0] == '1' or qiskit_version[0] == '2':
	
		circuit_qiskit.save_statevector()
	
		backend = Aer.AerSimulator(method='statevector')
		compiled_circuit = transpile(circuit_qiskit, backend)
		result = backend.run(compiled_circuit).result()
		
		transformed_state = result.get_statevector(compiled_circuit)		
       
        
	elif qiskit_version[0] == '0':
	
		# Select the StatevectorSimulator from the Aer provider
		simulator = Aer.get_backend('statevector_simulator')	
		
		backend = Aer.get_backend('aer_simulator')
		result = execute(circuit_qiskit, simulator).result()
		
		transformed_state = result.get_statevector(circuit_qiskit)



	t_qiskit = time.time() - t0
	#print( "Time elapsed QISKIT: ", t_qiskit, " at qbit_num = ", qbit_num )

	execution_times_qiskit[ qbit_num ] = t_qiskit
	transformed_states_qiskit[ qbit_num ] = np.array(transformed_state)

print("QISKIT execution times [s]:")
print( execution_times_qiskit )


from qulacs import Observable, QuantumCircuit, QuantumState
import qulacs

execution_times_qulacs = {}
transformed_states_qulacs = {}


for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):

	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	initial_state = initial_state_squander[ qbit_num ]

	parameters = parameters_squander[ qbit_num ]
	parameter_idx = 0

	# prepare circuit

	# creating qulacs quantum circuit
	state = QuantumState(qbit_num)
	state.load( initial_state )

	circuit_qulacs = QuantumCircuit(qbit_num)

	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_qulacs.add_U3_gate(target_qbit, parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2] )
				parameter_idx = parameter_idx+3
				circuit_qulacs.add_U3_gate( control_qbit, parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2] )
				parameter_idx = parameter_idx+3

				#circuit_qulacs.add_CNOT_gate( control_qbit, target_qbit )
				
				RY_gate = qulacs.gate.RotY( target_qbit, parameters[parameter_idx]*2 )
				RY_gate = qulacs.gate.to_matrix_gate( RY_gate )
				RY_gate.add_control_qubit( control_qbit, 1)
				circuit_qulacs.add_gate( RY_gate )
				#circuit_qulacs.add_RotY_gate( target_qbit, parameters[parameter_idx]*2 )
				parameter_idx = parameter_idx+1
				

	for target_qbit in range(qbit_num):
		circuit_qulacs.add_U3_gate( target_qbit, parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2] )
		parameter_idx = parameter_idx+3
		break
		

	t0 = time.time()
	# Execute and get the state vector
	circuit_qulacs.update_quantum_state( state )
	transformed_state = state.get_vector()
	t_qulacs = time.time() - t0
	#print( "Time elapsed qulacs: ", t_qulacs, " at qbit_num = ", qbit_num )

	execution_times_qulacs[ qbit_num ] = t_qulacs
	transformed_states_qulacs[ qbit_num ] = np.array(transformed_state)

print("Qulacs execution times [s]:")
print( execution_times_qulacs )
# check errors

print(' ')
print("Difference between the transformed state vectors:")
# SQUANDER-QISKIT-Qulacs comparision
keys = transformed_states_qiskit.keys()
for qbit_num in keys:
	state_squander = transformed_states_squander[ qbit_num ]
	state_qiskit   = transformed_states_qiskit[ qbit_num ]
	state_qulacs   = transformed_states_qulacs[ qbit_num ]

	print( "Squander vs QISKIT: ", np.linalg.norm( state_squander-state_qiskit ) )
	print( "Squander vs Qulacs: ", np.linalg.norm( state_squander-state_qulacs ) )
	

