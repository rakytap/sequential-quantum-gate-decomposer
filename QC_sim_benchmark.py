# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file test_optimization_problem_combined.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package

from squander import Gates_Block       


#from squander import nn

import numpy as np
import random
import scipy.linalg
import time
from scipy.fft import fft

import time

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


np.set_printoptions(linewidth=200) 


# number of qubits
qbit_num_min = 2
qbit_num_max = 22

# number of levels
levels = 4

##########################################################################################################
################################ SQUANER #################################################################

execution_times_squander = {}
transformed_states_squander = {}
parameters_squander = {}

for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):

	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	initial_state = np.zeros( (matrix_size, 1,), dtype=np.complex128 )
	initial_state[0] = 1+0j


	# prepare circuit

	circuit_squander = Gates_Block( qbit_num )

	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_squander.add_U3(target_qbit, True, True, True )
				circuit_squander.add_U3(control_qbit, True, True, True )
				circuit_squander.add_CNOT( target_qbit=target_qbit, control_qbit=control_qbit )

	for target_qbit in range(qbit_num):
		circuit_squander.add_U3(target_qbit, True, True, True )
		


	num_of_parameters = circuit_squander.get_Parameter_Num()
	#print("The number of free parameters at qubit_num= ", qbit_num, ": ", num_of_parameters )


	parameters = np.random.rand(num_of_parameters)*2*np.pi

	t0 = time.time()
	circuit_squander.apply_to( parameters, initial_state )
	t_SQUANDER = time.time() - t0
	print( "Time elapsed SQUANDER: ", t_SQUANDER, " at qbit_num = ", qbit_num )

	execution_times_squander[ qbit_num ] = t_SQUANDER
	transformed_states_squander[ qbit_num ] = np.reshape(initial_state, (initial_state.size,) )
	parameters_squander[ qbit_num ] = parameters


print( execution_times_squander )


##########################################################################################################
################################ QISKIT #################################################################

execution_times_qiskit = {}
transformed_states_qiskit = {}

from qiskit import QuantumCircuit
from qiskit import Aer, execute

for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):

	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	initial_state = np.zeros( (matrix_size, 1,), dtype=np.complex128 )
	initial_state[0] = 1+0j

	parameters = parameters_squander[ qbit_num ]
	parameter_idx = parameters.size-1

	# prepare circuit

	# creating Qiskit quantum circuit
	circuit_qiskit = QuantumCircuit(qbit_num)

	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_qiskit.u(parameters[parameter_idx-2]*2, parameters[parameter_idx-1], parameters[parameter_idx], target_qbit )
				parameter_idx = parameter_idx-3
				circuit_qiskit.u(parameters[parameter_idx-2]*2, parameters[parameter_idx-1], parameters[parameter_idx], control_qbit )
				parameter_idx = parameter_idx-3
				circuit_qiskit.cx( control_qbit, target_qbit )

	for target_qbit in range(qbit_num):
		circuit_qiskit.u(parameters[parameter_idx-2]*2, parameters[parameter_idx-1], parameters[parameter_idx], target_qbit )
		parameter_idx = parameter_idx-3
		


	# Select the StatevectorSimulator from the Aer provider
	simulator = Aer.get_backend('statevector_simulator')

	t0 = time.time()
	# Execute and get the state vector
	result = execute(circuit_qiskit, simulator).result()
	transformed_state = result.get_statevector(circuit_qiskit)
	t_qiskit = time.time() - t0
	print( "Time elapsed QISKIT: ", t_qiskit, " at qbit_num = ", qbit_num )

	execution_times_qiskit[ qbit_num ] = t_qiskit
	transformed_states_qiskit[ qbit_num ] = np.array(transformed_state)

print( execution_times_qiskit )



# check errors

# SQUANDER-QISKIT
keys = transformed_states_qiskit.keys()
for qbit_num in keys:
	state_squander = transformed_states_squander[ qbit_num ]
	state_qiskit   = transformed_states_qiskit[ qbit_num ]

	print( np.linalg.norm( state_squander-state_qiskit ) )

