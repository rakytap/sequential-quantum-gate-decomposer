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

@author: Peter Rakyta, Ph.D.
"""
## \file example_evolutionary.py
## \brief Simple example python code demonstrating a state preparation experiment

from squander import N_Qubit_State_Preparation_adaptive

import numpy as np
from qiskit import QuantumCircuit


#Here we provide an example to use the SQUANDER package. The following python interface is accessible from version 1.8.8. In this example we use two optimization engines for the sate preparation:

#   An evolutionary engine called AGENTS
#   Second order gradient descend algorithm (BFGS)
    
# A state to be generated will be a single column of a unitary 

# to construct the unitary we load the quantum circuit from a file and retrieve the unitary of the circuit
qc = QuantumCircuit.from_qasm_file( '../../benchmarks/IBM/alu-v4_37.qasm') 


from qiskit import execute
from qiskit import Aer
import numpy.linalg as LA
    
# test the decomposition of the matrix
## Qiskit backend for simulator
backend = Aer.get_backend('unitary_simulator')
    
## job execution and getting the result as an object
job = execute(qc, backend)
## the result of the Qiskit job
result = job.result()
    
## the unitary matrix from the result object
Umtx = result.get_unitary(qc)
Umtx = np.asarray(Umtx)


# now take the first column of the unitary to get the state
State = np.ascontiguousarray( Umtx[:,0].copy() )



################################# state preparation #####################################

# Firstly we construct a Python map to set hyper-parameters during the gate synthesis.

# Python map containing hyper-parameters
config = {      'agent_lifetime':1000,
                'agent_num': 64,
                'max_inner_iterations_agent': 10000,
                'max_inner_iterations_compression': 1000,
                'max_inner_iterations' : 100,
                'max_inner_iterations_final': 100, 		
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-8,
                'optimization_tolerance': 1e-16,
		'convergence_length': 50}
                

# Next we initialize the decomposition class with the unitary Umtx to be decomposed.

# creating a class to decompose the unitary
cStatePrep = N_Qubit_State_Preparation_adaptive( State, config=config )

# The verbosity of the execution output can be controlled by the function call

# setting the verbosity of the decomposition
# set -1 to fully suppress verbosity
cStatePrep.set_Verbose( 3 )

# We construct the initial trial gate structure for the optimization consisting of 2 levels of adaptive layer. (a level is made of qubit_num*(qubit_num-1) two-qubit building blocks if all-to-all connectivity is assumed)

# add initial decomposing layers to the gate structure
levels = 2
for idx in range(levels):
    cStatePrep.add_Adaptive_Layers()

cStatePrep.add_Finalyzing_Layer_To_Gate_Structure()

# We can construct an initial parameters set for the optimization by retrieving the number of free parameters. If the initial parameter set is not set, random parameters are used by default.
# setting intial parameter set
parameter_num = cStatePrep.get_Parameter_Num()
#parameters    = np.zeros( (parameter_num,1), dtype=np.float64 )
parameters    = np.random.randn( parameter_num )
cStatePrep.set_Optimized_Parameters( parameters )


# adding new layer to the decomposition until a threshold reached
#The optimization process terminates by either reaching the tolerance 'optimization_tolerance_agent' or by reaching the maximal iteration number 'max_inner_iterations_agent', or if the engines identifies a convergence to a local minimum. The SQUANDER framework enables one to continue the optimization using a different engine. In particular we set a second order gradient descend method 'BFGS' In order to achieve the best performance one can play around with the hyper-parameters in the map 'config'. (Optimization strategy AGENTS is good in avoiding local minima or get through flat areas of the optimization landscape. Then a gradient descend method can be used for faster convergence toward a solution.)

for new_layer in range(4):

	print(' ')
	print(' ')
	print(' Adding new layer to the gate structure')
	print(' ')
	print(' ')


	# We can choose between several engines to solve the optimization problem. Here we use an evolutionary based algorithm named 'AGENTS'


	# setting optimizer
	cStatePrep.set_Optimizer("AGENTS")

	# starting the decomposition
	cStatePrep.get_Initial_Circuit()

	# store parameters after the evolutionary optimization 
	# (After BFGS converge to local minimum, evolutionary optimization wont make it better)
	params_AGENTS = cStatePrep.get_Optimized_Parameters()

	# setting optimizer
	cStatePrep.set_Optimizer("BFGS")

	# continue the decomposition with a second optimizer method
	cStatePrep.get_Initial_Circuit()   

	params_BFGS  = cStatePrep.get_Optimized_Parameters()
	decomp_error = cStatePrep.Optimization_Problem( params_BFGS )		

	if decomp_error <= config['optimization_tolerance_agent']:
		break
	else:
		# restore parameters to the evolutionary ones since BFGS iterates to local minima
		cStatePrep.set_Optimized_Parameters( params_AGENTS )

		# add new layer to the decomposition if toleranace was not reached
		cStatePrep.add_Layer_To_Imported_Gate_Structure()



#After solving the optimization problem for the initial gate structure, we can initiate gate compression iterations. (This step can be omited.)

# starting compression iterations
cStatePrep.Compress_Circuit()

#By finalizing the gate structure we replace the CRY gates with CNOT gates. (CRY gates with small rotation angle are approximately expressed with a single CNOT gate, so further optimization process needs to be initiated.)

# finalize the gate structure (replace CRY gates with CNOT gates)
cStatePrep.Finalize_Circuit()

# Finally, we can retrieve the decomposed quantum circuit in QISKIT format.

print(' ')
print('Constructing quantum circuit:')
print(' ')
## Qiskit quantum circuit
quantum_circuit = cStatePrep.get_Quantum_Circuit()

print( quantum_circuit )


