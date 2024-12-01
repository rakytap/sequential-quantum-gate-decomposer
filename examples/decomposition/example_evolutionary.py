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
## \brief Simple example python code demonstrating the a decomposiiton of a deep circuit.

## [import adaptive]
from squander import N_Qubit_Decomposition_adaptive       
## [import adaptive]

import numpy as np
from qiskit import QuantumCircuit


#Here we provide an example to use the SQUANDER package. The following python interface is accessible from version 1.8.0. In this example we use two optimization engines for the decomposition:

#   An evolutionary engine called AGENTS
#   Second order gradient descend algorithm (BFGS)
    

# load the quantum circuit from a file andretrieve the unitary of the circuit
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
Umtx = Umtx*np.exp(-1j*(qc.global_phase))



################################# gate systhesis #####################################

# Firstly we construct a Python map to set hyper-parameters during the gate synthesis.

# Python map containing hyper-parameters
config = {      'agent_lifetime':200,
                'max_inner_iterations_agent': 100000,
                'max_inner_iterations_compression': 100000,
                'max_inner_iterations' : 10000,
                'max_inner_iterations_final': 10000, 		
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-3,
                'optimization_tolerance_': 1e-8}
                

# Next we initialize the decomposition class with the unitary Umtx to be decomposed.

# creating a class to decompose the unitary
## [creating decomp class]
cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config )
## [creating decomp class]

# The verbosity of the execution output can be controlled by the function call

# setting the verbosity of the decomposition
## [set verbosity]
cDecompose.set_Verbose( 3 )
## [set verbosity]

# We construct the initial trial gate structure for the optimization consisting of 2 levels of adaptive layer. (1 level is made of qubit_num*(qubit_num-1) two-qubit building blocks if all-to-all connectivity is assumed)

# adding decomposing layers to the gate structure
## [set adaptive gate structure]
levels = 5
for idx in range(levels):
    cDecompose.add_Adaptive_Layers()

cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
## [set adaptive gate structure]

# We can construct an initial parameters set for the optimization by retrieving the number of free parameters. If the initial parameter set is not set, random parameters are used by default.

# setting intial parameter set
## [set parameters]
parameter_num = cDecompose.get_Parameter_Num()
parameters = np.zeros( (parameter_num,1), dtype=np.float64 )
cDecompose.set_Optimized_Parameters( parameters )
## [set parameters]


# We can choose between several engines to solve the optimization problem. Here we use an evolutionary based algorithm named 'AGENTS'

# setting optimizer
## [set AGENTS optimizer]
cDecompose.set_Optimizer("AGENTS")
## [set AGENTS optimizer]

# The optimization process is started by the function call


## [start optimization]
# starting the decomposition
cDecompose.get_Initial_Circuit()
## [start optimization]

#The optimization process terminates by either reaching the tolerance 'optimization_tolerance_agent' or by reaching the maximal iteration number 'max_inner_iterations_agent', or if the engines identifies a convergence to a local minimum. The SQUANDER framework enables one to continue the optimization using a different engine. In particular we set a second order gradient descend method 'BFGS' In order to achieve the best performance one can play around with the hyper-parameters in the map 'config'. (Optimization strategy AGENTS is good in avoiding local minima or get through flat areas of the optimization landscape. Then a gradient descend method can be used for faster convergence toward a solution.)


# setting optimizer
## [set BFGS optimizer]
cDecompose.set_Optimizer("BFGS")
## [set BFGS optimizer]

# continue the decomposition with a second optimizer method
## [start BFGS optimization]
cDecompose.get_Initial_Circuit()
## [start BFGS optimization]

#After solving the optimization problem for the initial gate structure, we can initiate gate compression iterations. (This step can be omited.)

# starting compression iterations
## [start compression phase]
cDecompose.Compress_Circuit()
## [start compression phase]

#By finalizing the gate structure we replace the CRY gates with CNOT gates. (CRY gates with small rotation angle are approximately expressed with a single CNOT gate, so further optimization process needs to be initiated.)

# finalize the gate structure (replace CRY gates with CNOT gates)
## [finalyze circuit]
cDecompose.Finalize_Circuit()
## [finalyze circuit]

# Finally, we can retrieve the decomposed quantum circuit in QISKIT format.

## [qiskit]
print(' ')
print('Constructing quantum circuit:')
print(' ')
## Qiskit quantum circuit
quantum_circuit = cDecompose.get_Qiskit_Circuit()

print(quantum_circuit)

import numpy.linalg as LA
    
## the unitary matrix from the result object
from squander import utils
decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
phase = np.angle(product_matrix[0,0])
product_matrix = product_matrix*np.exp(-1j*phase)
    
product_matrix = np.eye(product_matrix.shape[0])*2 - product_matrix - product_matrix.conj().T
# the error of the decomposition
decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
print('The error of the decomposition is ' + str(decomposition_error))

## [qiskit]
