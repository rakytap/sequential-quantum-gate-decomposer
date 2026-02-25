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
## \file example.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package

## [import]
from squander import N_Qubit_Decomposition 
## [import]
## [import adaptive]
from squander import N_Qubit_Decomposition_adaptive       
## [import adaptive]


print(' ')
print(' ')
print(' ')
print('**********************************************************************************')
print('**********************************************************************************')
print('******************** Solving the 4th IBM chellenge *******************************')
print(' ')
print(' ')
print(' ')


#******************************
## [load Umtx]
from scipy.io import loadmat
import numpy as np


from squander import utils
    
## load the unitary from file
data = loadmat('examples/decomposition/Umtx.mat')  
## The unitary to be decomposed  
Umtx = data['Umtx']
## [load Umtx]


# determine the size of the unitary to be decomposed
matrix_size = len(Umtx)

## [create decomposition class]
## creating a class to decompose the unitary
method = "basin_hopping"
#method = "differential_evolution"
#method = "dual_annealing"
iteration_loops = {4: 3, 3: 3, 2: 3}
if method == "basin_hopping":
    config = {"use_basin_hopping": 1,
              "bh_T": 1.0, "bh_stepsize": 0.5, "bh_interval": 50,
              "bh_target_accept_rate": 0.5, "bh_stepwise_factor": 0.9}
elif method == "differential_evolution":
    config = {"use_differential_evolution": 1, 
              "de_strategy": 0, #0 is STRAT_BEST1BIN, 1 is STRAT_RAND1BIN
              "de_popsize": 15, "de_mutation": 0.8, "de_recombination": 0.7,
              "de_tol": 1e-6, "de_init": 0, #0 is INIT_RANDOM, 1 is INIT_LHS
              "de_polish": 1
              }
elif method == "dual_annealing": #default
    config = {"use_dual_annealing": 1,
              "da_initial_temp": 5330.0, "da_restart_temp_ratio": 2e-5, "da_visit": 2.62, "da_accept": -5.0,
              "da_maxiter": max(iteration_loops.values()), "da_maxfun": 10000000, "da_tol": 1e-6, "da_no_local_search": 0
              }
print("Using config:", config)
cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config, level_limit_max=5, level_limit_min=1 )
cDecompose.set_Iteration_Loops(iteration_loops)
## [create decomposition class]

# set the optimization engine to be used

cDecompose.set_Optimizer("BFGS2")


## [start decomposition]
# starting the decomposition
cDecompose.Start_Decomposition()

# list the decomposing operations
cDecompose.List_Gates()
## [start decomposition]


## [qiskit]
print(' ')
print('Constructing quantum circuit:')
print(' ')
## Qiskit quantum circuit
quantum_circuit = cDecompose.get_Qiskit_Circuit()

print(quantum_circuit)

import numpy.linalg as LA
    
## the unitary matrix from the result object
decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
phase = np.angle(product_matrix[0,0])
product_matrix = product_matrix*np.exp(-1j*phase)
    
product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T
# the error of the decomposition
decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
print('The error of the decomposition is ' + str(decomposition_error))

## [qiskit]








