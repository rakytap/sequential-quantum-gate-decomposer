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
## \file example.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package



## [import]
from qgd_python.N_Qubit_Decomposition import N_Qubit_Decomposition 
## [import]

print('******************** Decomposing general 3-qubit matrix *******************************')

# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np

    
# the number of qubits spanning the unitary
qbit_num = 4

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)
   
# creating a random unitary to be decomposed
Umtx = unitary_group.rvs(matrix_size)

# creating a class to decompose the 
cDecompose = N_Qubit_Decomposition( Umtx.conj().T )

# starting the decomposition
cDecompose.start_decomposition()

# list the decomposing operations
#cDecompose.list_operations()

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
    
## load the unitary from file
data = loadmat('Umtx.mat')  
## The unitary to be decomposed  
Umtx = data['Umtx']
## [load Umtx]

## [create decomposition class]
## creating a class to decompose the unitary
cDecompose = N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=True, initial_guess="zeros" )
## [create decomposition class]

## [set parameters]
# set the number of successive identical blocks in the optimalization of disentanglement of the n-th qubits
cDecompose.set_identical_blocks( {4: 2, 3: 1} )

# set the maximal number of layers in the decomposition
cDecompose.set_max_layer_num( {4: 9, 3:4})

# set the number of iteration loops in the decomposition
cDecompose.set_iteration_loops({4: 3, 3: 3, 2: 3})

# setting the verbosity of the decomposition
cDecompose.set_verbose( True )
## [set parameters]

## [start decomposition]
# starting the decomposition
cDecompose.start_decomposition()

# list the decomposing operations
cDecompose.list_operations()
## [start decomposition]


## [qiskit]
print(' ')
print('Constructing quantum circuit:')
print(' ')
## Qiskit quantum circuit
quantum_circuit = cDecompose.get_quantum_circuit()

print(quantum_circuit)

from qiskit import execute
from qiskit import Aer
import numpy.linalg as LA
    
# test the decomposition of the matrix
## Qiskit backend for simulator
backend = Aer.get_backend('unitary_simulator')
    
## job execution and getting the result as an object
job = execute(quantum_circuit, backend)
## the result of the Qiskit job
result = job.result()
    
## the unitary matrix from the result object
decomposed_matrix = result.get_unitary(quantum_circuit)

    
## the Umtx*Umtx' matrix
product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
## the error of the decomposition
decomposition_error =  LA.norm(product_matrix - np.identity(16)*product_matrix[0,0], 2)

print('The error of the decomposition is ' + str(decomposition_error))

## [qiskit]

