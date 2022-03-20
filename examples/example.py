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
from qgd_python.decomposition.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition 
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive       
## [import]

print('******************** Decomposing general 3-qubit matrix *******************************')

# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np

    
# the number of qubits spanning the unitary
qbit_num = 3

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)
   
# creating a random unitary to be decomposed
Umtx = unitary_group.rvs(matrix_size)

# creating a class to decompose the 
cDecompose = qgd_N_Qubit_Decomposition( Umtx.conj().T )

# setting the tolerance of the optimization process. The final error of the decomposition would scale with the square root of this value.
cDecompose.set_Optimization_Tolerance( 1e-12 )

# set the number of block to be optimized in one shot
cDecompose.set_Optimization_Blocks( 20 )

# starting the decomposition
cDecompose.Start_Decomposition()

# list the decomposing operations
cDecompose.List_Gates()

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


# determine the size of the unitary to be decomposed
matrix_size = len(Umtx)

## [create decomposition class]
## creating a class to decompose the unitary
cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=0 )
## [create decomposition class]

## [set parameters]

# setting the verbosity of the decomposition
cDecompose.set_Verbose( 3 )

# setting the debugfile name
cDecompose.set_Debugfile( "debug" )

## [set parameters]

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
quantum_circuit = cDecompose.get_Quantum_Circuit()

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
decomposed_matrix = np.asarray( result.get_unitary(quantum_circuit) )
product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
phase = np.angle(product_matrix[0,0])
product_matrix = product_matrix*np.exp(-1j*phase)
    
product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T
# the error of the decomposition
decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
print('The error of the decomposition is ' + str(decomposition_error))

## [qiskit]





