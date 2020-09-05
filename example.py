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

# @brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package


from qgd_python.N_Qubit_Decomposition import N_Qubit_Decomposition 

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
cDecompose = N_Qubit_Decomposition( Umtx )

cDecompose.set_identical_blocks( 3, 2 )

# starting the decomposition
cDecompose.start_decomposition()

print(' ')
print(' ')
print(' ')
print('*******************************************************')
print('*******************************************************')

#exit()

print('******************** Solving the 4th IBM chellenge *******************************')

#******************************
from scipy.io import loadmat
    
# load the matrix from file
data = loadmat('Umtx.mat')    
Umtx = data['Umtx']

# the number of qubits spanning the unitary
qbit_num = 4

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)

# creating a class to decompose the 
cDecompose = N_Qubit_Decomposition( Umtx, optimize_layer_num=True, initial_guess="zeros" )

# set the number of successive identical blocks in the optimalization of disentanglement of the n-th qubits
cDecompose.set_identical_blocks( 4, 2 )
cDecompose.set_identical_blocks( 3, 1 )

# set the maximal number of layers in the decomposition
cDecompose.set_max_layer_num(4, 9)
cDecompose.set_max_layer_num(3, 4)

# set the number of iteration loops in the decomposition
cDecompose.set_iteration_loops(4, 3)
cDecompose.set_iteration_loops(3, 3)
cDecompose.set_iteration_loops(2, 3)

# starting the decomposition
cDecompose.start_decomposition()

