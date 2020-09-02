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
import ctypes


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

# starting the decomposition
cDecompose.start_decomposition()
