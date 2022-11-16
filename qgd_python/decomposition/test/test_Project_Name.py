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
## \file test_decomposition.py
## \brief Functionality test cases for the qgd_N_Qubit_Decomposition class.



# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np
import os

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False

from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
from scipy.io import loadmat

def test_Project_Name():
	# load the unitary from file
	data = loadmat('Umtx.mat')
	# The unitary to be decomposed  
	Umtx = data['Umtx']
	# creating a class to decompose the unitary
	cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=0 )
	project_name = cDecompose.get_Project_Name()
	assert(project_name=="")
	cDecompose.set_Project_Name("TEST_")
	project_name = cDecompose.get_Project_Name()
	assert(project_name=="TEST_")
	cDecompose.export_Unitary("unitary_project_name_test.binary")
	assert(os.path.exists(project_name+"unitary_project_name_test.binary"))
	
	
