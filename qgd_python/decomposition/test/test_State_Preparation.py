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
import pytest
from qgd_python.utils import get_unitary_from_qiskit_circuit

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False

class Test_State_Preparation:
	 def test_State_Preparation_adaptive_false(self): #atnevezni
		    r"""
		    This method is called by pytest. 
		    Test to decompose a 4-qubit unitary State

		    """

		    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_qubit_State_Preparation_adaptive       
		    from scipy.io import loadmat
		
		    # load the unitary from file
		    data = loadmat('Umtx.mat')  
		    # The unitary to be decomposed
		    Umtx = data['Umtx'].conj().T

		    # creating a class to decompose the unitary
		    with pytest.raises(Exception):
		    	 cDecompose = qgd_N_qubit_State_Preparation_adaptive( Umtx, level_limit_max=5, level_limit_min=0 )
		    
	 def test_State_Preparation_adaptive(self):
		    r"""
		    This method is called by pytest. 
		    Test to decompose a 4-qubit State 

		    """

		    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_qubit_State_Preparation_adaptive       
		    from scipy.io import loadmat
		
		    # load the unitary from file
		    data = loadmat('Umtx.mat')  
		    # The unitary to be decomposed  
		    Umtx = data['Umtx']
		    State = Umtx[:,0]
		    40

		    # creating a class to decompose the unitary
		    cDecompose = qgd_N_qubit_State_Preparation_adaptive( State, level_limit_max=5, level_limit_min=0 )

		    # setting the verbosity of the decomposition
		    cDecompose.set_Verbose( 3 )

		    # setting the verbosity of the decomposition
		    cDecompose.set_Cost_Function_Variant( 3 )
		    
		    # starting the decomposition
		    cDecompose.Start_Decomposition()

		    # list the decomposing operations
		    cDecompose.List_Gates()

		    # get the decomposing operations
		    quantum_circuit = cDecompose.get_Quantum_Circuit()

		    # print the quantum circuit
		    print(quantum_circuit)
		
		
		    # the unitary matrix from the result object
		    decomposed_matrix = np.asarray( get_unitary_from_qiskit_circuit( quantum_circuit ))
		    prepared_State = decomposed_matrix[:,0]
		    dot_prod = np.dot(State.conj().T, prepared_State)
		    print(np.abs(dot_prod))
		    
