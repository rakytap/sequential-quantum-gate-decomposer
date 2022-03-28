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


class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""

    def ctest_N_Qubit_Decomposition_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of class N_Qubit_Decomposition.

        """

        from qgd_python.decomposition.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)
    
        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=False, initial_guess="random" )

    def test_N_Qubit_Decomposition_3qubit(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 3-qubit unitary

        """

        from qgd_python.decomposition.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)

        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx )

        # start the decomposition
        decomp.Start_Decomposition(prepare_export=True)

        decomp.List_Gates()

    def test_N_Qubit_Decomposition_Qiskit_export(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 2-qubit unitary and retrive the corresponding Qiskit circuit

        """

        from qgd_python.decomposition.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 2

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)

        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T )

        # set the number of block to be optimized in one shot
        decomp.set_Optimization_Blocks( 20 )

        # setting the verbosity of the decomposition
        decomp.set_Verbose( 3 )

        # setting the debugfile name. If it is not set, the program will not debug.
        decomp.set_Debugfile( "debugfile.txt" )

        # start the decomposition
        decomp.Start_Decomposition()

        # get the decomposing operations
        quantum_circuit = decomp.get_Quantum_Circuit()

        # print the list of decomposing operations
        decomp.List_Gates()

        # print the quantum circuit
        print(quantum_circuit)

        from qiskit import execute
        from qiskit import Aer
        import numpy.linalg as LA
    
        # test the decomposition of the matrix
        # Qiskit backend for simulator
        backend = Aer.get_backend('unitary_simulator')
     
        # job execution and getting the result as an object
        job = execute(quantum_circuit, backend)
        # the result of the Qiskit job
        result = job.result()
    
        # the unitary matrix from the result object
        decomposed_matrix = np.asarray( result.get_unitary(quantum_circuit) )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )
