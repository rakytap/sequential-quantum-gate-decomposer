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
## \brief Functionality test cases for the N_Qubit_Decomposition class.



# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np


class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""

    def test_N_Qubit_Decomposition_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of class N_Qubit_Decomposition.

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)
    
        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=False, initial_guess="randomc" )


    def test_N_Qubit_Decomposition_3qubit(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 3-qubit unitary

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)

        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx, optimize_layer_num=False, initial_guess="zeros" )

        # start the decomposition
        decomp.Start_Decomposition(finalize_decomp=True, prepare_export=True)



    def test_N_Qubit_Decomposition_Qiskit_export(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 2-qubit unitary and retrive the corresponding Qiskit circuit

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 2

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)

        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=False, initial_guess="zeros" )

        # start the decomposition
        decomp.Start_Decomposition(finalize_decomp=True, prepare_export=True)

        # get the decomposing operations
        quantum_circuit = decomp.get_Quantum_Circuit()

        # print the list of decomposing operations
        decomp.List_Operations()

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
        decomposed_matrix = result.get_unitary(quantum_circuit)

    
        # the Umtx*Umtx' matrix
        product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
        # the error of the decomposition
        decomposition_error =  LA.norm(product_matrix - np.identity(4)*product_matrix[0,0], 2)

        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )

    def test_IBM_Chellenge(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 4-qubit unitary of the IBM chellenge

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition        
        from scipy.io import loadmat
    
        # load the unitary from file
        data = loadmat('Umtx.mat')  
        # The unitary to be decomposed  
        Umtx = data['Umtx']
        

        # creating a class to decompose the unitary
        cDecompose = qgd_N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=True, initial_guess="ZEROS" )
        

        # set the number of successive identical blocks in the optimalization of disentanglement of the n-th qubits
        cDecompose.set_Identical_Blocks( {4: 2, 3: 1} )

        # set the maximal number of layers in the decomposition
        cDecompose.set_Max_Layer_Num( {4: 9, 3:4} )

        # set the number of iteration loops in the decomposition
        cDecompose.set_Iteration_Loops({4: 3, 3: 3, 2: 3})

        # setting the verbosity of the decomposition
        cDecompose.set_Verbose( True )

        # starting the decomposition
        cDecompose.Start_Decomposition(finalize_decomp=True, prepare_export=True)

        # list the decomposing operations
        cDecompose.List_Operations()

        # get the decomposing operations
        quantum_circuit = cDecompose.get_Quantum_Circuit()

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
        decomposed_matrix = result.get_unitary(quantum_circuit)

    
        # the Umtx*Umtx' matrix
        product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
        # the error of the decomposition
        decomposition_error =  LA.norm(product_matrix - np.identity(16)*product_matrix[0,0], 2)

        print('The error of the decomposition is ' + str(decomposition_error))


      
    def test_N_Qubit_Decomposition_define_structure(self):
        r"""
        This method is called by pytest. 
        Test to define custom gate structure in the decomposition

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition
        from qgd_python.gates.qgd_Operation_Block import qgd_Operation_Block

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)
    
        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=False, initial_guess="random" )

        # create custom gate structure
        gate_structure = { 4: self.create_custom_gate_structure(4), 3: self.create_custom_gate_structure(3) }        


        # adding custom gate structure to the decomposition
        decomp.set_Gate_Structure( gate_structure )


        # starting the decomposition
        decomp.Start_Decomposition(finalize_decomp=True, prepare_export=True)

        # list the decomposing operations
        decomp.List_Operations()

      


    def test_N_Qubit_Decomposition_reorder_qbits(self):
        r"""
        This method is called by pytest. 
        Test to define custom gate structure in the decomposition

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition
        from qgd_python.gates.qgd_Operation_Block import qgd_Operation_Block

        # the number of qubits spanning the unitary
        qbit_num = 4

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)
    
        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=False, initial_guess="random" )


        # create custom gate structure
        gate_structure = { 4: self.create_custom_gate_structure(4), 3: self.create_custom_gate_structure(3)}        


        # adding custom gate structure to the decomposition
        decomp.set_Gate_Structure( gate_structure )


        # list of reordered qubits (original: (3,2,1,0) )
        reordered_qbits = (0,1,3,2)

        # adding custom gate structure to the decomposition
        decomp.Reorder_Qubits( reordered_qbits )




        # starting the decomposition
        decomp.Start_Decomposition(finalize_decomp=True, prepare_export=True)

        # list of reordered qubits to revert the initial order
        revert_qbits = (1,0,2,3)             

        # adding custom gate structure to the decomposition
        decomp.Reorder_Qubits( revert_qbits )

        # list the decomposing operations
        decomp.List_Operations()

        # get the decomposing operations
        quantum_circuit = decomp.get_Quantum_Circuit()


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
        decomposed_matrix = result.get_unitary(quantum_circuit)
    
        # the Umtx*Umtx' matrix
        product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)

        # the error of the decomposition
        decomposition_error =  LA.norm(product_matrix - np.identity(int(2**qbit_num))*product_matrix[0,0], 2)

        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )




    def create_custom_gate_structure(self, qbit_num):
        r"""
        This method is called to create custom gate structure for the decomposition

        """

        from qgd_python.gates.qgd_Operation_Block import qgd_Operation_Block

        # creating an instance of the wrapper class qgd_Operation_Block
        Operation_Block_ret = qgd_Operation_Block( qbit_num )

        control_qbit = qbit_num - 1 

        for target_qbit in range(0, control_qbit ):

            # creating an instance of the wrapper class qgd_Operation_Block
            Operation_Block_inner = qgd_Operation_Block( qbit_num )

            if target_qbit == 1:

                # add CNOT gate to the block
                Operation_Block_inner.add_CNOT_To_End( 0, 1)

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Operation_Block_inner.add_U3_To_End( 0, Theta, Phi, Lambda )                 
                Operation_Block_inner.add_U3_To_End( 1, Theta, Phi, Lambda ) 

            else:

                # add CNOT gate to the block
                Operation_Block_inner.add_CNOT_To_End( control_qbit, target_qbit )

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Operation_Block_inner.add_U3_To_End( target_qbit, Theta, Phi, Lambda )    
                Operation_Block_inner.add_U3_To_End( control_qbit, Theta, Phi, Lambda )  


            Operation_Block_ret.add_Operation_Block_To_End( Operation_Block_inner )

        return Operation_Block_ret
    
