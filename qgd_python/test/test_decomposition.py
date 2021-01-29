# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np


class Test_Example:
    """This is an example class to demonstrate how to interface with a C++ part of the piquasso project."""

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
        decomp = qgd_N_Qubit_Decomposition( Umtx, qbit_num ) #, False, "ZEROS" )


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
        decomp = qgd_N_Qubit_Decomposition( Umtx, qbit_num )#, False, "ZEROS" )

        # start the decomposition
        decomp.Start_Decomposition(1,1)



      

