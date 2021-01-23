# distutils: language = c++
#
# Copyright (C) 2020 by TODO - All rights reserved.
#



import numpy as np
cimport numpy as np
np.import_array()


from libcpp.vector cimport vector
from qgd_python.gates.qgd_Operation_Block cimport Operation_block # import the C++ version of class Operation_block
#from qgd_python.common.matrix cimport matrix # import the C++ version of class matrix



cdef class qgd_Operation_Block:
    """Object to represent a CNOT gate of the QGD package."""

    cdef Operation_block cOperation_block # class representing the C++ implementation of Operation_block class


    def __init__(self, qbit_num ):
        r"""
        A Python binding of the class Operation_block from the QGD package 
        Operation_block describes a block a gates.

        qbit_num (int): the number of qubits spanning the CNOT gate


    """
       
        self.cOperation_block = Operation_block( qbit_num )

    def add_U3_To_End(self, target_qbit, Theta, Phi, Lambda):
        r"""
        Append a U3 gate to the block of gates


        qbit_num (int): the number of qubits spanning the CNOT gate


        target_qbit (int): The id of the target qubit (0<=target_qbit<qbit_num)

        Theta (bool) logical value indicating whether the matrix creation takes an argument theta.

        Phi (bool) logical value indicating whether the matrix creation takes an argument phi

        Lambda (bool) logical value indicating whether the matrix creation takes an argument lambda


    """

        # add U3 gate to the block of operations
        self.cOperation_block.add_u3_to_end(target_qbit, Theta, Phi, Lambda)


    def add_CNOT_To_End(self, target_qbit, control_qbit):
        r"""
        Append a CNOT gate to the block of gates


        qbit_num (int): the number of qubits spanning the CNOT gate

        target_qbit (int): The id of the target qubit (0<=target_qbit<qbit_num)

        control_qbit (int): The id of the target qubit (0<=control_qbit<qbit_num)

    """

        # add CNOT gate to the block of operations
        self.cOperation_block.add_cnot_to_end(target_qbit, control_qbit)
