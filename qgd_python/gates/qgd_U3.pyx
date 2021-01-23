# distutils: language = c++
#
# Copyright (C) 2020 by TODO - All rights reserved.
#



import numpy as np
cimport numpy as np
np.import_array()


from libcpp.vector cimport vector
from qgd_python.gates.qgd_U3 cimport U3 # import the C++ version of class U3
#from qgd_python.common.matrix cimport matrix # import the C++ version of class matrix



cdef class qgd_U3:
    """Object to represent a CNOT gate of the QGD package."""

    cdef U3 cU3 # class representing the C++ implementation of U3 gate


    def __init__(self, qbit_num, target_qbit, Theta, Phi, Lambda ):
        r"""
        A Python binding of the CNOT gate from the QGD package


        qbit_num (int): the number of qubits spanning the CNOT gate


        target_qbit (int): The id of the target qubit (0<=target_qbit<qbit_num)

        Theta (bool) logical value indicating whether the matrix creation takes an argument theta.

        Phi (bool) logical value indicating whether the matrix creation takes an argument phi

        Lambda (bool) logical value indicating whether the matrix creation takes an argument lambda


    """
       
        self.cU3 = U3( qbit_num, target_qbit, Theta, Phi, Lambda )

