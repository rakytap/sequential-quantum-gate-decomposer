# distutils: language = c++
#
# Copyright (C) 2020 by TODO - All rights reserved.
#



import numpy as np
cimport numpy as np
np.import_array()


from libcpp.vector cimport vector
from qgd_python.gates.CNOT cimport CNOT # import the C++ version of class CNOT
#from qgd_python.common.matrix cimport matrix # import the C++ version of class matrix



cdef class qgd_CNOT:
    """Object to represent a CNOT gate of the QGD package."""

    cdef CNOT cCNOT # class representing the C++ implementation of CNOT gate


    def __init__(self, qbit_num, target_qbit, control_qbit ):
        r"""
        A Python binding of the CNOT gate from the QGD package


        qbit_num (int): the number of qubits spanning the CNOT gate

        target_qbit (int): The id of the target qubit (0<=target_qbit<qbit_num)

        control_qbit (int): The id of the target qubit (0<=control_qbit<qbit_num)

    """
       
        self.cCNOT = CNOT( qbit_num, target_qbit, control_qbit )

