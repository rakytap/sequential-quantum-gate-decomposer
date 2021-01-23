from libcpp cimport bool
from libcpp.vector cimport vector
#from piquasso.common.matrix cimport matrix

cdef extern from "../../operations/include/CNOT.h":

    cdef cppclass CNOT:
        CNOT() except +
        CNOT(int qbit_num_in, int target_qbit_in,  int control_qbit_in) except +
