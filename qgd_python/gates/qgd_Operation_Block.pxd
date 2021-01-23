from libcpp cimport bool
from libcpp.vector cimport vector
#from piquasso.common.matrix cimport matrix

cdef extern from "../../operations/include/Operation_block.h":

    cdef cppclass Operation_block:
        Operation_block() except +
        Operation_block(int qbit_num_in) except +
        void add_u3_to_end(int target_qbit, bool Theta, bool Phi, bool Lambda)
        void add_cnot_to_end( int control_qbit, int target_qbit)
