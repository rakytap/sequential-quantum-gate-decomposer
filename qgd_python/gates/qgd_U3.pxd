from libcpp cimport bool
from libcpp.vector cimport vector
#from piquasso.common.matrix cimport matrix

cdef extern from "../../operations/include/U3.h":

    cdef cppclass U3:
        U3() except +
        U3(int qbit_num_in, int target_qbit_in, bool theta_in, bool phi_in, bool lambda_in) except +
