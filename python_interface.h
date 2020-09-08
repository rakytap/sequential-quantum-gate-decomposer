/*
Created on Fri Jun 26 14:13:26 2020
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
*/

#include "qgd/Operation.h"


extern "C" {

// @brief Creates an instance of class N_Qubit_Decomposition and return with a void pointer pointing to the class instance
void* iface_new_N_Qubit_Decomposition( double* mtx_real, double* mtx_imag, int qbit_num, bool optimize_layer_num, int initial_guess_num );

// @brief Starts the decomposition of the unitary
int iface_start_decomposition( void* ptr );

// @brief Deallocate the N_Qubit_Decomposition class
int iface_delete_N_Qubit_Decomposition( void* ptr );

// @brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
int iface_set_identical_blocks( void* ptr, int qbit, int identical_blocks_in );

// @brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
int iface_set_iteration_loops( void* ptr, int qbit, int iteration_loops_in );

// @brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
int iface_set_max_layer_num( void* ptr, int qbit, int max_layer_num_in );

// @brief Call to list the operations giving the decomposition of the unitary
void iface_list_operations( void* ptr, int start_index );

// @brief Call to get the n-th optimized operation. The values are returned via the input parameter references, and (non preallocated) pointers
// @param n Integer labeling the n-th oepration  (n>=0).
// @param type The type of operation. (Possible values: CNOT_OPERATION=2, U3_OPERATION=3)
// @param target_qbit The id of the target qubit.
// @param control_qbit The id of the control qubit.
// @return Returns with 0 if the export of the n-th operation was successful. If the n-th operation does not exists, -1 is returned. If the operation is not allowed to be exported, i.e. it is not a CNOT or U3 operation, then -2 is returned.
int iface_get_operation( void* ptr, int n, operation_type &type, int &target_qbit, int &control_qbit, double* &parameters );


}







