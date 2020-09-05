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


#define QGD_VERSION "1.0"

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

// @brief Set the type of initial guess
//int iface_set_max_layer_num( void* ptr, int qbit, int max_layer_num_in );

}







