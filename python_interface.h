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
/*! \file qgd/python_interface.h
    \brief Header file for a methods of the @python_iface.
*/

#include "qgd/Operation.h"


extern "C" {

/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a void pointer pointing to the class instance
@param mtx_real Pointer to the real part of the unitary to be decomposed
@param mtx_imag Pointer to the imaginary part of the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param optimize_layer_num Logical value. Set true to optimize the number of decomposing layers during the decomposition procedure, or false otherwise.
@param initial_guess_num Integer encoding the method to guess initial values for the optimalization. Possible values: 'zeros=0','random=1', 'close_to_zero=2'
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void* iface_new_N_Qubit_Decomposition( double* mtx_real, double* mtx_imag, int qbit_num, bool optimize_layer_num, int initial_guess_num );

/**
@brief Starts the decomposition of the unitary
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
int iface_start_decomposition( void* ptr );

/**
@brief Call to deallocate the N_Qubit_Decomposition class
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
int iface_delete_N_Qubit_Decomposition( void* ptr );

/**
@brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param qbit The number of qubits for which the subdecomposition should contain identical_blocks successive identical blocks.
@param identical_blocks Number of successive identical blocks in the decomposition.
*/
int iface_set_identical_blocks( void* ptr, int qbit, int identical_blocks );

/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param qbit The number of qubits for which the subdecomposition should run iteration_loops number of iterations in each step of the optimization.
@param iteration_loops Number of iteration in each step of the subdecomposition.
*/
int iface_set_iteration_loops( void* ptr, int qbit, int iteration_loops );

/**
@brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param qbit The number of qubits for which the subdecomposition should contain maximum max_layer_num layers of operation blocks.
@param max_layer_num The number of maximal number of layers used in the subdecomposition.
*/
int iface_set_max_layer_num( void* ptr, int qbit, int max_layer_num );

/**
@brief Call to list the operations giving the decomposition of the unitary
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param start_index The starting number of the operations to be listed.
*/
void iface_list_operations( void* ptr, int start_index );

/**
@brief Call to get the number of decomposing operations
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
int iface_get_operation_num( void* ptr );

/**
@brief Call to get the n-th optimized operation. The values are returned via the input parameter references, and pointers
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param n Integer labeling the n-th oepration  (n>=0).
@param op_type The type of operation. (Possible values: CNOT_OPERATION=2, U3_OPERATION=3)
@param target_qbit The id of the target qubit.
@param control_qbit The id of the control qubit.
@param parameters A pointer pointing to the 3-component array conatining the parameters of the U3 operation.
@return Returns with 0 if the export of the n-th operation was successful. If the n-th operation does not exists, -1 is returned. If the operation is not allowed to be exported, i.e. it is not a CNOT or U3 operation, then -2 is returned.
*/
int iface_get_operation( void* ptr, int n, int &op_type, int &target_qbit, int &control_qbit, double* parameters );

/**
@brief Call to set the verbosity of the N_Qubit_Decomposition class
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param verbose Set False to suppress the output messages of the decompostion, or True (deafult) otherwise.
@return Returns with 0 on success
*/
int iface_set_verbose( void* ptr, bool verbose );

/**
@brief Call to set the number of blocks to be optimized in one shot
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param optimalization_block The number of blocks to be optimized in one shot
@return Returns with 0 on success
*/
int iface_set_optimalization_block( void* ptr, int optimalization_block );


}







