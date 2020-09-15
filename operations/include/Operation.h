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

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space


#pragma once
#include <vector> 


#include "qgd/common.h"

using namespace std;

/// @brief Type definition of operation types (also generalized for decomposition classes derived from the class Operation_Block)
typedef enum operation_type {GENERAL_OPERATION, CNOT_OPERATION, U3_OPERATION, BLOCK_OPERATION, DECOMPOSITION_BASE_CLASS, SUB_MATRIX_DECOMPOSITION_CLASS, N_QUBIT_DECOMPOSITION_CLASS} operation_type;



/**
@brief Base class for the representation of one- and two-qubit operations.
*/
class Operation {


protected:

    /// number of qubits spanning the matrix of the operation
    int qbit_num;
    /// The type of the operation (see enumeration operation_type)
    operation_type type;
    /// The index of the qubit on which the operation acts (target_qbit >= 0) 
    int target_qbit;
    /// The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    int control_qbit;
    /// The size N of the NxN matrix associated with the operations.
    int matrix_size;
    /// Pointer to the operatrion matrix (if it is a constant general matrix)
    QGD_Complex16* matrix_alloc;
    /// the number of free parameters of the operation
    unsigned int parameter_num;
    


public:

/**
@brief Default constructor of the class.
@return An instance of the class
*/
Operation();

/**
@brief Destructor of the class
*/
virtual ~Operation();


/**
@brief Constructor of the class.
@param qbit_num The number of qubits spanning the unitaries
@return An instance of the class
*/
Operation(int qbit_num_in);

/**
@brief Call to terive the operation matrix
@return Returns with a pointer to the operation matrix
*/
virtual QGD_Complex16* matrix();


/**
@brief Call to retrieve the operation matrix
@param retrieve_matrix Preallocated array where the operation matrix is copied
@return Returns with 0 on success.
*/
virtual int matrix(QGD_Complex16* retrieve_matrix );

/**
@brief Call to set the stored matrix in the operation.
@param input a pointer to the operation matrix to be stored. The matrix is copied into the storage pointed by @matrix_alloc.
@return Returns with 0 on success.
*/
void set_matrix( QGD_Complex16* input );
   
/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num The number of qubits spanning the matrix
*/
virtual void set_qbit_num( int qbit_num_in );
     
/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( vector<int> qbit_list );


/**
@brief Call to get the index of the target qubit
@return Return with the index of the target qubit (return with -1 if target qubit was not set)
*/
int get_target_qbit();


/**
@brief Call to get the index of the control qubit
@return Return with the index of the control qubit (return with -1 if control qubit was not set)
*/
int get_control_qbit();


/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
unsigned int get_parameter_num();


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see operation_type for details) 
*/
operation_type get_type();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Operation* clone();

};

        
