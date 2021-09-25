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
/*! \file Operation.h
    \brief Header file for a class for the representation of general gate operations.
*/

#ifndef GATE_H
#define GATE_H

#include <vector>
#include "common.h"
#include "matrix.h"


/// @brief Type definition of operation types (also generalized for decomposition classes derived from the class Operation_Block)
typedef enum gate_type {GENERAL_OPERATION, CZ_OPERATION, CNOT_OPERATION, CH_OPERATION, U3_OPERATION, BLOCK_OPERATION, DECOMPOSITION_BASE_CLASS, SUB_MATRIX_DECOMPOSITION_CLASS, N_QUBIT_DECOMPOSITION_CLASS_BASE, N_QUBIT_DECOMPOSITION_CLASS} gate_type;



/**
@brief Base class for the representation of general gate operations.
*/
class Gate {


protected:

    /// number of qubits spanning the matrix of the operation
    int qbit_num;
    /// The type of the operation (see enumeration gate_type)
    gate_type type;
    /// The index of the qubit on which the operation acts (target_qbit >= 0)
    int target_qbit;
    /// The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    int control_qbit;
    /// The size N of the NxN matrix associated with the operations.
    int matrix_size;
    /// the number of free parameters of the operation
    unsigned int parameter_num;

private:

    /// Matrix of the operation
    Matrix matrix_alloc;

public:

/**
@brief Default constructor of the class.
@return An instance of the class
*/
Gate();

/**
@brief Destructor of the class
*/
virtual ~Gate();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
Gate(int qbit_num_in);

/**
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix get_matrix();


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
*/
void apply_to( Matrix& input );

/**
@brief Call to set the stored matrix in the operation.
@param input The operation matrix to be stored. The matrix is stored by attribute matrix_alloc.
@return Returns with 0 on success.
*/
void set_matrix( Matrix input );


/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
virtual void set_qbit_num( int qbit_num_in );

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list );


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
@return Return with the type of the operation (see gate_type for details)
*/
gate_type get_type();

/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int get_qbit_num();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Gate* clone();

};


#endif //OPERATION
