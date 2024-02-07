/*
Created on Fri Jun 26 14:13:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
#include "logging.h"
#include "matrix_real.h"


/// @brief Type definition of operation types (also generalized for decomposition classes derived from the class Operation_Block)
typedef enum gate_type {GENERAL_OPERATION, UN_OPERATION, ON_OPERATION, CZ_OPERATION, CNOT_OPERATION, CH_OPERATION, U3_OPERATION, RY_OPERATION, RX_OPERATION, RZ_OPERATION, RZ_P_OPERATION, X_OPERATION, SX_OPERATION, CRY_OPERATION, SYC_OPERATION, BLOCK_OPERATION, COMPOSITE_OPERATION, ADAPTIVE_OPERATION, DECOMPOSITION_BASE_CLASS, SUB_MATRIX_DECOMPOSITION_CLASS, N_QUBIT_DECOMPOSITION_CLASS_BASE, N_QUBIT_DECOMPOSITION_CLASS, Y_OPERATION, Z_OPERATION, CUSTOM_KERNEL_1QUBIT_GATE_OPERATION} gate_type;



/**
@brief Base class for the representation of general gate operations.
*/
class Gate : public logging {


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
    int parameter_num;

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
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_to_list( std::vector<Matrix>& input );


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set true to apply parallel kernels, false otherwise (optional)
*/
virtual void apply_to( Matrix& input, bool parallel=false );


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
virtual void apply_from_right( Matrix& input );

/**
@brief Call to set the stored matrix in the operation.
@param input The operation matrix to be stored. The matrix is stored by attribute matrix_alloc.
@return Returns with 0 on success.
*/
void set_matrix( Matrix input );


/**
@brief Call to set the control qubit for the gate operation
@param control_qbit_in The control qubit. Should be: 0 <= control_qbit_in < qbit_num
*/
void set_control_qbit(int control_qbit_in);

/**
@brief Call to set the target qubit for the gate operation
@param target_qbit_in The target qubit on which the gate is applied. Should be: 0 <= target_qbit_in < qbit_num
*/
void set_target_qbit(int target_qbit_in);

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
int get_parameter_num();


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
virtual Gate* clone();

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix calc_one_qubit_u3(double Theta, double Phi, double Lambda );

/**
@brief Calculate the matrix of the constans gates.
@return Returns with the matrix of the one-qubit matrix.
*/
virtual Matrix calc_one_qubit_u3( );

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
virtual void parameters_for_calc_one_qubit(double& ThetaOver2, double& Phi, double& Lambda);






protected:
/**
@brief Call to apply the gate kernel on the input state or unitary with optional AVX support
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise (optional)
@param parallel Set true to apply parallel kernels, false otherwise (optional)
*/
void apply_kernel_to( Matrix& u3_1qbit, Matrix& input, bool deriv=false, bool parallel=false );



/**
@brief Call to apply the gate kernel on the input state or unitary from right (no AVX support)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
*/
void apply_kernel_from_right( Matrix& u3_1qbit, Matrix& input );


};

#endif //OPERATION
