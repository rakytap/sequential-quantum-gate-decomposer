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
/*! \file UN.cpp
    \brief Class for the representation of general unitary operation on the first qbit_num-1 qubits.
*/


#include "N_Qubit_Permutation.h"
#include "common.h"
#include "Random_Unitary.h"
#include "dot.h"




/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
N_Qubit_Permutation::N_Qubit_Permutation() {

    // A string labeling the gate operation
    name = "N_Qubit_Permutation";
 
    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = N_QUBIT_PERMUTATION_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // the number of free parameters of the operation
    parameter_num = 0;
    pattern = std::vector<int>{};
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
N_Qubit_Permutation::N_Qubit_Permutation(int qbit_num_in, std::vector<int> pattern_in) {

    // A string labeling the gate operation
    name = "N_Qubit_Permutation";
    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = N_QUBIT_PERMUTATION_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The number of parameters
    parameter_num = 0;
    pattern = pattern_in;
}


/**
@brief Destructor of the class
*/
N_Qubit_Permutation::~N_Qubit_Permutation() {
}


/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the UN gate.
@return Returns with a matrix of the operation
*/
Matrix
N_Qubit_Permutation::get_matrix(  ) {


        return get_matrix( false );
}


/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the UN gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
N_Qubit_Permutation::get_matrix( int parallel ) {

        Matrix P_matrix = create_identity(matrix_size);
        apply_to(P_matrix, parallel);

#ifdef DEBUG
        if (P_matrix.isnan()) {
            std::stringstream sstream;
	    sstream << "U3::get_matrix: P contains NaN." << std::endl;
            print(sstream, 1);	         
        }
#endif

        return P_matrix;
}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
N_Qubit_Permutation::apply_to(Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::string err("UN::apply_to: Wrong input size in UN gate apply");     
        throw(err);
    }



    Matrix &&P = construct_matrix_from_pattern( pattern );
    Matrix transformed_input = dot(P,input);
    
    memcpy( input.get_data(), transformed_input.get_data(), transformed_input.size()*sizeof(QGD_Complex16) );
     
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
N_Qubit_Permutation::apply_from_right( Matrix& input ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in UN gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }

    Matrix &&P = construct_matrix_from_pattern( pattern );
    Matrix transformed_input = dot(input,P);
    
    memcpy( input.get_data(), transformed_input.get_data(), transformed_input.size()*sizeof(QGD_Complex16) );
}


/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void N_Qubit_Permutation::reorder_qubits( std::vector<int> qbit_list ) {

    return;
}



/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type N_Qubit_Permutation::get_type() {
    return type;
}

/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
std::vector<int> N_Qubit_Permutation::get_pattern() {
    return pattern;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int N_Qubit_Permutation::get_qbit_num() {
    return qbit_num;
}

Matrix N_Qubit_Permutation::construct_matrix_from_pattern(std::vector<int> pattern){

    Matrix perm_matrix(matrix_size,matrix_size);
    memset( perm_matrix.get_data(), 0.0, 2*perm_matrix.size()*sizeof(double) );

    for (int r = 0; r < matrix_size; r++) {
        int pr = 0;
        for (int out = 0; out < qbit_num; out++) {
            int in = pattern[out];
            int bit = (r >> in) & 1;
            pr |= (bit << out);
        }
        perm_matrix[pr*matrix_size+r].real = 1;
        }
    
    return perm_matrix;
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
N_Qubit_Permutation* N_Qubit_Permutation::clone() {

    N_Qubit_Permutation* ret = new N_Qubit_Permutation( qbit_num, pattern );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



