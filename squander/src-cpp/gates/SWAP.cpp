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
/*! \file SWAP.cpp
    \brief Class representing a SWAP gate.
*/

#include "SWAP.h"
#include "apply_dedicated_gate_kernel_to_input.h"

using namespace std;

/**
@brief Nullary constructor of the class.
*/
SWAP::SWAP() : Gate() {

    // A string labeling the gate operation
    name = "SWAP";

    // A string describing the type of the gate
    type = SWAP_OPERATION;

    // Initialize target qubits vector (empty for nullary constructor)
    target_qbits.clear();

    parameter_num = 0;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements for SWAP)
*/
SWAP::SWAP(int qbit_num_in, const std::vector<int>& target_qbits_in)
    : Gate(qbit_num_in, target_qbits_in) {

    // A string labeling the gate operation
    name = "SWAP";

    // A string describing the type of the gate
    type = SWAP_OPERATION;

    // Validate that we have exactly 2 target qubits
    if (target_qbits_in.size() != 2) {
        std::stringstream sstream;
        sstream << "SWAP gate requires exactly 2 target qubits, got " << target_qbits_in.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Check that target qubits are unique
    if (target_qbits_in[0] == target_qbits_in[1]) {
        std::stringstream sstream;
        sstream << "The two target qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    parameter_num = 0;
}

/**
@brief Destructor of the class
*/
SWAP::~SWAP() {

}

/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
Matrix
SWAP::get_matrix() {
    return get_matrix(false);
}

/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
SWAP::get_matrix(int parallel) {
    Matrix SWAP_matrix = create_identity(matrix_size);
    apply_to(SWAP_matrix, parallel);

#ifdef DEBUG
    if (SWAP_matrix.isnan()) {
        std::stringstream sstream;
        sstream << "SWAP::get_matrix: SWAP_matrix contains NaN." << std::endl;
        print(sstream, 1);
    }
#endif

    return SWAP_matrix;
}

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void
SWAP::apply_to(Matrix& input, int parallel) {

    if (input.rows != matrix_size) {
        std::string err("SWAP::apply_to: Wrong input size in SWAP gate apply");
        throw(err);
    }

    // Use the dedicated SWAP kernel with target_qbits vector (no control qubits)
    std::vector<int> empty_control_qbits;
    switch (parallel){
        case 0:
            apply_SWAP_kernel_to_input(input, target_qbits, empty_control_qbits, matrix_size); break;
        case 1:
            apply_SWAP_kernel_to_input_omp(input, target_qbits, empty_control_qbits, matrix_size); break;
        case 2:
            apply_SWAP_kernel_to_input_tbb(input, target_qbits, empty_control_qbits, matrix_size); break;
    }
}

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void SWAP::apply_to(Matrix& input, const Matrix_real& parameters, int parallel) {

    int matrix_size = input.rows;

    // Apply the dedicated SWAP kernel with target_qbits vector (no control qubits)
    std::vector<int> empty_control_qbits;
    switch (parallel){
        case 0:
            apply_SWAP_kernel_to_input(input, target_qbits, empty_control_qbits, matrix_size); break;
        case 1:
            apply_SWAP_kernel_to_input_omp(input, target_qbits, empty_control_qbits, matrix_size); break;
        case 2:
            apply_SWAP_kernel_to_input_tbb(input, target_qbits, empty_control_qbits, matrix_size); break;
    }

}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SWAP* SWAP::clone() {

    SWAP* ret = new SWAP(qbit_num, target_qbits);

    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);

    return ret;
}

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void SWAP::reorder_qubits(std::vector<int> qbit_list) {
    // Use Gate's implementation which now handles vectors
    Gate::reorder_qubits(qbit_list);
}

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void SWAP::set_qbit_num(int qbit_num_in) {
    // setting the number of qubits
    Gate::set_qbit_num(qbit_num_in);
}

/**
@brief Get list of involved qubits
@param only_target If true, return only target qubits, otherwise include control qubits too
@return Vector of qubit indices
*/
std::vector<int> SWAP::get_involved_qubits(bool only_target) {
    // Use Gate's implementation which now handles vectors
    return Gate::get_involved_qubits(only_target);
}