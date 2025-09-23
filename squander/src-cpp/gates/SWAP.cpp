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
SWAP::SWAP() {

    // A string labeling the gate operation
    name = "SWAP";

    // number of qubits spanning the matrix of the gate
    qbit_num = -1;
    // the size of the matrix
    matrix_size = -1;
    // A string describing the type of the gate
    type = SWAP_OPERATION;

    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (used as second target for SWAP)
    control_qbit = -1;

    parameter_num = 0;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the first target qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit2_in The identification number of the second target qubit. (0 <= target_qbit2 <= qbit_num-1)
*/
SWAP::SWAP(int qbit_num_in, int target_qbit_in, int target_qbit2_in) {

    // A string labeling the gate operation
    name = "SWAP";

    // number of qubits spanning the matrix of the gate
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the gate
    type = SWAP_OPERATION;

    if (target_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the first target qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    if (target_qbit2_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the second target qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    if (target_qbit_in == target_qbit2_in) {
        std::stringstream sstream;
        sstream << "The two target qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = target_qbit_in;
    // Store second target qubit in control_qbit for kernel compatibility
    control_qbit = target_qbit2_in;

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

    // Use the dedicated SWAP kernel (using control_qbit as target_qbit2)
    switch (parallel){
        case 0:
            apply_SWAP_kernel_to_input(input, target_qbit, control_qbit, -1, matrix_size); break;
        case 1:
            apply_SWAP_kernel_to_input_omp(input, target_qbit, control_qbit, -1, matrix_size); break;
        case 2:
            apply_SWAP_kernel_to_input_tbb(input, target_qbit, control_qbit, -1, matrix_size); break;
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

    // Apply the dedicated X kernel with both control qubits
    switch (parallel){
        case 0:
            apply_SWAP_kernel_to_input(input, target_qbit, control_qbit, -1, matrix_size); break;
        case 1:
            apply_SWAP_kernel_to_input_omp(input, target_qbit, control_qbit, -1, matrix_size); break;
        case 2:
            apply_SWAP_kernel_to_input_tbb(input, target_qbit, control_qbit, -1, matrix_size); break;
    }

}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SWAP* SWAP::clone() {

    SWAP* ret = new SWAP(qbit_num, target_qbit, control_qbit);

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
    std::vector<int> involved_qubits;

    involved_qubits.push_back(target_qbit);
    involved_qubits.push_back(control_qbit);

    return involved_qubits;
}