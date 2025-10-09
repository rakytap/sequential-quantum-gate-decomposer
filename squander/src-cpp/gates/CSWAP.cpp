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
/*! \file CSWAP.cpp
    \brief Class representing a CSWAP (Controlled SWAP) gate.
*/

#include "CSWAP.h"
#include "apply_dedicated_gate_kernel_to_input.h"

using namespace std;

/**
@brief Nullary constructor of the class.
*/
CSWAP::CSWAP() : SWAP() {

    // A string labeling the gate operation
    name = "CSWAP";

    // A string describing the type of the gate
    type = CSWAP_OPERATION;

    // Clear control qubits for nullary constructor
    control_qbits.clear();
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements)
@param control_qbits_in Vector of control qubit indices (should contain exactly 1 element)
*/
CSWAP::CSWAP(int qbit_num_in, const std::vector<int>& target_qbits_in, const std::vector<int>& control_qbits_in)
    : SWAP(qbit_num_in, target_qbits_in) {

    // A string labeling the gate operation
    name = "CSWAP";

    // A string describing the type of the gate
    type = CSWAP_OPERATION;

    // Validate that we have exactly 1 control qubit
    if (control_qbits_in.size() != 1) {
        std::stringstream sstream;
        sstream << "CSWAP gate requires exactly 1 control qubit, got " << control_qbits_in.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Validate control qubit index
    if (control_qbits_in[0] >= qbit_num_in) {
        std::stringstream sstream;
        sstream << "Control qubit index " << control_qbits_in[0] << " is larger than or equal to the number of qubits" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Check that control qubit doesn't overlap with target qubits
    for (int tq : target_qbits_in) {
        if (control_qbits_in[0] == tq) {
            std::stringstream sstream;
            sstream << "The control qubit cannot be the same as any target qubit" << std::endl;
            print(sstream, 0);
            throw sstream.str();
        }
    }

    // Store control qubit
    control_qbits = control_qbits_in;
    control_qbit = control_qbits[0];  // For backward compatibility
}

/**
@brief Destructor of the class
*/
CSWAP::~CSWAP() {
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CSWAP* CSWAP::clone() {

    CSWAP* ret = new CSWAP(qbit_num, target_qbits, control_qbits);

    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);

    return ret;
}

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void
CSWAP::apply_to(Matrix& input, int parallel) {

    if (input.rows != matrix_size) {
        std::string err("CSWAP::apply_to: Wrong input size in CSWAP gate apply");
        throw(err);
    }
    // Use the dedicated SWAP kernel with control_qbits vector
    switch (parallel){
    case 0:
        apply_SWAP_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
    case 1:
        apply_SWAP_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
    case 2:
        apply_SWAP_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
    }
}

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void CSWAP::apply_to(Matrix& input, const Matrix_real& parameters, int parallel) {

    int matrix_size = input.rows;

    // Use the dedicated SWAP kernel with control_qbits vector
    switch (parallel){
    case 0:
        apply_SWAP_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
    case 1:
        apply_SWAP_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
    case 2:
        apply_SWAP_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
    }
}

/**
@brief Get list of involved qubits
@param only_target If true, return only target qubits, otherwise include control qubits too
@return Vector of qubit indices
*/
std::vector<int> CSWAP::get_involved_qubits(bool only_target) {
    // Use Gate's implementation which now handles vectors
    return Gate::get_involved_qubits(only_target);
}