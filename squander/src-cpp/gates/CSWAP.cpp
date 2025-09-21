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

    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
    control_qbit = -1;

    // The actual control qubit for CSWAP
    control_qbit2 = -1;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the first target qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit2_in The identification number of the second target qubit. (0 <= target_qbit2 <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= control_qbit <= qbit_num-1)
*/
CSWAP::CSWAP(int qbit_num_in, int target_qbit_in, int target_qbit2_in, int control_qbit_in)
    : SWAP(qbit_num_in, target_qbit_in, target_qbit2_in) {

    // A string labeling the gate operation
    name = "CSWAP";

    // A string describing the type of the gate
    type = CSWAP_OPERATION;

    if (control_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    if (control_qbit_in == target_qbit_in || control_qbit_in == target_qbit2_in) {
        std::stringstream sstream;
        sstream << "The control qubit cannot be the same as any target qubit" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // The actual control qubit for CSWAP (stored in control_qbit2)
    control_qbit2 = control_qbit_in;
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

    CSWAP* ret = new CSWAP(qbit_num, target_qbit, control_qbit, control_qbit2);

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

    // Debug output for CSWAP parameters
    std::cout << "CSWAP::apply_to: target_qbit=" << target_qbit
              << ", control_qbit=" << control_qbit
              << ", control_qbit2=" << control_qbit2 << std::endl;

    // Use the dedicated SWAP kernel with control_qbit2 as the actual control
    apply_SWAP_kernel_to_input(input, target_qbit, control_qbit, control_qbit2, matrix_size);
}

/**
@brief Get list of involved qubits
@param only_target If true, return only target qubits, otherwise include control qubits too
@return Vector of qubit indices
*/
std::vector<int> CSWAP::get_involved_qubits(bool only_target) {
    std::vector<int> involved_qubits;

    involved_qubits.push_back(target_qbit);
    involved_qubits.push_back(control_qbit);

    if (!only_target && control_qbit2 >= 0) {
        involved_qubits.push_back(control_qbit2);
    }

    return involved_qubits;
}

/**
@brief Get the control qubit
@return The index of the control qubit
*/
int CSWAP::get_control_qbit2() {
    return control_qbit2;
}

/**
@brief Set the control qubit
@param control_qbit2_in The index of the control qubit
*/
void CSWAP::set_control_qbit2(int control_qbit2_in) {
    control_qbit2 = control_qbit2_in;
}