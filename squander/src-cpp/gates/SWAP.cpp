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

