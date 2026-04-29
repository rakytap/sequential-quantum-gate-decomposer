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
/*! \file CCX.cpp
    \brief Class representing a CCX (Toffoli) gate.
*/

#include "CCX.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CCX::CCX() : Gate(){

    // A string labeling the gate operation
    name = "CCX";

    // A string describing the type of the gate
    type = CCX_OPERATION;

    // Initialize control qubits vector (empty for nullary constructor)
    control_qbits.clear();
}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbits_in Vector of control qubit indices (should contain exactly 2 elements for CCX)
*/
CCX::CCX(int qbit_num_in, int target_qbit_in, const std::vector<int>& control_qbits_in)
    : Gate(qbit_num_in, std::vector<int>{target_qbit_in}, control_qbits_in) {

    // A string labeling the gate operation
    name = "CCX";

    // A string describing the type of the gate
    type = CCX_OPERATION;

    // Validate that we have exactly 2 control qubits
    if (control_qbits_in.size() != 2) {
        std::stringstream sstream;
        sstream << "CCX gate requires exactly 2 control qubits, got " << control_qbits_in.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Check that control qubits are unique
    if (control_qbits_in[0] == control_qbits_in[1]) {
        std::stringstream sstream;
        sstream << "The two control qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }
}

/**
@brief Destructor of the class
*/
CCX::~CCX() {
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CCX* CCX::clone() {

    CCX* ret = new CCX( qbit_num, target_qbit, control_qbits );

    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}

