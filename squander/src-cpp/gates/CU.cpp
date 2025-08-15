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
/*! \file CU.cpp
    \brief Class representing a CNOT gate.
*/

#include "CU.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CU::CU() : U3() {

    // A string labeling the gate operation
    name = "CU";

    // number of qubits spanning the matrix of the gate
    qbit_num = -1;

    // the size of the matrix
    matrix_size = -1;

    // A string describing the type of the gate
    type = CU_OPERATION;

    // The number of free parameters
    parameter_num = 0;

    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = -1;

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CU::CU(int qbit_num_in,  int target_qbit_in, int control_qbit_in) : U3(qbit_num_in,  target_qbit_in) {


    // A string labeling the gate operation
    name = "CU";

    // A string describing the type of the gate
    type = CU_OPERATION;


    if (control_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);	    	
        throw sstream.str();
    }
    
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = control_qbit_in;


}

/**
@brief Destructor of the class
*/
CU::~CU() {
}







/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CU* CU::clone() {

    CU* ret = new CU( qbit_num, target_qbit, control_qbit );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



