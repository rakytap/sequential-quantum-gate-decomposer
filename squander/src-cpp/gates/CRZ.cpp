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
/*! \file CRZ.cpp
    \brief Class representing a controlled Y rotattion gate.
*/

#include "CRZ.h"



//static tbb::spin_mutex my_mutex;
/**
@brief NullaRZ constructor of the class.
*/
CRZ::CRZ() : RZ() {

    // A string labeling the gate operation
    name = "CRZ";

    // A string describing the type of the gate
    type = CRZ_OPERATION;

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
CRZ::CRZ(int qbit_num_in, int target_qbit_in, int control_qbit_in) : RZ(qbit_num_in, target_qbit_in) {

    // A string labeling the gate operation
    name = "CRZ";

    // A string describing the type of the gate
    type = CRZ_OPERATION;


    if (control_qbit_in >= qbit_num) {
       std::stringstream sstream;
       sstream << "The index of the control qubit is larger than the number of qubits in CRZ gate." << std::endl;
       print(sstream, 0);	  
        throw sstream.str();
    }

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
    control_qbit = control_qbit_in;


}

/**
@brief Destructor of the class
*/
CRZ::~CRZ() {

}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CRZ* CRZ::clone() {

    CRZ* ret = new CRZ(qbit_num, target_qbit, control_qbit);

    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}


