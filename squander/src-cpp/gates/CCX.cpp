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
#include "apply_dedicated_gate_kernel_to_input.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CCX::CCX() : X(){

    // A string labeling the gate operation
    name = "CCX";

    // A string describing the type of the gate
    type = CCX_OPERATION;

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = -1;

    // The index of the second control qubit
    control_qbit2 = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the first control qubit. (0 <= control_qbit <= qbit_num-1)
@param control_qbit2_in The identification number of the second control qubit. (0 <= control_qbit2 <= qbit_num-1)
*/
CCX::CCX(int qbit_num_in, int target_qbit_in, int control_qbit_in, int control_qbit2_in) : X(qbit_num_in, target_qbit_in) {


    // A string labeling the gate operation
    name = "CCX";

    // A string describing the type of the gate
    type = CCX_OPERATION;


    if (control_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the first control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    if (control_qbit2_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the second control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    if (control_qbit_in == control_qbit2_in) {
        std::stringstream sstream;
        sstream << "The two control qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = control_qbit_in;

    // The index of the second control qubit
    control_qbit2 = control_qbit2_in;


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

    CCX* ret = new CCX( qbit_num, target_qbit, control_qbit, control_qbit2 );

    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}

/**
@brief Call to apply the gate operation on the input matrix (without parameters)
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void CCX::apply_to(Matrix& input, int parallel) {

    int matrix_size = input.rows;

    // Apply the dedicated X kernel with both control qubits
    switch (parallel){
        case 0:
            apply_X_kernel_to_input(input, target_qbit, control_qbit, control_qbit2, matrix_size); break;
        case 1:
            apply_X_kernel_to_input_omp(input, target_qbit, control_qbit, control_qbit2, matrix_size); break;
        case 2:
            apply_X_kernel_to_input_tbb(input, target_qbit, control_qbit, control_qbit2, matrix_size); break;
    }

}

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void CCX::apply_to(Matrix& input, const Matrix_real& parameters, int parallel) {

    int matrix_size = input.rows;

    // Apply the dedicated X kernel with both control qubits
    switch (parallel){
        case 0:
            apply_X_kernel_to_input(input, target_qbit, control_qbit, control_qbit2, matrix_size); break;
        case 1:
            apply_X_kernel_to_input_omp(input, target_qbit, control_qbit, control_qbit2, matrix_size); break;
        case 2:
            apply_X_kernel_to_input_tbb(input, target_qbit, control_qbit, control_qbit2, matrix_size); break;
    }
}

/**
@brief Call to get the qubits involved in the gate operation.
@return Return with a list of the involved qubits
*/
std::vector<int> CCX::get_involved_qubits(bool only_target) {

    std::vector<int> involved_qbits;
    
    if( target_qbit != -1 ) {
        involved_qbits.push_back( target_qbit );
    }
    
    if (!only_target){
        if( control_qbit != -1 ) {
            involved_qbits.push_back( control_qbit );
        }
        if( control_qbit2 != -1 ) {
            involved_qbits.push_back( control_qbit2 );
        }    
    }
    
    return involved_qbits;
    

}


int CCX::get_control_qbit2(){

    return control_qbit2;
}

void CCX::set_control_qbit2(int control_qbit2_in){
    control_qbit2 = control_qbit2_in;
    return;
}