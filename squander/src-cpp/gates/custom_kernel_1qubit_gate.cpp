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
/*! \file custom_kernel_1qubit_gate.cpp
    \brief Class representing a single qubit gate with custom gate kernel
*/

#include "custom_kernel_1qubit_gate.h"



//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
custom_kernel_1qubit_gate::custom_kernel_1qubit_gate() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = CUSTOM_KERNEL_1QUBIT_GATE_OPERATION;

        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = -1;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;

        parameter_num = 0;



}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param kernel_in The 2x2 matrix kernel of the gate
*/
custom_kernel_1qubit_gate::custom_kernel_1qubit_gate(int qbit_num_in, int target_qbit_in, Matrix& kernel_in) {

	//The stringstream input to store the output messages.
	std::stringstream sstream;

	//Integer value to set the verbosity level of the output messages.
	int verbose_level;

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = CUSTOM_KERNEL_1QUBIT_GATE_OPERATION;

        kernel = kernel_in;


        if (target_qbit_in >= qbit_num) {
	   verbose_level=1;
	   sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
	   print(sstream,verbose_level);	    	
	            
	   throw "The index of the target qubit is larger than the number of qubits";
        }
	
        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;

        parameter_num = 0;


}


/**
@brief Destructor of the class
*/
custom_kernel_1qubit_gate::~custom_kernel_1qubit_gate() {


}




/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
custom_kernel_1qubit_gate::apply_to( Matrix& input ) {


    if (input.rows != matrix_size ) {
        std::string err("Wrong matrix size in custom_kernel_1qubit_gate gate apply");
        throw err;
    }


    apply_kernel_to( kernel, input );


}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
custom_kernel_1qubit_gate::apply_from_right( Matrix_real& parameters, Matrix& input ) {


    if (input.cols != matrix_size ) {
        std::string err("Wrong matrix size in U3 apply_from_right");
        throw err;    
    }


    apply_kernel_from_right(kernel, input);



}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
custom_kernel_1qubit_gate* custom_kernel_1qubit_gate::clone() {

    custom_kernel_1qubit_gate* ret = new custom_kernel_1qubit_gate(qbit_num, target_qbit, kernel);

    ret->set_parameter_start_idx( get_parameter_start_idx() );

    return ret;

}




