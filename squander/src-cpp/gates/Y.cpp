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
/*! \file Y.cpp
    \brief Class representing a Y gate.
*/

#include "Y.h"



//static tbb::spin_mutex my_mutex;
/**
@brief NullaRX constructor of the class.
*/
Y::Y() {

    // A string labeling the gate operation
    name = "Y";

    // number of qubits spanning the matrix of the gate
    qbit_num = -1;
    // the size of the matrix
    matrix_size = -1;
    // A string describing the type of the gate
    type = Y_OPERATION;

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
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
Y::Y(int qbit_num_in, int target_qbit_in) {

    // A string labeling the gate operation
    name = "Y";
    // number of qubits spanning the matrix of the gate
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the gate
    type = Y_OPERATION;


    if (target_qbit_in >= qbit_num) {
       std::stringstream sstream;
       sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
       print(sstream, 0);
	   
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
Y::~Y() {

}


/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
Matrix
Y::get_matrix() {

        return get_matrix( false );

}


/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
Y::get_matrix( int parallel) {

        Matrix Y_matrix = create_identity(matrix_size);
        apply_to(Y_matrix, parallel);

#ifdef DEBUG
        if (Y_matrix.isnan()) {
            std::stringstream sstream;
	    sstream << "Y::get_matrix: Y_matrix contains NaN." << std::endl;
            print(sstream, 1);	          
        }
#endif

        return Y_matrix;

}



/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Y::apply_to( Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
        std::stringstream sstream;
	sstream << "Wrong matrix size in X gate apply" << std::endl;
        print(sstream, 0);	       
        exit(-1);
    }

    Matrix u3_1qbit = calc_one_qubit_u3();

    //apply_kernel_to function to Y gate 
    apply_kernel_to( u3_1qbit, input, false, parallel );
   


}



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
Y::apply_from_right( Matrix& input ) {

    if (input.rows != matrix_size ) {
        std::string err("Y::apply_to: Wrong input size in Y gate apply");     
        throw(err);
    }

    Matrix u3_1qbit = calc_one_qubit_u3();   
    //apply_kernel_from_right function to X gate 
    apply_kernel_from_right(u3_1qbit, input);

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Y* Y::clone() {

    Y* ret = new Y(qbit_num, target_qbit);
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}




/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Y::reorder_qubits( std::vector<int> qbit_list) {

    Gate::reorder_qubits(qbit_list);

}


/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void Y::set_qbit_num(int qbit_num_in) {

        // setting the number of qubits
        Gate::set_qbit_num(qbit_num_in);
}

/**
@brief Set static values for matrix of the gates.
@param u3_1qbit Matrix parameter for the gate.

*/
Matrix
Y::calc_one_qubit_u3( ){

    Matrix u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = 0.0; u3_1qbit[0].imag = 0.0; 
    u3_1qbit[1].real = 0.0; u3_1qbit[1].imag = -1.0;
    u3_1qbit[2].real = 0.0; u3_1qbit[2].imag = 1.0;
    u3_1qbit[3].real = 0.0;u3_1qbit[3].imag = 0.0;
    return u3_1qbit;

}
