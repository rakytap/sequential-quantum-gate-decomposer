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
/*! \file Composite.cpp
    \brief Class for the representation of composite gate operation.
*/


#include "CNZ.h"
#include "common.h"
#include "dot.h"
#include "Random_Unitary.h"
#include "apply_large_kernel_to_input.h"

static double M_PIOver2 = M_PI/2;
/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
CNZ::CNZ() {

    // A string labeling the gate operation
    name = "CNZ";

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = CNZ_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // the number of free parameters of the operation
    parameter_num = 0;
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
CNZ::CNZ(int qbit_num_in) {

    // A string labeling the gate operation
    name = "CNZ";

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = CNZ_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The number of parameters
    parameter_num = 0;
}


/**
@brief Destructor of the class
*/
CNZ::~CNZ() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void CNZ::set_qbit_num( int qbit_num_in ) {
    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

}

/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the composite gate.
@return Returns with a matrix of the operation
*/
Matrix
CNZ::get_matrix( ) {

        return get_matrix( false );
}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CNZ::apply_to_list(  std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to(  *it, 0);
    }

}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CNZ::apply_to_list(std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = inputs.size();
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,inputs.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to(*input, parallel );

        }

    });

}


/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix of the composite gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix
CNZ::get_matrix(  int parallel ) {

    Matrix com_matrix(matrix_size,1);
    memset(com_matrix.get_data(),0.0,(com_matrix.size()*2)*sizeof(double));
    for (int idx = 0; idx<matrix_size; idx++){
        com_matrix[idx].real = 1.;
    }
    com_matrix[matrix_size-1].real=-1.;
//com_matrix.print_matrix();
#ifdef DEBUG
        if (com_matrix.isnan()) {
	    std::stringstream sstream;
	    sstream << "Composite::get_matrix: UN_matrix contains NaN." << std::endl;
            print(sstream, 1);	           
        }
#endif

        return com_matrix;
}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CNZ::apply_to(Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
        std::string err("Composite::apply_to: Wrong matrix size in Composite gate apply.");
        throw err;    
    }

    if (parameters.size() < parameter_num) {
	std::stringstream sstream;
	sstream << "Not enough parameters given for the Composite gate" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    Matrix com_matrix = get_matrix( );
    apply_diagonal_gate_to_matrix_input(com_matrix,input,input.rows);

//std::cout << "Composite::apply_to" << std::endl;
//exit(-1);
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
CNZ::apply_from_right(Matrix& input ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in Composite gate apply" << std::endl;
        print(sstream, 0);	        
        exit(-1);
    }

    if (parameters.size() < parameter_num) {
	std::stringstream sstream;
        sstream << "Not enough parameters given for the Composite gate" << std::endl;
        print(sstream, 0);	 
        exit(-1);
    }

    Matrix com_matrix = get_matrix(  );
    apply_diagonal_gate_to_matrix_input(com_matrix,input,input.rows);


//std::cout << "Composite::apply_to" << std::endl;
//exit(-1);

}


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type 
CNZ::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int 
CNZ::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CNZ* CNZ::clone() {

    CNZ* ret = new CNZ( qbit_num );

    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}



