/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Operation.cpp
    \brief Class for the representation of general gate operations.
*/


#include "Gate.h"
#include "common.h"



/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
Gate::Gate() {

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = GENERAL_OPERATION;
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
Gate::Gate(int qbit_num_in) {

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = GENERAL_OPERATION;
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
Gate::~Gate() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void Gate::set_qbit_num( int qbit_num_in ) {
    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

}


/**
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix
Gate::get_matrix() {

    return matrix_alloc;
}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
Gate::apply_to_list( std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( *it );
    }

}


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
*/
void 
Gate::apply_to( Matrix& input ) {

    Matrix ret = dot(matrix_alloc, input);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );
    //input = ret;
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
Gate::apply_from_right( Matrix& input ) {

    Matrix ret = dot(input, matrix_alloc);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );

}


/**
@brief Call to set the stored matrix in the operation.
@param input The operation matrix to be stored. The matrix is stored by attribute matrix_alloc.
@return Returns with 0 on success.
*/
void
Gate::set_matrix( Matrix input ) {
    matrix_alloc = input;
}


/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Gate::reorder_qubits( std::vector<int> qbit_list ) {

    // check the number of qubits
    if ((int)qbit_list.size() != qbit_num ) {
	std::stringstream sstream;
	sstream << "Wrong number of qubits" << std::endl;
	print(sstream, 0);	    	
       
        exit(-1);
    }


    int control_qbit_new = control_qbit;
    int target_qbit_new = target_qbit;

    // setting the new value for the target qubit
    for (int idx=0; idx<qbit_num; idx++) {
        if (target_qbit == qbit_list[idx]) {
            target_qbit_new = qbit_num-1-idx;
        }
        if (control_qbit == qbit_list[idx]) {
            control_qbit_new = qbit_num-1-idx;
        }
    }

    control_qbit = control_qbit_new;
    target_qbit = target_qbit_new;
}


/**
@brief Call to get the index of the target qubit
@return Return with the index of the target qubit (return with -1 if target qubit was not set)
*/
int Gate::get_target_qbit() {
    return target_qbit;
}

/**
@brief Call to get the index of the control qubit
@return Return with the index of the control qubit (return with -1 if control qubit was not set)
*/
int Gate::get_control_qbit()  {
    return control_qbit;
}

/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
int Gate::get_parameter_num() {
    return parameter_num;
}


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type Gate::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int Gate::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Gate* Gate::clone() {

    Gate* ret = new Gate( qbit_num );
    ret->set_matrix( matrix_alloc );

    return ret;

}



/**
@brief ???????????
*/
void 
Gate::apply_kernel_to(Matrix& u3_1qbit, Matrix& input, bool deriv) {


    
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step_target;

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < matrix_size ) {

        for(int idx=0; idx<index_step_target; idx++) {  
        //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            int row_offset = current_idx_loc*input.stride;
            int row_offset_pair = current_idx_pair_loc*input.stride;

           if ( control_qbit<0 || ((current_idx_loc >> control_qbit) & 1) ) {

                for ( int col_idx=0; col_idx<input.cols; col_idx++) {
   			
                    int index      = row_offset+col_idx;
                    int index_pair = row_offset_pair+col_idx;                

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);
 
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit[2], element);
                    tmp2 = mult(u3_1qbit[3], element_pair);

                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;

                }

            }
            else if (deriv) {
                // when calculating derivatives, the constant element should be zeros
                memset( input.get_data()+row_offset, 0.0, input.cols*sizeof(QGD_Complex16));
                memset( input.get_data()+row_offset_pair, 0.0, input.cols*sizeof(QGD_Complex16));
            }
            else {
                // leave the state as it is
                continue; 
            }


//std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;

        
        //});
        }


        current_idx = current_idx + (index_step_target << 1);
        current_idx_pair = current_idx_pair + (index_step_target << 1);


    }


}





/**
@brief Call to apply the gate on the input array/matrix by input*CNOT
@param input The input array on which the gate is applied
*/
void 
Gate::apply_kernel_from_right( Matrix& u3_1qbit, Matrix& input ) {

   
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step_target;

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < input.cols ) {

        for(int idx=0; idx<index_step_target; idx++) { 
        //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            // determine the action according to the state of the control qubit
            if ( control_qbit<0 || ((current_idx_loc >> control_qbit) & 1) ) {

                for ( int row_idx=0; row_idx<matrix_size; row_idx++) {

                    int row_offset = row_idx*input.stride;


                    int index      = row_offset+current_idx_loc;
                    int index_pair = row_offset+current_idx_pair_loc;

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];

                    QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit[2], element_pair);
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit[1], element);
                    tmp2 = mult(u3_1qbit[3], element_pair);
                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;

                }

            }
            else {
                // leave the state as it is
                continue;
            }        


//std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


        //});
        }


        current_idx = current_idx + (index_step_target << 1);
        current_idx_pair = current_idx_pair + (index_step_target << 1);


    }


}


