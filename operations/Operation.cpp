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

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space


#include "qgd/Operation.h"


using namespace std;

//
// @brief Constructor of the class.
// @return An instance of the class
Operation::Operation() {

    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // the size of the matrix
    matrix_size = -1;
    // A string describing the type of the operation
    type = "general";
    // The index of the qubit on which the operation acts (target_qbit >= 0) 
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The matrix (or function handle to generate the matrix) of the operation
    matrix_alloc = NULL;
    // The number of parameters
    parameter_num = 0;
}



//
// @brief Constructor of the class.
// @param qbit_num The number of qubits in the unitaries
// @return An instance of the class
Operation::Operation(int qbit_num_in) {

    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = "general";
    // The index of the qubit on which the operation acts (target_qbit >= 0) 
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // The matrix (or function handle to generate the matrix) of the operation
    matrix_alloc = NULL;
    // The number of parameters
    parameter_num = 0;
}


//
// @brief Destructor of the class
Operation::~Operation() {

    if ( matrix_alloc != NULL ) {
        qgd_free(matrix_alloc);
        matrix_alloc = NULL;
    }
}
   
//
// @brief Set the number of qubits spanning the matrix of the operation
// @param qbit_num The number of qubits spanning the matrix
void Operation::set_qbit_num( int qbit_num_in ) {
    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

}

//
// @brief Call to terive the operation matrix
QGD_Complex16* Operation::matrix() {
    return matrix_alloc;
}

//
// @brief Call to terive the operation matrix
int Operation::matrix(QGD_Complex16* retrive_matrix ) {
    memcpy( retrive_matrix, matrix_alloc, matrix_size*matrix_size*sizeof(QGD_Complex16) );
    return 0;
}


//
// @brief Call to set the stored matrix in the operation
// @param The pointer pointing to the matrix to be set
void Operation::set_matrix( QGD_Complex16* input) {
    if ( matrix_alloc == NULL ) {
        matrix_alloc = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size,sizeof(QGD_Complex16), 64);
    }
    memcpy( matrix_alloc, input, matrix_size*matrix_size*sizeof(QGD_Complex16) );
}

     
//
// @brief Call to reorder the qubits in the matrix of the operation
// @param qbit_list The list of qubits spanning the matrix

void Operation::reorder_qubits( vector<int> qbit_list ) {
      
    // check the number of qubits
    if (qbit_list.size() != qbit_num ) {
        printf("Wrong number of qubits\n");
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


//
// @brief Call to get the index of the target qubit
// @return Return with the index of the target qubit (return with -1 if target qubit was not set)
int Operation::get_target_qbit() {
    return target_qbit;
}

//
// @brief Call to get the index of the control qubit
// @return Return with the index of the control qubit (return with -1 if control qubit was not set)
int Operation::get_control_qbit()  {
    return control_qbit;
}

//
// @brief Call to get the number of free parameters
// @return Return with the index of the target qubit (return with -1 if target qubit was not set)
int Operation::get_parameter_num() {
    return parameter_num;
}


//
// @brief Call to get the type of the operation
// @return Return with the string indicating the type of the operation
string Operation::get_type() {
    return type;
}


//
// @brief Create a clone of the present class
// @return Return with a pointer pointing to the cloned object
Operation* Operation::clone() {

    Operation* ret = new Operation( qbit_num );
 
    if (matrix_alloc != NULL) {
        ret->set_matrix( matrix_alloc );
    }
    

    return ret;

}

  
