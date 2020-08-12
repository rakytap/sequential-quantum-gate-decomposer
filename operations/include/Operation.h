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


#pragma once
#include <vector> 


#include <common.h> 

using namespace std;

class Operation {


protected:

    // number of qubits spanning the matrix of the operation
    int qbit_num;
    // A string describing the type of the operation
    string type;
    // The index of the qubit on which the operation acts (target_qbit >= 0) 
    int target_qbit;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    int control_qbit;
    // The size of teh matrx
    int matrix_size;
    // The allocated array of the operatrion matrix
    MKL_Complex16* matrix_alloc;
    // the number of free parameters
    int parameter_num;
    


public:
    //
    // @brief Deafult constructor of the class.
    // @return An instance of the class
    Operation();

    //
    // @brief Destructor of the class
    ~Operation();


    //
    // @brief Constructor of the class.
    // @param qbit_num The number of qubits in the unitaries
    // @return An instance of the class
    Operation(int);


    //
    // @brief Call to terive the operation matrix
    MKL_Complex16* matrix();
   
    //
    // @brief Set the number of qubits spanning the matrix of the operation
    // @param qbit_num The number of qubits spanning the matrix
    void set_qbit_num( int qbit_num_in );
     
    //
    // @brief Call to reorder the qubits in the matrix of the operation
    // @param qbit_list The list of qubits spanning the matrix
    void reorder_qubits( vector<int> qbit_list );


    //
    // @brief Call to get the index of the target qubit
    // @return Return with the index of the target qubit (return with -1 if target qubit was not set)
    int get_target_qbit();


    //
    // @brief Call to get the index of the control qubit
    // @return Return with the index of the control qubit (return with -1 if control qubit was not set)
    int get_control_qbit();


    //
    // @brief Call to get the number of free parameters
    // @return Return with the index of the target qubit (return with -1 if target qubit was not set)
    int get_parameter_num();


    //
    // @brief Call to get the type of the operation
    // @return Return with the string indicating the type of the operation
    string get_type();

};

        
