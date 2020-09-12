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
// @brief A base class responsible for constructing matrices of C-NOT gates
// gates acting on the N-qubit space

#pragma once
#include "qgd/Operation.h"
#include <math.h>

using namespace std;


/**
@brief A class representing a CNOT operation.
*/
class CNOT: public Operation {

protected:   
        

public: 
/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CNOT(int qbit_num_in, int target_qbit_in,  int control_qbit_in);

/**
@brief Destructor of the class
*/
~CNOT();

/**
@brief Call to retrieve the operation matrix
@return Returns with a pointer to the operation matrix
*/
QGD_Complex16* matrix();

/**
@brief Call to retrieve the operation matrix
@param retrieve_matrix A pointer to the preallocated array of the operation matrix.
@return Returns with 0 on success.
*/
int matrix(QGD_Complex16* retrieve_matrix );

/**
@brief Call to set the number of qubits spanning the matrix of the operation
@param qbit_num The number of qubits
*/
void set_qbit_num(int qbit_num);

/**
@brief Calculate the matrix of a CNOT gate operation acting on the space of qbit_num qubits.
@return Returns with a pointer to the operation matrix
*/
QGD_Complex16* composite_cnot();

/**
@brief Calculate the matrix of a CNOT gate operation acting on the space of qbit_num qubits.
@param CNOT_mtx A pointer to the preallocated array of the operation matrix.
@return Returns with 0 on success.
*/
int composite_cnot(QGD_Complex16* CNOT_mtx);



/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( vector<int> qbit_list);

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CNOT* clone();

};

                   
