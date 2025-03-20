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
/*! \file SYC.h
    \brief Header file for a class representing a Sycamore gate.
*/

#ifndef SYC_H
#define SYC_H

#include "matrix.h"
#include "Gate.h"
#define _USE_MATH_DEFINES
#include <math.h>



/**
@brief A class representing a SYC operation.
*/
class SYC: public Gate {

protected:


public:

/**
@brief Nullary constructor of the class.
*/
SYC();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
SYC(int qbit_num_in, int target_qbit_in,  int control_qbit_in);

/**
@brief Destructor of the class
*/
~SYC();

/**
@brief Call to retrieve the operation matrix
@return Returns with the matrix of the operation
*/
Matrix get_matrix();


/**
@brief Call to retrieve the operation matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with the matrix of the operation
*/
Matrix get_matrix(int parallel);

/**
@brief Call to apply the gate on the input array/matrix by SYC*input
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void apply_to( Matrix& input, int parallel );


/**
@brief Call to apply the gate on the input array/matrix by input*SYC
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix& input );


/**
@brief Call to set the number of qubits spanning the matrix of the operation
@param qbit_num The number of qubits
*/
void set_qbit_num(int qbit_num);

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits( std::vector<int> qbit_list);

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SYC* clone();


};

#endif //SYC
