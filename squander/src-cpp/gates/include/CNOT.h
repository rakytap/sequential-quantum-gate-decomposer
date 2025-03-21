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
/*! \file CNOT.h
    \brief Header file for a class representing a CNOT operation.
*/

#ifndef CNOT_H
#define CNOT_H

#include "matrix.h"
#include "Gate.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a CNOT operation.
*/
class CNOT: public Gate {

protected:


public:

/**
@brief Nullary constructor of the class.
*/
CNOT();


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
virtual ~CNOT();

/**
@brief Call to retrieve the operation matrix
@param parallel Set true to apply parallel kernels, false otherwise
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
@brief Call to apply the gate on the input array/matrix by CNOT*input
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual void apply_to( Matrix& input, int parallel );


/**
@brief Call to apply the gate on the input array/matrix by input*CNOT
@param input The input array on which the gate is applied
*/
virtual void apply_from_right( Matrix& input );


/**
@brief Call to set the number of qubits spanning the matrix of the operation
@param qbit_num The number of qubits
*/
virtual void set_qbit_num(int qbit_num);

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list);


/**
@brief Set static values for matrix of the gates.
@param u3_1qbit Matrix parameter for the gate.

*/
Matrix calc_one_qubit_u3( );

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual CNOT* clone();

};

#endif //CNOT
