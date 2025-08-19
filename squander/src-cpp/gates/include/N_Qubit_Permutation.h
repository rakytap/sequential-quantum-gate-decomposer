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
/*! \file UN.h
    \brief Header file for a class for the representation of general gate operations on the first qbit_num-1 qubits.
*/

#ifndef N_Qubit_Permutation_H
#define N_Qubit_Permutation_H

#include <vector>
#include "common.h"
#include "matrix.h"
#include "matrix_real.h"
#include "Gate.h"

/**
@brief Base class for the representation of general gate operations.
*/
class N_Qubit_Permutation : public Gate {


protected:
   
   std::vector<int> pattern;

public:

/**
@brief Default constructor of the class.
@return An instance of the class
*/
N_Qubit_Permutation();

/**
@brief Destructor of the class
*/
virtual ~N_Qubit_Permutation();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
N_Qubit_Permutation(int qbit_num_in, std::vector<int> pattern_in);


/**
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix get_matrix();


/**
@brief Call to retrieve the operation matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix get_matrix( int parallel);


/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void apply_to( Matrix& input, int parallel );


/**
@brief Call to retrieve the qbit_num-1 kernel of the UN gate.
*/
Matrix construct_matrix_from_pattern( std::vector<int> pattern );

/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix& input );


/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list );


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type get_type();

std::vector<int> get_pattern();

/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int get_qbit_num();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
N_Qubit_Permutation* clone();

};


#endif //OPERATION
