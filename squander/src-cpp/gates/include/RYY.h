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
/*! \file RYY.h
    \brief Class representing a RYY gate.
*/

#ifndef RYY_H
#define RYY_H

#include "Gate.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>
/**
@brief A class representing a RYY gate.
*/
class RYY: public Gate {


public:

/**
@brief Nullary constructor of the class.
*/
RYY();

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements for RYY)
*/
RYY(int qbit_num_in, const std::vector<int>& target_qbits_in);

/**
@brief Destructor of the class
*/
~RYY();

/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
Matrix
get_matrix(Matrix_real& parameters);

/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
get_matrix(Matrix_real& parameters, int parallel);


/**
@brief Call to apply the gate operation on the input matrix
@param parameters An array of parameters to calculate the matrix elements
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
void apply_to(Matrix_real& parameters, Matrix& input, int parallel);
/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/

std::vector<Matrix> apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel );
/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RYY* clone();

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void reorder_qubits(std::vector<int> qbit_list);

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void set_qbit_num(int qbit_num_in);

/**
@brief Get list of involved qubits
@param only_target If true, return only target qubits, otherwise include control qubits too
@return Vector of qubit indices
*/
std::vector<int> get_involved_qubits(bool only_target);
};
#endif //RYY
