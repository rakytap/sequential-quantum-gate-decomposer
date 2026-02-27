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
/*! \file CCX.h
    \brief Header file for a class representing a CCX (Toffoli) operation.
*/

#ifndef CCX_H
#define CCX_H

#include "matrix.h"
#include "Gate.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a CCX (Toffoli) operation.
*/
class CCX: public Gate {

protected:

public:

/**
@brief Nullary constructor of the class.
*/
CCX();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbits_in Vector of control qubit indices (should contain exactly 2 elements for CCX)
*/
CCX(int qbit_num_in, int target_qbit_in, const std::vector<int>& control_qbits_in);

/**
@brief Destructor of the class
*/
virtual ~CCX();




/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual CCX* clone() override;

/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
virtual Matrix get_matrix() override;

/**
@brief Call to retrieve the gate matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB (optional)
@return Returns with a matrix of the gate
*/
virtual Matrix get_matrix(int parallel) override;

/**
@brief Call to apply the gate operation on the input matrix (without parameters)
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
virtual void apply_to(Matrix& input, int parallel = 0) override;

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
virtual void apply_to(Matrix& input, const Matrix_real& parameters, int parallel = 0);

std::vector<int> get_involved_qubits(bool only_target) override;

};

#endif //CCX