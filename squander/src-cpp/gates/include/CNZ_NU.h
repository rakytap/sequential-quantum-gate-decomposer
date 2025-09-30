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
/*! \file CNZ_NU.h
    \brief Header file for a class representing N-qubit non-uniform controlled-Z gate operation.
*/

#ifndef CNZNU_H
#define CNZNU_H

#include <vector>
#include "common.h"
#include "matrix.h"
#include "matrix_real.h"
#include "Gate.h"

/**
@brief Base class for the representation of general gate operations.
*/
class CNZ_NU : public Gate {


protected:

   /// Centers for computing distance-based logits
   std::vector<double> centers;

   /// Temperature parameter for softmax (lower = more discrete)
   double temperature;

public:

/**
@brief Default constructor of the class.
@return An instance of the class
*/
CNZ_NU();

/**
@brief Destructor of the class
*/
virtual ~CNZ_NU();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
CNZ_NU(int qbit_num_in);


/**
@brief Call to retrieve the operation matrix
@param parallel Set true to apply parallel kernels, false otherwise
@return Returns with a matrix of the operation
*/
Matrix get_matrix();

/**
@brief Call to retrieve the operation matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix get_matrix(int parallel);

/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix
@return Returns with a matrix of the operation
*/
Matrix get_matrix(Matrix_real& parameters);

/**
@brief Call to retrieve the operation matrix
@param parameters An array of parameters to calculate the matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the operation
*/
Matrix get_matrix(Matrix_real& parameters, int parallel);

void apply_to_list(Matrix_real& parameters_mtx, std::vector<Matrix>& input );

void apply_to_list(Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel );

/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void apply_to(Matrix_real& parameters_mtx, Matrix& input, int parallel );

/**
@brief Call to apply the derivative of the gate on the input array/matrix
@param parameters_mtx The parameters of the gate
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns vector of matrices with the derivatives
*/
std::vector<Matrix> apply_derivate_to(Matrix_real& parameters_mtx, Matrix& input, int parallel );


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void apply_from_right(Matrix_real& parameters_mtx, Matrix& input );

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
virtual void set_qbit_num( int qbit_num_in );

/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type get_type();

/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int get_qbit_num();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CNZ_NU* clone();

protected:

/**
@brief Compute softmax probability for position k given parameter x
@param x Input parameter (determines which position to prefer)
@param k Index of the position
@return Softmax probability for position k
*/
double softmax_k(double x, int k);

/**
@brief Derivative of softmax probability w.r.t. x
@param x Input parameter
@param k Index of the position
@return Derivative of softmax probability
*/
double softmax_k_derivative(double x, int k);

/**
@brief Set the temperature parameter for softmax
@param temp Temperature value (lower = more discrete, higher = more smooth)
*/
void set_temperature(double temp);

};


#endif //OPERATION
