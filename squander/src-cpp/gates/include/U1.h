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
/*! \file U1.h
    \brief Header file for a class representing a U1 gate.
*/

#ifndef U1_H
#define U1_H

#include "Gate.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>

/**
@brief A class representing a U1 gate.
*/
class U1: public Gate {

protected:
   /// Parameters of the U1 gate after the decomposition of the unitary is done
   Matrix_real parameters;

public:

/**
@brief Nullary constructor of the class.
*/
U1();

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
*/
U1(int qbit_num_in, int target_qbit_in);

/**
@brief Destructor of the class
*/
virtual ~U1();

/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@return Returns with a matrix of the gate
*/
virtual Matrix get_matrix( Matrix_real& parameters  );

/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
virtual Matrix get_matrix( Matrix_real& parameters, int parallel  );

/**
@brief Call to apply the gate on a list of inputs
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual void apply_to_list( Matrix_real& parameters, std::vector<Matrix>& inputs, int parallel );

/**
@brief Call to apply the gate on the input array/matrix
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual void apply_to( Matrix_real& parameters, Matrix& input, int parallel );

/**
@brief Call to apply the gate on the input array/matrix by input*U1
@param parameters An array of parameters to calculate the matrix of the U1 gate.
@param input The input array on which the gate is applied
*/
virtual void apply_from_right( Matrix_real& parameters, Matrix& input );

/**
@brief Call to evaluate the derivate of the circuit on an input with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual std::vector<Matrix> apply_derivate_to( Matrix_real& parameters, Matrix& input, int parallel );

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
virtual void set_qbit_num(int qbit_num_in);

/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list);

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual U1* clone();

/**
@brief Call to set the final optimized parameters of the gate.
@param Lambda Real parameter standing for the parameter lambda.
*/
void set_optimized_parameters(double Lambda );

/**
@brief Call to get the final optimized parameters of the gate.
@return Returns with the parameters of the U1 gate.
*/
Matrix_real get_optimized_parameters();

/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
virtual Matrix_real extract_parameters( Matrix_real& parameters );

/**
@brief Calculate the matrix of a U1 gate corresponding to the given parameters acting on a single qbit space. (Virtual method override)
@param Theta Real parameter standing for the parameter theta (ignored for U1).
@param Phi Real parameter standing for the parameter phi (ignored for U1).
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
virtual Matrix calc_one_qubit_u3(double Theta, double Phi, double Lambda) override;

/**
@brief Calculate the matrix of a U1 gate corresponding to the given parameters acting on a single qbit space. (Convenience method)
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix calc_one_qubit_u3(double Lambda);

};

#endif //U1