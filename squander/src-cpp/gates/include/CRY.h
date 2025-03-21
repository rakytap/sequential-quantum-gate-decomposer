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
/*! \file CRY.h
    \brief Header file for a class representing a controlled rotation gate around the Y axis.
*/

#ifndef CRZ_H
#define CRZ_H

#include "RY.h"
#include "CNOT.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a CRY gate.
*/
class CRY: public RY {


public:

/**
@brief Nullary constructor of the class.
*/
CRY();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
*/
CRY(int qbit_num_in, int target_qbit_in, int control_qbit_in);

/**
@brief Destructor of the class
*/
virtual ~CRY();


/**
@brief Call to apply the gate on the input array/matrix by CRY*input
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual void apply_to( Matrix_real& parameters, Matrix& input, int parallel  );


/**
@brief Call to apply the gate on the input array/matrix by input*CRY
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
virtual void apply_from_right( Matrix_real& parameters, Matrix& input );

/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual std::vector<Matrix> apply_derivate_to( Matrix_real& parameters, Matrix& input, int parallel );


/**
@brief Call to set the final optimized parameters of the gate.
@param Theta Real parameter standing for the parameter theta.
*/
void set_optimized_parameters(double Theta );

/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
*/
Matrix_real get_optimized_parameters();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual CRY* clone();


/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
virtual Matrix_real extract_parameters( Matrix_real& parameters );

};


#endif //CRY

