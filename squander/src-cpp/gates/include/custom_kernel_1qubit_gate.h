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
/*! \file custom_kernel_1qubit_gate.h
    \brief Header file for a class representing a single qubit gate with custom gate kernel
*/

#ifndef CUSTOM_KERNEL_1QUBIT_GATE_H
#define CUSTOM_KERNEL_1QUBIT_GATE_H

#include "U3.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a custom_kernel_1qubit_gate gate.  The matrix of the gate is [exp(i*varphi/2),0;0,exp(i*varphi/2) ]. The input rotation angle is varphi/2.
*/
class custom_kernel_1qubit_gate: public U3 {

    // the lernel of the gate operation
    Matrix kernel;

public:

/**
@brief Nullacustom_kernel_1qubit_gate constructor of the class.
*/
custom_kernel_1qubit_gate();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param kernel_in The 2x2 matrix kernel of the gate
*/
custom_kernel_1qubit_gate(int qbit_num_in, int target_qbit_in, Matrix& kernel_in);

/**
@brief Destructor of the class
*/
~custom_kernel_1qubit_gate();


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_to( Matrix& input );


/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void apply_from_right( Matrix_real& parameters, Matrix& input );




/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
custom_kernel_1qubit_gate* clone();



};


#endif //U3

