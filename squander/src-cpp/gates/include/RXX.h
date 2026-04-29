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
/*! \file RXX.cpp
    \brief Class representing a RXX gate.
*/

#ifndef RXX_H
#define RXX_H

#include "Gate.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>
/**
@brief A class representing a RXX gate.
*/
class RXX: public Gate {


public:

/**
@brief Nullary constructor of the class.
*/
RXX();

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements for RXX)
*/
RXX(int qbit_num_in, const std::vector<int>& target_qbits_in);

/**
@brief Destructor of the class
*/
~RXX();

/// Build and return the 4x4 RXX unitary for the given parameter.
Matrix gate_kernel(const Matrix_real& precomputed_sincos) override;
Matrix_float gate_kernel(const Matrix_real_float& precomputed_sincos) override;
Matrix inverse_gate_kernel(const Matrix_real& precomputed_sincos) override;
Matrix_float inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) override;
Matrix derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) override;
Matrix_float derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) override;

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RXX* clone() override;


/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
std::vector<double> get_parameter_multipliers() const override;

};
#endif //RXX,
