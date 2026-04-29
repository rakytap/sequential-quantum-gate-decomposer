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

#include "RXX.h"
#include "gate_kernel_templates.h"
using namespace std;

// Build the 4x4 RXX kernel matrix for either float32 or float64 precision.
/**
@brief Nullary constructor of the class.
*/
RXX::RXX() : Gate() {

    // A string labeling the gate operation
    name = "RXX";

    // A string describing the type of the gate
    type = RXX_OPERATION;

    // Initialize target qubits vector (empty for nullary constructor)
    target_qbits.clear();

    parameter_num = 1;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements for RXX)
*/
RXX::RXX(int qbit_num_in, const std::vector<int>& target_qbits_in)
    : Gate(qbit_num_in, target_qbits_in) {

    // A string labeling the gate operation
    name = "RXX";

    // A string describing the type of the gate
    type = RXX_OPERATION;

    // Validate that we have exactly 2 target qubits
    if (target_qbits_in.size() != 2) {
        std::stringstream sstream;
        sstream << "RXX gate requires exactly 2 target qubits, got " << target_qbits_in.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // Check that target qubits are unique
    if (target_qbits_in[0] == target_qbits_in[1]) {
        std::stringstream sstream;
        sstream << "The two target qubits cannot be the same" << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    parameter_num = 1;
}

/**
@brief Destructor of the class
*/
RXX::~RXX() {

}

/**
@brief Call to retrieve the gate matrix
@return Returns with a matrix of the gate
*/
/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parameters An array of parameters to calculate the matrix elements
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
Matrix RXX::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return build_rxx_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RXX::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return build_rxx_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

Matrix RXX::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return build_rxx_kernel_from_trig<Matrix, double>(-s_theta, c_theta);
}

Matrix_float RXX::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return build_rxx_kernel_from_trig<Matrix_float, float>(-s_theta, c_theta);
}

Matrix RXX::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return build_rxx_derivative_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RXX::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix_float();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return build_rxx_derivative_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RXX* RXX::clone() {

    RXX* ret = new RXX(qbit_num, target_qbits);

    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);

    return ret;
}

std::vector<double>
RXX::get_parameter_multipliers() const {
    return {2.0};
}
