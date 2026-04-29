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
/*! \file RYY.cpp
    \brief Class representing a RYY gate.
*/

#include "RYY.h"
#include "gate_kernel_templates.h"
using namespace std;

/**
@brief Nullary constructor of the class.
*/
RYY::RYY() : Gate() {

    // A string labeling the gate operation
    name = "RYY";

    // A string describing the type of the gate
    type = RYY_OPERATION;

    // Initialize target qubits vector (empty for nullary constructor)
    target_qbits.clear();

    parameter_num = 1;
}

/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbits_in Vector of target qubit indices (should contain exactly 2 elements for RYY)
*/
RYY::RYY(int qbit_num_in, const std::vector<int>& target_qbits_in)
    : Gate(qbit_num_in, target_qbits_in) {

    // A string labeling the gate operation
    name = "RYY";

    // A string describing the type of the gate
    type = RYY_OPERATION;

    // Validate that we have exactly 2 target qubits
    if (target_qbits_in.size() != 2) {
        std::stringstream sstream;
        sstream << "RYY gate requires exactly 2 target qubits, got " << target_qbits_in.size() << std::endl;
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
RYY::~RYY() {

}

Matrix RYY::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return build_ryy_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RYY::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return build_ryy_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

Matrix RYY::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return build_ryy_kernel_from_trig<Matrix, double>(-s_theta, c_theta);
}

Matrix_float RYY::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return build_ryy_kernel_from_trig<Matrix_float, float>(-s_theta, c_theta);
}

Matrix RYY::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return build_ryy_derivative_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RYY::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix_float();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return build_ryy_derivative_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
RYY* RYY::clone() {

    RYY* ret = new RYY(qbit_num, target_qbits);

    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);

    return ret;
}

std::vector<double>
RYY::get_parameter_multipliers() const {
    return {2.0};
}
