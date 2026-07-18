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
/*! \file CROT.cpp
    \brief Class representing a controlled Y rotattion gate.
*/

#include "CROT.h"
#include "gate_kernel_templates.h"

//static tbb::spin_mutex my_mutex;

namespace {
template<typename RealMatrixT, typename RealT>
void read_crot_trig(
    const RealMatrixT& precomputed_sincos,
    RealT& s_theta,
    RealT& c_theta,
    RealT& s_phi,
    RealT& c_phi) {

    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    s_theta = precomputed_sincos[theta_offset + 0];
    c_theta = precomputed_sincos[theta_offset + 1];
    s_phi = precomputed_sincos[phi_offset + 0];
    c_phi = precomputed_sincos[phi_offset + 1];
}
}
/**
@brief Nullary constructor of the class.
*/
CROT::CROT(){

        // A string describing the type of the gate
        type = CROT_OPERATION;
        
         // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        
        target_qbit = -1;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;
        parameter_num = 2;

        name = "CROT";

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
CROT::CROT(int qbit_num_in, int target_qbit_in, int control_qbit_in) {

        name = "CROT";

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate

        // A string describing the type of the gate
        type = CROT_OPERATION;
        



        if (control_qbit_in >= qbit_num) {
	    std::stringstream sstream;
	    sstream << "The index of the control qubit is larger than the number of qubits in CROT gate." << std::endl;
	    print(sstream, 0);	  
            throw sstream.str();
        }

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = control_qbit_in;


        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;
        

        parameter_num=2;
}

/**
@brief Destructor of the class
*/
CROT::~CROT() {

}

Matrix
CROT::gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float
CROT::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix
CROT::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float
CROT::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix
CROT::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    Matrix ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

Matrix_float
CROT::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    Matrix_float ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

Matrix
CROT::derivative_aux_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    Matrix ret;
    derivative_aux_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

Matrix_float
CROT::derivative_aux_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    Matrix_float ret;
    derivative_aux_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

void
CROT::gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    double s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real, double>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    build_crot_gate_kernel_from_trig_to<Matrix, double>(output, s_theta, c_theta, s_phi, c_phi);
}

void
CROT::gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    float s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real_float, float>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    build_crot_gate_kernel_from_trig_to<Matrix_float, float>(output, s_theta, c_theta, s_phi, c_phi);
}

void
CROT::inverse_gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    double s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real, double>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    build_crot_inverse_gate_kernel_from_trig_to<Matrix, double>(output, s_theta, c_theta, s_phi, c_phi);
}

void
CROT::inverse_gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    float s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real_float, float>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    build_crot_inverse_gate_kernel_from_trig_to<Matrix_float, float>(output, s_theta, c_theta, s_phi, c_phi);
}

void
CROT::derivative_kernel_to(const Matrix_real& precomputed_sincos, int param_idx, Matrix& output) {
    double s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real, double>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    if (param_idx == 0) {
        build_crot_theta_derivative_kernel_from_trig_to<Matrix, double>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    if (param_idx == 1) {
        build_crot_phi_derivative_kernel_from_trig_to<Matrix, double>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    output = Matrix();
}

void
CROT::derivative_kernel_to(const Matrix_real_float& precomputed_sincos, int param_idx, Matrix_float& output) {
    float s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real_float, float>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    if (param_idx == 0) {
        build_crot_theta_derivative_kernel_from_trig_to<Matrix_float, float>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    if (param_idx == 1) {
        build_crot_phi_derivative_kernel_from_trig_to<Matrix_float, float>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    output = Matrix_float();
}

void
CROT::derivative_aux_kernel_to(const Matrix_real& precomputed_sincos, int param_idx, Matrix& output) {
    double s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real, double>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    if (param_idx == 0) {
        build_crot_theta_derivative_aux_kernel_from_trig_to<Matrix, double>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    if (param_idx == 1) {
        build_crot_phi_derivative_aux_kernel_from_trig_to<Matrix, double>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    output = Matrix();
}

void
CROT::derivative_aux_kernel_to(const Matrix_real_float& precomputed_sincos, int param_idx, Matrix_float& output) {
    float s_theta, c_theta, s_phi, c_phi;
    read_crot_trig<Matrix_real_float, float>(precomputed_sincos, s_theta, c_theta, s_phi, c_phi);
    if (param_idx == 0) {
        build_crot_theta_derivative_aux_kernel_from_trig_to<Matrix_float, float>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    if (param_idx == 1) {
        build_crot_phi_derivative_aux_kernel_from_trig_to<Matrix_float, float>(output, s_theta, c_theta, s_phi, c_phi);
        return;
    }
    output = Matrix_float();
}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CROT* CROT::clone() {

    CROT* ret = new CROT(qbit_num, target_qbit, control_qbit);

    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}

std::vector<double>
CROT::get_parameter_multipliers() const {
    return {2.0, 1.0};
}
