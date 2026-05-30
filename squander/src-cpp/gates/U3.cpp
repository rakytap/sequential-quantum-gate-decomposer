/**
 * Copyright (C) Miklos Maroti, 2021
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gate_kernel_templates.h"
#include "U3.h"

namespace {

template <typename RealMatrixT, typename RealT>
void read_u3_trig(const RealMatrixT& precomputed_sincos, RealT& sin_theta, RealT& cos_theta, RealT& sin_phi, RealT& cos_phi, RealT& sin_lambda, RealT& cos_lambda) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    sin_theta = precomputed_sincos[theta_offset + 0];
    cos_theta = precomputed_sincos[theta_offset + 1];
    sin_phi = precomputed_sincos[phi_offset + 0];
    cos_phi = precomputed_sincos[phi_offset + 1];
    sin_lambda = precomputed_sincos[lambda_offset + 0];
    cos_lambda = precomputed_sincos[lambda_offset + 1];
}

}

U3::U3() : U3(-1, -1) {}

U3::U3(int qbit_num_in, int target_qbit_in) : Gate(qbit_num_in) {
    name        = "U3";
    type        = U3_OPERATION;
    target_qbit = target_qbit_in;
    control_qbit = -1;
    parameter_num = 3;
    if (qbit_num > 0 && target_qbit >= qbit_num) {
        std::string err("U3: target qubit index out of range.");
        throw err;
    }
}

U3::~U3() {}

U3* U3::clone() {
    U3* ret = new U3(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> U3::get_parameter_multipliers() const {
    return {2.0, 1.0, 1.0};
}

Matrix U3::gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float U3::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix U3::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float U3::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix U3::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    Matrix ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

Matrix_float U3::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    Matrix_float ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

void U3::gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    double sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig<Matrix_real, double>(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    calc_one_qubit_u3_from_trig_to<Matrix, double>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

void U3::gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    float sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig<Matrix_real_float, float>(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    calc_one_qubit_u3_from_trig_to<Matrix_float, float>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

void U3::inverse_gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    double sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig<Matrix_real, double>(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    calc_one_qubit_u3_inverse_from_trig_to<Matrix, double>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

void U3::inverse_gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    float sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig<Matrix_real_float, float>(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    calc_one_qubit_u3_inverse_from_trig_to<Matrix_float, float>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

void U3::derivative_kernel_to(const Matrix_real& precomputed_sincos, int param_idx, Matrix& output) {
    double sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig<Matrix_real, double>(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    if (param_idx == 0) {
        u3_derivative_kernel_theta_from_trig_to<Matrix, double>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        return;
    }
    if (param_idx == 1) {
        u3_derivative_kernel_phi_from_trig_to<Matrix, double>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        return;
    }
    if (param_idx == 2) {
        u3_derivative_kernel_lambda_from_trig_to<Matrix, double>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        return;
    }
    output = Matrix();
}

void U3::derivative_kernel_to(const Matrix_real_float& precomputed_sincos, int param_idx, Matrix_float& output) {
    float sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig<Matrix_real_float, float>(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    if (param_idx == 0) {
        u3_derivative_kernel_theta_from_trig_to<Matrix_float, float>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        return;
    }
    if (param_idx == 1) {
        u3_derivative_kernel_phi_from_trig_to<Matrix_float, float>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        return;
    }
    if (param_idx == 2) {
        u3_derivative_kernel_lambda_from_trig_to<Matrix_float, float>(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        return;
    }
    output = Matrix_float();
}

Matrix u3_matrix_kernel(double ThetaOver2, double Phi, double Lambda) {
    double sin_theta, cos_theta;
    double sin_phi, cos_phi;
    double sin_lambda, cos_lambda;
    qgd_sincos<double>(ThetaOver2, &sin_theta, &cos_theta);
    qgd_sincos<double>(Phi, &sin_phi, &cos_phi);
    qgd_sincos<double>(Lambda, &sin_lambda, &cos_lambda);
    return calc_one_qubit_u3_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

Matrix_float u3_matrix_kernel_f(float ThetaOver2, float Phi, float Lambda) {
    float sin_theta, cos_theta;
    float sin_phi, cos_phi;
    float sin_lambda, cos_lambda;
    qgd_sincos<float>(ThetaOver2, &sin_theta, &cos_theta);
    qgd_sincos<float>(Phi, &sin_phi, &cos_phi);
    qgd_sincos<float>(Lambda, &sin_lambda, &cos_lambda);
    return calc_one_qubit_u3_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}
