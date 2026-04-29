#include "gate_kernel_templates.h"
#include "U2.h"

static const double M_PIOver4 = M_PI / 4.0;

U2::U2() : U2(-1, -1) {}

U2::U2(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name          = "U2";
    type          = U2_OPERATION;
    parameter_num = 2;
    control_qbit  = -1;
}

U2::~U2() {}

U2* U2::clone() {
    U2* ret = new U2(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> U2::get_parameter_multipliers() const {
    return {1.0, 1.0};
}

Matrix U2::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const int lambda_offset = 1 * precomputed_sincos.stride;
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    const double s_lambda = precomputed_sincos[lambda_offset + 0];
    const double c_lambda = precomputed_sincos[lambda_offset + 1];
    return u2_gate_kernel_from_trig<Matrix, double>(s_phi, c_phi, s_lambda, c_lambda);
}

Matrix_float U2::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const int lambda_offset = 1 * precomputed_sincos.stride;
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    const float s_lambda = precomputed_sincos[lambda_offset + 0];
    const float c_lambda = precomputed_sincos[lambda_offset + 1];
    return u2_gate_kernel_from_trig<Matrix_float, float>(s_phi, c_phi, s_lambda, c_lambda);
}

Matrix U2::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const int lambda_offset = 1 * precomputed_sincos.stride;
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    const double s_lambda = precomputed_sincos[lambda_offset + 0];
    const double c_lambda = precomputed_sincos[lambda_offset + 1];
    return u2_inverse_gate_kernel_from_trig<Matrix, double>(s_phi, c_phi, s_lambda, c_lambda);
}

Matrix_float U2::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const int lambda_offset = 1 * precomputed_sincos.stride;
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    const float s_lambda = precomputed_sincos[lambda_offset + 0];
    const float c_lambda = precomputed_sincos[lambda_offset + 1];
    return u2_inverse_gate_kernel_from_trig<Matrix_float, float>(s_phi, c_phi, s_lambda, c_lambda);
}

Matrix U2::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const int lambda_offset = 1 * precomputed_sincos.stride;
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    const double s_lambda = precomputed_sincos[lambda_offset + 0];
    const double c_lambda = precomputed_sincos[lambda_offset + 1];

    if (param_idx == 0) {
        return u2_derivative_kernel_phi_from_trig<Matrix, double>(s_phi, c_phi, s_lambda, c_lambda);
    }
    if (param_idx == 1) {
        return u2_derivative_kernel_lambda_from_trig<Matrix, double>(s_phi, c_phi, s_lambda, c_lambda);
    }

    return Matrix();
}

Matrix_float U2::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const int lambda_offset = 1 * precomputed_sincos.stride;
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    const float s_lambda = precomputed_sincos[lambda_offset + 0];
    const float c_lambda = precomputed_sincos[lambda_offset + 1];

    if (param_idx == 0) {
        return u2_derivative_kernel_phi_from_trig<Matrix_float, float>(s_phi, c_phi, s_lambda, c_lambda);
    }
    if (param_idx == 1) {
        return u2_derivative_kernel_lambda_from_trig<Matrix_float, float>(s_phi, c_phi, s_lambda, c_lambda);
    }

    return Matrix_float();
}
