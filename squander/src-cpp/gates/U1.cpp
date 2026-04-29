#include "gate_kernel_templates.h"
#include "U1.h"

U1::U1() : U1(-1, -1) {}

U1::U1(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name          = "U1";
    type          = U1_OPERATION;
    parameter_num = 1;
    control_qbit  = -1;
}

U1::~U1() {}

U1* U1::clone() {
    U1* ret = new U1(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> U1::get_parameter_multipliers() const {
    return {1.0};
}

Matrix U1::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int lambda_offset = 0 * precomputed_sincos.stride;
    const double s_lambda = precomputed_sincos[lambda_offset + 0];
    const double c_lambda = precomputed_sincos[lambda_offset + 1];
    return u1_gate_kernel_from_trig<Matrix, double>(s_lambda, c_lambda);
}

Matrix_float U1::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int lambda_offset = 0 * precomputed_sincos.stride;
    const float s_lambda = precomputed_sincos[lambda_offset + 0];
    const float c_lambda = precomputed_sincos[lambda_offset + 1];
    return u1_gate_kernel_from_trig<Matrix_float, float>(s_lambda, c_lambda);
}

Matrix U1::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int lambda_offset = 0 * precomputed_sincos.stride;
    const double s_lambda = precomputed_sincos[lambda_offset + 0];
    const double c_lambda = precomputed_sincos[lambda_offset + 1];
    return u1_inverse_gate_kernel_from_trig<Matrix, double>(s_lambda, c_lambda);
}

Matrix_float U1::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int lambda_offset = 0 * precomputed_sincos.stride;
    const float s_lambda = precomputed_sincos[lambda_offset + 0];
    const float c_lambda = precomputed_sincos[lambda_offset + 1];
    return u1_inverse_gate_kernel_from_trig<Matrix_float, float>(s_lambda, c_lambda);
}

Matrix U1::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix();
    }
    const int lambda_offset = 0 * precomputed_sincos.stride;
    const double s_lambda = precomputed_sincos[lambda_offset + 0];
    const double c_lambda = precomputed_sincos[lambda_offset + 1];
    return u1_derivative_kernel_from_trig<Matrix, double>(s_lambda, c_lambda);
}

Matrix_float U1::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix_float();
    }
    const int lambda_offset = 0 * precomputed_sincos.stride;
    const float s_lambda = precomputed_sincos[lambda_offset + 0];
    const float c_lambda = precomputed_sincos[lambda_offset + 1];
    return u1_derivative_kernel_from_trig<Matrix_float, float>(s_lambda, c_lambda);
}
