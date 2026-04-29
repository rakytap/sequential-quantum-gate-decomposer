#include "gate_kernel_templates.h"
#include "R.h"

R::R() : R(-1, -1) {}

R::R(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name          = "R";
    type          = R_OPERATION;
    parameter_num = 2;
    control_qbit  = -1;
}

R::~R() {}

R* R::clone() {
    R* ret = new R(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> R::get_parameter_multipliers() const {
    return {2.0, 1.0};
}

Matrix R::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    return r_gate_kernel_from_trig<Matrix, double>(s_theta, c_theta, s_phi, c_phi);
}

Matrix_float R::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    return r_gate_kernel_from_trig<Matrix_float, float>(s_theta, c_theta, s_phi, c_phi);
}

Matrix R::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    return r_inverse_gate_kernel_from_trig<Matrix, double>(s_theta, c_theta, s_phi, c_phi);
}

Matrix_float R::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    return r_inverse_gate_kernel_from_trig<Matrix_float, float>(s_theta, c_theta, s_phi, c_phi);
}

Matrix R::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];

    if (param_idx == 0) {
        return r_derivative_kernel_theta_from_trig<Matrix, double>(s_theta, c_theta, s_phi, c_phi);
    }
    if (param_idx == 1) {
        return r_derivative_kernel_phi_from_trig<Matrix, double>(s_theta, s_phi, c_phi);
    }

    return Matrix();
}

Matrix_float R::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];

    if (param_idx == 0) {
        return r_derivative_kernel_theta_from_trig<Matrix_float, float>(s_theta, c_theta, s_phi, c_phi);
    }
    if (param_idx == 1) {
        return r_derivative_kernel_phi_from_trig<Matrix_float, float>(s_theta, s_phi, c_phi);
    }

    return Matrix_float();
}

