#include "gate_kernel_templates.h"
#include "RY.h"

RY::RY() : RY(-1, -1) {}

RY::RY(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name          = "RY";
    type          = RY_OPERATION;
    parameter_num = 1;
    control_qbit  = -1;
}

RY::~RY() {}

RY* RY::clone() {
    RY* ret = new RY(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> RY::get_parameter_multipliers() const {
    return {2.0};
}

Matrix RY::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return ry_gate_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RY::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return ry_gate_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

Matrix RY::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return ry_inverse_gate_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RY::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return ry_inverse_gate_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

Matrix RY::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return ry_derivative_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RY::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix_float();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return ry_derivative_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}
