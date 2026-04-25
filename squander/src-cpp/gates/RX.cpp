#include "gate_kernel_templates.h"
#include "RX.h"

RX::RX() : RX(-1, -1) {}

RX::RX(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name          = "RX";
    type          = RX_OPERATION;
    parameter_num = 1;
    control_qbit  = -1;
}

RX::~RX() {}

RX* RX::clone() {
    RX* ret = new RX(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> RX::get_parameter_multipliers() const {
    return {2.0};
}

Matrix RX::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return rx_gate_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RX::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return rx_gate_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

Matrix RX::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return rx_inverse_gate_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RX::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return rx_inverse_gate_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}

Matrix RX::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const double s_theta = precomputed_sincos[theta_offset + 0];
    const double c_theta = precomputed_sincos[theta_offset + 1];
    return rx_derivative_kernel_from_trig<Matrix, double>(s_theta, c_theta);
}

Matrix_float RX::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix_float();
    }
    const int theta_offset = 0 * precomputed_sincos.stride;
    const float s_theta = precomputed_sincos[theta_offset + 0];
    const float c_theta = precomputed_sincos[theta_offset + 1];
    return rx_derivative_kernel_from_trig<Matrix_float, float>(s_theta, c_theta);
}
