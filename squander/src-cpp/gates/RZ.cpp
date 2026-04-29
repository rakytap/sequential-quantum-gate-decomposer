#include "gate_kernel_templates.h"
#include "RZ.h"

RZ::RZ() : RZ(-1, -1) {}

RZ::RZ(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name          = "RZ";
    type          = RZ_OPERATION;
    parameter_num = 1;
    control_qbit  = -1;
}

RZ::~RZ() {}

RZ* RZ::clone() {
    RZ* ret = new RZ(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> RZ::get_parameter_multipliers() const {
    return {2.0};
}

Matrix RZ::gate_kernel(const Matrix_real& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    return rz_gate_kernel_from_trig<Matrix, double>(s_phi, c_phi);
}

Matrix_float RZ::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    return rz_gate_kernel_from_trig<Matrix_float, float>(s_phi, c_phi);
}

Matrix RZ::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    return rz_inverse_gate_kernel_from_trig<Matrix, double>(s_phi, c_phi);
}

Matrix_float RZ::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int phi_offset = 0 * precomputed_sincos.stride;
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    return rz_inverse_gate_kernel_from_trig<Matrix_float, float>(s_phi, c_phi);
}

Matrix RZ::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix();
    }
    const int phi_offset = 0 * precomputed_sincos.stride;
    const double s_phi = precomputed_sincos[phi_offset + 0];
    const double c_phi = precomputed_sincos[phi_offset + 1];
    return rz_derivative_kernel_from_trig<Matrix, double>(s_phi, c_phi);
}

Matrix_float RZ::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    if (param_idx != 0) {
        return Matrix_float();
    }
    const int phi_offset = 0 * precomputed_sincos.stride;
    const float s_phi = precomputed_sincos[phi_offset + 0];
    const float c_phi = precomputed_sincos[phi_offset + 1];
    return rz_derivative_kernel_from_trig<Matrix_float, float>(s_phi, c_phi);
}
