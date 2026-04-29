#include "gate_kernel_templates.h"
#include "Z.h"

Z::Z() : Z(-1, -1) {}

Z::Z(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "Z";
    type         = Z_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

Z::~Z() {}

Z* Z::clone() {
    Z* ret = new Z(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix Z::gate_kernel(const Matrix_real& /*parameters*/) {
    return z_gate_kernel<Matrix, double>();
}

Matrix_float Z::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return z_gate_kernel<Matrix_float, float>();
}

Matrix Z::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return z_gate_kernel<Matrix, double>();
}

Matrix_float Z::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return z_gate_kernel<Matrix_float, float>();
}
