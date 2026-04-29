#include "gate_kernel_templates.h"
#include "X.h"

X::X() : X(-1, -1) {}

X::X(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "X";
    type         = X_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

X::~X() {}

X* X::clone() {
    X* ret = new X(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix X::gate_kernel(const Matrix_real& /*parameters*/) {
    return x_gate_kernel<Matrix, double>();
}

Matrix_float X::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return x_gate_kernel<Matrix_float, float>();
}

Matrix X::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return x_gate_kernel<Matrix, double>();
}

Matrix_float X::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return x_gate_kernel<Matrix_float, float>();
}
