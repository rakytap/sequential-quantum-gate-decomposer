#include "gate_kernel_templates.h"
#include "Y.h"

Y::Y() : Y(-1, -1) {}

Y::Y(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "Y";
    type         = Y_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

Y::~Y() {}

Y* Y::clone() {
    Y* ret = new Y(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix Y::gate_kernel(const Matrix_real& /*parameters*/) {
    return y_gate_kernel<Matrix, double>();
}

Matrix_float Y::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return y_gate_kernel<Matrix_float, float>();
}

Matrix Y::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return y_gate_kernel<Matrix, double>();
}

Matrix_float Y::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return y_gate_kernel<Matrix_float, float>();
}
