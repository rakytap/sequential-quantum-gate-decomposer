#include "gate_kernel_templates.h"
#include "S.h"

S::S() : S(-1, -1) {}

S::S(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "S";
    type         = S_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

S::~S() {}

S* S::clone() {
    S* ret = new S(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix S::gate_kernel(const Matrix_real& /*parameters*/) {
    return s_gate_kernel<Matrix, double>();
}

Matrix_float S::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return s_gate_kernel<Matrix_float, float>();
}

Matrix S::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return sdg_gate_kernel<Matrix, double>();
}

Matrix_float S::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return sdg_gate_kernel<Matrix_float, float>();
}
