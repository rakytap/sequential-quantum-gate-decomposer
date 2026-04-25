#include "gate_kernel_templates.h"
#include "T.h"

T::T() : T(-1, -1) {}

T::T(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "T";
    type         = T_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

T::~T() {}

T* T::clone() {
    T* ret = new T(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix T::gate_kernel(const Matrix_real& /*parameters*/) {
    return t_gate_kernel<Matrix, double>();
}

Matrix_float T::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return t_gate_kernel<Matrix_float, float>();
}

Matrix T::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return tdg_gate_kernel<Matrix, double>();
}

Matrix_float T::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return tdg_gate_kernel<Matrix_float, float>();
}
