#include "gate_kernel_templates.h"
#include "Tdg.h"

Tdg::Tdg() : Tdg(-1, -1) {}

Tdg::Tdg(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "Tdg";
    type         = TDG_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

Tdg::~Tdg() {}

Tdg* Tdg::clone() {
    Tdg* ret = new Tdg(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix Tdg::gate_kernel(const Matrix_real& /*parameters*/) {
    return tdg_gate_kernel<Matrix, double>();
}

Matrix_float Tdg::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return tdg_gate_kernel<Matrix_float, float>();
}

Matrix Tdg::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return t_gate_kernel<Matrix, double>();
}

Matrix_float Tdg::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return t_gate_kernel<Matrix_float, float>();
}
