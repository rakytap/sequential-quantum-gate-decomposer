#include "gate_kernel_templates.h"
#include "SDG.h"

SDG::SDG() : SDG(-1, -1) {}

SDG::SDG(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "SDG";
    type         = SDG_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

SDG::~SDG() {}

SDG* SDG::clone() {
    SDG* ret = new SDG(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix SDG::gate_kernel(const Matrix_real& /*parameters*/) {
    return sdg_gate_kernel<Matrix, double>();
}

Matrix_float SDG::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return sdg_gate_kernel<Matrix_float, float>();
}

Matrix SDG::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return s_gate_kernel<Matrix, double>();
}

Matrix_float SDG::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return s_gate_kernel<Matrix_float, float>();
}
