#include "gate_kernel_templates.h"
#include "SXdg.h"

SXdg::SXdg() : SXdg(-1, -1) {}

SXdg::SXdg(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "SXdg";
    type         = SXDG_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

SXdg::~SXdg() {}

SXdg* SXdg::clone() {
    SXdg* ret = new SXdg(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix SXdg::gate_kernel(const Matrix_real& /*parameters*/) {
    return sxdg_gate_kernel<Matrix, double>();
}

Matrix_float SXdg::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return sxdg_gate_kernel<Matrix_float, float>();
}

Matrix SXdg::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return sx_gate_kernel<Matrix, double>();
}

Matrix_float SXdg::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return sx_gate_kernel<Matrix_float, float>();
}
