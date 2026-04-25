#include "gate_kernel_templates.h"
#include "SX.h"

SX::SX() : SX(-1, -1) {}

SX::SX(int qbit_num_in, int target_qbit_in) : U3(qbit_num_in, target_qbit_in) {
    name         = "SX";
    type         = SX_OPERATION;
    parameter_num = 0;
    control_qbit  = -1;
}

SX::~SX() {}

SX* SX::clone() {
    SX* ret = new SX(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

Matrix SX::gate_kernel(const Matrix_real& /*parameters*/) {
    return sx_gate_kernel<Matrix, double>();
}

Matrix_float SX::gate_kernel(const Matrix_real_float& /*parameters*/) {
    return sx_gate_kernel<Matrix_float, float>();
}

Matrix SX::inverse_gate_kernel(const Matrix_real& /*parameters*/) {
    return sxdg_gate_kernel<Matrix, double>();
}

Matrix_float SX::inverse_gate_kernel(const Matrix_real_float& /*parameters*/) {
    return sxdg_gate_kernel<Matrix_float, float>();
}
