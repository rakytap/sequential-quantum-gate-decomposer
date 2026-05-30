#include "gate_kernel_templates.h"
#include "CU.h"

static const double M_PIOver2 = M_PI / 2.0;

namespace {
template<typename RealMatrixT, typename RealT>
void read_cu_trig(
    const RealMatrixT& precomputed_sincos,
    RealT& sin_theta,
    RealT& cos_theta,
    RealT& sin_phi,
    RealT& cos_phi,
    RealT& sin_lambda,
    RealT& cos_lambda,
    RealT& sin_gamma,
    RealT& cos_gamma) {

    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    sin_theta = precomputed_sincos[theta_offset + 0];
    cos_theta = precomputed_sincos[theta_offset + 1];
    sin_phi = precomputed_sincos[phi_offset + 0];
    cos_phi = precomputed_sincos[phi_offset + 1];
    sin_lambda = precomputed_sincos[lambda_offset + 0];
    cos_lambda = precomputed_sincos[lambda_offset + 1];
    sin_gamma = precomputed_sincos[gamma_offset + 0];
    cos_gamma = precomputed_sincos[gamma_offset + 1];
}
}

CU::CU() : CU(-1, -1, -1) {}

CU::CU(int qbit_num_in, int target_qbit_in, int control_qbit_in)
    : U3(qbit_num_in, target_qbit_in) {
    name          = "CU";
    type          = CU_OPERATION;
    parameter_num = 4;
    if (qbit_num > 0 && (control_qbit_in < 0 || control_qbit_in >= qbit_num)) {
        std::string err("CU: control qubit index out of range.");
        throw err;
    }
    control_qbit = control_qbit_in;
}

CU::~CU() {}

CU* CU::clone() {
    CU* ret = new CU(qbit_num, target_qbit, control_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> CU::get_parameter_multipliers() const {
    return {2.0, 1.0, 1.0, 1.0};
}

Matrix CU::gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float CU::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix CU::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float CU::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix CU::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    Matrix ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

Matrix_float CU::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    Matrix_float ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

void
CU::gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    double sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma;
    read_cu_trig<Matrix_real, double>(
        precomputed_sincos,
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
    cu_gate_kernel_from_trig_to<Matrix, double>(
        output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
}

void
CU::gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    float sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma;
    read_cu_trig<Matrix_real_float, float>(
        precomputed_sincos,
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
    cu_gate_kernel_from_trig_to<Matrix_float, float>(
        output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
}

void
CU::inverse_gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    double sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma;
    read_cu_trig<Matrix_real, double>(
        precomputed_sincos,
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
    cu_inverse_gate_kernel_from_trig_to<Matrix, double>(
        output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
}

void
CU::inverse_gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    float sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma;
    read_cu_trig<Matrix_real_float, float>(
        precomputed_sincos,
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
    cu_inverse_gate_kernel_from_trig_to<Matrix_float, float>(
        output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );
}

void
CU::derivative_kernel_to(const Matrix_real& precomputed_sincos, int param_idx, Matrix& output) {
    double sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma;
    read_cu_trig<Matrix_real, double>(
        precomputed_sincos,
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );

    if (param_idx == 0) {
        u3_derivative_kernel_theta_from_trig_to<Matrix, double>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda
        );
        multiply_2x2_by_phase<Matrix, double>(output, sin_gamma, cos_gamma);
        return;
    }
    if (param_idx == 1) {
        u3_derivative_kernel_phi_from_trig_to<Matrix, double>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda
        );
        multiply_2x2_by_phase<Matrix, double>(output, sin_gamma, cos_gamma);
        return;
    }
    if (param_idx == 2) {
        u3_derivative_kernel_lambda_from_trig_to<Matrix, double>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda
        );
        multiply_2x2_by_phase<Matrix, double>(output, sin_gamma, cos_gamma);
        return;
    }
    if (param_idx == 3) {
        cu_derivative_kernel_gamma_from_trig_to<Matrix, double>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
        );
        return;
    }

    output = Matrix();
}

void
CU::derivative_kernel_to(const Matrix_real_float& precomputed_sincos, int param_idx, Matrix_float& output) {
    float sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma;
    read_cu_trig<Matrix_real_float, float>(
        precomputed_sincos,
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
    );

    if (param_idx == 0) {
        u3_derivative_kernel_theta_from_trig_to<Matrix_float, float>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda
        );
        multiply_2x2_by_phase<Matrix_float, float>(output, sin_gamma, cos_gamma);
        return;
    }
    if (param_idx == 1) {
        u3_derivative_kernel_phi_from_trig_to<Matrix_float, float>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda
        );
        multiply_2x2_by_phase<Matrix_float, float>(output, sin_gamma, cos_gamma);
        return;
    }
    if (param_idx == 2) {
        u3_derivative_kernel_lambda_from_trig_to<Matrix_float, float>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda
        );
        multiply_2x2_by_phase<Matrix_float, float>(output, sin_gamma, cos_gamma);
        return;
    }
    if (param_idx == 3) {
        cu_derivative_kernel_gamma_from_trig_to<Matrix_float, float>(
            output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma
        );
        return;
    }

    output = Matrix_float();
}
