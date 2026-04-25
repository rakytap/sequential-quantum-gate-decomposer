#include "gate_kernel_templates.h"
#include "CU.h"

static const double M_PIOver2 = M_PI / 2.0;

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
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    const double sin_theta = precomputed_sincos[theta_offset + 0];
    const double cos_theta = precomputed_sincos[theta_offset + 1];
    const double sin_phi = precomputed_sincos[phi_offset + 0];
    const double cos_phi = precomputed_sincos[phi_offset + 1];
    const double sin_lambda = precomputed_sincos[lambda_offset + 0];
    const double cos_lambda = precomputed_sincos[lambda_offset + 1];
    const double sin_gamma = precomputed_sincos[gamma_offset + 0];
    const double cos_gamma = precomputed_sincos[gamma_offset + 1];
    return cu_gate_kernel_from_trig<Matrix, double>(
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
}

Matrix_float CU::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    const float sin_theta = precomputed_sincos[theta_offset + 0];
    const float cos_theta = precomputed_sincos[theta_offset + 1];
    const float sin_phi = precomputed_sincos[phi_offset + 0];
    const float cos_phi = precomputed_sincos[phi_offset + 1];
    const float sin_lambda = precomputed_sincos[lambda_offset + 0];
    const float cos_lambda = precomputed_sincos[lambda_offset + 1];
    const float sin_gamma = precomputed_sincos[gamma_offset + 0];
    const float cos_gamma = precomputed_sincos[gamma_offset + 1];
    return cu_gate_kernel_from_trig<Matrix_float, float>(
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
}

Matrix CU::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    const double sin_theta = precomputed_sincos[theta_offset + 0];
    const double cos_theta = precomputed_sincos[theta_offset + 1];
    const double sin_phi = precomputed_sincos[phi_offset + 0];
    const double cos_phi = precomputed_sincos[phi_offset + 1];
    const double sin_lambda = precomputed_sincos[lambda_offset + 0];
    const double cos_lambda = precomputed_sincos[lambda_offset + 1];
    const double sin_gamma = precomputed_sincos[gamma_offset + 0];
    const double cos_gamma = precomputed_sincos[gamma_offset + 1];
    return cu_inverse_gate_kernel_from_trig<Matrix, double>(
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
}

Matrix_float CU::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    const float sin_theta = precomputed_sincos[theta_offset + 0];
    const float cos_theta = precomputed_sincos[theta_offset + 1];
    const float sin_phi = precomputed_sincos[phi_offset + 0];
    const float cos_phi = precomputed_sincos[phi_offset + 1];
    const float sin_lambda = precomputed_sincos[lambda_offset + 0];
    const float cos_lambda = precomputed_sincos[lambda_offset + 1];
    const float sin_gamma = precomputed_sincos[gamma_offset + 0];
    const float cos_gamma = precomputed_sincos[gamma_offset + 1];
    return cu_inverse_gate_kernel_from_trig<Matrix_float, float>(
        sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
}

Matrix CU::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    const double sin_theta = precomputed_sincos[theta_offset + 0];
    const double cos_theta = precomputed_sincos[theta_offset + 1];
    const double sin_phi = precomputed_sincos[phi_offset + 0];
    const double cos_phi = precomputed_sincos[phi_offset + 1];
    const double sin_lambda = precomputed_sincos[lambda_offset + 0];
    const double cos_lambda = precomputed_sincos[lambda_offset + 1];
    const double sin_gamma = precomputed_sincos[gamma_offset + 0];
    const double cos_gamma = precomputed_sincos[gamma_offset + 1];

    if (param_idx == 0) {
        Matrix kernel = u3_derivative_kernel_theta_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        multiply_2x2_by_phase<Matrix, double>(kernel, sin_gamma, cos_gamma);
        return kernel;
    }
    if (param_idx == 1) {
        Matrix kernel = u3_derivative_kernel_phi_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        multiply_2x2_by_phase<Matrix, double>(kernel, sin_gamma, cos_gamma);
        return kernel;
    }
    if (param_idx == 2) {
        Matrix kernel = u3_derivative_kernel_lambda_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        multiply_2x2_by_phase<Matrix, double>(kernel, sin_gamma, cos_gamma);
        return kernel;
    }
    if (param_idx == 3) {
        return cu_derivative_kernel_gamma_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
    }

    return Matrix();
}

Matrix_float CU::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    const int gamma_offset = 3 * precomputed_sincos.stride;
    const float sin_theta = precomputed_sincos[theta_offset + 0];
    const float cos_theta = precomputed_sincos[theta_offset + 1];
    const float sin_phi = precomputed_sincos[phi_offset + 0];
    const float cos_phi = precomputed_sincos[phi_offset + 1];
    const float sin_lambda = precomputed_sincos[lambda_offset + 0];
    const float cos_lambda = precomputed_sincos[lambda_offset + 1];
    const float sin_gamma = precomputed_sincos[gamma_offset + 0];
    const float cos_gamma = precomputed_sincos[gamma_offset + 1];

    if (param_idx == 0) {
        Matrix_float kernel = u3_derivative_kernel_theta_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        multiply_2x2_by_phase<Matrix_float, float>(kernel, sin_gamma, cos_gamma);
        return kernel;
    }
    if (param_idx == 1) {
        Matrix_float kernel = u3_derivative_kernel_phi_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        multiply_2x2_by_phase<Matrix_float, float>(kernel, sin_gamma, cos_gamma);
        return kernel;
    }
    if (param_idx == 2) {
        Matrix_float kernel = u3_derivative_kernel_lambda_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
        multiply_2x2_by_phase<Matrix_float, float>(kernel, sin_gamma, cos_gamma);
        return kernel;
    }
    if (param_idx == 3) {
        return cu_derivative_kernel_gamma_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
    }

    return Matrix_float();
}
