/**
 * Copyright (C) Miklos Maroti, 2021
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gate_kernel_templates.h"
#include "U3.h"

namespace {

template <typename MatrixT>
void ensure_2x2(MatrixT& output) {
    if (output.rows != 2 || output.cols != 2 || output.stride != 2) {
        output = MatrixT(2, 2);
    }
}

template <typename MatrixT, typename RealT>
void set_u3_kernel(MatrixT& output, RealT sin_theta, RealT cos_theta, RealT sin_phi, RealT cos_phi, RealT sin_lambda, RealT cos_lambda) {
    ensure_2x2(output);
    const RealT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RealT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;

    output[0].real =  cos_theta;
    output[0].imag =  (RealT)0;
    output[1].real = -sin_theta * cos_lambda;
    output[1].imag = -sin_theta * sin_lambda;
    output[2].real =  sin_theta * cos_phi;
    output[2].imag =  sin_theta * sin_phi;
    output[3].real =  cos_theta * cos_phi_lambda;
    output[3].imag =  cos_theta * sin_phi_lambda;
}

template <typename MatrixT, typename RealT>
void set_u3_inverse_kernel(MatrixT& output, RealT sin_theta, RealT cos_theta, RealT sin_phi, RealT cos_phi, RealT sin_lambda, RealT cos_lambda) {
    ensure_2x2(output);
    const RealT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RealT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;

    output[0].real =  cos_theta;
    output[0].imag =  (RealT)0;
    output[1].real =  sin_theta * cos_phi;
    output[1].imag = -sin_theta * sin_phi;
    output[2].real = -sin_theta * cos_lambda;
    output[2].imag =  sin_theta * sin_lambda;
    output[3].real =  cos_theta * cos_phi_lambda;
    output[3].imag = -cos_theta * sin_phi_lambda;
}

template <typename MatrixT, typename RealT>
void set_u3_derivative_kernel(MatrixT& output, RealT sin_theta, RealT cos_theta, RealT sin_phi, RealT cos_phi, RealT sin_lambda, RealT cos_lambda, int param_idx) {
    ensure_2x2(output);
    const RealT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RealT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;

    if (param_idx == 0) {
        output[0].real = -sin_theta; output[0].imag = (RealT)0;
        output[1].real = -cos_theta * cos_lambda; output[1].imag = -cos_theta * sin_lambda;
        output[2].real =  cos_theta * cos_phi; output[2].imag =  cos_theta * sin_phi;
        output[3].real = -sin_theta * cos_phi_lambda;
        output[3].imag = -sin_theta * sin_phi_lambda;
        return;
    }

    if (param_idx == 1) {
        output[0].real = (RealT)0; output[0].imag = (RealT)0;
        output[1].real = (RealT)0; output[1].imag = (RealT)0;
        output[2].real = -sin_theta * sin_phi; output[2].imag =  sin_theta * cos_phi;
        output[3].real = -cos_theta * sin_phi_lambda;
        output[3].imag =  cos_theta * cos_phi_lambda;
        return;
    }

    if (param_idx == 2) {
        output[0].real = (RealT)0; output[0].imag = (RealT)0;
        output[1].real =  sin_theta * sin_lambda; output[1].imag = -sin_theta * cos_lambda;
        output[2].real = (RealT)0; output[2].imag = (RealT)0;
        output[3].real = -cos_theta * sin_phi_lambda;
        output[3].imag =  cos_theta * cos_phi_lambda;
        return;
    }

    output = MatrixT();
}

template <typename RealMatrixT, typename RealT>
void read_u3_trig(const RealMatrixT& precomputed_sincos, RealT& sin_theta, RealT& cos_theta, RealT& sin_phi, RealT& cos_phi, RealT& sin_lambda, RealT& cos_lambda) {
    const int theta_offset = 0 * precomputed_sincos.stride;
    const int phi_offset = 1 * precomputed_sincos.stride;
    const int lambda_offset = 2 * precomputed_sincos.stride;
    sin_theta = precomputed_sincos[theta_offset + 0];
    cos_theta = precomputed_sincos[theta_offset + 1];
    sin_phi = precomputed_sincos[phi_offset + 0];
    cos_phi = precomputed_sincos[phi_offset + 1];
    sin_lambda = precomputed_sincos[lambda_offset + 0];
    cos_lambda = precomputed_sincos[lambda_offset + 1];
}

template <typename RealMatrixT, typename MatrixT, typename RealT>
void u3_family_gate_kernel_to(gate_type type, const RealMatrixT& precomputed_sincos, MatrixT& output) {
    switch (type) {
        case H_OPERATION:
            h_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case X_OPERATION:
            x_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case Y_OPERATION:
            y_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case Z_OPERATION:
            z_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case S_OPERATION:
            s_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case SDG_OPERATION:
            sdg_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case T_OPERATION:
            t_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case TDG_OPERATION:
            tdg_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case SX_OPERATION:
            sx_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case SXDG_OPERATION:
            sxdg_gate_kernel_to<MatrixT, RealT>(output);
            return;
        default:
            break;
    }

    if (type == RX_OPERATION || type == RY_OPERATION || type == RZ_OPERATION || type == U1_OPERATION) {
        const RealT s0 = precomputed_sincos[0];
        const RealT c0 = precomputed_sincos[1];
        if (type == RX_OPERATION) {
            rx_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else if (type == RY_OPERATION) {
            ry_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else if (type == RZ_OPERATION) {
            rz_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else {
            u1_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        return;
    }

    if (type == R_OPERATION) {
        const int phi_offset = precomputed_sincos.stride;
        r_gate_kernel_from_trig_to<MatrixT, RealT>(
            output,
            precomputed_sincos[0],
            precomputed_sincos[1],
            precomputed_sincos[phi_offset + 0],
            precomputed_sincos[phi_offset + 1]
        );
        return;
    }

    if (type == U2_OPERATION) {
        const int lambda_offset = precomputed_sincos.stride;
        u2_gate_kernel_from_trig_to<MatrixT, RealT>(
            output,
            precomputed_sincos[0],
            precomputed_sincos[1],
            precomputed_sincos[lambda_offset + 0],
            precomputed_sincos[lambda_offset + 1]
        );
        return;
    }

    RealT sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    set_u3_kernel(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

template <typename RealMatrixT, typename MatrixT, typename RealT>
void u3_family_inverse_gate_kernel_to(gate_type type, const RealMatrixT& precomputed_sincos, MatrixT& output) {
    switch (type) {
        case H_OPERATION:
            h_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case X_OPERATION:
            x_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case Y_OPERATION:
            y_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case Z_OPERATION:
            z_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case S_OPERATION:
            sdg_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case SDG_OPERATION:
            s_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case T_OPERATION:
            tdg_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case TDG_OPERATION:
            t_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case SX_OPERATION:
            sxdg_gate_kernel_to<MatrixT, RealT>(output);
            return;
        case SXDG_OPERATION:
            sx_gate_kernel_to<MatrixT, RealT>(output);
            return;
        default:
            break;
    }

    if (type == RX_OPERATION || type == RY_OPERATION || type == RZ_OPERATION || type == U1_OPERATION) {
        const RealT s0 = precomputed_sincos[0];
        const RealT c0 = precomputed_sincos[1];
        if (type == RX_OPERATION) {
            rx_inverse_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else if (type == RY_OPERATION) {
            ry_inverse_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else if (type == RZ_OPERATION) {
            rz_inverse_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else {
            u1_inverse_gate_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        return;
    }

    if (type == R_OPERATION) {
        const int phi_offset = precomputed_sincos.stride;
        r_inverse_gate_kernel_from_trig_to<MatrixT, RealT>(
            output,
            precomputed_sincos[0],
            precomputed_sincos[1],
            precomputed_sincos[phi_offset + 0],
            precomputed_sincos[phi_offset + 1]
        );
        return;
    }

    if (type == U2_OPERATION) {
        const int lambda_offset = precomputed_sincos.stride;
        u2_inverse_gate_kernel_from_trig_to<MatrixT, RealT>(
            output,
            precomputed_sincos[0],
            precomputed_sincos[1],
            precomputed_sincos[lambda_offset + 0],
            precomputed_sincos[lambda_offset + 1]
        );
        return;
    }

    RealT sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    set_u3_inverse_kernel(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

template <typename RealMatrixT, typename MatrixT, typename RealT>
void u3_family_derivative_kernel_to(gate_type type, const RealMatrixT& precomputed_sincos, int param_idx, MatrixT& output) {
    if (type == RX_OPERATION || type == RY_OPERATION || type == RZ_OPERATION || type == U1_OPERATION) {
        if (param_idx != 0) {
            output = MatrixT();
            return;
        }
        const RealT s0 = precomputed_sincos[0];
        const RealT c0 = precomputed_sincos[1];
        if (type == RX_OPERATION) {
            rx_derivative_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else if (type == RY_OPERATION) {
            ry_derivative_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else if (type == RZ_OPERATION) {
            rz_derivative_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        else {
            u1_derivative_kernel_from_trig_to<MatrixT, RealT>(output, s0, c0);
        }
        return;
    }

    if (type == R_OPERATION) {
        const int phi_offset = precomputed_sincos.stride;
        const RealT s_theta = precomputed_sincos[0];
        const RealT c_theta = precomputed_sincos[1];
        const RealT s_phi = precomputed_sincos[phi_offset + 0];
        const RealT c_phi = precomputed_sincos[phi_offset + 1];
        if (param_idx == 0) {
            r_derivative_kernel_theta_from_trig_to<MatrixT, RealT>(output, s_theta, c_theta, s_phi, c_phi);
            return;
        }
        if (param_idx == 1) {
            r_derivative_kernel_phi_from_trig_to<MatrixT, RealT>(output, s_theta, s_phi, c_phi);
            return;
        }
        output = MatrixT();
        return;
    }

    if (type == U2_OPERATION) {
        const int lambda_offset = precomputed_sincos.stride;
        const RealT s_phi = precomputed_sincos[0];
        const RealT c_phi = precomputed_sincos[1];
        const RealT s_lambda = precomputed_sincos[lambda_offset + 0];
        const RealT c_lambda = precomputed_sincos[lambda_offset + 1];
        if (param_idx == 0) {
            u2_derivative_kernel_phi_from_trig_to<MatrixT, RealT>(output, s_phi, c_phi, s_lambda, c_lambda);
            return;
        }
        if (param_idx == 1) {
            u2_derivative_kernel_lambda_from_trig_to<MatrixT, RealT>(output, s_phi, c_phi, s_lambda, c_lambda);
            return;
        }
        output = MatrixT();
        return;
    }

    RealT sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda;
    read_u3_trig(precomputed_sincos, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    set_u3_derivative_kernel(output, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, param_idx);
}

}

U3::U3() : U3(-1, -1) {}

U3::U3(int qbit_num_in, int target_qbit_in) : Gate(qbit_num_in) {
    name        = "U3";
    type        = U3_OPERATION;
    target_qbit = target_qbit_in;
    control_qbit = -1;
    parameter_num = 3;
    if (qbit_num > 0 && target_qbit >= qbit_num) {
        std::string err("U3: target qubit index out of range.");
        throw err;
    }
}

U3::~U3() {}

U3* U3::clone() {
    U3* ret = new U3(qbit_num, target_qbit);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

std::vector<double> U3::get_parameter_multipliers() const {
    return {2.0, 1.0, 1.0};
}

Matrix U3::gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float U3::gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix U3::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix_float U3::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float ret;
    inverse_gate_kernel_to(precomputed_sincos, ret);
    return ret;
}

Matrix U3::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    Matrix ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

Matrix_float U3::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    Matrix_float ret;
    derivative_kernel_to(precomputed_sincos, param_idx, ret);
    return ret;
}

void U3::gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    u3_family_gate_kernel_to<Matrix_real, Matrix, double>(type, precomputed_sincos, output);
}

void U3::gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    u3_family_gate_kernel_to<Matrix_real_float, Matrix_float, float>(type, precomputed_sincos, output);
}

void U3::inverse_gate_kernel_to(const Matrix_real& precomputed_sincos, Matrix& output) {
    u3_family_inverse_gate_kernel_to<Matrix_real, Matrix, double>(type, precomputed_sincos, output);
}

void U3::inverse_gate_kernel_to(const Matrix_real_float& precomputed_sincos, Matrix_float& output) {
    u3_family_inverse_gate_kernel_to<Matrix_real_float, Matrix_float, float>(type, precomputed_sincos, output);
}

void U3::derivative_kernel_to(const Matrix_real& precomputed_sincos, int param_idx, Matrix& output) {
    u3_family_derivative_kernel_to<Matrix_real, Matrix, double>(type, precomputed_sincos, param_idx, output);
}

void U3::derivative_kernel_to(const Matrix_real_float& precomputed_sincos, int param_idx, Matrix_float& output) {
    u3_family_derivative_kernel_to<Matrix_real_float, Matrix_float, float>(type, precomputed_sincos, param_idx, output);
}

Matrix u3_matrix_kernel(double ThetaOver2, double Phi, double Lambda) {
    double sin_theta, cos_theta;
    double sin_phi, cos_phi;
    double sin_lambda, cos_lambda;
    qgd_sincos<double>(ThetaOver2, &sin_theta, &cos_theta);
    qgd_sincos<double>(Phi, &sin_phi, &cos_phi);
    qgd_sincos<double>(Lambda, &sin_lambda, &cos_lambda);
    return calc_one_qubit_u3_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}

Matrix_float u3_matrix_kernel_f(float ThetaOver2, float Phi, float Lambda) {
    float sin_theta, cos_theta;
    float sin_phi, cos_phi;
    float sin_lambda, cos_lambda;
    qgd_sincos<float>(ThetaOver2, &sin_theta, &cos_theta);
    qgd_sincos<float>(Phi, &sin_phi, &cos_phi);
    qgd_sincos<float>(Lambda, &sin_lambda, &cos_lambda);
    return calc_one_qubit_u3_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}
