/**
 * Shared inline templates for single-qubit gate kernel matrices.
 * Each function builds the 2x2 unitary matrix for the corresponding gate.
 * MT = matrix type (Matrix or Matrix_float), RT = real scalar type (double or float).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <math.h>
#include <utility>
#include "../../common/include/qgd_math.h"

template<typename MT>
inline void ensure_kernel_storage(MT& kernel, int rows, int cols) {
    if (kernel.rows != rows || kernel.cols != cols || kernel.stride != cols) {
        kernel = MT(rows, cols);
    }
}

// ---------------------------------------------------------------------------
// Zero-parameter gates
// ---------------------------------------------------------------------------

template<typename MT, typename RT>
inline void h_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real =  sq; u3[0].imag = (RT)0;
    u3[1].real =  sq; u3[1].imag = (RT)0;
    u3[2].real =  sq; u3[2].imag = (RT)0;
    u3[3].real = -sq; u3[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline MT h_gate_kernel() {
    MT u3(2, 2);
    h_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void x_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)0; u3[0].imag = (RT)0;
    u3[1].real = (RT)1; u3[1].imag = (RT)0;
    u3[2].real = (RT)1; u3[2].imag = (RT)0;
    u3[3].real = (RT)0; u3[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline MT x_gate_kernel() {
    MT u3(2, 2);
    x_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void y_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)0; u3[0].imag =  (RT)0;
    u3[1].real = (RT)0; u3[1].imag = -(RT)1;
    u3[2].real = (RT)0; u3[2].imag =  (RT)1;
    u3[3].real = (RT)0; u3[3].imag =  (RT)0;
}

template<typename MT, typename RT>
inline MT y_gate_kernel() {
    MT u3(2, 2);
    y_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void z_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real =  (RT)1; u3[0].imag = (RT)0;
    u3[1].real =  (RT)0; u3[1].imag = (RT)0;
    u3[2].real =  (RT)0; u3[2].imag = (RT)0;
    u3[3].real = -(RT)1; u3[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline MT z_gate_kernel() {
    MT u3(2, 2);
    z_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void s_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)1; u3[0].imag = (RT)0;
    u3[1].real = (RT)0; u3[1].imag = (RT)0;
    u3[2].real = (RT)0; u3[2].imag = (RT)0;
    u3[3].real = (RT)0; u3[3].imag = (RT)1;
}

template<typename MT, typename RT>
inline MT s_gate_kernel() {
    MT u3(2, 2);
    s_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void sdg_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)1; u3[0].imag =  (RT)0;
    u3[1].real = (RT)0; u3[1].imag =  (RT)0;
    u3[2].real = (RT)0; u3[2].imag =  (RT)0;
    u3[3].real = (RT)0; u3[3].imag = -(RT)1;
}

template<typename MT, typename RT>
inline MT sdg_gate_kernel() {
    MT u3(2, 2);
    sdg_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void t_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real = (RT)1; u3[0].imag = (RT)0;
    u3[1].real = (RT)0; u3[1].imag = (RT)0;
    u3[2].real = (RT)0; u3[2].imag = (RT)0;
    u3[3].real =    sq; u3[3].imag =    sq;
}

template<typename MT, typename RT>
inline MT t_gate_kernel() {
    MT u3(2, 2);
    t_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void tdg_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real = (RT)1; u3[0].imag =  (RT)0;
    u3[1].real = (RT)0; u3[1].imag =  (RT)0;
    u3[2].real = (RT)0; u3[2].imag =  (RT)0;
    u3[3].real =    sq; u3[3].imag =    -sq;
}

template<typename MT, typename RT>
inline MT tdg_gate_kernel() {
    MT u3(2, 2);
    tdg_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void sx_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)0.5; u3[0].imag =  (RT)0.5;
    u3[1].real = (RT)0.5; u3[1].imag = -(RT)0.5;
    u3[2].real = (RT)0.5; u3[2].imag = -(RT)0.5;
    u3[3].real = (RT)0.5; u3[3].imag =  (RT)0.5;
}

template<typename MT, typename RT>
inline MT sx_gate_kernel() {
    MT u3(2, 2);
    sx_gate_kernel_to<MT, RT>(u3);
    return u3;
}

template<typename MT, typename RT>
inline void sxdg_gate_kernel_to(MT& u3) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)0.5; u3[0].imag = -(RT)0.5;
    u3[1].real = (RT)0.5; u3[1].imag =  (RT)0.5;
    u3[2].real = (RT)0.5; u3[2].imag =  (RT)0.5;
    u3[3].real = (RT)0.5; u3[3].imag = -(RT)0.5;
}

template<typename MT, typename RT>
inline MT sxdg_gate_kernel() {
    MT u3(2, 2);
    sxdg_gate_kernel_to<MT, RT>(u3);
    return u3;
}

// ---------------------------------------------------------------------------
// Parametric gates
// ---------------------------------------------------------------------------

template<typename MT, typename RT>
inline void rx_gate_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real =  (RT)c; u3[0].imag =  (RT)0;
    u3[1].real =  (RT)0; u3[1].imag = -(RT)s;
    u3[2].real =  (RT)0; u3[2].imag = -(RT)s;
    u3[3].real =  (RT)c; u3[3].imag =  (RT)0;
}

template<typename MT, typename RT>
inline MT rx_gate_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    rx_gate_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void rx_inverse_gate_kernel_from_trig_to(MT& u3, RT s, RT c) {
    rx_gate_kernel_from_trig_to<MT, RT>(u3, -s, c);
}

template<typename MT, typename RT>
inline MT rx_inverse_gate_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    rx_inverse_gate_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void rx_derivative_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = -(RT)s; u3[0].imag =  (RT)0;
    u3[1].real =  (RT)0; u3[1].imag = -(RT)c;
    u3[2].real =  (RT)0; u3[2].imag = -(RT)c;
    u3[3].real = -(RT)s; u3[3].imag =  (RT)0;
}

template<typename MT, typename RT>
inline MT rx_derivative_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    rx_derivative_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void ry_gate_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real =  (RT)c; u3[0].imag = (RT)0;
    u3[1].real = -(RT)s; u3[1].imag = (RT)0;
    u3[2].real =  (RT)s; u3[2].imag = (RT)0;
    u3[3].real =  (RT)c; u3[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline MT ry_gate_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    ry_gate_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void ry_inverse_gate_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ry_gate_kernel_from_trig_to<MT, RT>(u3, -s, c);
}

template<typename MT, typename RT>
inline MT ry_inverse_gate_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    ry_inverse_gate_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void ry_derivative_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = -(RT)s; u3[0].imag = (RT)0;
    u3[1].real = -(RT)c; u3[1].imag = (RT)0;
    u3[2].real =  (RT)c; u3[2].imag = (RT)0;
    u3[3].real = -(RT)s; u3[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline MT ry_derivative_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    ry_derivative_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void rz_gate_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real =  (RT)c; u3[0].imag = -(RT)s;
    u3[1].real =  (RT)0; u3[1].imag =  (RT)0;
    u3[2].real =  (RT)0; u3[2].imag =  (RT)0;
    u3[3].real =  (RT)c; u3[3].imag =  (RT)s;
}

template<typename MT, typename RT>
inline MT rz_gate_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    rz_gate_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void rz_inverse_gate_kernel_from_trig_to(MT& u3, RT s, RT c) {
    rz_gate_kernel_from_trig_to<MT, RT>(u3, -s, c);
}

template<typename MT, typename RT>
inline MT rz_inverse_gate_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    rz_inverse_gate_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void rz_derivative_kernel_from_trig_to(MT& u3, RT s, RT c) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = -(RT)s; u3[0].imag = -(RT)c;
    u3[1].real =  (RT)0; u3[1].imag =  (RT)0;
    u3[2].real =  (RT)0; u3[2].imag =  (RT)0;
    u3[3].real = -(RT)s; u3[3].imag =  (RT)c;
}

template<typename MT, typename RT>
inline MT rz_derivative_kernel_from_trig(RT s, RT c) {
    MT u3(2, 2);
    rz_derivative_kernel_from_trig_to<MT, RT>(u3, s, c);
    return u3;
}

template<typename MT, typename RT>
inline void r_gate_kernel_from_trig_to(MT& u3, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real =  (RT)c_theta;
    u3[0].imag =  (RT)0;
    u3[1].real = -(RT)s_theta * s_phi;
    u3[1].imag = -(RT)s_theta * c_phi;
    u3[2].real =  (RT)s_theta * s_phi;
    u3[2].imag = -(RT)s_theta * c_phi;
    u3[3].real =  (RT)c_theta;
    u3[3].imag =  (RT)0;
}

template<typename MT, typename RT>
inline MT r_gate_kernel_from_trig(RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    MT u3(2, 2);
    r_gate_kernel_from_trig_to<MT, RT>(u3, s_theta, c_theta, s_phi, c_phi);
    return u3;
}

template<typename MT, typename RT>
inline void r_inverse_gate_kernel_from_trig_to(MT& u3, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    r_gate_kernel_from_trig_to<MT, RT>(u3, -s_theta, c_theta, s_phi, c_phi);
}

template<typename MT, typename RT>
inline MT r_inverse_gate_kernel_from_trig(RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    MT u3(2, 2);
    r_inverse_gate_kernel_from_trig_to<MT, RT>(u3, s_theta, c_theta, s_phi, c_phi);
    return u3;
}

template<typename MT, typename RT>
inline void r_derivative_kernel_theta_from_trig_to(MT& u3, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = -(RT)s_theta;     u3[0].imag =  (RT)0;
    u3[1].real = -(RT)c_theta * s_phi; u3[1].imag = -(RT)c_theta * c_phi;
    u3[2].real =  (RT)c_theta * s_phi; u3[2].imag = -(RT)c_theta * c_phi;
    u3[3].real = -(RT)s_theta;     u3[3].imag =  (RT)0;
}

template<typename MT, typename RT>
inline MT r_derivative_kernel_theta_from_trig(RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    MT u3(2, 2);
    r_derivative_kernel_theta_from_trig_to<MT, RT>(u3, s_theta, c_theta, s_phi, c_phi);
    return u3;
}

template<typename MT, typename RT>
inline void r_derivative_kernel_phi_from_trig_to(MT& u3, RT s_theta, RT s_phi, RT c_phi) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real =  (RT)0;              u3[0].imag = (RT)0;
    u3[1].real = -(RT)s_theta * c_phi; u3[1].imag =  (RT)s_theta * s_phi;
    u3[2].real =  (RT)s_theta * c_phi; u3[2].imag =  (RT)s_theta * s_phi;
    u3[3].real =  (RT)0;              u3[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline MT r_derivative_kernel_phi_from_trig(RT s_theta, RT s_phi, RT c_phi) {
    MT u3(2, 2);
    r_derivative_kernel_phi_from_trig_to<MT, RT>(u3, s_theta, s_phi, c_phi);
    return u3;
}

template<typename MT, typename RT>
inline void u1_gate_kernel_from_trig_to(MT& u3, RT s_lambda, RT c_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)1; u3[0].imag = (RT)0;
    u3[1].real = (RT)0; u3[1].imag = (RT)0;
    u3[2].real = (RT)0; u3[2].imag = (RT)0;
    u3[3].real = (RT)c_lambda; u3[3].imag = (RT)s_lambda;
}

template<typename MT, typename RT>
inline MT u1_gate_kernel_from_trig(RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u1_gate_kernel_from_trig_to<MT, RT>(u3, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u1_inverse_gate_kernel_from_trig_to(MT& u3, RT s_lambda, RT c_lambda) {
    u1_gate_kernel_from_trig_to<MT, RT>(u3, -s_lambda, c_lambda);
}

template<typename MT, typename RT>
inline MT u1_inverse_gate_kernel_from_trig(RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u1_inverse_gate_kernel_from_trig_to<MT, RT>(u3, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u1_derivative_kernel_from_trig_to(MT& u3, RT s_lambda, RT c_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    u3[0].real = (RT)0; u3[0].imag = (RT)0;
    u3[1].real = (RT)0; u3[1].imag = (RT)0;
    u3[2].real = (RT)0; u3[2].imag = (RT)0;
    u3[3].real = -(RT)s_lambda; u3[3].imag = (RT)c_lambda;
}

template<typename MT, typename RT>
inline MT u1_derivative_kernel_from_trig(RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u1_derivative_kernel_from_trig_to<MT, RT>(u3, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u2_gate_kernel_from_trig_to(MT& u3, RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT s_pl = s_phi * c_lambda + c_phi * s_lambda;
    const RT c_pl = c_phi * c_lambda - s_phi * s_lambda;
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real =  sq;             u3[0].imag =  (RT)0;
    u3[1].real = -sq * c_lambda;  u3[1].imag = -sq * s_lambda;
    u3[2].real =  sq * c_phi;     u3[2].imag =  sq * s_phi;
    u3[3].real =  sq * c_pl;      u3[3].imag =  sq * s_pl;
}

template<typename MT, typename RT>
inline MT u2_gate_kernel_from_trig(RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u2_gate_kernel_from_trig_to<MT, RT>(u3, s_phi, c_phi, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u2_inverse_gate_kernel_from_trig_to(MT& u3, RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT s_pl = s_phi * c_lambda + c_phi * s_lambda;
    const RT c_pl = c_phi * c_lambda - s_phi * s_lambda;
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real =  sq;             u3[0].imag =  (RT)0;
    u3[1].real =  sq * c_phi;     u3[1].imag = -sq * s_phi;
    u3[2].real = -sq * c_lambda;  u3[2].imag =  sq * s_lambda;
    u3[3].real =  sq * c_pl;      u3[3].imag = -sq * s_pl;
}

template<typename MT, typename RT>
inline MT u2_inverse_gate_kernel_from_trig(RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u2_inverse_gate_kernel_from_trig_to<MT, RT>(u3, s_phi, c_phi, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u2_derivative_kernel_phi_from_trig_to(MT& u3, RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT s_pl = s_phi * c_lambda + c_phi * s_lambda;
    const RT c_pl = c_phi * c_lambda - s_phi * s_lambda;
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real = (RT)0;          u3[0].imag = (RT)0;
    u3[1].real = (RT)0;          u3[1].imag = (RT)0;
    u3[2].real = -sq * s_phi;    u3[2].imag =  sq * c_phi;
    u3[3].real = -sq * s_pl;     u3[3].imag =  sq * c_pl;
}

template<typename MT, typename RT>
inline MT u2_derivative_kernel_phi_from_trig(RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u2_derivative_kernel_phi_from_trig_to<MT, RT>(u3, s_phi, c_phi, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u2_derivative_kernel_lambda_from_trig_to(MT& u3, RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT s_pl = s_phi * c_lambda + c_phi * s_lambda;
    const RT c_pl = c_phi * c_lambda - s_phi * s_lambda;
    const RT sq = (RT)M_SQRT1_2;
    u3[0].real = (RT)0;           u3[0].imag = (RT)0;
    u3[1].real =  sq * s_lambda;  u3[1].imag = -sq * c_lambda;
    u3[2].real = (RT)0;           u3[2].imag = (RT)0;
    u3[3].real = -sq * s_pl;      u3[3].imag =  sq * c_pl;
}

template<typename MT, typename RT>
inline MT u2_derivative_kernel_lambda_from_trig(RT s_phi, RT c_phi, RT s_lambda, RT c_lambda) {
    MT u3(2, 2);
    u2_derivative_kernel_lambda_from_trig_to<MT, RT>(u3, s_phi, c_phi, s_lambda, c_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void calc_one_qubit_u3_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;

    u3[0].real =  cos_theta;
    u3[0].imag =  0;
    u3[1].real = -sin_theta * cos_lambda;
    u3[1].imag = -sin_theta * sin_lambda;
    u3[2].real =  sin_theta * cos_phi;
    u3[2].imag =  sin_theta * sin_phi;
    u3[3].real =  cos_theta * cos_phi_lambda;
    u3[3].imag =  cos_theta * sin_phi_lambda;
}

template<typename MT, typename RT>
inline MT calc_one_qubit_u3_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    MT u3(2, 2);
    calc_one_qubit_u3_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void calc_one_qubit_u3_inverse_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;

    u3[0].real =  cos_theta;
    u3[0].imag =  (RT)0;
    u3[1].real =  sin_theta * cos_phi;
    u3[1].imag = -sin_theta * sin_phi;
    u3[2].real = -sin_theta * cos_lambda;
    u3[2].imag =  sin_theta * sin_lambda;
    u3[3].real =  cos_theta * cos_phi_lambda;
    u3[3].imag = -cos_theta * sin_phi_lambda;
}

template<typename MT, typename RT>
inline MT calc_one_qubit_u3_inverse_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    MT u3(2, 2);
    calc_one_qubit_u3_inverse_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u3_derivative_kernel_theta_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;
    u3[0].real = -sin_theta;               u3[0].imag = (RT)0;
    u3[1].real = -cos_theta * cos_lambda; u3[1].imag = -cos_theta * sin_lambda;
    u3[2].real =  cos_theta * cos_phi;    u3[2].imag =  cos_theta * sin_phi;
    u3[3].real = -sin_theta * cos_phi_lambda;
    u3[3].imag = -sin_theta * sin_phi_lambda;
}

template<typename MT, typename RT>
inline MT u3_derivative_kernel_theta_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    MT u3(2, 2);
    u3_derivative_kernel_theta_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u3_derivative_kernel_phi_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;
    u3[0].real = (RT)0;                  u3[0].imag = (RT)0;
    u3[1].real = (RT)0;                  u3[1].imag = (RT)0;
    u3[2].real = -sin_theta * sin_phi;   u3[2].imag =  sin_theta * cos_phi;
    u3[3].real = -cos_theta * sin_phi_lambda;
    u3[3].imag =  cos_theta * cos_phi_lambda;
}

template<typename MT, typename RT>
inline MT u3_derivative_kernel_phi_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    MT u3(2, 2);
    u3_derivative_kernel_phi_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void u3_derivative_kernel_lambda_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    ensure_kernel_storage(u3, 2, 2);
    const RT sin_phi_lambda = sin_phi * cos_lambda + cos_phi * sin_lambda;
    const RT cos_phi_lambda = cos_phi * cos_lambda - sin_phi * sin_lambda;
    u3[0].real = (RT)0;                 u3[0].imag = (RT)0;
    u3[1].real =  sin_theta * sin_lambda; u3[1].imag = -sin_theta * cos_lambda;
    u3[2].real = (RT)0;                 u3[2].imag = (RT)0;
    u3[3].real = -cos_theta * sin_phi_lambda;
    u3[3].imag =  cos_theta * cos_phi_lambda;
}

template<typename MT, typename RT>
inline MT u3_derivative_kernel_lambda_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda) {
    MT u3(2, 2);
    u3_derivative_kernel_lambda_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    return u3;
}

template<typename MT, typename RT>
inline void multiply_2x2_by_phase(MT& u3, RT sin_gamma, RT cos_gamma) {
    for (int k = 0; k < 4; ++k) {
        const RT real = u3[k].real;
        const RT imag = u3[k].imag;
        u3[k].real = real * cos_gamma - imag * sin_gamma;
        u3[k].imag = real * sin_gamma + imag * cos_gamma;
    }
}

template<typename MT, typename RT>
inline void cu_gate_kernel_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda, RT sin_gamma, RT cos_gamma) {
    calc_one_qubit_u3_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    multiply_2x2_by_phase<MT, RT>(u3, sin_gamma, cos_gamma);
}

template<typename MT, typename RT>
inline MT cu_gate_kernel_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda, RT sin_gamma, RT cos_gamma) {
    MT u3(2, 2);
    cu_gate_kernel_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
    return u3;
}

template<typename MT, typename RT>
inline void cu_inverse_gate_kernel_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda, RT sin_gamma, RT cos_gamma) {
    calc_one_qubit_u3_inverse_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
    multiply_2x2_by_phase<MT, RT>(u3, -sin_gamma, cos_gamma);
}

template<typename MT, typename RT>
inline MT cu_inverse_gate_kernel_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda, RT sin_gamma, RT cos_gamma) {
    MT u3(2, 2);
    cu_inverse_gate_kernel_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
    return u3;
}

template<typename MT, typename RT>
inline void cu_derivative_kernel_gamma_from_trig_to(MT& u3, RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda, RT sin_gamma, RT cos_gamma) {
    cu_gate_kernel_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
    for (int k = 0; k < 4; ++k) {
        const RT real = u3[k].real;
        const RT imag = u3[k].imag;
        u3[k].real = -imag;
        u3[k].imag = real;
    }
}

template<typename MT, typename RT>
inline MT cu_derivative_kernel_gamma_from_trig(RT sin_theta, RT cos_theta, RT sin_phi, RT cos_phi, RT sin_lambda, RT cos_lambda, RT sin_gamma, RT cos_gamma) {
    MT u3(2, 2);
    cu_derivative_kernel_gamma_from_trig_to<MT, RT>(u3, sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda, sin_gamma, cos_gamma);
    return u3;
}

// ---------------------------------------------------------------------------
// Two-qubit parametric gate kernels (4x4)
// ---------------------------------------------------------------------------

/// RXX(theta): exp(-i * theta/2 * XX)
template<typename MT, typename RT>
inline void build_rxx_kernel_from_trig_to(MT& U, RT s, RT c) {
    ensure_kernel_storage(U, 4, 4);
    for (int i = 0; i < 16; ++i) { U[i].real = RT(0); U[i].imag = RT(0); }
    U[0].real = c;    U[3].imag  = -s;
    U[5].real = c;    U[6].imag  = -s;
    U[10].real = c;   U[9].imag  = -s;
    U[15].real = c;   U[12].imag = -s;
}

template<typename MT, typename RT>
inline MT build_rxx_kernel_from_trig(RT s, RT c) {
    MT U(4, 4);
    build_rxx_kernel_from_trig_to<MT, RT>(U, s, c);
    return U;
}

template<typename MT, typename RT>
inline void build_rxx_derivative_kernel_from_trig_to(MT& U, RT s, RT c) {
    ensure_kernel_storage(U, 4, 4);
    for (int i = 0; i < 16; ++i) { U[i].real = RT(0); U[i].imag = RT(0); }
    U[0].real = -s;   U[3].imag  = -c;
    U[5].real = -s;   U[6].imag  = -c;
    U[10].real = -s;  U[9].imag  = -c;
    U[15].real = -s;  U[12].imag = -c;
}

template<typename MT, typename RT>
inline MT build_rxx_derivative_kernel_from_trig(RT s, RT c) {
    MT U(4, 4);
    build_rxx_derivative_kernel_from_trig_to<MT, RT>(U, s, c);
    return U;
}

/// RYY(theta): exp(-i * theta/2 * YY)
template<typename MT, typename RT>
inline void build_ryy_kernel_from_trig_to(MT& U, RT s, RT c) {
    ensure_kernel_storage(U, 4, 4);
    for (int i = 0; i < 16; ++i) { U[i].real = RT(0); U[i].imag = RT(0); }
    U[0].real = c;    U[3].imag  = +s;
    U[5].real = c;    U[6].imag  = -s;
    U[10].real = c;   U[9].imag  = -s;
    U[15].real = c;   U[12].imag = +s;
}

template<typename MT, typename RT>
inline MT build_ryy_kernel_from_trig(RT s, RT c) {
    MT U(4, 4);
    build_ryy_kernel_from_trig_to<MT, RT>(U, s, c);
    return U;
}

template<typename MT, typename RT>
inline void build_ryy_derivative_kernel_from_trig_to(MT& U, RT s, RT c) {
    ensure_kernel_storage(U, 4, 4);
    for (int i = 0; i < 16; ++i) { U[i].real = RT(0); U[i].imag = RT(0); }
    U[0].real = -s;   U[3].imag  = +c;
    U[5].real = -s;   U[6].imag  = -c;
    U[10].real = -s;  U[9].imag  = -c;
    U[15].real = -s;  U[12].imag = +c;
}

template<typename MT, typename RT>
inline MT build_ryy_derivative_kernel_from_trig(RT s, RT c) {
    MT U(4, 4);
    build_ryy_derivative_kernel_from_trig_to<MT, RT>(U, s, c);
    return U;
}

/// RZZ(theta): exp(-i * theta/2 * ZZ)
template<typename MT, typename RT>
inline void build_rzz_kernel_from_trig_to(MT& U, RT s, RT c) {
    ensure_kernel_storage(U, 4, 4);
    for (int i = 0; i < 16; ++i) { U[i].real = RT(0); U[i].imag = RT(0); }
    U[0].real  = c;   U[0].imag  = -s;
    U[5].real  = c;   U[5].imag  = +s;
    U[10].real = c;   U[10].imag = +s;
    U[15].real = c;   U[15].imag = -s;
}

template<typename MT, typename RT>
inline MT build_rzz_kernel_from_trig(RT s, RT c) {
    MT U(4, 4);
    build_rzz_kernel_from_trig_to<MT, RT>(U, s, c);
    return U;
}

template<typename MT, typename RT>
inline void build_rzz_derivative_kernel_from_trig_to(MT& U, RT s, RT c) {
    ensure_kernel_storage(U, 4, 4);
    for (int i = 0; i < 16; ++i) { U[i].real = RT(0); U[i].imag = RT(0); }
    U[0].real  = -s;  U[0].imag  = -c;
    U[5].real  = -s;  U[5].imag  = +c;
    U[10].real = -s;  U[10].imag = +c;
    U[15].real = -s;  U[15].imag = -c;
}

template<typename MT, typename RT>
inline MT build_rzz_derivative_kernel_from_trig(RT s, RT c) {
    MT U(4, 4);
    build_rzz_derivative_kernel_from_trig_to<MT, RT>(U, s, c);
    return U;
}

// ---------------------------------------------------------------------------
// CROT helper kernels (two 2x2 branches)
// ---------------------------------------------------------------------------

template<typename MT, typename RT>
inline void build_crot_gate_kernel_from_trig_to(MT& forward, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    // CROT uses U3(theta, phi-pi/2, -phi+pi/2) on one branch and U3(-theta, ...) on the other.
    // Convert shifted angles directly from sin(phi), cos(phi):
    // sin(phi-pi/2)=-cos(phi), cos(phi-pi/2)=sin(phi)
    // sin(-phi+pi/2)= cos(phi), cos(-phi+pi/2)=sin(phi)
    const RT s_phi_m_pi2 = -c_phi;
    const RT c_phi_m_pi2 =  s_phi;
    const RT s_nphi_p_pi2 =  c_phi;
    const RT c_nphi_p_pi2 =  s_phi;

    calc_one_qubit_u3_from_trig_to<MT, RT>(
        forward,
        s_theta, c_theta, s_phi_m_pi2, c_phi_m_pi2, s_nphi_p_pi2, c_nphi_p_pi2
    );
}

template<typename MT, typename RT>
inline void build_crot_inverse_gate_kernel_from_trig_to(MT& inverse, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    const RT s_phi_m_pi2 = -c_phi;
    const RT c_phi_m_pi2 =  s_phi;
    const RT s_nphi_p_pi2 =  c_phi;
    const RT c_nphi_p_pi2 =  s_phi;

    calc_one_qubit_u3_from_trig_to<MT, RT>(
        inverse,
        -s_theta, c_theta, s_phi_m_pi2, c_phi_m_pi2, s_nphi_p_pi2, c_nphi_p_pi2
    );
}

template<typename MT, typename RT>
inline void build_crot_gate_kernels_from_trig_to(MT& forward, MT& inverse, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    build_crot_gate_kernel_from_trig_to<MT, RT>(forward, s_theta, c_theta, s_phi, c_phi);
    build_crot_inverse_gate_kernel_from_trig_to<MT, RT>(inverse, s_theta, c_theta, s_phi, c_phi);
}

template<typename MT, typename RT>
inline std::pair<MT, MT> build_crot_gate_kernels_from_trig(RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    MT forward(2, 2);
    MT inverse(2, 2);
    build_crot_gate_kernels_from_trig_to<MT, RT>(forward, inverse, s_theta, c_theta, s_phi, c_phi);
    return std::make_pair(forward, inverse);
}

template<typename MT, typename RT>
inline void build_crot_theta_derivative_kernels_from_trig_to(MT& forward, MT& inverse, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    // d/dtheta implemented via phase shift theta -> theta + pi/2 on both branches.
    const RT s_theta_shift = c_theta;
    const RT c_theta_shift = -s_theta;
    build_crot_gate_kernels_from_trig_to<MT, RT>(forward, inverse, s_theta_shift, c_theta_shift, s_phi, c_phi);
}

template<typename MT, typename RT>
inline void build_crot_theta_derivative_kernel_from_trig_to(MT& forward, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    const RT s_theta_shift = c_theta;
    const RT c_theta_shift = -s_theta;
    build_crot_gate_kernel_from_trig_to<MT, RT>(forward, s_theta_shift, c_theta_shift, s_phi, c_phi);
}

template<typename MT, typename RT>
inline void build_crot_theta_derivative_aux_kernel_from_trig_to(MT& inverse, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    const RT s_theta_shift = c_theta;
    const RT c_theta_shift = -s_theta;
    build_crot_inverse_gate_kernel_from_trig_to<MT, RT>(inverse, s_theta_shift, c_theta_shift, s_phi, c_phi);
}

template<typename MT, typename RT>
inline std::pair<MT, MT> build_crot_theta_derivative_kernels_from_trig(RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    MT forward(2, 2);
    MT inverse(2, 2);
    build_crot_theta_derivative_kernels_from_trig_to<MT, RT>(forward, inverse, s_theta, c_theta, s_phi, c_phi);
    return std::make_pair(forward, inverse);
}

template<typename MT, typename RT>
inline void build_crot_phi_derivative_kernels_from_trig_to(MT& forward, MT& inverse, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    // Matches legacy implementation: U3(theta, phi, -phi) and U3(-theta, phi, -phi)
    // with zeroed diagonal entries.
    calc_one_qubit_u3_from_trig_to<MT, RT>(forward, s_theta, c_theta, s_phi, c_phi, -s_phi, c_phi);
    calc_one_qubit_u3_from_trig_to<MT, RT>(inverse, -s_theta, c_theta, s_phi, c_phi, -s_phi, c_phi);

    forward[0].real = (RT)0; forward[0].imag = (RT)0;
    forward[3].real = (RT)0; forward[3].imag = (RT)0;
    inverse[0].real = (RT)0; inverse[0].imag = (RT)0;
    inverse[3].real = (RT)0; inverse[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline void build_crot_phi_derivative_kernel_from_trig_to(MT& forward, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    calc_one_qubit_u3_from_trig_to<MT, RT>(forward, s_theta, c_theta, s_phi, c_phi, -s_phi, c_phi);
    forward[0].real = (RT)0; forward[0].imag = (RT)0;
    forward[3].real = (RT)0; forward[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline void build_crot_phi_derivative_aux_kernel_from_trig_to(MT& inverse, RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    calc_one_qubit_u3_from_trig_to<MT, RT>(inverse, -s_theta, c_theta, s_phi, c_phi, -s_phi, c_phi);
    inverse[0].real = (RT)0; inverse[0].imag = (RT)0;
    inverse[3].real = (RT)0; inverse[3].imag = (RT)0;
}

template<typename MT, typename RT>
inline std::pair<MT, MT> build_crot_phi_derivative_kernels_from_trig(RT s_theta, RT c_theta, RT s_phi, RT c_phi) {
    MT forward(2, 2);
    MT inverse(2, 2);
    build_crot_phi_derivative_kernels_from_trig_to<MT, RT>(forward, inverse, s_theta, c_theta, s_phi, c_phi);
    return std::make_pair(forward, inverse);
}
