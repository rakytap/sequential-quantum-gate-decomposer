/**
 * Copyright (C) Miklos Maroti, 2021
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef U3_H
#define U3_H

#include "Gate.h"
#include "matrix.h"
#include "matrix_real.h"
#include "matrix_real_float.h"
#include "matrix_float.h"
#include <math.h>

/**
@brief Class representing a general 3-parameter single-qubit gate U3(θ,φ,λ).
All constant and rotation single-qubit gates inherit from U3.
*/
class U3: public Gate {
public:
    U3();
    U3(int qbit_num_in, int target_qbit_in);
    ~U3() override;

    virtual U3* clone() override;
    virtual std::vector<double> get_parameter_multipliers() const override;
    virtual Matrix gate_kernel(const Matrix_real& precomputed_sincos) override;
    virtual Matrix_float gate_kernel(const Matrix_real_float& precomputed_sincos) override;
    virtual Matrix inverse_gate_kernel(const Matrix_real& precomputed_sincos) override;
    virtual Matrix_float inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) override;
    virtual Matrix derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) override;
    virtual Matrix_float derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) override;

};

// Free helper functions — compute a 2×2 U3 matrix without constructing a gate object.
Matrix       u3_matrix_kernel(double ThetaOver2, double Phi, double Lambda);
Matrix_float u3_matrix_kernel_f(float ThetaOver2, float Phi, float Lambda);

// Free helpers — compute a 2x2 U3 kernel without constructing a gate object.
Matrix       u3_matrix_kernel(double ThetaOver2, double Phi, double Lambda);
Matrix_float u3_matrix_kernel_f(float ThetaOver2, float Phi, float Lambda);

#endif //U3_H
