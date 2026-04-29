#ifndef RZ_H
#define RZ_H

#include "U3.h"
#include "matrix.h"
#include "matrix_real.h"
#include "matrix_real_float.h"
#include "matrix_float.h"
#include <math.h>

class RZ : public U3 {
public:
    RZ();
    RZ(int qbit_num_in, int target_qbit_in);
    ~RZ() override;
    virtual RZ* clone() override;
    virtual std::vector<double> get_parameter_multipliers() const override;
    virtual Matrix       gate_kernel(const Matrix_real&       precomputed_sincos) override;
    virtual Matrix_float gate_kernel(const Matrix_real_float& precomputed_sincos) override;
    virtual Matrix       inverse_gate_kernel(const Matrix_real&       precomputed_sincos) override;
    virtual Matrix_float inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) override;
    virtual Matrix       derivative_kernel(const Matrix_real&       precomputed_sincos, int param_idx) override;
    virtual Matrix_float derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) override;
};

#endif //RZ_H
