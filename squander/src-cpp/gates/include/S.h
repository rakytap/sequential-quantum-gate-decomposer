#ifndef S_H
#define S_H

#include "U3.h"
#include "matrix.h"
#include "matrix_real.h"
#include "matrix_real_float.h"
#include "matrix_float.h"
#include <math.h>

class S : public U3 {
public:
    using Gate::apply_to;
    using Gate::apply_from_right;
    using Gate::get_matrix;

    S();
    S(int qbit_num_in, int target_qbit_in);
    ~S() override;
    virtual S* clone() override;
    virtual Matrix       gate_kernel(const Matrix_real&       parameters) override;
    virtual Matrix_float gate_kernel(const Matrix_real_float& parameters) override;
    virtual Matrix       inverse_gate_kernel(const Matrix_real&       parameters) override;
    virtual Matrix_float inverse_gate_kernel(const Matrix_real_float& parameters) override;
};

#endif //S_H
