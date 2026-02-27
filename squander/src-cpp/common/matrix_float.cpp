#ifndef ENABLE_FLOAT32
  #error "matrix_float.cpp compiled without ENABLE_FLOAT32 defined"
#endif

#include "matrix_float.h"
#include "matrix.h"

/**
@brief Convert to double precision
@return Matrix with converted data
*/
Matrix Matrix_float::to_float64() const {
    Matrix ret(rows, cols, stride);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = row * stride + col;
            ret.data[idx].real = static_cast<double>(data[idx].real);
            ret.data[idx].imag = static_cast<double>(data[idx].imag);
        }
    }
    if (is_conjugated()) ret.conjugate();
    if (is_transposed()) ret.transpose();
    return ret;
}
