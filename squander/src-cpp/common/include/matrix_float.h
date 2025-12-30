#ifndef ENABLE_FLOAT32_MATRIX
  #error "matrix_float.h requires ENABLE_FLOAT32_MATRIX. Compile with -DENABLE_FLOAT32_MATRIX"
#endif

#ifndef MATRIX_FLOAT_H
#define MATRIX_FLOAT_H

#include "matrix_base.hpp"
#include <cmath>
#include <type_traits>

class Matrix;  // Forward declaration

/**
@brief Single-precision complex matrix (float32)
@note BLAS operations NOT supported. Convert to Matrix for multiplication
@note Conversions only between complex types (Matrix <-> MatrixFloat)
@note Python bindings NOT yet implemented
*/
// WARNING: matrix_base has no virtual destructor. Never delete through base pointer.
class MatrixFloat : public matrix_base<QGD_Complex8> {
  char padding[CACHELINE-48];

public:
    MatrixFloat();
    MatrixFloat(QGD_Complex8* data_in, int rows_in, int cols_in);
    MatrixFloat(QGD_Complex8* data_in, int rows_in, int cols_in, int stride_in);
    MatrixFloat(int rows_in, int cols_in);
    MatrixFloat(int rows_in, int cols_in, int stride_in);
    MatrixFloat(const MatrixFloat &in);
    MatrixFloat& operator=(const MatrixFloat &mtx);

    MatrixFloat copy();
    bool isnan();
    void print_matrix() const;

    /**
    @brief Convert to double precision
    @return Matrix with converted data
    @throws std::bad_alloc if memory allocation fails
    */
    Matrix to_float64() const;
};

// Compile-time type checks
static_assert(sizeof(QGD_Complex8) == 8, "QGD_Complex8 must be 8 bytes");
static_assert(std::is_standard_layout<QGD_Complex8>::value &&
              std::is_trivial<QGD_Complex8>::value,
              "QGD_Complex8 must be standard layout and trivial");

#endif // MATRIX_FLOAT_H