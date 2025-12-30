#ifndef ENABLE_FLOAT32_MATRIX
  #error "matrix_float.cpp compiled without ENABLE_FLOAT32_MATRIX defined"
#endif

#include "matrix_float.h"
#include "matrix.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"
#include <math.h>

MatrixFloat::MatrixFloat() : matrix_base<QGD_Complex8>() {
}

MatrixFloat::MatrixFloat(QGD_Complex8* data_in, int rows_in, int cols_in)
    : matrix_base<QGD_Complex8>(data_in, rows_in, cols_in) {
}

MatrixFloat::MatrixFloat(QGD_Complex8* data_in, int rows_in, int cols_in, int stride_in)
    : matrix_base<QGD_Complex8>(data_in, rows_in, cols_in, stride_in) {
}

MatrixFloat::MatrixFloat(int rows_in, int cols_in)
    : matrix_base<QGD_Complex8>(rows_in, cols_in) {
}

MatrixFloat::MatrixFloat(int rows_in, int cols_in, int stride_in)
    : matrix_base<QGD_Complex8>(rows_in, cols_in, stride_in) {
}

MatrixFloat::MatrixFloat(const MatrixFloat &in)
    : matrix_base<QGD_Complex8>(in) {
}

MatrixFloat& MatrixFloat::operator=(const MatrixFloat &mtx) {
    matrix_base<QGD_Complex8>::operator=(mtx);
    return *this;
}

MatrixFloat MatrixFloat::copy() {
    MatrixFloat ret = MatrixFloat(rows, cols, stride);
    ret.conjugated = conjugated;
    ret.transposed = transposed;
    ret.owner = true;
    memcpy(ret.data, data, rows*stride*sizeof(QGD_Complex8));
    return ret;
}

bool MatrixFloat::isnan() {
    for (int idx=0; idx < rows*cols; idx++) {
        if (std::isnan(data[idx].real) || std::isnan(data[idx].imag)) {
            return true;
        }
    }
    return false;
}

void MatrixFloat::print_matrix() const {
    std::cout << std::endl << "The stored matrix:" << std::endl;
    for (int row_idx=0; row_idx < rows; row_idx++) {
        for (int col_idx=0; col_idx < cols; col_idx++) {
            int element_idx = row_idx*stride + col_idx;
            std::cout << " (" << data[element_idx].real << ", "
                      << data[element_idx].imag << "*i)";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl << std::endl;
}

Matrix MatrixFloat::to_float64() const {
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

// Explicit template instantiation
template class matrix_base<QGD_Complex8>;
