#include "include/matrix_template.hpp"
#include "include/QGDTypes.h"
#include <cstring>
#include <iostream>
#include <cmath>

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T() : matrix_base<ComplexType>() {}

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T(ComplexType* data_in, int rows_in, int cols_in)
    : matrix_base<ComplexType>(data_in, rows_in, cols_in) {}

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T(ComplexType* data_in, int rows_in, int cols_in, int stride_in)
    : matrix_base<ComplexType>(data_in, rows_in, cols_in, stride_in) {}

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T(int rows_in, int cols_in)
    : matrix_base<ComplexType>(rows_in, cols_in) {}

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T(int rows_in, int cols_in, int stride_in)
    : matrix_base<ComplexType>(rows_in, cols_in, stride_in) {}

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T(const Matrix_T &other)
    : matrix_base<ComplexType>(other) {}

template<typename ComplexType>
Matrix_T<ComplexType>::Matrix_T(Matrix_T &&other) noexcept
    : matrix_base<ComplexType>(std::move(other)) {}

template<typename ComplexType>
Matrix_T<ComplexType>& Matrix_T<ComplexType>::operator=(const Matrix_T &other) {
    if (this != &other) {
        matrix_base<ComplexType>::operator=(other);
    }
    return *this;
}

template<typename ComplexType>
Matrix_T<ComplexType>& Matrix_T<ComplexType>::operator=(Matrix_T &&other) noexcept {
    if (this != &other) {
        matrix_base<ComplexType>::operator=(std::move(other));
    }
    return *this;
}

template<typename ComplexType>
Matrix_T<ComplexType> Matrix_T<ComplexType>::copy() const {
    Matrix_T<ComplexType> result(this->rows, this->cols, this->stride);
    result.conjugated = this->conjugated;
    result.transposed = this->transposed;
    std::memcpy(result.data, this->data, this->rows * this->stride * sizeof(ComplexType));
    return result;
}

template<typename ComplexType>
bool Matrix_T<ComplexType>::isnan() {
    for (int idx = 0; idx < this->rows * this->cols; idx++) {
        if (std::isnan(this->data[idx].real) || std::isnan(this->data[idx].imag)) {
            return true;
        }
    }
    return false;
}

template<typename ComplexType>
void Matrix_T<ComplexType>::print_matrix() const {
    std::cout << "\nThe stored matrix:\n";
    for (int row_idx = 0; row_idx < this->rows; row_idx++) {
        for (int col_idx = 0; col_idx < this->cols; col_idx++) {
            int element_idx = row_idx * this->stride + col_idx;
            std::cout << " (" << this->data[element_idx].real
                      << ", " << this->data[element_idx].imag << "*i)";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Explicit instantiation
template class Matrix_T<QGD_Complex16>;
#ifdef ENABLE_FLOAT32
template class Matrix_T<QGD_Complex8>;
#endif

// Verify size after instantiation
static_assert(sizeof(Matrix_T<QGD_Complex16>) == CACHELINE,
              "Matrix_T<QGD_Complex16> size must equal exactly one CACHELINE");
#ifdef ENABLE_FLOAT32
static_assert(sizeof(Matrix_T<QGD_Complex8>) == CACHELINE,
              "Matrix_T<QGD_Complex8> size must equal exactly one CACHELINE");
#endif
