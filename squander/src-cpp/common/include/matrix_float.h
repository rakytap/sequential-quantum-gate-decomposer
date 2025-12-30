/*
Created on Fri Jun 26 14:13:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Peter Rakyta, Ph.D.
*/
/*! \file matrix_float.h
    \brief Header file of single-precision complex array storage with automatic and thread safe reference counting.
*/

#ifndef ENABLE_FLOAT32_MATRIX
  #error "matrix_float.h requires ENABLE_FLOAT32_MATRIX. Compile with -DENABLE_FLOAT32_MATRIX"
#endif

#ifndef MATRIX_FLOAT_H
#define MATRIX_FLOAT_H

#include "matrix_base.hpp"
#include "matrix_template.hpp"
#include <cmath>
#include <type_traits>

extern template class Matrix_T<QGD_Complex8>;

class Matrix;

/**
@brief Single-precision complex matrix (float32). Class to store data of single-precision complex arrays and its properties.
@note BLAS operations NOT supported. Convert to Matrix for multiplication
@note Conversions only between complex types (Matrix <-> Matrix_float)
@note Python bindings NOT yet implemented
*/
class Matrix_float : public Matrix_T<QGD_Complex8> {
public:
    using Matrix_T<QGD_Complex8>::Matrix_T;

    explicit Matrix_float(Matrix_T<QGD_Complex8>&& base) noexcept
        : Matrix_T<QGD_Complex8>(std::move(base)) {}

    Matrix_float(const Matrix_float&) = default;
    Matrix_float(Matrix_float&&) noexcept = default;
    Matrix_float& operator=(const Matrix_float&) = default;
    Matrix_float& operator=(Matrix_float&&) noexcept = default;
    ~Matrix_float() = default;

    /**
    @brief Call to create a copy of the matrix
    @return Returns with the instance of the class.
    */
    Matrix_float copy() const {
        return Matrix_float(Matrix_T<QGD_Complex8>::copy());
    }

    /**
    @brief Convert to double precision
    @return Matrix with converted data
    @throws std::bad_alloc if memory allocation fails
    @note Values outside [-FLT_MAX, FLT_MAX] saturate to infinity per IEEE 754
    */
    Matrix to_float64() const;
};

// ABI compatibility check
static_assert(sizeof(Matrix_float) == sizeof(Matrix_T<QGD_Complex8>),
              "ABI: Matrix_float size mismatch");

// Compile-time type checks
static_assert(sizeof(QGD_Complex8) == 8, "QGD_Complex8 must be 8 bytes");
static_assert(std::is_standard_layout<QGD_Complex8>::value &&
              std::is_trivial<QGD_Complex8>::value,
              "QGD_Complex8 must be standard layout and trivial");

#endif // MATRIX_FLOAT_H