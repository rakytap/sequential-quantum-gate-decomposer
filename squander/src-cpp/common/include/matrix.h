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
/*! \file matrix.h
    \brief Header file of complex array storage array with automatic and thread safe reference counting.
*/


#ifndef matrix_H
#define matrix_H

#include "matrix_base.hpp"
#include "matrix_template.hpp"
#include <type_traits>

extern template class Matrix_T<QGD_Complex16>;

#ifdef ENABLE_FLOAT32_MATRIX
class Matrix_float;
#endif

/**
@brief Double-precision complex matrix (float64). Class to store data of complex arrays and its properties. Compatible with the Picasso numpy interface.
*/
class Matrix : public Matrix_T<QGD_Complex16> {
public:
    using Matrix_T<QGD_Complex16>::Matrix_T;

    Matrix() = default;

    explicit Matrix(Matrix_T<QGD_Complex16>&& base) noexcept
        : Matrix_T<QGD_Complex16>(std::move(base)) {}

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix() = default;

    /**
    @brief Call to create a copy of the matrix
    @return Returns with the instance of the class.
    */
    Matrix copy() const {
        return Matrix(Matrix_T<QGD_Complex16>::copy());
    }

    #ifdef ENABLE_FLOAT32_MATRIX
    /**
    @brief Convert to single precision
    @return Matrix_float with converted data
    @throws std::bad_alloc if memory allocation fails
    @note Values outside [-FLT_MAX, FLT_MAX] saturate to infinity per IEEE 754
    */
    Matrix_float to_float32() const;
    #endif
};

// ABI compatibility check
static_assert(sizeof(Matrix) == sizeof(Matrix_T<QGD_Complex16>),
              "ABI: Matrix size mismatch");

// Compile-time type checks
static_assert(sizeof(QGD_Complex16) == 16, "QGD_Complex16 must be 16 bytes");
static_assert(std::is_standard_layout<QGD_Complex16>::value &&
              std::is_trivial<QGD_Complex16>::value,
              "QGD_Complex16 must be standard layout and trivial");

#endif // MATRIX_H
