/*
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef MATRIX_ANY_H
#define MATRIX_ANY_H

#include "matrix.h"
#include "matrix_float.h"
#include <string>

enum class matrix_precision {
    float64 = 0,
    float32 = 1
};

/**
@brief Non-owning carrier that references either Matrix (complex128) or
Matrix_float (complex64) without data conversion.

@note Intended for precision dispatch at API boundaries only.
Hot kernels should still consume concrete Matrix or Matrix_float types.
*/
class Matrix_any {
public:
    Matrix_any(Matrix& m)
        : precision_(matrix_precision::float64), m64_(&m), m32_(NULL) {}

    Matrix_any(Matrix_float& m)
        : precision_(matrix_precision::float32), m64_(NULL), m32_(&m) {}

    matrix_precision precision() const { return precision_; }

    bool is_float64() const { return precision_ == matrix_precision::float64; }
    bool is_float32() const { return precision_ == matrix_precision::float32; }

    int rows() const { return is_float64() ? m64_->rows : m32_->rows; }
    int cols() const { return is_float64() ? m64_->cols : m32_->cols; }
    int stride() const { return is_float64() ? m64_->stride : m32_->stride; }
    int size() const { return is_float64() ? m64_->size() : m32_->size(); }

    Matrix& as_float64() {
        if (!m64_) {
            throw std::string("Matrix_any::as_float64 called on float32 matrix");
        }
        return *m64_;
    }

    Matrix_float& as_float32() {
        if (!m32_) {
            throw std::string("Matrix_any::as_float32 called on float64 matrix");
        }
        return *m32_;
    }

private:
    matrix_precision precision_;
    Matrix* m64_;
    Matrix_float* m32_;
};

#endif
