/*
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef MATRIX_REAL_ANY_H
#define MATRIX_REAL_ANY_H

#include "matrix_real.h"
#include "matrix_real_float.h"
#include <string>

enum class matrix_real_precision {
    float64 = 0,
    float32 = 1
};

/**
@brief Non-owning carrier that can reference either Matrix_real or Matrix_real_float.
*/
class Matrix_real_any {
public:
    Matrix_real_any(Matrix_real& m)
        : precision_(matrix_real_precision::float64), m64_(&m), m32_(NULL) {}

    Matrix_real_any(Matrix_real_float& m)
        : precision_(matrix_real_precision::float32), m64_(NULL), m32_(&m) {}

    matrix_real_precision precision() const { return precision_; }

    bool is_float32() const { return precision_ == matrix_real_precision::float32; }
    bool is_float64() const { return precision_ == matrix_real_precision::float64; }

    int rows() const { return is_float64() ? m64_->rows : m32_->rows; }
    int cols() const { return is_float64() ? m64_->cols : m32_->cols; }
    int stride() const { return is_float64() ? m64_->stride : m32_->stride; }

    Matrix_real& as_float64() {
        if (!m64_) {
            throw std::string("Matrix_real_any::as_float64 called on float32 matrix");
        }
        return *m64_;
    }

    Matrix_real_float& as_float32() {
        if (!m32_) {
            throw std::string("Matrix_real_any::as_float32 called on float64 matrix");
        }
        return *m32_;
    }

private:
    matrix_real_precision precision_;
    Matrix_real* m64_;
    Matrix_real_float* m32_;
};

#endif
