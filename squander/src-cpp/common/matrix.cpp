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
/*! \file matrix.cpp
    \brief Implementation of complex array storage array with automatic and thread safe reference counting.
*/

#include "matrix.h"

#ifdef ENABLE_FLOAT32_MATRIX
#include "matrix_float.h"

/**
@brief Convert to single precision
@return Matrix_float with converted data
*/
Matrix_float Matrix::to_float32() const {
    Matrix_float ret(rows, cols, stride);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = row * stride + col;
            ret.data[idx].real = static_cast<float>(data[idx].real);
            ret.data[idx].imag = static_cast<float>(data[idx].imag);
        }
    }
    if (is_conjugated()) ret.conjugate();
    if (is_transposed()) ret.transpose();
    return ret;
}
#endif

