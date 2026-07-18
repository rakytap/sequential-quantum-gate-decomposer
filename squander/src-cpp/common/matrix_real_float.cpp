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

#include "matrix_real_float.h"
#include <cstring>
#include <cmath>


Matrix_real_float::Matrix_real_float() : matrix_base<float>() {
}

Matrix_real_float::Matrix_real_float(float* data_in, int rows_in, int cols_in)
    : matrix_base<float>(data_in, rows_in, cols_in) {
}

Matrix_real_float::Matrix_real_float(float* data_in, int rows_in, int cols_in, int stride_in)
    : matrix_base<float>(data_in, rows_in, cols_in, stride_in) {
}

Matrix_real_float::Matrix_real_float(int rows_in, int cols_in)
    : matrix_base<float>(rows_in, cols_in) {
}

Matrix_real_float::Matrix_real_float(int rows_in, int cols_in, int stride_in)
    : matrix_base<float>(rows_in, cols_in, stride_in) {
}

Matrix_real_float::Matrix_real_float(const Matrix_real_float &in)
    : matrix_base<float>(in) {
}

Matrix_real_float&
Matrix_real_float::operator=(const Matrix_real_float &mtx) {
    matrix_base<float>::operator=(mtx);
    return *this;
}

Matrix_real_float
Matrix_real_float::copy() const {

  Matrix_real_float ret = Matrix_real_float(rows, cols, stride);

  ret.conjugated = conjugated;
  ret.transposed = transposed;
  ret.owner = true;

  memcpy(ret.data, data, rows * stride * sizeof(float));

  return ret;

}

bool
Matrix_real_float::isnan() {

    for (int idx=0; idx < rows*cols; idx++) {
        if (std::isnan(data[idx])) {
            return true;
        }
    }

    return false;

}
