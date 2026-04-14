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

#ifndef MATRIX_REAL_FLOAT_H
#define MATRIX_REAL_FLOAT_H

#include "matrix_base.hpp"
#include <cmath>


/**
@brief Class to store single-precision real arrays and properties.
*/
class Matrix_real_float : public matrix_base<float> {

    /// padding class object to cache line borders
    char padding[CACHELINE-48];

public:

Matrix_real_float();
Matrix_real_float(float* data_in, int rows_in, int cols_in);
Matrix_real_float(float* data_in, int rows_in, int cols_in, int stride_in);
Matrix_real_float(int rows_in, int cols_in);
Matrix_real_float(int rows_in, int cols_in, int stride_in);
Matrix_real_float(const Matrix_real_float &in);
Matrix_real_float& operator=(const Matrix_real_float &mtx);
Matrix_real_float copy() const;
bool isnan();

};


#endif
