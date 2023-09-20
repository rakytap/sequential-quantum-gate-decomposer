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
/*! \file apply_kerel_to_input_AVX.cpp
    \brief ????????????????
*/


#ifndef apply_kerel_to_input_AVX_H
#define apply_kerel_to_input_AVX_H

#include "matrix.h"
#include "common.h"

/**
@brief AVX kernel to apply single qubit gate kernel on an input matrix -- optimal for small number of qubits
@param ????????
@param ?????????
*/
void apply_kernel_to_input_AVX_small(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);



/**
@brief AVX kernel to apply single qubit gate kernel on an input matrix
@param ????????
@param ?????????
*/
void apply_kernel_to_input_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);


/**
@brief parallel AVX kernel to apply single qubit gate kernel on an input matrix
@param ????????
@param ?????????
*/
void apply_kernel_to_input_AVX_parallel(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);


#endif
