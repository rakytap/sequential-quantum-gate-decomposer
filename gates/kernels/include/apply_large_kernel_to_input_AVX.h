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
/*! \file apply_kerel_to_input_AVX.h
    \brief ????????????????
*/


#ifndef apply_large_kerel_to_input_AVX_H
#define apply_large_kerel_to_input_AVX_H

#include <immintrin.h>
#include "matrix.h"
#include "common.h"

#include "apply_large_kernel_to_input.h"

__m256d get_AVX_vector(double* element_outer, double* element_inner);

__m256d complex_mult_AVX(__m256d input_vec, __m256d unitary_row_vec, __m256d neg);

void apply_large_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_2qbit_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size);

void apply_2qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_3qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_4qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

#endif
