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

//AVX auxillary funcs

__m256d get_AVX_vector(double* element_outer, double* element_inner);

__m256d complex_mult_AVX(__m256d input_vec, __m256d unitary_row_vec, __m256d neg);

//main function to call
void apply_large_kernel_to_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_large_kernel_to_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_large_kernel_to_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// general N qubit kernel functions
inline void get_block_indices(int N, const std::vector<int> &tq, const std::vector<int> &non_targets,int iter_idx, std::vector<int> &indices);

inline void get_block_indices_fast(int iter_idx, const std::vector<int>& target_qubits, const std::vector<int>& non_targets, const std::vector<int>& block_pattern, std::vector<int>& indices);

void precompute_index_mapping(const std::vector<int>& target_qubits, const std::vector<int>& non_targets, std::vector<int>& block_pattern);

inline void write_out_block(Matrix& input, const std::vector<double>& new_block_real, const std::vector<double>& new_block_imag, const std::vector<int>& indices);

inline void complex_prod_AVX(const __m256d* mv_xy, int rdx, int cdx,  const std::vector<int>& indices, const Matrix& input, __m256d& result);

inline __m256d* construct_mv_xy_vectors(const Matrix& gate_kernel_unitary, const int& matrix_size);

void apply_nqbit_unitary_AVX( Matrix& gate_kernel_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size );

void apply_nqbit_unitary_parallel_AVX( Matrix& gate_kernel_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size );

// 2 qubit kernel functions
void apply_2qbit_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_2qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits,  const int& matrix_size);

void apply_2qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// 3 qubit kernel functions
void apply_3qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_3qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_3qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// 4 qubit kernel functions
void apply_4qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_4qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_4qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// 5 qubit kernel functions
void apply_5qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_5qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

void apply_5qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

#endif
