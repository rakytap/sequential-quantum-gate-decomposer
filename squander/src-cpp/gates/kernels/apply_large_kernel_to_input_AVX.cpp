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
/*! \file apply_kerel_to_state_vector_input.cpp
    \brief ????????????????
*/


#include "apply_large_kernel_to_input_AVX.h"
#include "tbb/tbb.h"
#include "omp.h"

inline __m256d get_AVX_vector(double* element_outer, double* element_inner){

    __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
    element_outer_vec = _mm256_permute4x64_pd(element_outer_vec,0b11011000);
    __m256d element_inner_vec = _mm256_loadu_pd(element_inner);
    element_inner_vec = _mm256_permute4x64_pd(element_inner_vec,0b11011000);
    __m256d outer_inner_vec = _mm256_shuffle_pd(element_outer_vec,element_inner_vec,0b0000);
    outer_inner_vec = _mm256_permute4x64_pd(outer_inner_vec,0b11011000);
    
    return outer_inner_vec;
}

inline __m256d complex_mult_AVX(__m256d input_vec, __m256d unitary_row_vec, __m256d neg){

    __m256d vec3 = _mm256_mul_pd(input_vec, unitary_row_vec);
    __m256d unitary_row_switched = _mm256_permute_pd(unitary_row_vec, 0x5);
    unitary_row_switched = _mm256_mul_pd(unitary_row_switched, neg);
    __m256d vec4 = _mm256_mul_pd(input_vec, unitary_row_switched);
    __m256d result_vec = _mm256_hsub_pd(vec3, vec4);
    result_vec = _mm256_permute4x64_pd(result_vec,0b11011000);
    
    return result_vec;
}

/**
@brief Call to apply kernel to apply multi qubit gate kernel on an input matrix
@param unitary The 2^Nx2^N kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The involved qubits in the operation
@param matrix_size The size of the input
*/
void apply_large_kernel_to_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    if (input.cols==1){
       switch(involved_qbits.size()){
      case 2:{
              apply_2qbit_kernel_to_state_vector_input_AVX(unitary, input, involved_qbits, matrix_size);
              break;
      }
      case 3:{
          apply_3qbit_kernel_to_state_vector_input_AVX(unitary,input,involved_qbits,matrix_size);
          break;
      }
      case 4:{
              apply_4qbit_kernel_to_state_vector_input_AVX(unitary,input,involved_qbits,matrix_size);
              break;
      }
      }
  }
  else{
      apply_2qbit_kernel_to_matrix_input_AVX(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size);
  }
}

void precompute_index_mapping(const std::vector<int>& target_qubits,
                              const std::vector<int>& non_targets,
                              std::vector<int>& block_pattern) {
    int block_size = 1 << target_qubits.size();
    
    for (int k = 0; k < block_size; ++k) {
        int idx = 0;
        for (int bit = 0; bit < target_qubits.size(); ++bit) {
            if (k & (1 << bit)) {
                idx |= (1 << target_qubits[bit]);
            }
        }
        block_pattern[k] = idx;
    }
}

inline void get_block_indices_fast(int iter_idx,
                                   const std::vector<int>& target_qubits,
                                   const std::vector<int>& non_targets,
                                   const std::vector<int>& block_pattern,
                                   std::vector<int>& indices) {
    int base = 0;
    for (int i = 0; i < non_targets.size(); ++i) {
        if (iter_idx & (1 << i)) {
            base |= (1 << non_targets[i]);
        }
    }
    for (int k = 0; k < block_pattern.size(); ++k) {
        indices[k] = base | block_pattern[k];  
    }
}

inline void write_out_block(Matrix& input, const std::vector<double>& new_block_real,const std::vector<double>& new_block_imag, const std::vector<int>& indices){

    const double* real_ptr = new_block_real.data();
    const double* imag_ptr = new_block_imag.data();
    const int* idx_ptr = indices.data();

    const int block_size = (int)new_block_real.size();
    for (int k = 0; k < block_size; ++k) {
        auto& elem = input[idx_ptr[k]];
        elem.real = real_ptr[k];
        elem.imag = imag_ptr[k];
    }
    return;

}

inline void complex_prod_AVX(const __m256d* mv_xy, int rdx, int cdx,  const std::vector<int>& indices, const Matrix& input, __m256d& result){
    int block_size = (int)indices.size();
    int current_idx = indices[cdx];
    int current_idx_pair = indices[cdx+1];

    const double* data_ptr = (const double*)input.get_data();

    __m256d data = _mm256_set_pd(
        data_ptr[2*current_idx_pair + 1],
        data_ptr[2*current_idx_pair + 0],
        data_ptr[2*current_idx + 1],
        data_ptr[2*current_idx + 0]
    );

    __m256d mv_x0 = mv_xy[block_size*rdx + cdx];
    __m256d mv_x1 = mv_xy[block_size*rdx + cdx + 1];

    // Use fused multiply-add style manually
    __m256d data_u0 = _mm256_mul_pd(data, mv_x0);
    __m256d data_u1 = _mm256_mul_pd(data, mv_x1);
    __m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);

    result = _mm256_add_pd(result, data_u2);
        
        return;
}

inline __m256d* construct_mv_xy_vectors(const Matrix& gate_kernel_unitary, const int& matrix_size)
{
    // Allocate aligned memory for AVX (32-byte alignment)
    __m256d* mv_xy = (__m256d*) _mm_malloc(sizeof(__m256d) * matrix_size * matrix_size, 32);

    for (int rdx = 0; rdx < matrix_size; rdx++) {
        for (int cdx = 0; cdx < matrix_size; cdx += 2) {

            // Precompute both vectors in a single loop
            mv_xy[rdx * matrix_size + cdx] = _mm256_set_pd(
                -gate_kernel_unitary[matrix_size*rdx+cdx+1].imag,
                gate_kernel_unitary[matrix_size*rdx+cdx+1].real,
                -gate_kernel_unitary[matrix_size*rdx+cdx].imag,
                gate_kernel_unitary[matrix_size*rdx+cdx].real
            );

            mv_xy[rdx * matrix_size + cdx + 1] = _mm256_set_pd(
                gate_kernel_unitary[matrix_size*rdx+cdx+1].real,
                gate_kernel_unitary[matrix_size*rdx+cdx+1].imag,
                gate_kernel_unitary[matrix_size*rdx+cdx].real,
                gate_kernel_unitary[matrix_size*rdx+cdx].imag
            );
        }
    }

    return mv_xy;
}

void apply_nqbit_unitary_AVX( Matrix& gate_kernel_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size ) {

    int n = involved_qbits.size();
    int qubit_num = (int) std::log2(input.rows);
    int block_size = 1 << n;
    int num_blocks = 1 << (qubit_num - n);
    std::sort(involved_qbits.begin(), involved_qbits.end());

    std::vector<int> is_target(qubit_num, 0);
    for (int q : involved_qbits) is_target[q] = 1;
    std::vector<int> non_targets;
    non_targets.reserve(qubit_num - n);
    for (int q = 0; q < qubit_num; ++q)
    if (!is_target[q]) non_targets.push_back(q);
    std::vector<int> block_pattern(block_size);
    precompute_index_mapping(involved_qbits, non_targets, block_pattern);
    __m256d* mv_xy = construct_mv_xy_vectors(gate_kernel_unitary, gate_kernel_unitary.rows);
    std::vector<double> new_block_real(block_size,0.0);
    std::vector<double> new_block_imag(block_size,0.0);
    std::vector<int> indices(block_size);

    for (int iter_idx = 0; iter_idx < num_blocks; iter_idx++) {
        get_block_indices_fast(iter_idx, involved_qbits, non_targets, block_pattern, indices);
        std::fill(new_block_real.begin(), new_block_real.end(), 0.0);
        std::fill(new_block_imag.begin(), new_block_imag.end(), 0.0);
            
    for (int rdx = 0; rdx < block_size; rdx++) {
        __m256d result = _mm256_setzero_pd();

        for (int cdx = 0; cdx < block_size; cdx += 2) {
            complex_prod_AVX(mv_xy, rdx, cdx, indices, input, result);
        }

        // Step 1: swap 128-bit lanes: [a0,a1,a2,a3] -> [a2,a3,a0,a1]
        __m256d perm = _mm256_permute2f128_pd(result, result, 0x01);

        // Step 2: add to original to compute pairwise sums
        __m256d sum = _mm256_add_pd(result, perm);
        // sum = [a0+a2, a1+a3, a2+a0, a3+a1]

        // Step 3: extract the low 128 bits containing [a0+a2, a1+a3]
        __m128d low128 = _mm256_castpd256_pd128(sum);

        // Step 4: extract scalars correctly
        double real = _mm_cvtsd_f64(low128);                        // element 0 = a0+a2
        double imag = _mm_cvtsd_f64(_mm_unpackhi_pd(low128, low128)); // element 1 = a1+a3

        new_block_real[rdx] = real;
        new_block_imag[rdx] = imag;
    }

        
        write_out_block(input, new_block_real, new_block_imag, indices);

    }
  _mm_free(mv_xy);
}

void apply_nqbit_unitary_parallel_AVX( Matrix& gate_kernel_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size ) {

    int n = involved_qbits.size();
    int qubit_num = (int) std::log2(input.rows);
    int block_size = 1 << n;
    int num_blocks = 1 << (qubit_num - n);
    std::sort(involved_qbits.begin(), involved_qbits.end());

    std::vector<int> is_target(qubit_num, 0);
    for (int q : involved_qbits) is_target[q] = 1;
    std::vector<int> non_targets;
    non_targets.reserve(qubit_num - n);
    for (int q = 0; q < qubit_num; ++q)
    if (!is_target[q]) non_targets.push_back(q);
    std::vector<int> block_pattern(block_size);
    precompute_index_mapping(involved_qbits, non_targets, block_pattern);
    __m256d* mv_xy = construct_mv_xy_vectors(gate_kernel_unitary, gate_kernel_unitary.rows);
    #pragma omp parallel
    {
        std::vector<int> indices(block_size);
        std::vector<double> new_block_real(block_size,0.0);
        std::vector<double> new_block_imag(block_size,0.0);
        #pragma omp for schedule(static)
        for (int iter_idx = 0; iter_idx < num_blocks; iter_idx++) {
            get_block_indices_fast(iter_idx, involved_qbits, non_targets, block_pattern, indices);
            std::fill(new_block_real.begin(), new_block_real.end(), 0.0);
            std::fill(new_block_imag.begin(), new_block_imag.end(), 0.0);
        for (int rdx = 0; rdx < block_size; rdx++) {
            __m256d result = _mm256_setzero_pd();

            for (int cdx = 0; cdx < block_size; cdx += 2) {
                complex_prod_AVX(mv_xy, rdx, cdx, indices, input, result);
            }
            // Step 1: swap 128-bit lanes: [a0,a1,a2,a3] -> [a2,a3,a0,a1]
            __m256d perm = _mm256_permute2f128_pd(result, result, 0x01);

            // Step 2: add to original to compute pairwise sums
            __m256d sum = _mm256_add_pd(result, perm);
            // sum = [a0+a2, a1+a3, a2+a0, a3+a1]

            // Step 3: extract the low 128 bits containing [a0+a2, a1+a3]
            __m128d low128 = _mm256_castpd256_pd128(sum);

            // Step 4: extract scalars correctly
            double real = _mm_cvtsd_f64(low128);                        // element 0 = a0+a2
            double imag = _mm_cvtsd_f64(_mm_unpackhi_pd(low128, low128)); // element 1 = a1+a3

            new_block_real[rdx] = real;
            new_block_imag[rdx] = imag;
        }
        write_out_block(input, new_block_real, new_block_imag, indices);
        }
    } 
  _mm_free(mv_xy);

}

/**
@brief Call to apply kernel to apply two qubit gate kernel on a state vector using AVX
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param inner_qbit The lower significance qubit (little endian convention)
@param outer_qbit The higher significance qubit (little endian convention)
@param matrix_size The size of the input
*/
void apply_2qbit_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    int inner_qbit = involved_qbits[0];
    int outer_qbit = involved_qbits[1];
    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);



/*
AVX kernel developed according to https://github.com/qulacs/qulacs/blob/main/src/csim/update_ops_matrix_dense_single.cpp

under MIT License

Copyright (c) 2018 Qulacs Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
*/
if (inner_qbit==0){
    __m256d mv00 = _mm256_set_pd(-two_qbit_unitary[1].imag, two_qbit_unitary[1].real, -two_qbit_unitary[0].imag, two_qbit_unitary[0].real);
    __m256d mv01 = _mm256_set_pd( two_qbit_unitary[1].real, two_qbit_unitary[1].imag,  two_qbit_unitary[0].real, two_qbit_unitary[0].imag);
    __m256d mv20 = _mm256_set_pd(-two_qbit_unitary[3].imag, two_qbit_unitary[3].real, -two_qbit_unitary[2].imag, two_qbit_unitary[2].real);
    __m256d mv21 = _mm256_set_pd( two_qbit_unitary[3].real, two_qbit_unitary[3].imag,  two_qbit_unitary[2].real, two_qbit_unitary[2].imag);
    __m256d mv40 = _mm256_set_pd(-two_qbit_unitary[5].imag, two_qbit_unitary[5].real, -two_qbit_unitary[4].imag, two_qbit_unitary[4].real);
    __m256d mv41 = _mm256_set_pd( two_qbit_unitary[5].real, two_qbit_unitary[5].imag,  two_qbit_unitary[4].real, two_qbit_unitary[4].imag);     
    __m256d mv60 = _mm256_set_pd(-two_qbit_unitary[7].imag, two_qbit_unitary[7].real, -two_qbit_unitary[6].imag, two_qbit_unitary[6].real);
    __m256d mv61 = _mm256_set_pd( two_qbit_unitary[7].real, two_qbit_unitary[7].imag,  two_qbit_unitary[6].real, two_qbit_unitary[6].imag);     
    __m256d mv80 = _mm256_set_pd(-two_qbit_unitary[9].imag, two_qbit_unitary[9].real, -two_qbit_unitary[8].imag, two_qbit_unitary[8].real);
    __m256d mv81 = _mm256_set_pd( two_qbit_unitary[9].real, two_qbit_unitary[9].imag,  two_qbit_unitary[8].real, two_qbit_unitary[8].imag);
    __m256d mv100 = _mm256_set_pd(-two_qbit_unitary[11].imag, two_qbit_unitary[11].real, -two_qbit_unitary[10].imag, two_qbit_unitary[10].real);
    __m256d mv101 = _mm256_set_pd( two_qbit_unitary[11].real, two_qbit_unitary[11].imag,  two_qbit_unitary[10].real, two_qbit_unitary[10].imag);
    __m256d mv120 = _mm256_set_pd(-two_qbit_unitary[13].imag, two_qbit_unitary[13].real, -two_qbit_unitary[12].imag, two_qbit_unitary[12].real);
    __m256d mv121 = _mm256_set_pd( two_qbit_unitary[13].real, two_qbit_unitary[13].imag,  two_qbit_unitary[12].real, two_qbit_unitary[12].imag);
    __m256d mv140 = _mm256_set_pd(-two_qbit_unitary[15].imag, two_qbit_unitary[15].real, -two_qbit_unitary[14].imag, two_qbit_unitary[14].real);
    __m256d mv141 = _mm256_set_pd( two_qbit_unitary[15].real, two_qbit_unitary[15].imag,  two_qbit_unitary[14].real, two_qbit_unitary[14].imag);
    
    for (int current_idx = 0; current_idx < input.rows; current_idx += (index_step_outer << 1)) {
        int current_idx_pair_outer = current_idx + index_step_outer;
        
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner += (index_step_inner << 1)) {

            int current_idx_outer_loc = current_idx + current_idx_inner;
            int current_idx_outer_pair_loc = current_idx_pair_outer + current_idx_inner;
            
            // Load two consecutive complex numbers at each base location
            double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
            double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
            
            // Load 4 doubles = 2 complex numbers at each location
            __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
            __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
            
            // Compute the four matrix-vector products (one for each output element)
            
            // Result for current_idx_outer_loc (row 0 of matrix)
            __m256d data_u0 = _mm256_mul_pd(element_outer_vec, mv00);
            __m256d data_u1 = _mm256_mul_pd(element_outer_vec, mv01);
            __m256d data_u3 = _mm256_mul_pd(element_outer_pair_vec, mv20);
            __m256d data_u4 = _mm256_mul_pd(element_outer_pair_vec, mv21);
            __m256d data_u5 = _mm256_add_pd(data_u3, data_u0);
            __m256d data_u2 = _mm256_add_pd(data_u1, data_u4);
            __m256d data_u7 = _mm256_hadd_pd(data_u5, data_u2);
            __m256d data_u8 = _mm256_permute4x64_pd(data_u7, 0b11011000);
            __m256d data_u6 = _mm256_hadd_pd(data_u8, data_u8);

            __m256d data_d0 = _mm256_mul_pd(element_outer_vec, mv40);
            __m256d data_d1 = _mm256_mul_pd(element_outer_vec, mv41);
            __m256d data_d3 = _mm256_mul_pd(element_outer_pair_vec, mv60);
            __m256d data_d4 = _mm256_mul_pd(element_outer_pair_vec, mv61);
            __m256d data_d5 = _mm256_add_pd(data_d3, data_d0);
            __m256d data_d6 = _mm256_add_pd(data_d1, data_d4);
            data_d6 = _mm256_hadd_pd(data_d5, data_d6);
            data_d6 = _mm256_permute4x64_pd(data_d6, 0b11011000);
            data_d6 = _mm256_hadd_pd(data_d6, data_d6);

            // Result for row 2 of matrix
            __m256d data_e0 = _mm256_mul_pd(element_outer_vec, mv80);
            __m256d data_e1 = _mm256_mul_pd(element_outer_vec, mv81);
            __m256d data_e3 = _mm256_mul_pd(element_outer_pair_vec, mv100);
            __m256d data_e4 = _mm256_mul_pd(element_outer_pair_vec, mv101);
            __m256d data_e5 = _mm256_add_pd(data_e3, data_e0);
            __m256d data_e6 = _mm256_add_pd(data_e1, data_e4);
            data_e6 = _mm256_hadd_pd(data_e5, data_e6);
            data_e6 = _mm256_permute4x64_pd(data_e6, 0b11011000);
            data_e6 = _mm256_hadd_pd(data_e6, data_e6);

            // Result for row 3 of matrix
            __m256d data_f0 = _mm256_mul_pd(element_outer_vec, mv120);
            __m256d data_f1 = _mm256_mul_pd(element_outer_vec, mv121);
            __m256d data_f3 = _mm256_mul_pd(element_outer_pair_vec, mv140);
            __m256d data_f4 = _mm256_mul_pd(element_outer_pair_vec, mv141);
            __m256d data_f5 = _mm256_add_pd(data_f3, data_f0);
            __m256d data_f6 = _mm256_add_pd(data_f1, data_f4);
            data_f6 = _mm256_hadd_pd(data_f5, data_f6);
            data_f6 = _mm256_permute4x64_pd(data_f6, 0b11011000);
            data_f6 = _mm256_hadd_pd(data_f6, data_f6);

            // Store results back to the same locations where they were loaded
            __m128d low128u = _mm256_castpd256_pd128(data_u6);
            __m128d high128u = _mm256_extractf128_pd(data_u6, 1);

            input[current_idx_outer_loc].real = _mm_cvtsd_f64(low128u);
            input[current_idx_outer_loc].imag = _mm_cvtsd_f64(high128u);

            __m128d low128d = _mm256_castpd256_pd128(data_d6);
            __m128d high128d = _mm256_extractf128_pd(data_d6, 1);
            input[current_idx_outer_loc + 1].real = _mm_cvtsd_f64(low128d);
            input[current_idx_outer_loc + 1].imag = _mm_cvtsd_f64(high128d);

            __m128d low128e = _mm256_castpd256_pd128(data_e6);
            __m128d high128e = _mm256_extractf128_pd(data_e6, 1);
            input[current_idx_outer_pair_loc].real = _mm_cvtsd_f64(low128e);
            input[current_idx_outer_pair_loc].imag = _mm_cvtsd_f64(high128e);

            __m128d low128f = _mm256_castpd256_pd128(data_f6);
            __m128d high128f = _mm256_extractf128_pd(data_f6, 1);
            input[current_idx_outer_pair_loc + 1].real = _mm_cvtsd_f64(low128f);
            input[current_idx_outer_pair_loc + 1].imag = _mm_cvtsd_f64(high128f);
        }
    }
}
    else{
        __m256d mv00 = _mm256_set_pd(-two_qbit_unitary[0].imag, two_qbit_unitary[0].real, -two_qbit_unitary[0].imag, two_qbit_unitary[0].real);
        __m256d mv01 = _mm256_set_pd( two_qbit_unitary[0].real, two_qbit_unitary[0].imag,  two_qbit_unitary[0].real, two_qbit_unitary[0].imag);
        __m256d mv10 = _mm256_set_pd(-two_qbit_unitary[1].imag, two_qbit_unitary[1].real, -two_qbit_unitary[1].imag, two_qbit_unitary[1].real);
        __m256d mv11 = _mm256_set_pd( two_qbit_unitary[1].real, two_qbit_unitary[1].imag,  two_qbit_unitary[1].real, two_qbit_unitary[1].imag);
        __m256d mv20 = _mm256_set_pd(-two_qbit_unitary[2].imag, two_qbit_unitary[2].real, -two_qbit_unitary[2].imag, two_qbit_unitary[2].real);
        __m256d mv21 = _mm256_set_pd( two_qbit_unitary[2].real, two_qbit_unitary[2].imag,  two_qbit_unitary[2].real, two_qbit_unitary[2].imag);
        __m256d mv30 = _mm256_set_pd(-two_qbit_unitary[3].imag, two_qbit_unitary[3].real, -two_qbit_unitary[3].imag, two_qbit_unitary[3].real);
        __m256d mv31 = _mm256_set_pd( two_qbit_unitary[3].real, two_qbit_unitary[3].imag,  two_qbit_unitary[3].real, two_qbit_unitary[3].imag);
        __m256d mv40 = _mm256_set_pd(-two_qbit_unitary[4].imag, two_qbit_unitary[4].real, -two_qbit_unitary[4].imag, two_qbit_unitary[4].real);
        __m256d mv41 = _mm256_set_pd( two_qbit_unitary[4].real, two_qbit_unitary[4].imag,  two_qbit_unitary[4].real, two_qbit_unitary[4].imag);
        __m256d mv50 = _mm256_set_pd(-two_qbit_unitary[5].imag, two_qbit_unitary[5].real, -two_qbit_unitary[5].imag, two_qbit_unitary[5].real);
        __m256d mv51 = _mm256_set_pd( two_qbit_unitary[5].real, two_qbit_unitary[5].imag,  two_qbit_unitary[5].real, two_qbit_unitary[5].imag);
        __m256d mv60 = _mm256_set_pd(-two_qbit_unitary[6].imag, two_qbit_unitary[6].real, -two_qbit_unitary[6].imag, two_qbit_unitary[6].real);
        __m256d mv61 = _mm256_set_pd( two_qbit_unitary[6].real, two_qbit_unitary[6].imag,  two_qbit_unitary[6].real, two_qbit_unitary[6].imag);
        __m256d mv70 = _mm256_set_pd(-two_qbit_unitary[7].imag, two_qbit_unitary[7].real, -two_qbit_unitary[7].imag, two_qbit_unitary[7].real);
        __m256d mv71 = _mm256_set_pd( two_qbit_unitary[7].real, two_qbit_unitary[7].imag,  two_qbit_unitary[7].real, two_qbit_unitary[7].imag);
        __m256d mv80 = _mm256_set_pd(-two_qbit_unitary[8].imag, two_qbit_unitary[8].real, -two_qbit_unitary[8].imag, two_qbit_unitary[8].real);
        __m256d mv81 = _mm256_set_pd( two_qbit_unitary[8].real, two_qbit_unitary[8].imag,  two_qbit_unitary[8].real, two_qbit_unitary[8].imag);
        __m256d mv90 = _mm256_set_pd(-two_qbit_unitary[9].imag, two_qbit_unitary[9].real, -two_qbit_unitary[9].imag, two_qbit_unitary[9].real);
        __m256d mv91 = _mm256_set_pd( two_qbit_unitary[9].real, two_qbit_unitary[9].imag,  two_qbit_unitary[9].real, two_qbit_unitary[9].imag);
        __m256d mv100 = _mm256_set_pd(-two_qbit_unitary[10].imag, two_qbit_unitary[10].real, -two_qbit_unitary[10].imag, two_qbit_unitary[10].real);
        __m256d mv101 = _mm256_set_pd( two_qbit_unitary[10].real, two_qbit_unitary[10].imag,  two_qbit_unitary[10].real, two_qbit_unitary[10].imag);
        __m256d mv110 = _mm256_set_pd(-two_qbit_unitary[11].imag, two_qbit_unitary[11].real, -two_qbit_unitary[11].imag, two_qbit_unitary[11].real);
        __m256d mv111 = _mm256_set_pd( two_qbit_unitary[11].real, two_qbit_unitary[11].imag,  two_qbit_unitary[11].real, two_qbit_unitary[11].imag);
        __m256d mv120 = _mm256_set_pd(-two_qbit_unitary[12].imag, two_qbit_unitary[12].real, -two_qbit_unitary[12].imag, two_qbit_unitary[12].real);
        __m256d mv121 = _mm256_set_pd( two_qbit_unitary[12].real, two_qbit_unitary[12].imag,  two_qbit_unitary[12].real, two_qbit_unitary[12].imag);
        __m256d mv130 = _mm256_set_pd(-two_qbit_unitary[13].imag, two_qbit_unitary[13].real, -two_qbit_unitary[13].imag, two_qbit_unitary[13].real);
        __m256d mv131 = _mm256_set_pd( two_qbit_unitary[13].real, two_qbit_unitary[13].imag,  two_qbit_unitary[13].real, two_qbit_unitary[13].imag);
        __m256d mv140 = _mm256_set_pd(-two_qbit_unitary[14].imag, two_qbit_unitary[14].real, -two_qbit_unitary[14].imag, two_qbit_unitary[14].real);
        __m256d mv141 = _mm256_set_pd( two_qbit_unitary[14].real, two_qbit_unitary[14].imag,  two_qbit_unitary[14].real, two_qbit_unitary[14].imag);
        __m256d mv150 = _mm256_set_pd(-two_qbit_unitary[15].imag, two_qbit_unitary[15].real, -two_qbit_unitary[15].imag, two_qbit_unitary[15].real);
        __m256d mv151 = _mm256_set_pd( two_qbit_unitary[15].real, two_qbit_unitary[15].imag,  two_qbit_unitary[15].real, two_qbit_unitary[15].imag);
        for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<input.rows; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){

        for (int idx=0; idx<index_step_inner; idx=idx+2){
                
		    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
                
                double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                double* element_inner = element_outer + 2 * index_step_inner;
                
                double* element_outer_pair = element_outer + 2 * index_step_outer;
                double* element_inner_pair = element_outer_pair + 2 * index_step_inner;
                
                
                __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                __m256d element_inner_vec = _mm256_loadu_pd(element_inner);

                __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);

                __m256d data_u0 = _mm256_mul_pd(element_outer_vec, mv00);
                __m256d data_u1 = _mm256_mul_pd(element_inner_vec, mv10);
                __m256d data_u2 = _mm256_mul_pd(element_outer_vec, mv01);
                __m256d data_u3 = _mm256_mul_pd(element_inner_vec, mv11);
                __m256d data_u4 = _mm256_mul_pd(element_outer_pair_vec, mv20);
                __m256d data_u5 = _mm256_mul_pd(element_inner_pair_vec, mv30);
                __m256d data_u6 = _mm256_mul_pd(element_outer_pair_vec, mv21);
                __m256d data_u7 = _mm256_mul_pd(element_inner_pair_vec, mv31);
                __m256d data_u8 = _mm256_hadd_pd(data_u0, data_u2);
                __m256d data_u9 = _mm256_hadd_pd(data_u1, data_u3);
                __m256d data_u10 = _mm256_hadd_pd(data_u4, data_u6);
                __m256d data_u11 = _mm256_hadd_pd(data_u5, data_u7);
                __m256d data_u = _mm256_add_pd(data_u8, data_u9);
                data_u = _mm256_add_pd(data_u, data_u10);
                data_u = _mm256_add_pd(data_u, data_u11);

                __m256d data_d0 = _mm256_mul_pd(element_outer_vec, mv40);
                __m256d data_d1 = _mm256_mul_pd(element_inner_vec, mv50);
                __m256d data_d2 = _mm256_mul_pd(element_outer_vec, mv41);
                __m256d data_d3 = _mm256_mul_pd(element_inner_vec, mv51);
                __m256d data_d4 = _mm256_mul_pd(element_outer_pair_vec, mv60);
                __m256d data_d5 = _mm256_mul_pd(element_inner_pair_vec, mv70);
                __m256d data_d6 = _mm256_mul_pd(element_outer_pair_vec, mv61);
                __m256d data_d7 = _mm256_mul_pd(element_inner_pair_vec, mv71);
                __m256d data_d8 = _mm256_hadd_pd(data_d0, data_d2);
                __m256d data_d9 = _mm256_hadd_pd(data_d1, data_d3);
                __m256d data_d10 = _mm256_hadd_pd(data_d4, data_d6);
                __m256d data_d11 = _mm256_hadd_pd(data_d5, data_d7);
                __m256d data_d = _mm256_add_pd(data_d8, data_d9);
                data_d = _mm256_add_pd(data_d, data_d10);
                data_d = _mm256_add_pd(data_d, data_d11);

                __m256d data_e0 = _mm256_mul_pd(element_outer_vec, mv80);
                __m256d data_e1 = _mm256_mul_pd(element_inner_vec, mv90);
                __m256d data_e2 = _mm256_mul_pd(element_outer_vec, mv81);
                __m256d data_e3 = _mm256_mul_pd(element_inner_vec, mv91);
                __m256d data_e4 = _mm256_mul_pd(element_outer_pair_vec, mv100);
                __m256d data_e5 = _mm256_mul_pd(element_inner_pair_vec, mv110);
                __m256d data_e6 = _mm256_mul_pd(element_outer_pair_vec, mv101);
                __m256d data_e7 = _mm256_mul_pd(element_inner_pair_vec, mv111);
                __m256d data_e8 = _mm256_hadd_pd(data_e0, data_e2);
                __m256d data_e9 = _mm256_hadd_pd(data_e1, data_e3);
                __m256d data_e10 = _mm256_hadd_pd(data_e4, data_e6);
                __m256d data_e11 = _mm256_hadd_pd(data_e5, data_e7);
                __m256d data_e = _mm256_add_pd(data_e8, data_e9);
                data_e = _mm256_add_pd(data_e, data_e10);
                data_e = _mm256_add_pd(data_e, data_e11);

                __m256d data_f0 = _mm256_mul_pd(element_outer_vec, mv120);
                __m256d data_f1 = _mm256_mul_pd(element_inner_vec, mv130);
                __m256d data_f2 = _mm256_mul_pd(element_outer_vec, mv121);
                __m256d data_f3 = _mm256_mul_pd(element_inner_vec, mv131);
                __m256d data_f4 = _mm256_mul_pd(element_outer_pair_vec, mv140);
                __m256d data_f5 = _mm256_mul_pd(element_inner_pair_vec, mv150);
                __m256d data_f6 = _mm256_mul_pd(element_outer_pair_vec, mv141);
                __m256d data_f7 = _mm256_mul_pd(element_inner_pair_vec, mv151);
                __m256d data_f8 = _mm256_hadd_pd(data_f0, data_f2);
                __m256d data_f9 = _mm256_hadd_pd(data_f1, data_f3);
                __m256d data_f10 = _mm256_hadd_pd(data_f4, data_f6);
                __m256d data_f11 = _mm256_hadd_pd(data_f5, data_f7);
                __m256d data_f = _mm256_add_pd(data_f8, data_f9);
                data_f = _mm256_add_pd(data_f, data_f10);
                data_f = _mm256_add_pd(data_f, data_f11);

                _mm256_storeu_pd(element_outer, data_u);
                _mm256_storeu_pd(element_inner, data_d);
                _mm256_storeu_pd(element_outer_pair, data_e);
                _mm256_storeu_pd(element_inner_pair, data_f);



            }
             }
        current_idx = current_idx + (index_step_outer << 1);
    }
    
    }
    
}


/**
@brief Call to apply kernel to apply two qubit gate kernel on a state vector using AVX and TBB
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param inner_qbit The lower significance qubit (little endian convention)
@param outer_qbit The higher significance qubit (little endian convention)
@param matrix_size The size of the input
*/
void apply_2qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    
}

#define CREATE_MATRIX_VECTOR_CONSECUTIVE(base_idx) \
    __m256d mv##base_idx##0 = _mm256_set_pd(-unitary[base_idx+1].imag, unitary[base_idx+1].real, -unitary[base_idx].imag, unitary[base_idx].real); \
    __m256d mv##base_idx##1 = _mm256_set_pd( unitary[base_idx+1].real, unitary[base_idx+1].imag,  unitary[base_idx].real, unitary[base_idx].imag);

// Macro for 3-qubit row computation when inner_qbit == 0
#define COMPUTE_3QBIT_ROW_CONSECUTIVE(row_letter, mv00, mv20, mv40, mv60) \
    __m256d data_##row_letter##0 = _mm256_mul_pd(element_000_vec, mv##mv00##0); \
    __m256d data_##row_letter##1 = _mm256_mul_pd(element_000_vec, mv##mv00##1); \
    __m256d data_##row_letter##2 = _mm256_mul_pd(element_010_vec, mv##mv20##0); \
    __m256d data_##row_letter##3 = _mm256_mul_pd(element_010_vec, mv##mv20##1); \
    __m256d data_##row_letter##4 = _mm256_mul_pd(element_100_vec, mv##mv40##0); \
    __m256d data_##row_letter##5 = _mm256_mul_pd(element_100_vec, mv##mv40##1); \
    __m256d data_##row_letter##6 = _mm256_mul_pd(element_110_vec, mv##mv60##0); \
    __m256d data_##row_letter##7 = _mm256_mul_pd(element_110_vec, mv##mv60##1); \
    __m256d data_##row_letter##8 = _mm256_add_pd(data_##row_letter##0, data_##row_letter##2); \
    __m256d data_##row_letter##9 = _mm256_add_pd(data_##row_letter##1, data_##row_letter##3); \
    __m256d data_##row_letter##10 = _mm256_add_pd(data_##row_letter##4, data_##row_letter##6); \
    __m256d data_##row_letter##11 = _mm256_add_pd(data_##row_letter##5, data_##row_letter##7); \
    __m256d data_##row_letter##12 = _mm256_add_pd(data_##row_letter##8, data_##row_letter##10); \
    __m256d data_##row_letter##13 = _mm256_add_pd(data_##row_letter##9, data_##row_letter##11); \
    __m256d data_##row_letter = _mm256_hadd_pd(data_##row_letter##12, data_##row_letter##13); \
    data_##row_letter = _mm256_permute4x64_pd(data_##row_letter, 0b11011000); \
    data_##row_letter = _mm256_hadd_pd(data_##row_letter, data_##row_letter); \
    __m128d low128##row_letter = _mm256_castpd256_pd128(data_##row_letter); \
    __m128d high128##row_letter = _mm256_extractf128_pd(data_##row_letter, 1); \
    results[row_idx].real = _mm_cvtsd_f64(low128##row_letter); \
    results[row_idx].imag = _mm_cvtsd_f64(high128##row_letter);


#define CREATE_MATRIX_VECTOR(base_idx) \
    __m256d mv##base_idx##0 = _mm256_set_pd(-unitary[base_idx].imag, unitary[base_idx].real, -unitary[base_idx].imag, unitary[base_idx].real); \
    __m256d mv##base_idx##1 = _mm256_set_pd( unitary[base_idx].real, unitary[base_idx].imag,  unitary[base_idx].real, unitary[base_idx].imag);

// Macro for 3-qubit row computation
#define COMPUTE_3QBIT_ROW(row_letter, base0, base1, base2, base3, base4, base5, base6, base7) \
    __m256d data_##row_letter##0 = _mm256_mul_pd(element_outer_vec, mv##base0##0); \
    __m256d data_##row_letter##1 = _mm256_mul_pd(element_inner_vec, mv##base1##0); \
    __m256d data_##row_letter##2 = _mm256_mul_pd(element_outer_vec, mv##base0##1); \
    __m256d data_##row_letter##3 = _mm256_mul_pd(element_inner_vec, mv##base1##1); \
    __m256d data_##row_letter##4 = _mm256_mul_pd(element_middle_vec, mv##base2##0); \
    __m256d data_##row_letter##5 = _mm256_mul_pd(element_middle_inner_vec, mv##base3##0); \
    __m256d data_##row_letter##6 = _mm256_mul_pd(element_middle_vec, mv##base2##1); \
    __m256d data_##row_letter##7 = _mm256_mul_pd(element_middle_inner_vec, mv##base3##1); \
    __m256d data_##row_letter##8 = _mm256_mul_pd(element_outer_pair_vec, mv##base4##0); \
    __m256d data_##row_letter##9 = _mm256_mul_pd(element_inner_pair_vec, mv##base5##0); \
    __m256d data_##row_letter##10 = _mm256_mul_pd(element_outer_pair_vec, mv##base4##1); \
    __m256d data_##row_letter##11 = _mm256_mul_pd(element_inner_pair_vec, mv##base5##1); \
    __m256d data_##row_letter##12 = _mm256_mul_pd(element_middle_pair_vec, mv##base6##0); \
    __m256d data_##row_letter##13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv##base7##0); \
    __m256d data_##row_letter##14 = _mm256_mul_pd(element_middle_pair_vec, mv##base6##1); \
    __m256d data_##row_letter##15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv##base7##1); \
    __m256d data_##row_letter##16 = _mm256_hadd_pd(data_##row_letter##0, data_##row_letter##2); \
    __m256d data_##row_letter##17 = _mm256_hadd_pd(data_##row_letter##1, data_##row_letter##3); \
    __m256d data_##row_letter##18 = _mm256_hadd_pd(data_##row_letter##4, data_##row_letter##6); \
    __m256d data_##row_letter##19 = _mm256_hadd_pd(data_##row_letter##5, data_##row_letter##7); \
    __m256d data_##row_letter##20 = _mm256_hadd_pd(data_##row_letter##8, data_##row_letter##10); \
    __m256d data_##row_letter##21 = _mm256_hadd_pd(data_##row_letter##9, data_##row_letter##11); \
    __m256d data_##row_letter##22 = _mm256_hadd_pd(data_##row_letter##12, data_##row_letter##14); \
    __m256d data_##row_letter##23 = _mm256_hadd_pd(data_##row_letter##13, data_##row_letter##15); \
    __m256d data_##row_letter = _mm256_add_pd(data_##row_letter##16, data_##row_letter##17); \
    data_##row_letter = _mm256_add_pd(data_##row_letter, data_##row_letter##18); \
    data_##row_letter = _mm256_add_pd(data_##row_letter, data_##row_letter##19); \
    data_##row_letter = _mm256_add_pd(data_##row_letter, data_##row_letter##20); \
    data_##row_letter = _mm256_add_pd(data_##row_letter, data_##row_letter##21); \
    data_##row_letter = _mm256_add_pd(data_##row_letter, data_##row_letter##22); \
    data_##row_letter = _mm256_add_pd(data_##row_letter, data_##row_letter##23);

/**
@brief Call to apply kernel to apply three qubit gate kernel on a state vector using AVX
@param unitary The 8x8 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_3qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    int inner_qbit = involved_qbits[0];
    int middle_qbit = involved_qbits[1];
    int outer_qbit = involved_qbits[2];
    
    int index_step_inner = 1 << inner_qbit;
    int index_step_middle = 1 << middle_qbit;
    int index_step_outer = 1 << outer_qbit;
    
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

    if (inner_qbit == 0) {
        // Setup AVX vectors for 8x8 matrix when inner_qbit == 0 (consecutive elements)
        CREATE_MATRIX_VECTOR_CONSECUTIVE(0)     // mv00, mv01 from unitary[0,1]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(2)     // mv20, mv21 from unitary[2,3]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(4)     // mv40, mv41 from unitary[4,5]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(6)     // mv60, mv61 from unitary[6,7]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(8)     // mv80, mv81 from unitary[8,9]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(10)   // mv100, mv101 from unitary[10,11]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(12)   // mv120, mv121 from unitary[12,13]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(14)   // mv140, mv141 from unitary[14,15]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(16)   // mv160, mv161 from unitary[16,17]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(18)   // mv180, mv181 from unitary[18,19]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(20)   // mv200, mv201 from unitary[20,21]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(22)   // mv220, mv221 from unitary[22,23]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(24)   // mv240, mv241 from unitary[24,25]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(26)   // mv260, mv261 from unitary[26,27]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(28)   // mv280, mv281 from unitary[28,29]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(30)   // mv300, mv301 from unitary[30,31]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(32)   // mv320, mv321 from unitary[32,33]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(34)   // mv340, mv341 from unitary[34,35]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(36)   // mv360, mv361 from unitary[36,37]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(38)   // mv380, mv381 from unitary[38,39]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(40)   // mv400, mv401 from unitary[40,41]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(42)   // mv420, mv421 from unitary[42,43]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(44)   // mv440, mv441 from unitary[44,45]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(46)   // mv460, mv461 from unitary[46,47]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(48)   // mv480, mv481 from unitary[48,49]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(50)   // mv500, mv501 from unitary[50,51]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(52)   // mv520, mv521 from unitary[52,53]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(54)   // mv540, mv541 from unitary[54,55]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(56)   // mv560, mv561 from unitary[56,57]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(58)   // mv580, mv581 from unitary[58,59]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(60)   // mv600, mv601 from unitary[60,61]
        CREATE_MATRIX_VECTOR_CONSECUTIVE(62)   // mv620, mv621 from unitary[62,63]

        for (int current_idx = 0; current_idx < input.rows; current_idx += (index_step_outer << 1)) {
            int current_idx_pair_outer = current_idx + index_step_outer;
            
            for (int current_idx_middle = 0; current_idx_middle < index_step_outer; current_idx_middle += (index_step_middle << 1)) {
                int current_idx_pair_middle = current_idx_middle + index_step_middle;
                
                for (int current_idx_inner = 0; current_idx_inner < index_step_middle; current_idx_inner += (index_step_inner << 1)) {

                    int current_idx_000_loc = current_idx + current_idx_middle + current_idx_inner;
                    int current_idx_010_loc = current_idx + current_idx_pair_middle + current_idx_inner;
                    int current_idx_100_loc = current_idx_pair_outer + current_idx_middle + current_idx_inner;
                    int current_idx_110_loc = current_idx_pair_outer + current_idx_pair_middle + current_idx_inner;
                    
                    // Load consecutive complex numbers for each group
                    double* element_000 = (double*)input.get_data() + 2 * current_idx_000_loc;
                    double* element_010 = (double*)input.get_data() + 2 * current_idx_010_loc;
                    double* element_100 = (double*)input.get_data() + 2 * current_idx_100_loc;
                    double* element_110 = (double*)input.get_data() + 2 * current_idx_110_loc;
                    
                    __m256d element_000_vec = _mm256_loadu_pd(element_000);
                    __m256d element_010_vec = _mm256_loadu_pd(element_010);
                    __m256d element_100_vec = _mm256_loadu_pd(element_100);
                    __m256d element_110_vec = _mm256_loadu_pd(element_110);
                    
                    // Compute all 8 results (one for each row of 8x8 matrix)
                    QGD_Complex16 results[8];
                    
                    int row_idx = 0;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(u, 0, 2, 4, 6)
                    
                    // Row 1  
                    row_idx = 1;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(d, 8, 10, 12, 14)
                    
                    // Row 2
                    row_idx = 2;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(e, 16, 18, 20, 22)
                    
                    // Row 3
                    row_idx = 3;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(f, 24, 26, 28, 30)
                    
                    // Row 4
                    row_idx = 4;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(g, 32, 34, 36, 38)
                    
                    // Row 5
                    row_idx = 5;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(h, 40, 42, 44, 46)
                    
                    // Row 6
                    row_idx = 6;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(i, 48, 50, 52, 54)
                    
                    // Row 7
                    row_idx = 7;
                    COMPUTE_3QBIT_ROW_CONSECUTIVE(j, 56, 58, 60, 62)


                    // Store results
                    input[current_idx_000_loc] = results[0];
                    input[current_idx_000_loc + 1] = results[1];
                    input[current_idx_010_loc] = results[2];
                    input[current_idx_010_loc + 1] = results[3];
                    input[current_idx_100_loc] = results[4];
                    input[current_idx_100_loc + 1] = results[5];
                    input[current_idx_110_loc] = results[6];
                    input[current_idx_110_loc + 1] = results[7];
                }
            }
        }
    }
    else {
        
        
        // Create matrix vectors using macro
        CREATE_MATRIX_VECTOR(0)   CREATE_MATRIX_VECTOR(1)   CREATE_MATRIX_VECTOR(2)   CREATE_MATRIX_VECTOR(3)
        CREATE_MATRIX_VECTOR(4)   CREATE_MATRIX_VECTOR(5)   CREATE_MATRIX_VECTOR(6)   CREATE_MATRIX_VECTOR(7)
        CREATE_MATRIX_VECTOR(8)   CREATE_MATRIX_VECTOR(9)   CREATE_MATRIX_VECTOR(10)  CREATE_MATRIX_VECTOR(11)
        CREATE_MATRIX_VECTOR(12)  CREATE_MATRIX_VECTOR(13)  CREATE_MATRIX_VECTOR(14)  CREATE_MATRIX_VECTOR(15)
        CREATE_MATRIX_VECTOR(16)  CREATE_MATRIX_VECTOR(17)  CREATE_MATRIX_VECTOR(18)  CREATE_MATRIX_VECTOR(19)
        CREATE_MATRIX_VECTOR(20)  CREATE_MATRIX_VECTOR(21)  CREATE_MATRIX_VECTOR(22)  CREATE_MATRIX_VECTOR(23)
        CREATE_MATRIX_VECTOR(24)  CREATE_MATRIX_VECTOR(25)  CREATE_MATRIX_VECTOR(26)  CREATE_MATRIX_VECTOR(27)
        CREATE_MATRIX_VECTOR(28)  CREATE_MATRIX_VECTOR(29)  CREATE_MATRIX_VECTOR(30)  CREATE_MATRIX_VECTOR(31)
        CREATE_MATRIX_VECTOR(32)  CREATE_MATRIX_VECTOR(33)  CREATE_MATRIX_VECTOR(34)  CREATE_MATRIX_VECTOR(35)
        CREATE_MATRIX_VECTOR(36)  CREATE_MATRIX_VECTOR(37)  CREATE_MATRIX_VECTOR(38)  CREATE_MATRIX_VECTOR(39)
        CREATE_MATRIX_VECTOR(40)  CREATE_MATRIX_VECTOR(41)  CREATE_MATRIX_VECTOR(42)  CREATE_MATRIX_VECTOR(43)
        CREATE_MATRIX_VECTOR(44)  CREATE_MATRIX_VECTOR(45)  CREATE_MATRIX_VECTOR(46)  CREATE_MATRIX_VECTOR(47)
        CREATE_MATRIX_VECTOR(48)  CREATE_MATRIX_VECTOR(49)  CREATE_MATRIX_VECTOR(50)  CREATE_MATRIX_VECTOR(51)
        CREATE_MATRIX_VECTOR(52)  CREATE_MATRIX_VECTOR(53)  CREATE_MATRIX_VECTOR(54)  CREATE_MATRIX_VECTOR(55)
        CREATE_MATRIX_VECTOR(56)  CREATE_MATRIX_VECTOR(57)  CREATE_MATRIX_VECTOR(58)  CREATE_MATRIX_VECTOR(59)
        CREATE_MATRIX_VECTOR(60)  CREATE_MATRIX_VECTOR(61)  CREATE_MATRIX_VECTOR(62)  CREATE_MATRIX_VECTOR(63)
                
        int parallel_outer_cycles = matrix_size / (index_step_outer << 1);
        int parallel_inner_cycles = index_step_middle / (index_step_inner << 1);
        int parallel_middle_cycles = index_step_outer / (index_step_middle << 1);

        for (int outer_rdx = 0; outer_rdx < parallel_outer_cycles; outer_rdx++) {
            int current_idx = outer_rdx * (index_step_outer << 1);
            int current_idx_pair_outer = current_idx + index_step_outer;
            
            for (int middle_rdx = 0; middle_rdx < parallel_middle_cycles; middle_rdx++) {
                int current_idx_middle = middle_rdx * (index_step_middle << 1);
                
                for (int inner_rdx = 0; inner_rdx < parallel_inner_cycles; inner_rdx++) {
                    int current_idx_inner = inner_rdx * (index_step_inner << 1);

                    for (int idx = 0; idx < index_step_inner; idx += 2) {
                        int current_idx_outer_loc = current_idx + current_idx_inner + current_idx_middle + idx;
                        
                        double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                        double* element_inner = element_outer + 2 * index_step_inner;
                        double* element_middle = element_outer + 2 * index_step_middle;
                        double* element_middle_inner = element_outer + 2 * (index_step_middle + index_step_inner);
                        
                        double* element_outer_pair = element_outer + 2 * index_step_outer;
                        double* element_inner_pair = element_outer_pair + 2 * index_step_inner;
                        double* element_middle_pair = element_outer_pair + 2 * index_step_middle;
                        double* element_middle_inner_pair = element_outer_pair + 2 * (index_step_middle + index_step_inner);
                        
                        __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                        __m256d element_inner_vec = _mm256_loadu_pd(element_inner);
                        __m256d element_middle_vec = _mm256_loadu_pd(element_middle);
                        __m256d element_middle_inner_vec = _mm256_loadu_pd(element_middle_inner);
                        __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                        __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                        __m256d element_middle_pair_vec = _mm256_loadu_pd(element_middle_pair);
                        __m256d element_middle_inner_pair_vec = _mm256_loadu_pd(element_middle_inner_pair);

                        // Compute all 8 rows using macros
                        COMPUTE_3QBIT_ROW(u, 0, 1, 2, 3, 4, 5, 6, 7)       // Row 0
                        COMPUTE_3QBIT_ROW(d, 8, 9, 10, 11, 12, 13, 14, 15) // Row 1 
                        COMPUTE_3QBIT_ROW(e, 16, 17, 18, 19, 20, 21, 22, 23) // Row 2
                        COMPUTE_3QBIT_ROW(f, 24, 25, 26, 27, 28, 29, 30, 31) // Row 3
                        COMPUTE_3QBIT_ROW(g, 32, 33, 34, 35, 36, 37, 38, 39) // Row 4
                        COMPUTE_3QBIT_ROW(h, 40, 41, 42, 43, 44, 45, 46, 47) // Row 5
                        COMPUTE_3QBIT_ROW(i, 48, 49, 50, 51, 52, 53, 54, 55) // Row 6
                        COMPUTE_3QBIT_ROW(j, 56, 57, 58, 59, 60, 61, 62, 63) // Row 7

                        // Store results
                        _mm256_storeu_pd(element_outer, data_u);
                        _mm256_storeu_pd(element_inner, data_d);
                        _mm256_storeu_pd(element_middle, data_e);
                        _mm256_storeu_pd(element_middle_inner, data_f);
                        _mm256_storeu_pd(element_outer_pair, data_g);
                        _mm256_storeu_pd(element_inner_pair, data_h);
                        _mm256_storeu_pd(element_middle_pair, data_i);
                        _mm256_storeu_pd(element_middle_inner_pair, data_j);
                    }
                }
            }
        }
    }

}
/** 
@brief Call to apply kernel to apply four qubit gate kernel on a state vector using AVX and TBB
@param two_qbit_unitary The 16x16 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_4qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_inner   = 1 << involved_qbits[0];
    int index_step_middle1 = 1 << involved_qbits[1];
    int index_step_middle2 = 1 << involved_qbits[2];
    int index_step_outer   = 1 << involved_qbits[3];
                    
    for (int iter_idx = 0; iter_idx < matrix_size>>4; iter_idx++) {
        int base = iter_idx & ~(index_step_outer|index_step_middle2|index_step_middle1|index_step_inner);  

        int current_idx_outer_loc = base;
        int current_idx_inner_loc = base + index_step_inner;
        
        int current_idx_middle1_loc = base + index_step_middle1;
        int current_idx_middle1_inner_loc = base + index_step_middle1 + index_step_inner;
        
        int current_idx_middle2_loc = base + index_step_middle2;
        int current_idx_middle2_inner_loc = base + index_step_middle2 + index_step_inner;
        
        int current_idx_middle12_loc = base + index_step_middle1 + index_step_middle2;
        int current_idx_middle12_inner_loc = base + index_step_middle1 + index_step_middle2 + index_step_inner;
        
        int current_idx_outer_pair_loc = base + index_step_outer;
        int current_idx_inner_pair_loc = current_idx_outer_pair_loc + index_step_inner;
        
        int current_idx_middle1_pair_loc = current_idx_outer_pair_loc + index_step_middle1;
        int current_idx_middle1_inner_pair_loc = current_idx_outer_pair_loc + index_step_middle1 + index_step_inner;
        
        int current_idx_middle2_pair_loc = current_idx_outer_pair_loc + index_step_middle2;
        int current_idx_middle2_inner_pair_loc = current_idx_outer_pair_loc + index_step_middle2 + index_step_inner;
        
        int current_idx_middle12_pair_loc = current_idx_outer_pair_loc + index_step_middle1 + index_step_middle2;
        int current_idx_middle12_inner_pair_loc = current_idx_outer_pair_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
        if(involved_qbits[0] == 0){
            //preload all 16 elements instead of the unitary kernel matrix
            __m256d element_0000_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_outer_loc);
            __m256d element_0010_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_middle1_loc);
            __m256d element_0100_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_middle2_loc);
            __m256d element_0110_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_middle12_loc);
            __m256d element_1000_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_outer_pair_loc);
            __m256d element_1010_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_middle1_pair_loc);
            __m256d element_1100_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_middle2_pair_loc);
            __m256d element_1110_vec_real = _mm256_loadu_pd((double*)input.get_data() + 2 * current_idx_middle12_pair_loc);


            __m256d element_0000_vec_imag = _mm256_permute4x64_pd(element_0000_vec_real, 0b10110001);
            __m256d element_0010_vec_imag = _mm256_permute4x64_pd(element_0010_vec_real, 0b10110001);
            __m256d element_0100_vec_imag = _mm256_permute4x64_pd(element_0100_vec_real, 0b10110001);
            __m256d element_0110_vec_imag = _mm256_permute4x64_pd(element_0110_vec_real, 0b10110001);
            __m256d element_1000_vec_imag = _mm256_permute4x64_pd(element_1000_vec_real, 0b10110001);
            __m256d element_1010_vec_imag = _mm256_permute4x64_pd(element_1010_vec_real, 0b10110001);
            __m256d element_1100_vec_imag = _mm256_permute4x64_pd(element_1100_vec_real, 0b10110001);
            __m256d element_1110_vec_imag = _mm256_permute4x64_pd(element_1110_vec_real, 0b10110001); 


            element_0000_vec_real = _mm256_mul_pd(element_0000_vec_real,neg);
            element_0010_vec_real = _mm256_mul_pd(element_0010_vec_real,neg);
            element_0100_vec_real = _mm256_mul_pd(element_0100_vec_real,neg);
            element_0110_vec_real = _mm256_mul_pd(element_0110_vec_real,neg);
            element_1000_vec_real = _mm256_mul_pd(element_1000_vec_real,neg);
            element_1010_vec_real = _mm256_mul_pd(element_1010_vec_real,neg);
            element_1100_vec_real = _mm256_mul_pd(element_1100_vec_real,neg);
            element_1110_vec_real = _mm256_mul_pd(element_1110_vec_real,neg);

            QGD_Complex16 results[16];
            for (int mult_idx = 0; mult_idx < 16; mult_idx++) {
                double* unitary_row_1 = (double*)unitary.get_data() + 32*mult_idx;
                double* unitary_row_2 = unitary_row_1 + 4;
                double* unitary_row_3 = unitary_row_1 + 8;
                double* unitary_row_4 = unitary_row_1 + 12;
                double* unitary_row_5 = unitary_row_1 + 16;
                double* unitary_row_6 = unitary_row_1 + 20;
                double* unitary_row_7 = unitary_row_1 + 24;
                double* unitary_row_8 = unitary_row_1 + 28;

                __m256d row1_vec = _mm256_loadu_pd(unitary_row_1);
                __m256d row2_vec = _mm256_loadu_pd(unitary_row_2);
                __m256d row3_vec = _mm256_loadu_pd(unitary_row_3);
                __m256d row4_vec = _mm256_loadu_pd(unitary_row_4);
                __m256d row5_vec = _mm256_loadu_pd(unitary_row_5);
                __m256d row6_vec = _mm256_loadu_pd(unitary_row_6);
                __m256d row7_vec = _mm256_loadu_pd(unitary_row_7);
                __m256d row8_vec = _mm256_loadu_pd(unitary_row_8);

                __m256d data_u1 = _mm256_mul_pd(element_0000_vec_real, row1_vec);
                __m256d data_u2 = _mm256_mul_pd(element_0000_vec_imag, row1_vec);
                __m256d data_u3 = _mm256_mul_pd(element_0010_vec_real, row2_vec);
                __m256d data_u4 = _mm256_mul_pd(element_0010_vec_imag, row2_vec);
                __m256d data_u5 = _mm256_mul_pd(element_0100_vec_real, row3_vec);
                __m256d data_u6 = _mm256_mul_pd(element_0100_vec_imag, row3_vec);
                __m256d data_u7 = _mm256_mul_pd(element_0110_vec_real, row4_vec);
                __m256d data_u8 = _mm256_mul_pd(element_0110_vec_imag, row4_vec);
                __m256d data_u9 = _mm256_mul_pd(element_1000_vec_real, row5_vec);
                __m256d data_u10 = _mm256_mul_pd(element_1000_vec_imag, row5_vec);
                __m256d data_u11 = _mm256_mul_pd(element_1010_vec_real, row6_vec);
                __m256d data_u12 = _mm256_mul_pd(element_1010_vec_imag, row6_vec);
                __m256d data_u13 = _mm256_mul_pd(element_1100_vec_real, row7_vec);
                __m256d data_u14 = _mm256_mul_pd(element_1100_vec_imag, row7_vec);
                __m256d data_u15 = _mm256_mul_pd(element_1110_vec_real, row8_vec);
                __m256d data_u16 = _mm256_mul_pd(element_1110_vec_imag, row8_vec);

                __m256d sum1 = _mm256_add_pd(data_u1, data_u3);
                __m256d sum2 = _mm256_add_pd(data_u2, data_u4);
                __m256d sum3 = _mm256_add_pd(data_u5, data_u7);
                __m256d sum4 = _mm256_add_pd(data_u6, data_u8);
                __m256d sum5 = _mm256_add_pd(data_u9, data_u11);
                __m256d sum6 = _mm256_add_pd(data_u10, data_u12);
                __m256d sum7 = _mm256_add_pd(data_u13, data_u15);
                __m256d sum8 = _mm256_add_pd(data_u14, data_u16);
                __m256d sum9 = _mm256_add_pd(sum1, sum3);
                __m256d sum10 = _mm256_add_pd(sum2, sum4);
                __m256d sum11 = _mm256_add_pd(sum5, sum7);
                __m256d sum12 = _mm256_add_pd(sum6, sum8);
                __m256d final_sum1 = _mm256_add_pd(sum9, sum11);
                __m256d final_sum2 = _mm256_add_pd(sum10, sum12);

                __m256d final_vec = _mm256_hadd_pd(final_sum1, final_sum2);
                final_vec = _mm256_permute4x64_pd(final_vec, 0b11011000);
                final_vec = _mm256_hadd_pd(final_vec, final_vec);
                __m128d low128 = _mm256_castpd256_pd128(final_vec);
                __m128d high128 = _mm256_extractf128_pd(final_vec, 1);
                results[mult_idx].real = _mm_cvtsd_f64(low128);
                results[mult_idx].imag = _mm_cvtsd_f64(high128);

            }
            input[current_idx_outer_loc]           = results[0];
            input[current_idx_inner_loc]           = results[1];
            input[current_idx_middle1_loc]         = results[2];
            input[current_idx_middle1_inner_loc]   = results[3];
            input[current_idx_middle2_loc]         = results[4];
            input[current_idx_middle2_inner_loc]   = results[5];
            input[current_idx_middle12_loc]        = results[6];
            input[current_idx_middle12_inner_loc]  = results[7];
            input[current_idx_outer_pair_loc]      = results[8];
            input[current_idx_inner_pair_loc]      = results[9];
            input[current_idx_middle1_pair_loc]    = results[10];
            input[current_idx_middle1_inner_pair_loc] = results[11];
            input[current_idx_middle2_pair_loc]    = results[12];
            input[current_idx_middle2_inner_pair_loc] = results[13];
            input[current_idx_middle12_pair_loc]   = results[14];
            input[current_idx_middle12_inner_pair_loc] = results[15];

            
        }
        else{

        double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
        double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
        
        double* element_middle1 = (double*)input.get_data() + 2 * current_idx_middle1_loc;
        double* element_middle1_inner = (double*)input.get_data() + 2 * current_idx_middle1_inner_loc;
        
        double* element_middle2 = (double*)input.get_data() + 2 * current_idx_middle2_loc;
        double* element_middle2_inner = (double*)input.get_data() + 2 * current_idx_middle2_inner_loc;
        
        double* element_middle12 = (double*)input.get_data() + 2 * current_idx_middle12_loc;
        double* element_middle12_inner = (double*)input.get_data() + 2 * current_idx_middle12_inner_loc;
        
        double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
        double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
                                    
        double* element_middle1_pair = (double*)input.get_data() + 2 * current_idx_middle1_pair_loc;
        double* element_middle1_inner_pair = (double*)input.get_data() + 2 * current_idx_middle1_inner_pair_loc;
        
        double* element_middle2_pair = (double*)input.get_data() + 2 * current_idx_middle2_pair_loc;
        double* element_middle2_inner_pair = (double*)input.get_data() + 2 * current_idx_middle2_inner_pair_loc;
        
        double* element_middle12_pair = (double*)input.get_data() + 2 * current_idx_middle12_pair_loc;
        double* element_middle12_inner_pair = (double*)input.get_data() + 2 * current_idx_middle12_inner_pair_loc;
        
        __m256d element_0000_vec_real     = get_AVX_vector(element_outer, element_inner);
        __m256d element_0010_vec_real     = get_AVX_vector(element_middle1, element_middle1_inner);
        __m256d element_0100_vec_real     = get_AVX_vector(element_middle2, element_middle2_inner);
        __m256d element_0110_vec_real     = get_AVX_vector(element_middle12, element_middle12_inner);
        __m256d element_1000_vec_real     = get_AVX_vector(element_outer_pair, element_inner_pair);
        __m256d element_1010_vec_real     = get_AVX_vector(element_middle1_pair, element_middle1_inner_pair);
        __m256d element_1100_vec_real     = get_AVX_vector(element_middle2_pair, element_middle2_inner_pair);
        __m256d element_1110_vec_real     = get_AVX_vector(element_middle12_pair, element_middle12_inner_pair);

        __m256d element_0000_vec_imag = _mm256_permute4x64_pd(element_0000_vec_real, 0b10110001);
        __m256d element_0010_vec_imag = _mm256_permute4x64_pd(element_0010_vec_real, 0b10110001);
        __m256d element_0100_vec_imag = _mm256_permute4x64_pd(element_0100_vec_real, 0b10110001);
        __m256d element_0110_vec_imag = _mm256_permute4x64_pd(element_0110_vec_real, 0b10110001);
        __m256d element_1000_vec_imag = _mm256_permute4x64_pd(element_1000_vec_real, 0b10110001);
        __m256d element_1010_vec_imag = _mm256_permute4x64_pd(element_1010_vec_real, 0b10110001);
        __m256d element_1100_vec_imag = _mm256_permute4x64_pd(element_1100_vec_real, 0b10110001);
        __m256d element_1110_vec_imag = _mm256_permute4x64_pd(element_1110_vec_real, 0b10110001);

        element_0000_vec_real = _mm256_mul_pd(element_0000_vec_real,neg);
        element_0010_vec_real = _mm256_mul_pd(element_0010_vec_real,neg);
        element_0100_vec_real = _mm256_mul_pd(element_0100_vec_real,neg);
        element_0110_vec_real = _mm256_mul_pd(element_0110_vec_real,neg);
        element_1000_vec_real = _mm256_mul_pd(element_1000_vec_real,neg);
        element_1010_vec_real = _mm256_mul_pd(element_1010_vec_real,neg);
        element_1100_vec_real = _mm256_mul_pd(element_1100_vec_real,neg);
        element_1110_vec_real = _mm256_mul_pd(element_1110_vec_real,neg);

        QGD_Complex16 results[16];
        for (int mult_idx = 0; mult_idx < 16; mult_idx++) {
            double* unitary_row_1 = (double*)unitary.get_data() + 32*mult_idx;
            double* unitary_row_2 = unitary_row_1 + 4;
            double* unitary_row_3 = unitary_row_1 + 8;
            double* unitary_row_4 = unitary_row_1 + 12;
            double* unitary_row_5 = unitary_row_1 + 16;
            double* unitary_row_6 = unitary_row_1 + 20;
            double* unitary_row_7 = unitary_row_1 + 24;
            double* unitary_row_8 = unitary_row_1 + 28;

            __m256d row1_vec = _mm256_loadu_pd(unitary_row_1);
            __m256d row2_vec = _mm256_loadu_pd(unitary_row_2);
            __m256d row3_vec = _mm256_loadu_pd(unitary_row_3);
            __m256d row4_vec = _mm256_loadu_pd(unitary_row_4);
            __m256d row5_vec = _mm256_loadu_pd(unitary_row_5);
            __m256d row6_vec = _mm256_loadu_pd(unitary_row_6);
            __m256d row7_vec = _mm256_loadu_pd(unitary_row_7);
            __m256d row8_vec = _mm256_loadu_pd(unitary_row_8);

            __m256d data_u1 = _mm256_mul_pd(element_0000_vec_real, row1_vec);
            __m256d data_u2 = _mm256_mul_pd(element_0000_vec_imag, row1_vec);
            __m256d data_u3 = _mm256_mul_pd(element_0010_vec_real, row2_vec);
            __m256d data_u4 = _mm256_mul_pd(element_0010_vec_imag, row2_vec);
            __m256d data_u5 = _mm256_mul_pd(element_0100_vec_real, row3_vec);
            __m256d data_u6 = _mm256_mul_pd(element_0100_vec_imag, row3_vec);
            __m256d data_u7 = _mm256_mul_pd(element_0110_vec_real, row4_vec);
            __m256d data_u8 = _mm256_mul_pd(element_0110_vec_imag, row4_vec);
            __m256d data_u9 = _mm256_mul_pd(element_1000_vec_real, row5_vec);
            __m256d data_u10 = _mm256_mul_pd(element_1000_vec_imag, row5_vec);
            __m256d data_u11 = _mm256_mul_pd(element_1010_vec_real, row6_vec);
            __m256d data_u12 = _mm256_mul_pd(element_1010_vec_imag, row6_vec);
            __m256d data_u13 = _mm256_mul_pd(element_1100_vec_real, row7_vec);
            __m256d data_u14 = _mm256_mul_pd(element_1100_vec_imag, row7_vec);
            __m256d data_u15 = _mm256_mul_pd(element_1110_vec_real, row8_vec);
            __m256d data_u16 = _mm256_mul_pd(element_1110_vec_imag, row8_vec);

            __m256d sum1 = _mm256_add_pd(data_u1, data_u3);
            __m256d sum2 = _mm256_add_pd(data_u2, data_u4);
            __m256d sum3 = _mm256_add_pd(data_u5, data_u7);
            __m256d sum4 = _mm256_add_pd(data_u6, data_u8);
            __m256d sum5 = _mm256_add_pd(data_u9, data_u11);
            __m256d sum6 = _mm256_add_pd(data_u10, data_u12);
            __m256d sum7 = _mm256_add_pd(data_u13, data_u15);
            __m256d sum8 = _mm256_add_pd(data_u14, data_u16);
            __m256d sum9 = _mm256_add_pd(sum1, sum3);
            __m256d sum10 = _mm256_add_pd(sum2, sum4);
            __m256d sum11 = _mm256_add_pd(sum5, sum7);
            __m256d sum12 = _mm256_add_pd(sum6, sum8);
            __m256d final_sum1 = _mm256_add_pd(sum9, sum11);
            __m256d final_sum2 = _mm256_add_pd(sum10, sum12);

            __m256d final_vec = _mm256_hadd_pd(final_sum1, final_sum2);
            final_vec = _mm256_permute4x64_pd(final_vec, 0b11011000);
            final_vec = _mm256_hadd_pd(final_vec, final_vec);
            __m128d low128 = _mm256_castpd256_pd128(final_vec);
            __m128d high128 = _mm256_extractf128_pd(final_vec, 1);
            results[mult_idx].real = _mm_cvtsd_f64(low128);
            results[mult_idx].imag = _mm_cvtsd_f64(high128);

        }
        input[current_idx_outer_loc]           = results[0];
        input[current_idx_inner_loc]           = results[1];
        input[current_idx_middle1_loc]         = results[2];
        input[current_idx_middle1_inner_loc]   = results[3];
        input[current_idx_middle2_loc]         = results[4];
        input[current_idx_middle2_inner_loc]   = results[5];
        input[current_idx_middle12_loc]        = results[6];
        input[current_idx_middle12_inner_loc]  = results[7];
        input[current_idx_outer_pair_loc]      = results[8];
        input[current_idx_inner_pair_loc]      = results[9];
        input[current_idx_middle1_pair_loc]    = results[10];
        input[current_idx_middle1_inner_pair_loc] = results[11];
        input[current_idx_middle2_pair_loc]    = results[12];
        input[current_idx_middle2_inner_pair_loc] = results[13];
        input[current_idx_middle12_pair_loc]   = results[14];
        input[current_idx_middle12_inner_pair_loc] = results[15];

    }
    }
    
}

/**
@brief Call to apply kernel to apply two qubit gate kernel on an input matrix using AVX
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param inner_qbit The lower significance qubit (little endian convention)
@param outer_qbit The higher significance qubit (little endian convention)
@param matrix_size The size of the input
*/
void apply_2qbit_kernel_to_matrix_input_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){
    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<input.rows; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){

        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
    	    for (int idx=0; idx<index_step_inner; idx++){
            	
        	int current_idx_outer_loc = current_idx + current_idx_inner + idx;
	        int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
          	int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
	        int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
	        
	        int row_offset_outer = current_idx_outer_loc*input.cols;
	        int row_offset_inner = current_idx_inner_loc*input.cols;
	        int row_offset_outer_pair = current_idx_outer_pair_loc*input.cols;
	        int row_offset_inner_pair = current_idx_inner_pair_loc*input.cols;
	            #pragma omp parallel for
            	for (int col_idx=0; col_idx<input.cols;col_idx++){
            	
                int current_idx_outer = row_offset_outer + col_idx;
                int current_idx_inner = row_offset_inner + col_idx;
                int current_idx_outer_pair = row_offset_outer_pair + col_idx;
                int current_idx_inner_pair = row_offset_inner_pair + col_idx;
                
	            double results[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
			    
                double* element_outer = (double*)input.get_data() + 2 * current_idx_outer;
                double* element_inner = (double*)input.get_data() + 2 * current_idx_inner;
                double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair;
                double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair;
			    
                __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                element_outer_vec = _mm256_permute4x64_pd(element_outer_vec,0b11011000);
                __m256d element_inner_vec = _mm256_loadu_pd(element_inner);
                element_inner_vec = _mm256_permute4x64_pd(element_inner_vec,0b11011000);
                __m256d outer_inner_vec = _mm256_shuffle_pd(element_outer_vec,element_inner_vec,0b0000);
                outer_inner_vec = _mm256_permute4x64_pd(outer_inner_vec,0b11011000);


                __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                element_outer_pair_vec = _mm256_permute4x64_pd(element_outer_pair_vec,0b11011000);
                __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                element_inner_pair_vec = _mm256_permute4x64_pd(element_inner_pair_vec,0b11011000);
                __m256d outer_inner_pair_vec = _mm256_shuffle_pd(element_outer_pair_vec,element_inner_pair_vec,0b0000);
                outer_inner_pair_vec = _mm256_permute4x64_pd(outer_inner_pair_vec,0b11011000);




		    for (int mult_idx=0; mult_idx<4; mult_idx++){
	               double* unitary_row_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
	               double* unitary_row_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
			        
                    __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                    __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);

                    __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_01_vec,neg);
                                
                    __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_23_vec,neg);                   
                    
                    __m256d result_vec = _mm256_hadd_pd(result_upper_vec,result_lower_vec);
                    result_vec = _mm256_hadd_pd(result_vec,result_vec);
                    double* result = (double*)&result_vec;
                    results[mult_idx*2] = result[0];
                    results[mult_idx*2+1] = result[2];
			    }
		    input[current_idx_outer].real = results[0];
		    input[current_idx_outer].imag = results[1];
		    input[current_idx_inner].real = results[2];
		    input[current_idx_inner].imag = results[3];
		    input[current_idx_outer_pair].real = results[4];
		    input[current_idx_outer_pair].imag = results[5];
		    input[current_idx_inner_pair].real = results[6];
		    input[current_idx_inner_pair].imag = results[7];
            	  }
            	}
        
        }
        current_idx = current_idx + (index_step_outer << 1);
    }
}


/**
@brief Call to apply crot gate kernel on an input matrix using AVX and TBB
@param u3_1qbit1 The 2x2 kernel to be applied on target |1>
@param u3_1qbit2 The 2x2 kernel to be applied on target |0>
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit
@param control_qbit The control qubit
@param matrix_size The size of the input
*/
void
apply_crot_kernel_to_matrix_input_AVX_parallel(Matrix& u3_1qbit1,Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    input.ensure_aligned();
    
    int index_step_target = 1 << target_qbit;

    // load elements of the U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit1[0].real);
    __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit1[0].imag);
    __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit1[1].real);
    __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit1[1].imag);
    __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit1[2].real);
    __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit1[2].imag);
    __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit1[3].real);
    __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit1[3].imag);

    __m256d u3_1bit2_00r_vec = _mm256_broadcast_sd(&u3_1qbit2[0].real);
    __m256d u3_1bit2_00i_vec = _mm256_broadcast_sd(&u3_1qbit2[0].imag);
    __m256d u3_1bit2_01r_vec = _mm256_broadcast_sd(&u3_1qbit2[1].real);
    __m256d u3_1bit2_01i_vec = _mm256_broadcast_sd(&u3_1qbit2[1].imag);
    __m256d u3_1bit2_10r_vec = _mm256_broadcast_sd(&u3_1qbit2[2].real);
    __m256d u3_1bit2_10i_vec = _mm256_broadcast_sd(&u3_1qbit2[2].imag);
    __m256d u3_1bit2_11r_vec = _mm256_broadcast_sd(&u3_1qbit2[3].real);
    __m256d u3_1bit2_11i_vec = _mm256_broadcast_sd(&u3_1qbit2[3].imag);


    int parallel_outer_cycles = matrix_size/(index_step_target << 1);
    int outer_grain_size;
    if ( index_step_target <= 2 ) {
        outer_grain_size = 32;
    }
    else if ( index_step_target <= 4 ) {
        outer_grain_size = 16;
    }
    else if ( index_step_target <= 8 ) {
        outer_grain_size = 8;
    }
    else if ( index_step_target <= 16 ) {
        outer_grain_size = 4;
    }
    else {
        outer_grain_size = 2;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 

        int current_idx      = r.begin()*(index_step_target << 1);
        int current_idx_pair = index_step_target + r.begin()*(index_step_target << 1);

        for (int rdx=r.begin(); rdx<r.end(); rdx++) {
            

            tbb::parallel_for( tbb::blocked_range<int>(0,index_step_target,32), [&](tbb::blocked_range<int> r) {
	        for (int idx=r.begin(); idx<r.end(); ++idx) {


                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    int row_offset = current_idx_loc * input.stride;
                    int row_offset_pair = current_idx_pair_loc * input.stride;

                    if ((current_idx_loc >> control_qbit) & 1) {

    
                        double* element = (double*)input.get_data() + 2 * row_offset;
                        double* element_pair = (double*)input.get_data() + 2 * row_offset_pair;


                        for (int col_idx = 0; col_idx < 2 * (input.cols - 3); col_idx = col_idx + 8) {

                            // extract successive elements from arrays element, element_pair
                            __m256d element_vec = _mm256_load_pd(element + col_idx);
                            __m256d element_vec2 = _mm256_load_pd(element + col_idx + 4);
                            __m256d tmp = _mm256_shuffle_pd(element_vec, element_vec2, 0);
                            element_vec2 = _mm256_shuffle_pd(element_vec, element_vec2, 0xf);
                            element_vec = tmp;

                            __m256d element_pair_vec = _mm256_load_pd(element_pair + col_idx);
                            __m256d element_pair_vec2 = _mm256_load_pd(element_pair + col_idx + 4);
                            tmp = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0);
                            element_pair_vec2 = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0xf);
                            element_pair_vec = tmp;

                            __m256d vec3 = _mm256_mul_pd(u3_1bit_00r_vec, element_vec);
                            vec3 = _mm256_fnmadd_pd(u3_1bit_00i_vec, element_vec2, vec3);
                            __m256d vec4 = _mm256_mul_pd(u3_1bit_01r_vec, element_pair_vec);
                            vec4 = _mm256_fnmadd_pd(u3_1bit_01i_vec, element_pair_vec2, vec4);
                            vec3 = _mm256_add_pd(vec3, vec4);
                            __m256d vec5 = _mm256_mul_pd(u3_1bit_00r_vec, element_vec2);
                            vec5 = _mm256_fmadd_pd(u3_1bit_00i_vec, element_vec, vec5);
                            __m256d vec6 = _mm256_mul_pd(u3_1bit_01r_vec, element_pair_vec2);
                            vec6 = _mm256_fmadd_pd(u3_1bit_01i_vec, element_pair_vec, vec6);
                            vec5 = _mm256_add_pd(vec5, vec6);    

                            // 6 store the transformed elements in vec3
                            tmp = _mm256_shuffle_pd(vec3, vec5, 0);
                            vec5 = _mm256_shuffle_pd(vec3, vec5, 0xf);
                            vec3 = tmp;
                            _mm256_store_pd(element + col_idx, vec3);
                            _mm256_store_pd(element + col_idx + 4, vec5);

                            __m256d vec7 = _mm256_mul_pd(u3_1bit_10r_vec, element_vec);
                            vec7 = _mm256_fnmadd_pd(u3_1bit_10i_vec, element_vec2, vec7);
                            __m256d vec8 = _mm256_mul_pd(u3_1bit_11r_vec, element_pair_vec);
                            vec8 = _mm256_fnmadd_pd(u3_1bit_11i_vec, element_pair_vec2, vec8);
                            vec7 = _mm256_add_pd(vec7, vec8);
                            __m256d vec9 = _mm256_mul_pd(u3_1bit_10r_vec, element_vec2);
                            vec9 = _mm256_fmadd_pd(u3_1bit_10i_vec, element_vec, vec9);
                            __m256d vec10 = _mm256_mul_pd(u3_1bit_11r_vec, element_pair_vec2);
                            vec10 = _mm256_fmadd_pd(u3_1bit_11i_vec, element_pair_vec, vec10);
                            vec9 = _mm256_add_pd(vec9, vec10);

                            // 6 store the transformed elements in vec3
                            tmp = _mm256_shuffle_pd(vec7, vec9, 0);
                            vec9 = _mm256_shuffle_pd(vec7, vec9, 0xf);
                            vec7 = tmp;
                            _mm256_store_pd(element_pair + col_idx, vec7);
                            _mm256_store_pd(element_pair + col_idx + 4, vec9);
                        }

                        int remainder = input.cols % 4;
                        if (remainder != 0) {

                            for (int col_idx = input.cols-remainder; col_idx < input.cols; col_idx++) {
                                int index = row_offset + col_idx;
                                int index_pair = row_offset_pair + col_idx;
        
                                QGD_Complex16 element = input[index];
                                QGD_Complex16 element_pair = input[index_pair];

                                QGD_Complex16 tmp1 = mult(u3_1qbit1[0], element);
                                QGD_Complex16 tmp2 = mult(u3_1qbit1[1], element_pair);

                                input[index].real = tmp1.real + tmp2.real;
                                input[index].imag = tmp1.imag + tmp2.imag;

                                tmp1 = mult(u3_1qbit1[2], element);
                                tmp2 = mult(u3_1qbit1[3], element_pair);

                                input[index_pair].real = tmp1.real + tmp2.real;
                                input[index_pair].imag = tmp1.imag + tmp2.imag;
                            }
        
                        }

                    }

                    else {
                        
                        double* element = (double*)input.get_data() + 2 * row_offset;
                        double* element_pair = (double*)input.get_data() + 2 * row_offset_pair;


                        for (int col_idx = 0; col_idx < 2 * (input.cols - 3); col_idx = col_idx + 8) {

                              // extract successive elements from arrays element, element_pair
                              __m256d element_vec = _mm256_load_pd(element + col_idx);
                              __m256d element_vec2 = _mm256_load_pd(element + col_idx + 4);
                              __m256d tmp = _mm256_shuffle_pd(element_vec, element_vec2, 0);
                              element_vec2 = _mm256_shuffle_pd(element_vec, element_vec2, 0xf);
                              element_vec = tmp;

                              __m256d element_pair_vec = _mm256_load_pd(element_pair + col_idx);
                              __m256d element_pair_vec2 = _mm256_load_pd(element_pair + col_idx + 4);
                              tmp = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0);
                              element_pair_vec2 = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0xf);
                              element_pair_vec = tmp;

                              __m256d vec3 = _mm256_mul_pd(u3_1bit2_00r_vec, element_vec);
                              vec3 = _mm256_fnmadd_pd(u3_1bit2_00i_vec, element_vec2, vec3);
                              __m256d vec4 = _mm256_mul_pd(u3_1bit2_01r_vec, element_pair_vec);
                              vec4 = _mm256_fnmadd_pd(u3_1bit2_01i_vec, element_pair_vec2, vec4);
                              vec3 = _mm256_add_pd(vec3, vec4);
                              __m256d vec5 = _mm256_mul_pd(u3_1bit2_00r_vec, element_vec2);
                              vec5 = _mm256_fmadd_pd(u3_1bit2_00i_vec, element_vec, vec5);
                              __m256d vec6 = _mm256_mul_pd(u3_1bit2_01r_vec, element_pair_vec2);
                              vec6 = _mm256_fmadd_pd(u3_1bit2_01i_vec, element_pair_vec, vec6);
                              vec5 = _mm256_add_pd(vec5, vec6);    

                              // 6 store the transformed elements in vec3
                              tmp = _mm256_shuffle_pd(vec3, vec5, 0);
                              vec5 = _mm256_shuffle_pd(vec3, vec5, 0xf);
                              vec3 = tmp;
                              _mm256_store_pd(element + col_idx, vec3);
                              _mm256_store_pd(element + col_idx + 4, vec5);

                              __m256d vec7 = _mm256_mul_pd(u3_1bit2_10r_vec, element_vec);
                              vec7 = _mm256_fnmadd_pd(u3_1bit2_10i_vec, element_vec2, vec7);
                              __m256d vec8 = _mm256_mul_pd(u3_1bit2_11r_vec, element_pair_vec);
                              vec8 = _mm256_fnmadd_pd(u3_1bit2_11i_vec, element_pair_vec2, vec8);
                              vec7 = _mm256_add_pd(vec7, vec8);
                              __m256d vec9 = _mm256_mul_pd(u3_1bit2_10r_vec, element_vec2);
                              vec9 = _mm256_fmadd_pd(u3_1bit2_10i_vec, element_vec, vec9);
                              __m256d vec10 = _mm256_mul_pd(u3_1bit2_11r_vec, element_pair_vec2);
                              vec10 = _mm256_fmadd_pd(u3_1bit2_11i_vec, element_pair_vec, vec10);
                              vec9 = _mm256_add_pd(vec9, vec10);

                              // 6 store the transformed elements in vec3
                              tmp = _mm256_shuffle_pd(vec7, vec9, 0);
                              vec9 = _mm256_shuffle_pd(vec7, vec9, 0xf);
                              vec7 = tmp;
                              _mm256_store_pd(element_pair + col_idx, vec7);
                              _mm256_store_pd(element_pair + col_idx + 4, vec9);
                        }

                        int remainder = input.cols % 4;
                        if (remainder != 0) {

                            for (int col_idx = input.cols-remainder; col_idx < input.cols; col_idx++) {
                                int index = row_offset + col_idx;
                                int index_pair = row_offset_pair + col_idx;
        
                                QGD_Complex16 element = input[index];
                                QGD_Complex16 element_pair = input[index_pair];

                                QGD_Complex16 tmp1 = mult(u3_1qbit2[0], element);
                                QGD_Complex16 tmp2 = mult(u3_1qbit2[1], element_pair);

                                input[index].real = tmp1.real + tmp2.real;
                                input[index].imag = tmp1.imag + tmp2.imag;

                                tmp1 = mult(u3_1qbit2[2], element);
                                tmp2 = mult(u3_1qbit2[3], element_pair);

                                input[index_pair].real = tmp1.real + tmp2.real;
                                input[index_pair].imag = tmp1.imag + tmp2.imag;
                            }
        
                        }
                    }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;

                 
                }
            });
            


            current_idx = current_idx + (index_step_target << 1);
            current_idx_pair = current_idx_pair + (index_step_target << 1);

        }
    });
    

}

