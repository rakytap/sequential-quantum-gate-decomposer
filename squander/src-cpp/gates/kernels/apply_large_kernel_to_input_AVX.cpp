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
        __m256d mv00 = _mm256_set_pd(-unitary[1].imag, unitary[1].real, -unitary[0].imag, unitary[0].real);
        __m256d mv01 = _mm256_set_pd( unitary[1].real, unitary[1].imag,  unitary[0].real, unitary[0].imag);
        __m256d mv20 = _mm256_set_pd(-unitary[3].imag, unitary[3].real, -unitary[2].imag, unitary[2].real);
        __m256d mv21 = _mm256_set_pd( unitary[3].real, unitary[3].imag,  unitary[2].real, unitary[2].imag);
        __m256d mv40 = _mm256_set_pd(-unitary[5].imag, unitary[5].real, -unitary[4].imag, unitary[4].real);
        __m256d mv41 = _mm256_set_pd( unitary[5].real, unitary[5].imag,  unitary[4].real, unitary[4].imag);
        __m256d mv60 = _mm256_set_pd(-unitary[7].imag, unitary[7].real, -unitary[6].imag, unitary[6].real);
        __m256d mv61 = _mm256_set_pd( unitary[7].real, unitary[7].imag,  unitary[6].real, unitary[6].imag);
        __m256d mv80 = _mm256_set_pd(-unitary[9].imag, unitary[9].real, -unitary[8].imag, unitary[8].real);
        __m256d mv81 = _mm256_set_pd( unitary[9].real, unitary[9].imag,  unitary[8].real, unitary[8].imag);
        __m256d mv100 = _mm256_set_pd(-unitary[11].imag, unitary[11].real, -unitary[10].imag, unitary[10].real);
        __m256d mv101 = _mm256_set_pd( unitary[11].real, unitary[11].imag,  unitary[10].real, unitary[10].imag);
        __m256d mv120 = _mm256_set_pd(-unitary[13].imag, unitary[13].real, -unitary[12].imag, unitary[12].real);
        __m256d mv121 = _mm256_set_pd( unitary[13].real, unitary[13].imag,  unitary[12].real, unitary[12].imag);
        __m256d mv140 = _mm256_set_pd(-unitary[15].imag, unitary[15].real, -unitary[14].imag, unitary[14].real);
        __m256d mv141 = _mm256_set_pd( unitary[15].real, unitary[15].imag,  unitary[14].real, unitary[14].imag);
        __m256d mv160 = _mm256_set_pd(-unitary[17].imag, unitary[17].real, -unitary[16].imag, unitary[16].real);
        __m256d mv161 = _mm256_set_pd( unitary[17].real, unitary[17].imag,  unitary[16].real, unitary[16].imag);
        __m256d mv180 = _mm256_set_pd(-unitary[19].imag, unitary[19].real, -unitary[18].imag, unitary[18].real);
        __m256d mv181 = _mm256_set_pd( unitary[19].real, unitary[19].imag,  unitary[18].real, unitary[18].imag);
        __m256d mv200 = _mm256_set_pd(-unitary[21].imag, unitary[21].real, -unitary[20].imag, unitary[20].real);
        __m256d mv201 = _mm256_set_pd( unitary[21].real, unitary[21].imag,  unitary[20].real, unitary[20].imag);
        __m256d mv220 = _mm256_set_pd(-unitary[23].imag, unitary[23].real, -unitary[22].imag, unitary[22].real);
        __m256d mv221 = _mm256_set_pd( unitary[23].real, unitary[23].imag,  unitary[22].real, unitary[22].imag);
        __m256d mv240 = _mm256_set_pd(-unitary[25].imag, unitary[25].real, -unitary[24].imag, unitary[24].real);
        __m256d mv241 = _mm256_set_pd( unitary[25].real, unitary[25].imag,  unitary[24].real, unitary[24].imag);
        __m256d mv260 = _mm256_set_pd(-unitary[27].imag, unitary[27].real, -unitary[26].imag, unitary[26].real);
        __m256d mv261 = _mm256_set_pd( unitary[27].real, unitary[27].imag,  unitary[26].real, unitary[26].imag);
        __m256d mv280 = _mm256_set_pd(-unitary[29].imag, unitary[29].real, -unitary[28].imag, unitary[28].real);
        __m256d mv281 = _mm256_set_pd( unitary[29].real, unitary[29].imag,  unitary[28].real, unitary[28].imag);
        __m256d mv300 = _mm256_set_pd(-unitary[31].imag, unitary[31].real, -unitary[30].imag, unitary[30].real);
        __m256d mv301 = _mm256_set_pd( unitary[31].real, unitary[31].imag,  unitary[30].real, unitary[30].imag);
        __m256d mv320 = _mm256_set_pd(-unitary[33].imag, unitary[33].real, -unitary[32].imag, unitary[32].real);
        __m256d mv321 = _mm256_set_pd( unitary[33].real, unitary[33].imag,  unitary[32].real, unitary[32].imag);
        __m256d mv340 = _mm256_set_pd(-unitary[35].imag, unitary[35].real, -unitary[34].imag, unitary[34].real);
        __m256d mv341 = _mm256_set_pd( unitary[35].real, unitary[35].imag,  unitary[34].real, unitary[34].imag);
        __m256d mv360 = _mm256_set_pd(-unitary[37].imag, unitary[37].real, -unitary[36].imag, unitary[36].real);
        __m256d mv361 = _mm256_set_pd( unitary[37].real, unitary[37].imag,  unitary[36].real, unitary[36].imag);
        __m256d mv380 = _mm256_set_pd(-unitary[39].imag, unitary[39].real, -unitary[38].imag, unitary[38].real);
        __m256d mv381 = _mm256_set_pd( unitary[39].real, unitary[39].imag,  unitary[38].real, unitary[38].imag);
        __m256d mv400 = _mm256_set_pd(-unitary[41].imag, unitary[41].real, -unitary[40].imag, unitary[40].real);
        __m256d mv401 = _mm256_set_pd( unitary[41].real, unitary[41].imag,  unitary[40].real, unitary[40].imag);
        __m256d mv420 = _mm256_set_pd(-unitary[43].imag, unitary[43].real, -unitary[42].imag, unitary[42].real);
        __m256d mv421 = _mm256_set_pd( unitary[43].real, unitary[43].imag,  unitary[42].real, unitary[42].imag);
        __m256d mv440 = _mm256_set_pd(-unitary[45].imag, unitary[45].real, -unitary[44].imag, unitary[44].real);
        __m256d mv441 = _mm256_set_pd( unitary[45].real, unitary[45].imag,  unitary[44].real, unitary[44].imag);
        __m256d mv460 = _mm256_set_pd(-unitary[47].imag, unitary[47].real, -unitary[46].imag, unitary[46].real);
        __m256d mv461 = _mm256_set_pd( unitary[47].real, unitary[47].imag,  unitary[46].real, unitary[46].imag);
        __m256d mv480 = _mm256_set_pd(-unitary[49].imag, unitary[49].real, -unitary[48].imag, unitary[48].real);
        __m256d mv481 = _mm256_set_pd( unitary[49].real, unitary[49].imag,  unitary[48].real, unitary[48].imag);
        __m256d mv500 = _mm256_set_pd(-unitary[51].imag, unitary[51].real, -unitary[50].imag, unitary[50].real);
        __m256d mv501 = _mm256_set_pd( unitary[51].real, unitary[51].imag,  unitary[50].real, unitary[50].imag);
        __m256d mv520 = _mm256_set_pd(-unitary[53].imag, unitary[53].real, -unitary[52].imag, unitary[52].real);
        __m256d mv521 = _mm256_set_pd( unitary[53].real, unitary[53].imag,  unitary[52].real, unitary[52].imag);
        __m256d mv540 = _mm256_set_pd(-unitary[55].imag, unitary[55].real, -unitary[54].imag, unitary[54].real);
        __m256d mv541 = _mm256_set_pd( unitary[55].real, unitary[55].imag,  unitary[54].real, unitary[54].imag);
        __m256d mv560 = _mm256_set_pd(-unitary[57].imag, unitary[57].real, -unitary[56].imag, unitary[56].real);
        __m256d mv561 = _mm256_set_pd( unitary[57].real, unitary[57].imag,  unitary[56].real, unitary[56].imag);
        __m256d mv580 = _mm256_set_pd(-unitary[59].imag, unitary[59].real, -unitary[58].imag, unitary[58].real);
        __m256d mv581 = _mm256_set_pd( unitary[59].real, unitary[59].imag,  unitary[58].real, unitary[58].imag);
        __m256d mv600 = _mm256_set_pd(-unitary[61].imag, unitary[61].real, -unitary[60].imag, unitary[60].real);
        __m256d mv601 = _mm256_set_pd( unitary[61].real, unitary[61].imag,  unitary[60].real, unitary[60].imag);
        __m256d mv620 = _mm256_set_pd(-unitary[63].imag, unitary[63].real, -unitary[62].imag, unitary[62].real);
        __m256d mv621 = _mm256_set_pd( unitary[63].real, unitary[63].imag,  unitary[62].real, unitary[62].imag);

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
                    
                    // Row 0
                    __m256d data_u0 = _mm256_mul_pd(element_000_vec, mv00);
                    __m256d data_u1 = _mm256_mul_pd(element_000_vec, mv01);
                    __m256d data_u2 = _mm256_mul_pd(element_010_vec, mv20);
                    __m256d data_u3 = _mm256_mul_pd(element_010_vec, mv21);
                    __m256d data_u4 = _mm256_mul_pd(element_100_vec, mv40);
                    __m256d data_u5 = _mm256_mul_pd(element_100_vec, mv41);
                    __m256d data_u6 = _mm256_mul_pd(element_110_vec, mv60);
                    __m256d data_u7 = _mm256_mul_pd(element_110_vec, mv61);
                    __m256d data_u8 = _mm256_add_pd(data_u0, data_u2);
                    __m256d data_u9 = _mm256_add_pd(data_u1, data_u3);
                    __m256d data_u10 = _mm256_add_pd(data_u4, data_u6);
                    __m256d data_u11 = _mm256_add_pd(data_u5, data_u7);
                    __m256d data_u12 = _mm256_add_pd(data_u8, data_u10);
                    __m256d data_u13 = _mm256_add_pd(data_u9, data_u11);
                    __m256d data_u = _mm256_hadd_pd(data_u12, data_u13);
                    data_u = _mm256_permute4x64_pd(data_u, 0b11011000);
                    data_u = _mm256_hadd_pd(data_u, data_u);
                    __m128d low128u = _mm256_castpd256_pd128(data_u);
                    __m128d high128u = _mm256_extractf128_pd(data_u, 1);
                    results[0].real = _mm_cvtsd_f64(low128u);
                    results[0].imag = _mm_cvtsd_f64(high128u);

// Row 1
                    __m256d data_d0 = _mm256_mul_pd(element_000_vec, mv80);
                    __m256d data_d1 = _mm256_mul_pd(element_000_vec, mv81);
                    __m256d data_d2 = _mm256_mul_pd(element_010_vec, mv100);
                    __m256d data_d3 = _mm256_mul_pd(element_010_vec, mv101);
                    __m256d data_d4 = _mm256_mul_pd(element_100_vec, mv120);
                    __m256d data_d5 = _mm256_mul_pd(element_100_vec, mv121);
                    __m256d data_d6 = _mm256_mul_pd(element_110_vec, mv140);
                    __m256d data_d7 = _mm256_mul_pd(element_110_vec, mv141);
                    __m256d data_d8 = _mm256_add_pd(data_d0, data_d2);
                    __m256d data_d9 = _mm256_add_pd(data_d1, data_d3);
                    __m256d data_d10 = _mm256_add_pd(data_d4, data_d6);
                    __m256d data_d11 = _mm256_add_pd(data_d5, data_d7);
                    __m256d data_d12 = _mm256_add_pd(data_d8, data_d10);
                    __m256d data_d13 = _mm256_add_pd(data_d9, data_d11);
                    __m256d data_d = _mm256_hadd_pd(data_d12, data_d13);
                    data_d = _mm256_permute4x64_pd(data_d, 0b11011000);
                    data_d = _mm256_hadd_pd(data_d, data_d);
                    __m128d low128d = _mm256_castpd256_pd128(data_d);
                    __m128d high128d = _mm256_extractf128_pd(data_d, 1);
                    results[1].real = _mm_cvtsd_f64(low128d);
                    results[1].imag = _mm_cvtsd_f64(high128d);

                    // Row 2
                    __m256d data_e0 = _mm256_mul_pd(element_000_vec, mv160);
                    __m256d data_e1 = _mm256_mul_pd(element_000_vec, mv161);
                    __m256d data_e2 = _mm256_mul_pd(element_010_vec, mv180);
                    __m256d data_e3 = _mm256_mul_pd(element_010_vec, mv181);
                    __m256d data_e4 = _mm256_mul_pd(element_100_vec, mv200);
                    __m256d data_e5 = _mm256_mul_pd(element_100_vec, mv201);
                    __m256d data_e6 = _mm256_mul_pd(element_110_vec, mv220);
                    __m256d data_e7 = _mm256_mul_pd(element_110_vec, mv221);
                    __m256d data_e8 = _mm256_add_pd(data_e0, data_e2);
                    __m256d data_e9 = _mm256_add_pd(data_e1, data_e3);
                    __m256d data_e10 = _mm256_add_pd(data_e4, data_e6);
                    __m256d data_e11 = _mm256_add_pd(data_e5, data_e7);
                    __m256d data_e12 = _mm256_add_pd(data_e8, data_e10);
                    __m256d data_e13 = _mm256_add_pd(data_e9, data_e11);
                    __m256d data_e = _mm256_hadd_pd(data_e12, data_e13);
                    data_e = _mm256_permute4x64_pd(data_e, 0b11011000);
                    data_e = _mm256_hadd_pd(data_e, data_e);
                    __m128d low128e = _mm256_castpd256_pd128(data_e);
                    __m128d high128e = _mm256_extractf128_pd(data_e, 1);
                    results[2].real = _mm_cvtsd_f64(low128e);
                    results[2].imag = _mm_cvtsd_f64(high128e);

                    // Row 3
                    __m256d data_f0 = _mm256_mul_pd(element_000_vec, mv240);
                    __m256d data_f1 = _mm256_mul_pd(element_000_vec, mv241);
                    __m256d data_f2 = _mm256_mul_pd(element_010_vec, mv260);
                    __m256d data_f3 = _mm256_mul_pd(element_010_vec, mv261);
                    __m256d data_f4 = _mm256_mul_pd(element_100_vec, mv280);
                    __m256d data_f5 = _mm256_mul_pd(element_100_vec, mv281);
                    __m256d data_f6 = _mm256_mul_pd(element_110_vec, mv300);
                    __m256d data_f7 = _mm256_mul_pd(element_110_vec, mv301);
                    __m256d data_f8 = _mm256_add_pd(data_f0, data_f2);
                    __m256d data_f9 = _mm256_add_pd(data_f1, data_f3);
                    __m256d data_f10 = _mm256_add_pd(data_f4, data_f6);
                    __m256d data_f11 = _mm256_add_pd(data_f5, data_f7);
                    __m256d data_f12 = _mm256_add_pd(data_f8, data_f10);
                    __m256d data_f13 = _mm256_add_pd(data_f9, data_f11);
                    __m256d data_f = _mm256_hadd_pd(data_f12, data_f13);
                    data_f = _mm256_permute4x64_pd(data_f, 0b11011000);
                    data_f = _mm256_hadd_pd(data_f, data_f);
                    __m128d low128f = _mm256_castpd256_pd128(data_f);
                    __m128d high128f = _mm256_extractf128_pd(data_f, 1);
                    results[3].real = _mm_cvtsd_f64(low128f);
                    results[3].imag = _mm_cvtsd_f64(high128f);

                    // Row 4
                    __m256d data_g0 = _mm256_mul_pd(element_000_vec, mv320);
                    __m256d data_g1 = _mm256_mul_pd(element_000_vec, mv321);
                    __m256d data_g2 = _mm256_mul_pd(element_010_vec, mv340);
                    __m256d data_g3 = _mm256_mul_pd(element_010_vec, mv341);
                    __m256d data_g4 = _mm256_mul_pd(element_100_vec, mv360);
                    __m256d data_g5 = _mm256_mul_pd(element_100_vec, mv361);
                    __m256d data_g6 = _mm256_mul_pd(element_110_vec, mv380);
                    __m256d data_g7 = _mm256_mul_pd(element_110_vec, mv381);
                    __m256d data_g8 = _mm256_add_pd(data_g0, data_g2);
                    __m256d data_g9 = _mm256_add_pd(data_g1, data_g3);
                    __m256d data_g10 = _mm256_add_pd(data_g4, data_g6);
                    __m256d data_g11 = _mm256_add_pd(data_g5, data_g7);
                    __m256d data_g12 = _mm256_add_pd(data_g8, data_g10);
                    __m256d data_g13 = _mm256_add_pd(data_g9, data_g11);
                    __m256d data_g = _mm256_hadd_pd(data_g12, data_g13);
                    data_g = _mm256_permute4x64_pd(data_g, 0b11011000);
                    data_g = _mm256_hadd_pd(data_g, data_g);
                    __m128d low128g = _mm256_castpd256_pd128(data_g);
                    __m128d high128g = _mm256_extractf128_pd(data_g, 1);
                    results[4].real = _mm_cvtsd_f64(low128g);
                    results[4].imag = _mm_cvtsd_f64(high128g);

                    // Row 5
                    __m256d data_h0 = _mm256_mul_pd(element_000_vec, mv400);
                    __m256d data_h1 = _mm256_mul_pd(element_000_vec, mv401);
                    __m256d data_h2 = _mm256_mul_pd(element_010_vec, mv420);
                    __m256d data_h3 = _mm256_mul_pd(element_010_vec, mv421);
                    __m256d data_h4 = _mm256_mul_pd(element_100_vec, mv440);
                    __m256d data_h5 = _mm256_mul_pd(element_100_vec, mv441);
                    __m256d data_h6 = _mm256_mul_pd(element_110_vec, mv460);
                    __m256d data_h7 = _mm256_mul_pd(element_110_vec, mv461);
                    __m256d data_h8 = _mm256_add_pd(data_h0, data_h2);
                    __m256d data_h9 = _mm256_add_pd(data_h1, data_h3);
                    __m256d data_h10 = _mm256_add_pd(data_h4, data_h6);
                    __m256d data_h11 = _mm256_add_pd(data_h5, data_h7);
                    __m256d data_h12 = _mm256_add_pd(data_h8, data_h10);
                    __m256d data_h13 = _mm256_add_pd(data_h9, data_h11);
                    __m256d data_h = _mm256_hadd_pd(data_h12, data_h13);
                    data_h = _mm256_permute4x64_pd(data_h, 0b11011000);
                    data_h = _mm256_hadd_pd(data_h, data_h);
                    __m128d low128h = _mm256_castpd256_pd128(data_h);
                    __m128d high128h = _mm256_extractf128_pd(data_h, 1);
                    results[5].real = _mm_cvtsd_f64(low128h);
                    results[5].imag = _mm_cvtsd_f64(high128h);

                    // Row 6
                    __m256d data_i0 = _mm256_mul_pd(element_000_vec, mv480);
                    __m256d data_i1 = _mm256_mul_pd(element_000_vec, mv481);
                    __m256d data_i2 = _mm256_mul_pd(element_010_vec, mv500);
                    __m256d data_i3 = _mm256_mul_pd(element_010_vec, mv501);
                    __m256d data_i4 = _mm256_mul_pd(element_100_vec, mv520);
                    __m256d data_i5 = _mm256_mul_pd(element_100_vec, mv521);
                    __m256d data_i6 = _mm256_mul_pd(element_110_vec, mv540);
                    __m256d data_i7 = _mm256_mul_pd(element_110_vec, mv541);
                    __m256d data_i8 = _mm256_add_pd(data_i0, data_i2);
                    __m256d data_i9 = _mm256_add_pd(data_i1, data_i3);
                    __m256d data_i10 = _mm256_add_pd(data_i4, data_i6);
                    __m256d data_i11 = _mm256_add_pd(data_i5, data_i7);
                    __m256d data_i12 = _mm256_add_pd(data_i8, data_i10);
                    __m256d data_i13 = _mm256_add_pd(data_i9, data_i11);
                    __m256d data_i = _mm256_hadd_pd(data_i12, data_i13);
                    data_i = _mm256_permute4x64_pd(data_i, 0b11011000);
                    data_i = _mm256_hadd_pd(data_i, data_i);
                    __m128d low128i = _mm256_castpd256_pd128(data_i);
                    __m128d high128i = _mm256_extractf128_pd(data_i, 1);
                    results[6].real = _mm_cvtsd_f64(low128i);
                    results[6].imag = _mm_cvtsd_f64(high128i);

                    // Row 7
                    __m256d data_j0 = _mm256_mul_pd(element_000_vec, mv560);
                    __m256d data_j1 = _mm256_mul_pd(element_000_vec, mv561);
                    __m256d data_j2 = _mm256_mul_pd(element_010_vec, mv580);
                    __m256d data_j3 = _mm256_mul_pd(element_010_vec, mv581);
                    __m256d data_j4 = _mm256_mul_pd(element_100_vec, mv600);
                    __m256d data_j5 = _mm256_mul_pd(element_100_vec, mv601);
                    __m256d data_j6 = _mm256_mul_pd(element_110_vec, mv620);
                    __m256d data_j7 = _mm256_mul_pd(element_110_vec, mv621);
                    __m256d data_j8 = _mm256_add_pd(data_j0, data_j2);
                    __m256d data_j9 = _mm256_add_pd(data_j1, data_j3);
                    __m256d data_j10 = _mm256_add_pd(data_j4, data_j6);
                    __m256d data_j11 = _mm256_add_pd(data_j5, data_j7);
                    __m256d data_j12 = _mm256_add_pd(data_j8, data_j10);
                    __m256d data_j13 = _mm256_add_pd(data_j9, data_j11);
                    __m256d data_j = _mm256_hadd_pd(data_j12, data_j13);
                    data_j = _mm256_permute4x64_pd(data_j, 0b11011000);
                    data_j = _mm256_hadd_pd(data_j, data_j);
                    __m128d low128j = _mm256_castpd256_pd128(data_j);
                    __m128d high128j = _mm256_extractf128_pd(data_j, 1);
                    results[7].real = _mm_cvtsd_f64(low128j);
                    results[7].imag = _mm_cvtsd_f64(high128j);

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
        // Pre-cache all unitary matrix elements for inner_qbit != 0 case
        __m256d mv00 = _mm256_set_pd(-unitary[0].imag, unitary[0].real, -unitary[0].imag, unitary[0].real);
        __m256d mv01 = _mm256_set_pd( unitary[0].real, unitary[0].imag,  unitary[0].real, unitary[0].imag);
        __m256d mv10 = _mm256_set_pd(-unitary[1].imag, unitary[1].real, -unitary[1].imag, unitary[1].real);
        __m256d mv11 = _mm256_set_pd( unitary[1].real, unitary[1].imag,  unitary[1].real, unitary[1].imag);
        __m256d mv20 = _mm256_set_pd(-unitary[2].imag, unitary[2].real, -unitary[2].imag, unitary[2].real);
        __m256d mv21 = _mm256_set_pd( unitary[2].real, unitary[2].imag,  unitary[2].real, unitary[2].imag);
        __m256d mv30 = _mm256_set_pd(-unitary[3].imag, unitary[3].real, -unitary[3].imag, unitary[3].real);
        __m256d mv31 = _mm256_set_pd( unitary[3].real, unitary[3].imag,  unitary[3].real, unitary[3].imag);
        __m256d mv40 = _mm256_set_pd(-unitary[4].imag, unitary[4].real, -unitary[4].imag, unitary[4].real);
        __m256d mv41 = _mm256_set_pd( unitary[4].real, unitary[4].imag,  unitary[4].real, unitary[4].imag);
        __m256d mv50 = _mm256_set_pd(-unitary[5].imag, unitary[5].real, -unitary[5].imag, unitary[5].real);
        __m256d mv51 = _mm256_set_pd( unitary[5].real, unitary[5].imag,  unitary[5].real, unitary[5].imag);
        __m256d mv60 = _mm256_set_pd(-unitary[6].imag, unitary[6].real, -unitary[6].imag, unitary[6].real);
        __m256d mv61 = _mm256_set_pd( unitary[6].real, unitary[6].imag,  unitary[6].real, unitary[6].imag);
        __m256d mv70 = _mm256_set_pd(-unitary[7].imag, unitary[7].real, -unitary[7].imag, unitary[7].real);
        __m256d mv71 = _mm256_set_pd( unitary[7].real, unitary[7].imag,  unitary[7].real, unitary[7].imag);
        __m256d mv80 = _mm256_set_pd(-unitary[8].imag, unitary[8].real, -unitary[8].imag, unitary[8].real);
        __m256d mv81 = _mm256_set_pd( unitary[8].real, unitary[8].imag,  unitary[8].real, unitary[8].imag);
        __m256d mv90 = _mm256_set_pd(-unitary[9].imag, unitary[9].real, -unitary[9].imag, unitary[9].real);
        __m256d mv91 = _mm256_set_pd( unitary[9].real, unitary[9].imag,  unitary[9].real, unitary[9].imag);
        __m256d mv100 = _mm256_set_pd(-unitary[10].imag, unitary[10].real, -unitary[10].imag, unitary[10].real);
        __m256d mv101 = _mm256_set_pd( unitary[10].real, unitary[10].imag,  unitary[10].real, unitary[10].imag);
        __m256d mv110 = _mm256_set_pd(-unitary[11].imag, unitary[11].real, -unitary[11].imag, unitary[11].real);
        __m256d mv111 = _mm256_set_pd( unitary[11].real, unitary[11].imag,  unitary[11].real, unitary[11].imag);
        __m256d mv120 = _mm256_set_pd(-unitary[12].imag, unitary[12].real, -unitary[12].imag, unitary[12].real);
        __m256d mv121 = _mm256_set_pd( unitary[12].real, unitary[12].imag,  unitary[12].real, unitary[12].imag);
        __m256d mv130 = _mm256_set_pd(-unitary[13].imag, unitary[13].real, -unitary[13].imag, unitary[13].real);
        __m256d mv131 = _mm256_set_pd( unitary[13].real, unitary[13].imag,  unitary[13].real, unitary[13].imag);
        __m256d mv140 = _mm256_set_pd(-unitary[14].imag, unitary[14].real, -unitary[14].imag, unitary[14].real);
        __m256d mv141 = _mm256_set_pd( unitary[14].real, unitary[14].imag,  unitary[14].real, unitary[14].imag);
        __m256d mv150 = _mm256_set_pd(-unitary[15].imag, unitary[15].real, -unitary[15].imag, unitary[15].real);
        __m256d mv151 = _mm256_set_pd( unitary[15].real, unitary[15].imag,  unitary[15].real, unitary[15].imag);
        __m256d mv160 = _mm256_set_pd(-unitary[16].imag, unitary[16].real, -unitary[16].imag, unitary[16].real);
        __m256d mv161 = _mm256_set_pd( unitary[16].real, unitary[16].imag,  unitary[16].real, unitary[16].imag);
        __m256d mv170 = _mm256_set_pd(-unitary[17].imag, unitary[17].real, -unitary[17].imag, unitary[17].real);
        __m256d mv171 = _mm256_set_pd( unitary[17].real, unitary[17].imag,  unitary[17].real, unitary[17].imag);
        __m256d mv180 = _mm256_set_pd(-unitary[18].imag, unitary[18].real, -unitary[18].imag, unitary[18].real);
        __m256d mv181 = _mm256_set_pd( unitary[18].real, unitary[18].imag,  unitary[18].real, unitary[18].imag);
        __m256d mv190 = _mm256_set_pd(-unitary[19].imag, unitary[19].real, -unitary[19].imag, unitary[19].real);
        __m256d mv191 = _mm256_set_pd( unitary[19].real, unitary[19].imag,  unitary[19].real, unitary[19].imag);
        __m256d mv200 = _mm256_set_pd(-unitary[20].imag, unitary[20].real, -unitary[20].imag, unitary[20].real);
        __m256d mv201 = _mm256_set_pd( unitary[20].real, unitary[20].imag,  unitary[20].real, unitary[20].imag);
        __m256d mv210 = _mm256_set_pd(-unitary[21].imag, unitary[21].real, -unitary[21].imag, unitary[21].real);
        __m256d mv211 = _mm256_set_pd( unitary[21].real, unitary[21].imag,  unitary[21].real, unitary[21].imag);
        __m256d mv220 = _mm256_set_pd(-unitary[22].imag, unitary[22].real, -unitary[22].imag, unitary[22].real);
        __m256d mv221 = _mm256_set_pd( unitary[22].real, unitary[22].imag,  unitary[22].real, unitary[22].imag);
        __m256d mv230 = _mm256_set_pd(-unitary[23].imag, unitary[23].real, -unitary[23].imag, unitary[23].real);
        __m256d mv231 = _mm256_set_pd( unitary[23].real, unitary[23].imag,  unitary[23].real, unitary[23].imag);
        __m256d mv240 = _mm256_set_pd(-unitary[24].imag, unitary[24].real, -unitary[24].imag, unitary[24].real);
        __m256d mv241 = _mm256_set_pd( unitary[24].real, unitary[24].imag,  unitary[24].real, unitary[24].imag);
        __m256d mv250 = _mm256_set_pd(-unitary[25].imag, unitary[25].real, -unitary[25].imag, unitary[25].real);
        __m256d mv251 = _mm256_set_pd( unitary[25].real, unitary[25].imag,  unitary[25].real, unitary[25].imag);
        __m256d mv260 = _mm256_set_pd(-unitary[26].imag, unitary[26].real, -unitary[26].imag, unitary[26].real);
        __m256d mv261 = _mm256_set_pd( unitary[26].real, unitary[26].imag,  unitary[26].real, unitary[26].imag);
        __m256d mv270 = _mm256_set_pd(-unitary[27].imag, unitary[27].real, -unitary[27].imag, unitary[27].real);
        __m256d mv271 = _mm256_set_pd( unitary[27].real, unitary[27].imag,  unitary[27].real, unitary[27].imag);
        __m256d mv280 = _mm256_set_pd(-unitary[28].imag, unitary[28].real, -unitary[28].imag, unitary[28].real);
        __m256d mv281 = _mm256_set_pd( unitary[28].real, unitary[28].imag,  unitary[28].real, unitary[28].imag);
        __m256d mv290 = _mm256_set_pd(-unitary[29].imag, unitary[29].real, -unitary[29].imag, unitary[29].real);
        __m256d mv291 = _mm256_set_pd( unitary[29].real, unitary[29].imag,  unitary[29].real, unitary[29].imag);
        __m256d mv300 = _mm256_set_pd(-unitary[30].imag, unitary[30].real, -unitary[30].imag, unitary[30].real);
        __m256d mv301 = _mm256_set_pd( unitary[30].real, unitary[30].imag,  unitary[30].real, unitary[30].imag);
        __m256d mv310 = _mm256_set_pd(-unitary[31].imag, unitary[31].real, -unitary[31].imag, unitary[31].real);
        __m256d mv311 = _mm256_set_pd( unitary[31].real, unitary[31].imag,  unitary[31].real, unitary[31].imag);
        __m256d mv320 = _mm256_set_pd(-unitary[32].imag, unitary[32].real, -unitary[32].imag, unitary[32].real);
        __m256d mv321 = _mm256_set_pd( unitary[32].real, unitary[32].imag,  unitary[32].real, unitary[32].imag);
        __m256d mv330 = _mm256_set_pd(-unitary[33].imag, unitary[33].real, -unitary[33].imag, unitary[33].real);
        __m256d mv331 = _mm256_set_pd( unitary[33].real, unitary[33].imag,  unitary[33].real, unitary[33].imag);
        __m256d mv340 = _mm256_set_pd(-unitary[34].imag, unitary[34].real, -unitary[34].imag, unitary[34].real);
        __m256d mv341 = _mm256_set_pd( unitary[34].real, unitary[34].imag,  unitary[34].real, unitary[34].imag);
        __m256d mv350 = _mm256_set_pd(-unitary[35].imag, unitary[35].real, -unitary[35].imag, unitary[35].real);
        __m256d mv351 = _mm256_set_pd( unitary[35].real, unitary[35].imag,  unitary[35].real, unitary[35].imag);
        __m256d mv360 = _mm256_set_pd(-unitary[36].imag, unitary[36].real, -unitary[36].imag, unitary[36].real);
        __m256d mv361 = _mm256_set_pd( unitary[36].real, unitary[36].imag,  unitary[36].real, unitary[36].imag);
        __m256d mv370 = _mm256_set_pd(-unitary[37].imag, unitary[37].real, -unitary[37].imag, unitary[37].real);
        __m256d mv371 = _mm256_set_pd( unitary[37].real, unitary[37].imag,  unitary[37].real, unitary[37].imag);
        __m256d mv380 = _mm256_set_pd(-unitary[38].imag, unitary[38].real, -unitary[38].imag, unitary[38].real);
        __m256d mv381 = _mm256_set_pd( unitary[38].real, unitary[38].imag,  unitary[38].real, unitary[38].imag);
        __m256d mv390 = _mm256_set_pd(-unitary[39].imag, unitary[39].real, -unitary[39].imag, unitary[39].real);
        __m256d mv391 = _mm256_set_pd( unitary[39].real, unitary[39].imag,  unitary[39].real, unitary[39].imag);
        __m256d mv400 = _mm256_set_pd(-unitary[40].imag, unitary[40].real, -unitary[40].imag, unitary[40].real);
        __m256d mv401 = _mm256_set_pd( unitary[40].real, unitary[40].imag,  unitary[40].real, unitary[40].imag);
        __m256d mv410 = _mm256_set_pd(-unitary[41].imag, unitary[41].real, -unitary[41].imag, unitary[41].real);
        __m256d mv411 = _mm256_set_pd( unitary[41].real, unitary[41].imag,  unitary[41].real, unitary[41].imag);
        __m256d mv420 = _mm256_set_pd(-unitary[42].imag, unitary[42].real, -unitary[42].imag, unitary[42].real);
        __m256d mv421 = _mm256_set_pd( unitary[42].real, unitary[42].imag,  unitary[42].real, unitary[42].imag);
        __m256d mv430 = _mm256_set_pd(-unitary[43].imag, unitary[43].real, -unitary[43].imag, unitary[43].real);
        __m256d mv431 = _mm256_set_pd( unitary[43].real, unitary[43].imag,  unitary[43].real, unitary[43].imag);
        __m256d mv440 = _mm256_set_pd(-unitary[44].imag, unitary[44].real, -unitary[44].imag, unitary[44].real);
        __m256d mv441 = _mm256_set_pd( unitary[44].real, unitary[44].imag,  unitary[44].real, unitary[44].imag);
        __m256d mv450 = _mm256_set_pd(-unitary[45].imag, unitary[45].real, -unitary[45].imag, unitary[45].real);
        __m256d mv451 = _mm256_set_pd( unitary[45].real, unitary[45].imag,  unitary[45].real, unitary[45].imag);
        __m256d mv460 = _mm256_set_pd(-unitary[46].imag, unitary[46].real, -unitary[46].imag, unitary[46].real);
        __m256d mv461 = _mm256_set_pd( unitary[46].real, unitary[46].imag,  unitary[46].real, unitary[46].imag);
        __m256d mv470 = _mm256_set_pd(-unitary[47].imag, unitary[47].real, -unitary[47].imag, unitary[47].real);
        __m256d mv471 = _mm256_set_pd( unitary[47].real, unitary[47].imag,  unitary[47].real, unitary[47].imag);
        __m256d mv480 = _mm256_set_pd(-unitary[48].imag, unitary[48].real, -unitary[48].imag, unitary[48].real);
        __m256d mv481 = _mm256_set_pd( unitary[48].real, unitary[48].imag,  unitary[48].real, unitary[48].imag);
        __m256d mv490 = _mm256_set_pd(-unitary[49].imag, unitary[49].real, -unitary[49].imag, unitary[49].real);
        __m256d mv491 = _mm256_set_pd( unitary[49].real, unitary[49].imag,  unitary[49].real, unitary[49].imag);
        __m256d mv500 = _mm256_set_pd(-unitary[50].imag, unitary[50].real, -unitary[50].imag, unitary[50].real);
        __m256d mv501 = _mm256_set_pd( unitary[50].real, unitary[50].imag,  unitary[50].real, unitary[50].imag);
        __m256d mv510 = _mm256_set_pd(-unitary[51].imag, unitary[51].real, -unitary[51].imag, unitary[51].real);
        __m256d mv511 = _mm256_set_pd( unitary[51].real, unitary[51].imag,  unitary[51].real, unitary[51].imag);
        __m256d mv520 = _mm256_set_pd(-unitary[52].imag, unitary[52].real, -unitary[52].imag, unitary[52].real);
        __m256d mv521 = _mm256_set_pd( unitary[52].real, unitary[52].imag,  unitary[52].real, unitary[52].imag);
        __m256d mv530 = _mm256_set_pd(-unitary[53].imag, unitary[53].real, -unitary[53].imag, unitary[53].real);
        __m256d mv531 = _mm256_set_pd( unitary[53].real, unitary[53].imag,  unitary[53].real, unitary[53].imag);
        __m256d mv540 = _mm256_set_pd(-unitary[54].imag, unitary[54].real, -unitary[54].imag, unitary[54].real);
        __m256d mv541 = _mm256_set_pd( unitary[54].real, unitary[54].imag,  unitary[54].real, unitary[54].imag);
        __m256d mv550 = _mm256_set_pd(-unitary[55].imag, unitary[55].real, -unitary[55].imag, unitary[55].real);
        __m256d mv551 = _mm256_set_pd( unitary[55].real, unitary[55].imag,  unitary[55].real, unitary[55].imag);
        __m256d mv560 = _mm256_set_pd(-unitary[56].imag, unitary[56].real, -unitary[56].imag, unitary[56].real);
        __m256d mv561 = _mm256_set_pd( unitary[56].real, unitary[56].imag,  unitary[56].real, unitary[56].imag);
        __m256d mv570 = _mm256_set_pd(-unitary[57].imag, unitary[57].real, -unitary[57].imag, unitary[57].real);
        __m256d mv571 = _mm256_set_pd( unitary[57].real, unitary[57].imag,  unitary[57].real, unitary[57].imag);
        __m256d mv580 = _mm256_set_pd(-unitary[58].imag, unitary[58].real, -unitary[58].imag, unitary[58].real);
        __m256d mv581 = _mm256_set_pd( unitary[58].real, unitary[58].imag,  unitary[58].real, unitary[58].imag);
        __m256d mv590 = _mm256_set_pd(-unitary[59].imag, unitary[59].real, -unitary[59].imag, unitary[59].real);
        __m256d mv591 = _mm256_set_pd( unitary[59].real, unitary[59].imag,  unitary[59].real, unitary[59].imag);
        __m256d mv600 = _mm256_set_pd(-unitary[60].imag, unitary[60].real, -unitary[60].imag, unitary[60].real);
        __m256d mv601 = _mm256_set_pd( unitary[60].real, unitary[60].imag,  unitary[60].real, unitary[60].imag);
        __m256d mv610 = _mm256_set_pd(-unitary[61].imag, unitary[61].real, -unitary[61].imag, unitary[61].real);
        __m256d mv611 = _mm256_set_pd( unitary[61].real, unitary[61].imag,  unitary[61].real, unitary[61].imag);
        __m256d mv620 = _mm256_set_pd(-unitary[62].imag, unitary[62].real, -unitary[62].imag, unitary[62].real);
        __m256d mv621 = _mm256_set_pd( unitary[62].real, unitary[62].imag,  unitary[62].real, unitary[62].imag);
        __m256d mv630 = _mm256_set_pd(-unitary[63].imag, unitary[63].real, -unitary[63].imag, unitary[63].real);
        __m256d mv631 = _mm256_set_pd( unitary[63].real, unitary[63].imag,  unitary[63].real, unitary[63].imag);

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

                        // Row 0 computation
                        __m256d data_u0 = _mm256_mul_pd(element_outer_vec, mv00);
                        __m256d data_u1 = _mm256_mul_pd(element_inner_vec, mv10);
                        __m256d data_u2 = _mm256_mul_pd(element_outer_vec, mv01);
                        __m256d data_u3 = _mm256_mul_pd(element_inner_vec, mv11);
                        __m256d data_u4 = _mm256_mul_pd(element_middle_vec, mv20);
                        __m256d data_u5 = _mm256_mul_pd(element_middle_inner_vec, mv30);
                        __m256d data_u6 = _mm256_mul_pd(element_middle_vec, mv21);
                        __m256d data_u7 = _mm256_mul_pd(element_middle_inner_vec, mv31);
                        __m256d data_u8 = _mm256_mul_pd(element_outer_pair_vec, mv40);
                        __m256d data_u9 = _mm256_mul_pd(element_inner_pair_vec, mv50);
                        __m256d data_u10 = _mm256_mul_pd(element_outer_pair_vec, mv41);
                        __m256d data_u11 = _mm256_mul_pd(element_inner_pair_vec, mv51);
                        __m256d data_u12 = _mm256_mul_pd(element_middle_pair_vec, mv60);
                        __m256d data_u13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv70);
                        __m256d data_u14 = _mm256_mul_pd(element_middle_pair_vec, mv61);
                        __m256d data_u15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv71);
                        __m256d data_u16 = _mm256_hadd_pd(data_u0, data_u2);
                        __m256d data_u17 = _mm256_hadd_pd(data_u1, data_u3);
                        __m256d data_u18 = _mm256_hadd_pd(data_u4, data_u6);
                        __m256d data_u19 = _mm256_hadd_pd(data_u5, data_u7);
                        __m256d data_u20 = _mm256_hadd_pd(data_u8, data_u10);
                        __m256d data_u21 = _mm256_hadd_pd(data_u9, data_u11);
                        __m256d data_u22 = _mm256_hadd_pd(data_u12, data_u14);
                        __m256d data_u23 = _mm256_hadd_pd(data_u13, data_u15);
                        __m256d data_u = _mm256_add_pd(data_u16, data_u17);
                        data_u = _mm256_add_pd(data_u, data_u18);
                        data_u = _mm256_add_pd(data_u, data_u19);
                        data_u = _mm256_add_pd(data_u, data_u20);
                        data_u = _mm256_add_pd(data_u, data_u21);
                        data_u = _mm256_add_pd(data_u, data_u22);
                        data_u = _mm256_add_pd(data_u, data_u23);

                        // Row 1 computation
                        __m256d data_d0 = _mm256_mul_pd(element_outer_vec, mv80);
                        __m256d data_d1 = _mm256_mul_pd(element_inner_vec, mv90);
                        __m256d data_d2 = _mm256_mul_pd(element_outer_vec, mv81);
                        __m256d data_d3 = _mm256_mul_pd(element_inner_vec, mv91);
                        __m256d data_d4 = _mm256_mul_pd(element_middle_vec, mv100);
                        __m256d data_d5 = _mm256_mul_pd(element_middle_inner_vec, mv110);
                        __m256d data_d6 = _mm256_mul_pd(element_middle_vec, mv101);
                        __m256d data_d7 = _mm256_mul_pd(element_middle_inner_vec, mv111);
                        __m256d data_d8 = _mm256_mul_pd(element_outer_pair_vec, mv120);
                        __m256d data_d9 = _mm256_mul_pd(element_inner_pair_vec, mv130);
                        __m256d data_d10 = _mm256_mul_pd(element_outer_pair_vec, mv121);
                        __m256d data_d11 = _mm256_mul_pd(element_inner_pair_vec, mv131);
                        __m256d data_d12 = _mm256_mul_pd(element_middle_pair_vec, mv140);
                        __m256d data_d13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv150);
                        __m256d data_d14 = _mm256_mul_pd(element_middle_pair_vec, mv141);
                        __m256d data_d15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv151);
                        __m256d data_d16 = _mm256_hadd_pd(data_d0, data_d2);
                        __m256d data_d17 = _mm256_hadd_pd(data_d1, data_d3);
                        __m256d data_d18 = _mm256_hadd_pd(data_d4, data_d6);
                        __m256d data_d19 = _mm256_hadd_pd(data_d5, data_d7);
                        __m256d data_d20 = _mm256_hadd_pd(data_d8, data_d10);
                        __m256d data_d21 = _mm256_hadd_pd(data_d9, data_d11);
                        __m256d data_d22 = _mm256_hadd_pd(data_d12, data_d14);
                        __m256d data_d23 = _mm256_hadd_pd(data_d13, data_d15);
                        __m256d data_d = _mm256_add_pd(data_d16, data_d17);
                        data_d = _mm256_add_pd(data_d, data_d18);
                        data_d = _mm256_add_pd(data_d, data_d19);
                        data_d = _mm256_add_pd(data_d, data_d20);
                        data_d = _mm256_add_pd(data_d, data_d21);
                        data_d = _mm256_add_pd(data_d, data_d22);
                        data_d = _mm256_add_pd(data_d, data_d23);

                        // Row 2 computation
                        __m256d data_e0 = _mm256_mul_pd(element_outer_vec, mv160);
                        __m256d data_e1 = _mm256_mul_pd(element_inner_vec, mv170);
                        __m256d data_e2 = _mm256_mul_pd(element_outer_vec, mv161);
                        __m256d data_e3 = _mm256_mul_pd(element_inner_vec, mv171);
                        __m256d data_e4 = _mm256_mul_pd(element_middle_vec, mv180);
                        __m256d data_e5 = _mm256_mul_pd(element_middle_inner_vec, mv190);
                        __m256d data_e6 = _mm256_mul_pd(element_middle_vec, mv181);
                        __m256d data_e7 = _mm256_mul_pd(element_middle_inner_vec, mv191);
                        __m256d data_e8 = _mm256_mul_pd(element_outer_pair_vec, mv200);
                        __m256d data_e9 = _mm256_mul_pd(element_inner_pair_vec, mv210);
                        __m256d data_e10 = _mm256_mul_pd(element_outer_pair_vec, mv201);
                        __m256d data_e11 = _mm256_mul_pd(element_inner_pair_vec, mv211);
                        __m256d data_e12 = _mm256_mul_pd(element_middle_pair_vec, mv220);
                        __m256d data_e13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv230);
                        __m256d data_e14 = _mm256_mul_pd(element_middle_pair_vec, mv221);
                        __m256d data_e15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv231);
                        __m256d data_e16 = _mm256_hadd_pd(data_e0, data_e2);
                        __m256d data_e17 = _mm256_hadd_pd(data_e1, data_e3);
                        __m256d data_e18 = _mm256_hadd_pd(data_e4, data_e6);
                        __m256d data_e19 = _mm256_hadd_pd(data_e5, data_e7);
                        __m256d data_e20 = _mm256_hadd_pd(data_e8, data_e10);
                        __m256d data_e21 = _mm256_hadd_pd(data_e9, data_e11);
                        __m256d data_e22 = _mm256_hadd_pd(data_e12, data_e14);
                        __m256d data_e23 = _mm256_hadd_pd(data_e13, data_e15);
                        __m256d data_e = _mm256_add_pd(data_e16, data_e17);
                        data_e = _mm256_add_pd(data_e, data_e18);
                        data_e = _mm256_add_pd(data_e, data_e19);
                        data_e = _mm256_add_pd(data_e, data_e20);
                        data_e = _mm256_add_pd(data_e, data_e21);
                        data_e = _mm256_add_pd(data_e, data_e22);
                        data_e = _mm256_add_pd(data_e, data_e23);

                        // Row 3 computation
                        __m256d data_f0 = _mm256_mul_pd(element_outer_vec, mv240);
                        __m256d data_f1 = _mm256_mul_pd(element_inner_vec, mv250);
                        __m256d data_f2 = _mm256_mul_pd(element_outer_vec, mv241);
                        __m256d data_f3 = _mm256_mul_pd(element_inner_vec, mv251);
                        __m256d data_f4 = _mm256_mul_pd(element_middle_vec, mv260);
                        __m256d data_f5 = _mm256_mul_pd(element_middle_inner_vec, mv270);
                        __m256d data_f6 = _mm256_mul_pd(element_middle_vec, mv261);
                        __m256d data_f7 = _mm256_mul_pd(element_middle_inner_vec, mv271);
                        __m256d data_f8 = _mm256_mul_pd(element_outer_pair_vec, mv280);
                        __m256d data_f9 = _mm256_mul_pd(element_inner_pair_vec, mv290);
                        __m256d data_f10 = _mm256_mul_pd(element_outer_pair_vec, mv281);
                        __m256d data_f11 = _mm256_mul_pd(element_inner_pair_vec, mv291);
                        __m256d data_f12 = _mm256_mul_pd(element_middle_pair_vec, mv300);
                        __m256d data_f13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv310);
                        __m256d data_f14 = _mm256_mul_pd(element_middle_pair_vec, mv301);
                        __m256d data_f15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv311);
                        __m256d data_f16 = _mm256_hadd_pd(data_f0, data_f2);
                        __m256d data_f17 = _mm256_hadd_pd(data_f1, data_f3);
                        __m256d data_f18 = _mm256_hadd_pd(data_f4, data_f6);
                        __m256d data_f19 = _mm256_hadd_pd(data_f5, data_f7);
                        __m256d data_f20 = _mm256_hadd_pd(data_f8, data_f10);
                        __m256d data_f21 = _mm256_hadd_pd(data_f9, data_f11);
                        __m256d data_f22 = _mm256_hadd_pd(data_f12, data_f14);
                        __m256d data_f23 = _mm256_hadd_pd(data_f13, data_f15);
                        __m256d data_f = _mm256_add_pd(data_f16, data_f17);
                        data_f = _mm256_add_pd(data_f, data_f18);
                        data_f = _mm256_add_pd(data_f, data_f19);
                        data_f = _mm256_add_pd(data_f, data_f20);
                        data_f = _mm256_add_pd(data_f, data_f21);
                        data_f = _mm256_add_pd(data_f, data_f22);
                        data_f = _mm256_add_pd(data_f, data_f23);

                        // Row 4 computation
                        __m256d data_g0 = _mm256_mul_pd(element_outer_vec, mv320);
                        __m256d data_g1 = _mm256_mul_pd(element_inner_vec, mv330);
                        __m256d data_g2 = _mm256_mul_pd(element_outer_vec, mv321);
                        __m256d data_g3 = _mm256_mul_pd(element_inner_vec, mv331);
                        __m256d data_g4 = _mm256_mul_pd(element_middle_vec, mv340);
                        __m256d data_g5 = _mm256_mul_pd(element_middle_inner_vec, mv350);
                        __m256d data_g6 = _mm256_mul_pd(element_middle_vec, mv341);
                        __m256d data_g7 = _mm256_mul_pd(element_middle_inner_vec, mv351);
                        __m256d data_g8 = _mm256_mul_pd(element_outer_pair_vec, mv360);
                        __m256d data_g9 = _mm256_mul_pd(element_inner_pair_vec, mv370);
                        __m256d data_g10 = _mm256_mul_pd(element_outer_pair_vec, mv361);
                        __m256d data_g11 = _mm256_mul_pd(element_inner_pair_vec, mv371);
                        __m256d data_g12 = _mm256_mul_pd(element_middle_pair_vec, mv380);
                        __m256d data_g13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv390);
                        __m256d data_g14 = _mm256_mul_pd(element_middle_pair_vec, mv381);
                        __m256d data_g15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv391);
                        __m256d data_g16 = _mm256_hadd_pd(data_g0, data_g2);
                        __m256d data_g17 = _mm256_hadd_pd(data_g1, data_g3);
                        __m256d data_g18 = _mm256_hadd_pd(data_g4, data_g6);
                        __m256d data_g19 = _mm256_hadd_pd(data_g5, data_g7);
                        __m256d data_g20 = _mm256_hadd_pd(data_g8, data_g10);
                        __m256d data_g21 = _mm256_hadd_pd(data_g9, data_g11);
                        __m256d data_g22 = _mm256_hadd_pd(data_g12, data_g14);
                        __m256d data_g23 = _mm256_hadd_pd(data_g13, data_g15);
                        __m256d data_g = _mm256_add_pd(data_g16, data_g17);
                        data_g = _mm256_add_pd(data_g, data_g18);
                        data_g = _mm256_add_pd(data_g, data_g19);
                        data_g = _mm256_add_pd(data_g, data_g20);
                        data_g = _mm256_add_pd(data_g, data_g21);
                        data_g = _mm256_add_pd(data_g, data_g22);
                        data_g = _mm256_add_pd(data_g, data_g23);

                        // Row 5 computation
                        __m256d data_h0 = _mm256_mul_pd(element_outer_vec, mv400);
                        __m256d data_h1 = _mm256_mul_pd(element_inner_vec, mv410);
                        __m256d data_h2 = _mm256_mul_pd(element_outer_vec, mv401);
                        __m256d data_h3 = _mm256_mul_pd(element_inner_vec, mv411);
                        __m256d data_h4 = _mm256_mul_pd(element_middle_vec, mv420);
                        __m256d data_h5 = _mm256_mul_pd(element_middle_inner_vec, mv430);
                        __m256d data_h6 = _mm256_mul_pd(element_middle_vec, mv421);
                        __m256d data_h7 = _mm256_mul_pd(element_middle_inner_vec, mv431);
                        __m256d data_h8 = _mm256_mul_pd(element_outer_pair_vec, mv440);
                        __m256d data_h9 = _mm256_mul_pd(element_inner_pair_vec, mv450);
                        __m256d data_h10 = _mm256_mul_pd(element_outer_pair_vec, mv441);
                        __m256d data_h11 = _mm256_mul_pd(element_inner_pair_vec, mv451);
                        __m256d data_h12 = _mm256_mul_pd(element_middle_pair_vec, mv460);
                        __m256d data_h13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv470);
                        __m256d data_h14 = _mm256_mul_pd(element_middle_pair_vec, mv461);
                        __m256d data_h15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv471);
                        __m256d data_h16 = _mm256_hadd_pd(data_h0, data_h2);
                        __m256d data_h17 = _mm256_hadd_pd(data_h1, data_h3);
                        __m256d data_h18 = _mm256_hadd_pd(data_h4, data_h6);
                        __m256d data_h19 = _mm256_hadd_pd(data_h5, data_h7);
                        __m256d data_h20 = _mm256_hadd_pd(data_h8, data_h10);
                        __m256d data_h21 = _mm256_hadd_pd(data_h9, data_h11);
                        __m256d data_h22 = _mm256_hadd_pd(data_h12, data_h14);
                        __m256d data_h23 = _mm256_hadd_pd(data_h13, data_h15);
                        __m256d data_h = _mm256_add_pd(data_h16, data_h17);
                        data_h = _mm256_add_pd(data_h, data_h18);
                        data_h = _mm256_add_pd(data_h, data_h19);
                        data_h = _mm256_add_pd(data_h, data_h20);
                        data_h = _mm256_add_pd(data_h, data_h21);
                        data_h = _mm256_add_pd(data_h, data_h22);
                        data_h = _mm256_add_pd(data_h, data_h23);

                        // Row 6 computation
                        __m256d data_i0 = _mm256_mul_pd(element_outer_vec, mv480);
                        __m256d data_i1 = _mm256_mul_pd(element_inner_vec, mv490);
                        __m256d data_i2 = _mm256_mul_pd(element_outer_vec, mv481);
                        __m256d data_i3 = _mm256_mul_pd(element_inner_vec, mv491);
                        __m256d data_i4 = _mm256_mul_pd(element_middle_vec, mv500);
                        __m256d data_i5 = _mm256_mul_pd(element_middle_inner_vec, mv510);
                        __m256d data_i6 = _mm256_mul_pd(element_middle_vec, mv501);
                        __m256d data_i7 = _mm256_mul_pd(element_middle_inner_vec, mv511);
                        __m256d data_i8 = _mm256_mul_pd(element_outer_pair_vec, mv520);
                        __m256d data_i9 = _mm256_mul_pd(element_inner_pair_vec, mv530);
                        __m256d data_i10 = _mm256_mul_pd(element_outer_pair_vec, mv521);
                        __m256d data_i11 = _mm256_mul_pd(element_inner_pair_vec, mv531);
                        __m256d data_i12 = _mm256_mul_pd(element_middle_pair_vec, mv540);
                        __m256d data_i13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv550);
                        __m256d data_i14 = _mm256_mul_pd(element_middle_pair_vec, mv541);
                        __m256d data_i15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv551);
                        __m256d data_i16 = _mm256_hadd_pd(data_i0, data_i2);
                        __m256d data_i17 = _mm256_hadd_pd(data_i1, data_i3);
                        __m256d data_i18 = _mm256_hadd_pd(data_i4, data_i6);
                        __m256d data_i19 = _mm256_hadd_pd(data_i5, data_i7);
                        __m256d data_i20 = _mm256_hadd_pd(data_i8, data_i10);
                        __m256d data_i21 = _mm256_hadd_pd(data_i9, data_i11);
                        __m256d data_i22 = _mm256_hadd_pd(data_i12, data_i14);
                        __m256d data_i23 = _mm256_hadd_pd(data_i13, data_i15);
                        __m256d data_i = _mm256_add_pd(data_i16, data_i17);
                        data_i = _mm256_add_pd(data_i, data_i18);
                        data_i = _mm256_add_pd(data_i, data_i19);
                        data_i = _mm256_add_pd(data_i, data_i20);
                        data_i = _mm256_add_pd(data_i, data_i21);
                        data_i = _mm256_add_pd(data_i, data_i22);
                        data_i = _mm256_add_pd(data_i, data_i23);

                        // Row 7 computation
                        __m256d data_j0 = _mm256_mul_pd(element_outer_vec, mv560);
                        __m256d data_j1 = _mm256_mul_pd(element_inner_vec, mv570);
                        __m256d data_j2 = _mm256_mul_pd(element_outer_vec, mv561);
                        __m256d data_j3 = _mm256_mul_pd(element_inner_vec, mv571);
                        __m256d data_j4 = _mm256_mul_pd(element_middle_vec, mv580);
                        __m256d data_j5 = _mm256_mul_pd(element_middle_inner_vec, mv590);
                        __m256d data_j6 = _mm256_mul_pd(element_middle_vec, mv581);
                        __m256d data_j7 = _mm256_mul_pd(element_middle_inner_vec, mv591);
                        __m256d data_j8 = _mm256_mul_pd(element_outer_pair_vec, mv600);
                        __m256d data_j9 = _mm256_mul_pd(element_inner_pair_vec, mv610);
                        __m256d data_j10 = _mm256_mul_pd(element_outer_pair_vec, mv601);
                        __m256d data_j11 = _mm256_mul_pd(element_inner_pair_vec, mv611);
                        __m256d data_j12 = _mm256_mul_pd(element_middle_pair_vec, mv620);
                        __m256d data_j13 = _mm256_mul_pd(element_middle_inner_pair_vec, mv630);
                        __m256d data_j14 = _mm256_mul_pd(element_middle_pair_vec, mv621);
                        __m256d data_j15 = _mm256_mul_pd(element_middle_inner_pair_vec, mv631);
                        __m256d data_j16 = _mm256_hadd_pd(data_j0, data_j2);
                        __m256d data_j17 = _mm256_hadd_pd(data_j1, data_j3);
                        __m256d data_j18 = _mm256_hadd_pd(data_j4, data_j6);
                        __m256d data_j19 = _mm256_hadd_pd(data_j5, data_j7);
                        __m256d data_j20 = _mm256_hadd_pd(data_j8, data_j10);
                        __m256d data_j21 = _mm256_hadd_pd(data_j9, data_j11);
                        __m256d data_j22 = _mm256_hadd_pd(data_j12, data_j14);
                        __m256d data_j23 = _mm256_hadd_pd(data_j13, data_j15);
                        __m256d data_j = _mm256_add_pd(data_j16, data_j17);
                        data_j = _mm256_add_pd(data_j, data_j18);
                        data_j = _mm256_add_pd(data_j, data_j19);
                        data_j = _mm256_add_pd(data_j, data_j20);
                        data_j = _mm256_add_pd(data_j, data_j21);
                        data_j = _mm256_add_pd(data_j, data_j22);
                        data_j = _mm256_add_pd(data_j, data_j23);

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
        
        QGD_Complex16 results[16];
        
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
        
        __m256d outer_inner_vec     = get_AVX_vector(element_outer, element_inner);
        __m256d middle1_inner_vec   = get_AVX_vector(element_middle1, element_middle1_inner);
        __m256d middle2_inner_vec   = get_AVX_vector(element_middle2, element_middle2_inner);
        __m256d middle12_inner_vec  = get_AVX_vector(element_middle12, element_middle12_inner);
        __m256d outer_inner_pair_vec     = get_AVX_vector(element_outer_pair, element_inner_pair);
        __m256d middle1_inner_pair_vec   = get_AVX_vector(element_middle1_pair, element_middle1_inner_pair);
        __m256d middle2_inner_pair_vec   = get_AVX_vector(element_middle2_pair, element_middle2_inner_pair);
        __m256d middle12_inner_pair_vec  = get_AVX_vector(element_middle12_pair, element_middle12_inner_pair);

        for (int mult_idx = 0; mult_idx < 16; mult_idx++) {
            double* unitary_row_1 = (double*)unitary.get_data() + 32*mult_idx;
            double* unitary_row_2 = unitary_row_1 + 4;
            double* unitary_row_3 = unitary_row_1 + 8;
            double* unitary_row_4 = unitary_row_1 + 12;
            double* unitary_row_5 = unitary_row_1 + 16;
            double* unitary_row_6 = unitary_row_1 + 20;
            double* unitary_row_7 = unitary_row_1 + 24;
            double* unitary_row_8 = unitary_row_1 + 28;
            
            __m256d result_upper_vec        = complex_mult_AVX(outer_inner_vec,     _mm256_loadu_pd(unitary_row_1), neg);
            __m256d result_upper_middle1_vec= complex_mult_AVX(middle1_inner_vec,   _mm256_loadu_pd(unitary_row_2), neg);
            __m256d result_upper_middle2_vec= complex_mult_AVX(middle2_inner_vec,   _mm256_loadu_pd(unitary_row_3), neg);
            __m256d result_upper_middle12_vec=complex_mult_AVX(middle12_inner_vec,  _mm256_loadu_pd(unitary_row_4), neg);
            __m256d result_lower_vec        = complex_mult_AVX(outer_inner_pair_vec,_mm256_loadu_pd(unitary_row_5), neg);
            __m256d result_lower_middle1_vec= complex_mult_AVX(middle1_inner_pair_vec,_mm256_loadu_pd(unitary_row_6), neg);
            __m256d result_lower_middle2_vec= complex_mult_AVX(middle2_inner_pair_vec,_mm256_loadu_pd(unitary_row_7), neg);
            __m256d result_lower_middle12_vec=complex_mult_AVX(middle12_inner_pair_vec,_mm256_loadu_pd(unitary_row_8), neg);
            
            __m256d result1_vec = _mm256_hadd_pd(result_upper_vec,        result_upper_middle1_vec);
            __m256d result2_vec = _mm256_hadd_pd(result_upper_middle2_vec, result_upper_middle12_vec);
            __m256d result3_vec = _mm256_hadd_pd(result_lower_vec,        result_lower_middle1_vec);
            __m256d result4_vec = _mm256_hadd_pd(result_lower_middle2_vec, result_lower_middle12_vec);
            __m256d result5_vec = _mm256_hadd_pd(result1_vec, result2_vec);
            __m256d result6_vec = _mm256_hadd_pd(result3_vec, result4_vec);
            __m256d result_vec  = _mm256_hadd_pd(result5_vec, result6_vec);
            result_vec          = _mm256_hadd_pd(result_vec, result_vec);
            
            double* result = (double*)&result_vec;
            results[mult_idx].real = result[0];
            results[mult_idx].imag = result[2];
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

