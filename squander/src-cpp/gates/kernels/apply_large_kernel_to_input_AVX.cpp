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
      }
      case 3:{
          apply_3qbit_kernel_to_state_vector_input_AVX(unitary,input,involved_qbits,matrix_size);
      }
      case 4:{
              apply_4qbit_kernel_to_state_vector_input_AVX(unitary,input,involved_qbits,matrix_size);
      }
      }
  }
  else{
      apply_2qbit_kernel_to_matrix_input_parallel_AVX_OpenMP(unitary, input, involved_qbits, matrix_size);
  }
}

inline void get_block_indices(
    int N, const std::vector<int> &target_qubits, const std::vector<int> &non_targets,
    int iter_idx, std::vector<int> &indices
) {
    const int n = (int)target_qubits.size();
    const int block_size = 1 << n;

    // base: put iter_idx bits into non-target positions (little-endian)
    int base = 0;
    for (std::size_t i = 0; i < non_targets.size(); ++i) {
        if (iter_idx & (1 << i)) base |= (1 << non_targets[i]);
    }

    // enumerate local states k in 0..2^n-1 using the GIVEN target order
    for (int k = 0; k < block_size; ++k) {
        int idx = base;
        for (int bit = 0; bit < n; ++bit) {
            if (k & (1 << bit)) idx |= (1 << target_qubits[bit]); // crucial: bit->target_qubits[bit]
        }
        indices[k] = idx;
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


    __m256d* mv_xy = construct_mv_xy_vectors(gate_kernel_unitary, gate_kernel_unitary.rows);
    std::vector<int> indices(block_size);
    std::vector<double> new_block_real(block_size,0.0);
    std::vector<double> new_block_imag(block_size,0.0);
    
    for (int iter_idx = 0; iter_idx < num_blocks; iter_idx++) {
        get_block_indices(qubit_num, involved_qbits, non_targets, iter_idx, indices);
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
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<input.rows; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
            if (inner_qbit==0){
            	for (int idx=0; idx<index_step_inner; idx++){
            	

			    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			    int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
	        	int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			    int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			    double results[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
			    
                double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
			    
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
			    input[current_idx_outer_loc].real = results[0];
			    input[current_idx_outer_loc].imag = results[1];
			    input[current_idx_inner_loc].real = results[2];
			    input[current_idx_inner_loc].imag = results[3];
			    input[current_idx_outer_pair_loc].real = results[4];
			    input[current_idx_outer_pair_loc].imag = results[5];
			    input[current_idx_inner_pair_loc].real = results[6];
			    input[current_idx_inner_pair_loc].imag = results[7];
            	}
            }
            else {

                for (int idx=0; idx<index_step_inner; idx=idx+2){
                
		    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
		            
                    double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                    double* element_inner = element_outer + 2 * index_step_inner;
                    
                    double* element_outer_pair = element_outer + 2 * index_step_outer;
                    double* element_inner_pair = element_outer_pair + 2 * index_step_inner;
                    
                    double* indices[4] = {element_outer,element_inner,element_outer_pair,element_inner_pair};
                    
                    __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                    __m256d element_inner_vec = _mm256_loadu_pd(element_inner);

                    __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                    __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                    
                    __m256d result_vecs[4];
                    
                    for (int mult_idx=0; mult_idx<4; mult_idx++){
		                double* unitary_col_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
		                double* unitary_col_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
		                
		                __m256d unitary_col_01_vec     = _mm256_loadu_pd(unitary_col_01); // a,b,c,d
		                unitary_col_01_vec             = _mm256_permute4x64_pd(unitary_col_01_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_0_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_1_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b1111); // c, c, d, d
		                unitary_col_0_vec              = _mm256_permute4x64_pd(unitary_col_0_vec,0b11011000); // a, b, a, b
		                unitary_col_1_vec              = _mm256_permute4x64_pd(unitary_col_1_vec,0b11011000); // c, d, c, d

		                __m256d unitary_col_23_vec     = _mm256_loadu_pd(unitary_col_23); // a,b,c,d
		                unitary_col_23_vec             = _mm256_permute4x64_pd(unitary_col_23_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_2_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_3_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b1111); // c, c, d, d
		                unitary_col_2_vec              = _mm256_permute4x64_pd(unitary_col_2_vec,0b11011000); // a, a, b, b
		                unitary_col_3_vec              = _mm256_permute4x64_pd(unitary_col_3_vec,0b11011000); // c, c, d, d
		                
                        __m256d outer_vec_3            = _mm256_mul_pd(element_outer_vec, unitary_col_0_vec);
                        __m256d unitary_col_0_switched = _mm256_permute_pd(unitary_col_0_vec, 0x5);
                        unitary_col_0_switched         = _mm256_mul_pd( unitary_col_0_switched, neg);
                        __m256d outer_vec_4            = _mm256_mul_pd( element_outer_vec, unitary_col_0_switched);
                        __m256d result_outer_vec       = _mm256_hsub_pd( outer_vec_3, outer_vec_4);

                        
                        __m256d inner_vec_3            = _mm256_mul_pd(element_inner_vec, unitary_col_1_vec);
                        __m256d unitary_col_1_switched = _mm256_permute_pd(unitary_col_1_vec, 0x5);
                        unitary_col_1_switched         = _mm256_mul_pd( unitary_col_1_switched, neg);
                        __m256d inner_vec_4            = _mm256_mul_pd( element_inner_vec, unitary_col_1_switched);
                        __m256d result_inner_vec       = _mm256_hsub_pd( inner_vec_3, inner_vec_4);
                        
                        __m256d outer_pair_vec_3       = _mm256_mul_pd(element_outer_pair_vec, unitary_col_2_vec);
                        __m256d unitary_col_2_switched = _mm256_permute_pd(unitary_col_2_vec, 0x5);
                        unitary_col_2_switched         = _mm256_mul_pd( unitary_col_2_switched, neg);
                        __m256d outer_pair_vec_4       = _mm256_mul_pd( element_outer_pair_vec, unitary_col_2_switched);
                        __m256d result_outer_pair_vec  = _mm256_hsub_pd( outer_pair_vec_3, outer_pair_vec_4);

                        
                        __m256d inner_pair_vec_3       = _mm256_mul_pd(element_inner_pair_vec, unitary_col_3_vec);
                        __m256d unitary_col_3_switched = _mm256_permute_pd(unitary_col_3_vec, 0x5);
                        unitary_col_3_switched         = _mm256_mul_pd( unitary_col_3_switched, neg);
                        __m256d inner_pair_vec_4       = _mm256_mul_pd( element_inner_pair_vec, unitary_col_3_switched);
                        __m256d result_inner_pair_vec  = _mm256_hsub_pd( inner_pair_vec_3, inner_pair_vec_4);
                        
                        __m256d result_vec    = _mm256_add_pd(result_outer_vec,result_inner_vec);
                        result_vec            = _mm256_add_pd(result_vec,result_outer_pair_vec);
                        result_vec            = _mm256_add_pd(result_vec,result_inner_pair_vec);
                        result_vecs[mult_idx] = result_vec;
                    }
                    for (int row_idx=0; row_idx<4;row_idx++){
                        _mm256_storeu_pd(indices[row_idx], result_vecs[row_idx]);
                    }

                }
            
            }
        }
        current_idx = current_idx + (index_step_outer << 1);
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
void apply_2qbit_kernel_to_matrix_input_parallel_AVX_OpenMP(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    int inner_qbit = involved_qbits[0];
    int outer_qbit = involved_qbits[1];
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
@brief Call to apply kernel to apply two qubit gate kernel on a state vector using AVX and TBB
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param inner_qbit The lower significance qubit (little endian convention)
@param outer_qbit The higher significance qubit (little endian convention)
@param matrix_size The size of the input
*/
void apply_2qbit_kernel_to_state_vector_input_parallel_AVX_TBB(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int inner_qbit = involved_qbits[0];
    
    int outer_qbit = involved_qbits[1];
    
    int index_step_outer = 1 << outer_qbit;
    
    int index_step_inner = 1 << inner_qbit;
    
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_outer/(index_step_inner << 1);
    
    //int outer_grain_size = get_grain_size(index_step_outer);
    //int inner_grain_size = get_grain_size(index_step_inner);

    int outer_grain_size;
    if ( index_step_outer <= 2 ) {
        outer_grain_size = 64;
    }
    else if ( index_step_outer <= 4 ) {
        outer_grain_size = 32;
    }
    else if ( index_step_outer <= 8 ) {
        outer_grain_size = 16;
    }
    else if ( index_step_outer <= 16 ) {
        outer_grain_size = 8;
    }
    else {
        outer_grain_size = 2;
    }
outer_grain_size = 6000000;
    int inner_grain_size = 64000000;

    
    tbb::parallel_for( tbb::blocked_range<int>(0, parallel_outer_cycles, outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx            = r.begin()*(index_step_outer<<1);        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
            tbb::parallel_for( tbb::blocked_range<int>(0, parallel_inner_cycles, inner_grain_size), [&](tbb::blocked_range<int> r) {

                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){

                    if (inner_qbit<2){
                            for (int idx=0; idx<index_step_inner; ++idx){
        	
			        int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			        int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
	                        int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			        int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			        double results[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
			                
                                double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                                double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                                double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                                double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
			                
                                __m256d outer_inner_vec = get_AVX_vector(element_outer, element_inner);



                                __m256d outer_inner_pair_vec =  get_AVX_vector(element_outer_pair, element_inner_pair);
                            

			        __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

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

		                input[current_idx_outer_loc].real = results[0];
		                input[current_idx_outer_loc].imag = results[1];
		                input[current_idx_inner_loc].real = results[2];
		                input[current_idx_inner_loc].imag = results[3];
		                input[current_idx_outer_pair_loc].real = results[4];
		                input[current_idx_outer_pair_loc].imag = results[5];
		                input[current_idx_inner_pair_loc].real = results[6];
		                input[current_idx_inner_pair_loc].imag = results[7];
            	        }

                    }
                    else{

                        for (int alt_idx=0; alt_idx<index_step_inner/2; ++alt_idx){

                            int idx = alt_idx*2;
    			    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
		            
                            double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                            double* element_inner = element_outer + 2 * index_step_inner;
                    
                            double* element_outer_pair = element_outer + 2 * index_step_outer;
                            double* element_inner_pair = element_outer_pair + 2 * index_step_inner;
                    
                            double* indices[4] = {element_outer,element_inner,element_outer_pair,element_inner_pair};
                    
                            __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                            __m256d element_inner_vec = _mm256_loadu_pd(element_inner);

                            __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                            __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                        
                            __m256d result_vecs[4];
                    
                            for (int mult_idx=0; mult_idx<4; mult_idx++){
		                double* unitary_col_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
		                double* unitary_col_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
		                
		                __m256d unitary_col_01_vec     = _mm256_loadu_pd(unitary_col_01); // a,b,c,d
		                unitary_col_01_vec             = _mm256_permute4x64_pd(unitary_col_01_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_0_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_1_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b1111); // c, c, d, d
		                unitary_col_0_vec              = _mm256_permute4x64_pd(unitary_col_0_vec,0b11011000); // a, b, a, b
		                unitary_col_1_vec              = _mm256_permute4x64_pd(unitary_col_1_vec,0b11011000); // c, d, c, d

		                __m256d unitary_col_23_vec     = _mm256_loadu_pd(unitary_col_23); // a,b,c,d
		                unitary_col_23_vec             = _mm256_permute4x64_pd(unitary_col_23_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_2_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_3_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b1111); // c, c, d, d
		                unitary_col_2_vec              = _mm256_permute4x64_pd(unitary_col_2_vec,0b11011000); // a, a, b, b
                                unitary_col_3_vec              = _mm256_permute4x64_pd(unitary_col_3_vec,0b11011000); // c, c, d, d    
		                
                                __m256d outer_vec_3            = _mm256_mul_pd(element_outer_vec, unitary_col_0_vec);
                                __m256d unitary_col_0_switched = _mm256_permute_pd(unitary_col_0_vec, 0x5);
                                unitary_col_0_switched         = _mm256_mul_pd( unitary_col_0_switched, neg);
                                __m256d outer_vec_4            = _mm256_mul_pd( element_outer_vec, unitary_col_0_switched);
                                __m256d result_outer_vec       = _mm256_hsub_pd( outer_vec_3, outer_vec_4);

                        
                                __m256d inner_vec_3            = _mm256_mul_pd(element_inner_vec, unitary_col_1_vec);
                                __m256d unitary_col_1_switched = _mm256_permute_pd(unitary_col_1_vec, 0x5);
                                unitary_col_1_switched         = _mm256_mul_pd( unitary_col_1_switched, neg);
                                __m256d inner_vec_4            = _mm256_mul_pd( element_inner_vec, unitary_col_1_switched);
                                __m256d result_inner_vec       = _mm256_hsub_pd( inner_vec_3, inner_vec_4);
                        
                                __m256d outer_pair_vec_3       = _mm256_mul_pd(element_outer_pair_vec, unitary_col_2_vec);
                                __m256d unitary_col_2_switched = _mm256_permute_pd(unitary_col_2_vec, 0x5);
                                unitary_col_2_switched         = _mm256_mul_pd( unitary_col_2_switched, neg);
                                __m256d outer_pair_vec_4       = _mm256_mul_pd( element_outer_pair_vec, unitary_col_2_switched);
                                __m256d result_outer_pair_vec  = _mm256_hsub_pd( outer_pair_vec_3, outer_pair_vec_4);

                        
                                __m256d inner_pair_vec_3       = _mm256_mul_pd(element_inner_pair_vec, unitary_col_3_vec);
                                __m256d unitary_col_3_switched = _mm256_permute_pd(unitary_col_3_vec, 0x5);
                                unitary_col_3_switched         = _mm256_mul_pd( unitary_col_3_switched, neg);
                                __m256d inner_pair_vec_4       = _mm256_mul_pd( element_inner_pair_vec, unitary_col_3_switched);
                                __m256d result_inner_pair_vec  = _mm256_hsub_pd( inner_pair_vec_3, inner_pair_vec_4);
                        
                                __m256d result_vec    = _mm256_add_pd(result_outer_vec,result_inner_vec);
                                result_vec            = _mm256_add_pd(result_vec,result_outer_pair_vec);
                                result_vec            = _mm256_add_pd(result_vec,result_inner_pair_vec);
                                result_vecs[mult_idx] = result_vec;
                            }

                            for (int row_idx=0; row_idx<4;row_idx++){
                                _mm256_storeu_pd(indices[row_idx], result_vecs[row_idx]);
                            }

                        }
                    }

                    current_idx_inner = current_idx_inner +(index_step_inner << 1);

                }
            });

            current_idx = current_idx + (index_step_outer << 1);
            current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

        }   
    });
    
}

/**
@brief Call to apply kernel to apply three qubit gate kernel on a state vector using AVX and TBB
@param two_qbit_unitary The 8x8 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_3qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
__m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

int index_step_inner  = 1 << involved_qbits[0];
int index_step_middle = 1 << involved_qbits[1];
int index_step_outer  = 1 << involved_qbits[2];

int parallel_outer_cycles  = matrix_size / (index_step_outer << 1);
int parallel_inner_cycles  = index_step_middle / (index_step_inner << 1);
int parallel_middle_cycles = index_step_outer / (index_step_middle << 1);

for (int outer_rdx = 0; outer_rdx < parallel_outer_cycles; outer_rdx++) {

    int current_idx         = outer_rdx * (index_step_outer << 1);
    int current_idx_pair_outer = current_idx + index_step_outer;

    for (int middle_rdx = 0; middle_rdx < parallel_middle_cycles; middle_rdx++) {

        int current_idx_middle = middle_rdx * (index_step_middle << 1);

        for (int inner_rdx = 0; inner_rdx < parallel_inner_cycles; inner_rdx++) {

            int current_idx_inner = inner_rdx * (index_step_inner << 1);

            for (int idx = 0; idx < index_step_inner; ++idx) {

                int current_idx_loc       = current_idx + current_idx_middle + current_idx_inner + idx;
                int current_idx_pair_loc  = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle;

                int current_idx_outer_loc          = current_idx_loc;
                int current_idx_inner_loc          = current_idx_loc + index_step_inner;
                int current_idx_middle_loc         = current_idx_loc + index_step_middle;
                int current_idx_middle_inner_loc   = current_idx_loc + index_step_middle + index_step_inner;

                int current_idx_outer_pair_loc        = current_idx_pair_loc;
                int current_idx_inner_pair_loc        = current_idx_pair_loc + index_step_inner;
                int current_idx_middle_pair_loc       = current_idx_pair_loc + index_step_middle;
                int current_idx_middle_inner_pair_loc = current_idx_pair_loc + index_step_middle + index_step_inner;

                QGD_Complex16 results[8];

                double* element_outer         = (double*)input.get_data() + 2 * current_idx_outer_loc;
                double* element_inner         = (double*)input.get_data() + 2 * current_idx_inner_loc;
                double* element_middle        = (double*)input.get_data() + 2 * current_idx_middle_loc;
                double* element_middle_inner  = (double*)input.get_data() + 2 * current_idx_middle_inner_loc;
                double* element_outer_pair    = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                double* element_inner_pair    = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
                double* element_middle_pair   = (double*)input.get_data() + 2 * current_idx_middle_pair_loc;
                double* element_middle_inner_pair = (double*)input.get_data() + 2 * current_idx_middle_inner_pair_loc;

                __m256d outer_inner_vec       = get_AVX_vector(element_outer, element_inner);
                __m256d middle_inner_vec      = get_AVX_vector(element_middle, element_middle_inner);
                __m256d outer_inner_pair_vec  = get_AVX_vector(element_outer_pair, element_inner_pair);
                __m256d middle_inner_pair_vec = get_AVX_vector(element_middle_pair, element_middle_inner_pair);

                for (int mult_idx = 0; mult_idx < 8; mult_idx++) {

                    double* unitary_row_01 = (double*)unitary.get_data() + 16*mult_idx;
                    double* unitary_row_23 = unitary_row_01 + 4;
                    double* unitary_row_45 = unitary_row_01 + 8;
                    double* unitary_row_67 = unitary_row_01 + 12;

                    __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                    __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                    __m256d unitary_row_45_vec = _mm256_loadu_pd(unitary_row_45);
                    __m256d unitary_row_67_vec = _mm256_loadu_pd(unitary_row_67);

                    __m256d result_upper_vec        = complex_mult_AVX(outer_inner_vec, unitary_row_01_vec, neg);
                    __m256d result_upper_middle_vec = complex_mult_AVX(middle_inner_vec, unitary_row_23_vec, neg);
                    __m256d result_lower_vec        = complex_mult_AVX(outer_inner_pair_vec, unitary_row_45_vec, neg);
                    __m256d result_lower_middle_vec = complex_mult_AVX(middle_inner_pair_vec, unitary_row_67_vec, neg);

                    __m256d result1_vec = _mm256_hadd_pd(result_upper_vec, result_upper_middle_vec);
                    __m256d result2_vec = _mm256_hadd_pd(result_lower_vec, result_lower_middle_vec);
                    __m256d result_vec  = _mm256_hadd_pd(result1_vec, result2_vec);
                    result_vec          = _mm256_hadd_pd(result_vec, result_vec);

                    double* result = (double*)&result_vec;
                    results[mult_idx].real = result[0];
                    results[mult_idx].imag = result[2];
                }

                input[current_idx_outer_loc]          = results[0];
                input[current_idx_inner_loc]          = results[1];
                input[current_idx_middle_loc]         = results[2];
                input[current_idx_middle_inner_loc]   = results[3];
                input[current_idx_outer_pair_loc]     = results[4];
                input[current_idx_inner_pair_loc]     = results[5];
                input[current_idx_middle_pair_loc]    = results[6];
                input[current_idx_middle_inner_pair_loc] = results[7];
            }
        }
    }
}

}
/**
@brief Call to apply kernel to apply three qubit gate kernel on a state vector using AVX and TBB
@param two_qbit_unitary The 8x8 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_3qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_inner = 1 << involved_qbits[0];
    
    int index_step_middle = 1 << involved_qbits[1];
    
    int index_step_outer = 1 << involved_qbits[2];
    
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_middle/(index_step_inner << 1);
    
    int parallel_middle_cycles = index_step_outer/(index_step_middle << 1);
   
    
    int outer_grain_size = get_grain_size(index_step_outer);
    int inner_grain_size = get_grain_size(index_step_outer);
    int middle_grain_size = get_grain_size(index_step_middle);

    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx = r.begin()*(index_step_outer<<1);
        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
        tbb::parallel_for( tbb::blocked_range<int>(0,parallel_middle_cycles,middle_grain_size), [&](tbb::blocked_range<int> r) {
            
            int current_idx_middle = r.begin()*(index_step_middle<<1);
            for (int middle_rdx=r.begin(); middle_rdx<r.end(); middle_rdx++){
            
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_inner_cycles,inner_grain_size), [&](tbb::blocked_range<int> r) {
                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){

                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner,64),[&](tbb::blocked_range<int> r){
                    
        	            for (int idx=r.begin(); idx<r.end(); ++idx){
        	
                    	    int current_idx_loc = current_idx + current_idx_middle + current_idx_inner + idx;
                            int current_idx_pair_loc = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle;

		                    int current_idx_outer_loc = current_idx_loc;
		                    int current_idx_inner_loc = current_idx_loc + index_step_inner;
		                    
		                    int current_idx_middle_loc = current_idx_loc + index_step_middle;
		                    int current_idx_middle_inner_loc = current_idx_loc + index_step_middle + index_step_inner;
		                    
                        	int current_idx_outer_pair_loc = current_idx_pair_loc;
		                    int current_idx_inner_pair_loc = current_idx_pair_loc + index_step_inner;
		                    
		                    int current_idx_middle_pair_loc =current_idx_pair_loc + index_step_middle;
		                    int current_idx_middle_inner_pair_loc = current_idx_pair_loc + index_step_middle + index_step_inner;
			                
			                QGD_Complex16 results[8];
			                
                            double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                            double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                            
                            double* element_middle = (double*)input.get_data() + 2 * current_idx_middle_loc;
                            double* element_middle_inner = (double*)input.get_data() + 2 * current_idx_middle_inner_loc;
                            
                            double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                            double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
                                                        
                            double* element_middle_pair = (double*)input.get_data() + 2 * current_idx_middle_pair_loc;
                            double* element_middle_inner_pair = (double*)input.get_data() + 2 * current_idx_middle_inner_pair_loc;
			                
                            __m256d outer_inner_vec = get_AVX_vector(element_outer, element_inner);

                           __m256d middle_inner_vec = get_AVX_vector(element_middle,element_middle_inner);

                            __m256d outer_inner_pair_vec =  get_AVX_vector(element_outer_pair, element_inner_pair);
                            
                            __m256d middle_inner_pair_vec =  get_AVX_vector(element_middle_pair, element_middle_inner_pair);



			                for (int mult_idx=0; mult_idx<8; mult_idx++){
			                
			                    double* unitary_row_01 = (double*)unitary.get_data() + 16*mult_idx;
			                    double* unitary_row_23 = (double*)unitary.get_data() + 16*mult_idx + 4;
			                    double* unitary_row_45 = (double*)unitary.get_data() + 16*mult_idx + 8;
			                    double* unitary_row_67 = (double*)unitary.get_data() + 16*mult_idx + 12;
			                    
                                __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                                __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                                __m256d unitary_row_45_vec = _mm256_loadu_pd(unitary_row_45);
                                __m256d unitary_row_67_vec = _mm256_loadu_pd(unitary_row_67);
                                
                                
                                __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_01_vec,neg);
                                
                                __m256d result_upper_middle_vec = complex_mult_AVX(middle_inner_vec,unitary_row_23_vec,neg);
                                
                                __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_45_vec,neg);
                                
                                __m256d result_lower_middle_vec = complex_mult_AVX(middle_inner_pair_vec,unitary_row_67_vec,neg);
                                
                                
                                __m256d result1_vec = _mm256_hadd_pd(result_upper_vec,result_upper_middle_vec);
                                __m256d result2_vec = _mm256_hadd_pd(result_lower_vec,result_lower_middle_vec);
                                __m256d result_vec = _mm256_hadd_pd(result1_vec,result2_vec);
                                result_vec = _mm256_hadd_pd(result_vec,result_vec);
                                double* result = (double*)&result_vec;
                                results[mult_idx].real = result[0];
                                results[mult_idx].imag = result[2];
			                }
		                input[current_idx_outer_loc] = results[0];
		                input[current_idx_inner_loc] = results[1];
		                input[current_idx_middle_loc]  = results[2];
		                input[current_idx_middle_inner_loc] = results[3];
		                input[current_idx_outer_pair_loc] = results[4];
		                input[current_idx_inner_pair_loc] = results[5];
		                input[current_idx_middle_pair_loc] = results[6];
		                input[current_idx_middle_inner_pair_loc] = results[7];
            	    }
                   });
                
               
                current_idx_inner = current_idx_inner +(index_step_inner << 1);
                }
            });
                current_idx_middle = current_idx_middle +(index_step_middle << 1);
            }
            });
        current_idx = current_idx + (index_step_outer << 1);
        current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

    }
    });
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
    
    int parallel_outer_cycles   = matrix_size / (index_step_outer << 1);
    int parallel_inner_cycles   = index_step_middle1 / (index_step_inner << 1);
    int parallel_middle1_cycles = index_step_middle2 / (index_step_middle1 << 1);
    int parallel_middle2_cycles = index_step_outer   / (index_step_middle2 << 1);
    
    for (int outer_rdx = 0; outer_rdx < parallel_outer_cycles; outer_rdx++) {
        int current_idx = outer_rdx * (index_step_outer << 1);
        int current_idx_pair_outer = current_idx + index_step_outer;
        
        for (int middle2_rdx = 0; middle2_rdx < parallel_middle2_cycles; middle2_rdx++) {
            int current_idx_middle2 = middle2_rdx * (index_step_middle2 << 1);
            
            for (int middle1_rdx = 0; middle1_rdx < parallel_middle1_cycles; middle1_rdx++) {
                int current_idx_middle1 = middle1_rdx * (index_step_middle1 << 1);
                
                for (int inner_rdx = 0; inner_rdx < parallel_inner_cycles; inner_rdx++) {
                    int current_idx_inner = inner_rdx * (index_step_inner << 1);
                    
                    for (int idx = 0; idx < index_step_inner; ++idx) {
                        
                        int current_idx_loc = current_idx + idx + current_idx_inner + current_idx_middle1 + current_idx_middle2;
                        int current_idx_pair_loc = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle1 + current_idx_middle2;

                        int current_idx_outer_loc = current_idx_loc;
                        int current_idx_inner_loc = current_idx_loc + index_step_inner;
                        
                        int current_idx_middle1_loc = current_idx_loc + index_step_middle1;
                        int current_idx_middle1_inner_loc = current_idx_loc + index_step_middle1 + index_step_inner;
                        
                        int current_idx_middle2_loc = current_idx_loc + index_step_middle2;
                        int current_idx_middle2_inner_loc = current_idx_loc + index_step_middle2 + index_step_inner;
                        
                        int current_idx_middle12_loc = current_idx_loc + index_step_middle1 + index_step_middle2;
                        int current_idx_middle12_inner_loc = current_idx_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
                        
                        int current_idx_outer_pair_loc = current_idx_pair_loc;
                        int current_idx_inner_pair_loc = current_idx_pair_loc + index_step_inner;
                        
                        int current_idx_middle1_pair_loc = current_idx_pair_loc + index_step_middle1;
                        int current_idx_middle1_inner_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_inner;
                        
                        int current_idx_middle2_pair_loc = current_idx_pair_loc + index_step_middle2;
                        int current_idx_middle2_inner_pair_loc = current_idx_pair_loc + index_step_middle2 + index_step_inner;
                        
                        int current_idx_middle12_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_middle2;
                        int current_idx_middle12_inner_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
                        
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
void apply_4qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_inner = 1 << involved_qbits[0];
    
    int index_step_middle1 = 1 << involved_qbits[1];
    
    int index_step_middle2 = 1 << involved_qbits[2];
    
    int index_step_outer = 1 << involved_qbits[3];
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_middle1/(index_step_inner << 1);
    
    int parallel_middle1_cycles = index_step_middle2/(index_step_middle1 << 1);
    
    int parallel_middle2_cycles = index_step_outer/(index_step_middle2 << 1);
    
    int outer_grain_size = get_grain_size(index_step_outer);
    int inner_grain_size = get_grain_size(index_step_outer);
    int middle1_grain_size = get_grain_size(index_step_middle1);
    int middle2_grain_size = get_grain_size(index_step_middle2);
    
    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx = r.begin()*(index_step_outer<<1);
        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
        tbb::parallel_for( tbb::blocked_range<int>(0,parallel_middle2_cycles,middle2_grain_size), [&](tbb::blocked_range<int> r) {
            
            int current_idx_middle2 = r.begin()*(index_step_middle2<<1);
            
            for (int middle2_rdx=r.begin(); middle2_rdx<r.end(); middle2_rdx++){
            
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_middle1_cycles,middle1_grain_size), [&](tbb::blocked_range<int> r) {
            
            int current_idx_middle1 = r.begin()*(index_step_middle1<<1);
            
            for (int middle1_rdx=r.begin(); middle1_rdx<r.end(); middle1_rdx++){
            
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_inner_cycles,inner_grain_size), [&](tbb::blocked_range<int> r) {
                
                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){

                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner,64),[&](tbb::blocked_range<int> r){
                    
        	            for (int idx=r.begin(); idx<r.end(); ++idx){
        	
                    	    int current_idx_loc = current_idx + idx + current_idx_inner + current_idx_middle1 + current_idx_middle2  ;
                            int current_idx_pair_loc = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle1 + current_idx_middle2;

		                    int current_idx_outer_loc = current_idx_loc;
		                    int current_idx_inner_loc = current_idx_loc + index_step_inner;
		                    
		                    int current_idx_middle1_loc = current_idx_loc + index_step_middle1;
		                    int current_idx_middle1_inner_loc = current_idx_loc + index_step_middle1 + index_step_inner;
		                    
		                    int current_idx_middle2_loc = current_idx_loc + index_step_middle2;
		                    int current_idx_middle2_inner_loc = current_idx_loc + index_step_middle2 + index_step_inner;
		                    
		                    int current_idx_middle12_loc = current_idx_loc + index_step_middle1 + index_step_middle2;
		                    int current_idx_middle12_inner_loc = current_idx_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
		                    
                        	int current_idx_outer_pair_loc = current_idx_pair_loc;
		                    int current_idx_inner_pair_loc = current_idx_pair_loc + index_step_inner;
		                    
		                    int current_idx_middle1_pair_loc =current_idx_pair_loc + index_step_middle1;
		                    int current_idx_middle1_inner_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_inner;
		                    
		                    int current_idx_middle2_pair_loc = current_idx_pair_loc + index_step_middle2;
		                    int current_idx_middle2_inner_pair_loc = current_idx_pair_loc + index_step_middle2 + index_step_inner;
		                    
		                    int current_idx_middle12_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_middle2;
		                    int current_idx_middle12_inner_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
			                
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
                            
			                
                            __m256d outer_inner_vec = get_AVX_vector(element_outer, element_inner);

                           __m256d middle1_inner_vec = get_AVX_vector(element_middle1,element_middle1_inner);
                            
                            __m256d middle2_inner_vec = get_AVX_vector(element_middle2, element_middle2_inner);

                           __m256d middle12_inner_vec = get_AVX_vector(element_middle12,element_middle12_inner);

                            __m256d outer_inner_pair_vec =  get_AVX_vector(element_outer_pair, element_inner_pair);
                            
                            __m256d middle1_inner_pair_vec =  get_AVX_vector(element_middle1_pair, element_middle1_inner_pair);
                            
                            __m256d middle2_inner_pair_vec =  get_AVX_vector(element_middle2_pair, element_middle2_inner_pair);
                            
                            __m256d middle12_inner_pair_vec =  get_AVX_vector(element_middle12_pair, element_middle12_inner_pair);


			                for (int mult_idx=0; mult_idx<16; mult_idx++){
			                
			                    double* unitary_row_1 = (double*)unitary.get_data() + 32*mult_idx;
			                    double* unitary_row_2 = (double*)unitary.get_data() + 32*mult_idx + 4;
			                    double* unitary_row_3 = (double*)unitary.get_data() + 32*mult_idx + 8;
			                    double* unitary_row_4 = (double*)unitary.get_data() + 32*mult_idx + 12;
			                    double* unitary_row_5 = (double*)unitary.get_data() + 32*mult_idx + 16;
			                    double* unitary_row_6 = (double*)unitary.get_data() + 32*mult_idx + 20;
			                    double* unitary_row_7 = (double*)unitary.get_data() + 32*mult_idx + 24;
			                    double* unitary_row_8 = (double*)unitary.get_data() + 32*mult_idx + 28;
			                    
                                __m256d unitary_row_1_vec = _mm256_loadu_pd(unitary_row_1);
                                __m256d unitary_row_2_vec = _mm256_loadu_pd(unitary_row_2);
                                __m256d unitary_row_3_vec = _mm256_loadu_pd(unitary_row_3);
                                __m256d unitary_row_4_vec = _mm256_loadu_pd(unitary_row_4);
                                __m256d unitary_row_5_vec = _mm256_loadu_pd(unitary_row_5);
                                __m256d unitary_row_6_vec = _mm256_loadu_pd(unitary_row_6);
                                __m256d unitary_row_7_vec = _mm256_loadu_pd(unitary_row_7);
                                __m256d unitary_row_8_vec = _mm256_loadu_pd(unitary_row_8);
                                
                                __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_1_vec,neg);
                                
                                __m256d result_upper_middle1_vec = complex_mult_AVX(middle1_inner_vec,unitary_row_2_vec,neg);
                                
                                __m256d result_upper_middle2_vec = complex_mult_AVX(middle2_inner_vec,unitary_row_3_vec,neg);
                                
                                __m256d result_upper_middle12_vec = complex_mult_AVX(middle12_inner_vec,unitary_row_4_vec,neg);
                                
                                __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_5_vec,neg);
                                
                                __m256d result_lower_middle1_vec = complex_mult_AVX(middle1_inner_pair_vec,unitary_row_6_vec,neg);
                                
                                __m256d result_lower_middle2_vec = complex_mult_AVX(middle2_inner_pair_vec,unitary_row_7_vec,neg);
                                
                                __m256d result_lower_middle12_vec = complex_mult_AVX(middle12_inner_pair_vec,unitary_row_8_vec,neg);
                                
                                
                                __m256d result1_vec = _mm256_hadd_pd(result_upper_vec,result_upper_middle1_vec);
                                __m256d result2_vec = _mm256_hadd_pd(result_upper_middle2_vec,result_upper_middle12_vec);
                                __m256d result3_vec = _mm256_hadd_pd(result_lower_vec,result_lower_middle1_vec);
                                __m256d result4_vec = _mm256_hadd_pd(result_lower_middle2_vec,result_lower_middle12_vec);
                                __m256d result5_vec = _mm256_hadd_pd(result1_vec,result2_vec);
                                __m256d result6_vec = _mm256_hadd_pd(result3_vec,result4_vec);
                                __m256d result_vec  = _mm256_hadd_pd(result5_vec,result6_vec);
                                result_vec          = _mm256_hadd_pd(result_vec,result_vec);
                                double* result = (double*)&result_vec;
                                results[mult_idx].real = result[0];
                                results[mult_idx].imag = result[2];
			                }
		                input[current_idx_outer_loc] = results[0];
		                input[current_idx_inner_loc] = results[1];
		                input[current_idx_middle1_loc]  = results[2];
		                input[current_idx_middle1_inner_loc] = results[3];
		                input[current_idx_middle2_loc]  = results[4];
		                input[current_idx_middle2_inner_loc] = results[5];
		                input[current_idx_middle12_loc]  = results[6];
		                input[current_idx_middle12_inner_loc] = results[7];
		                input[current_idx_outer_pair_loc] = results[8];
		                input[current_idx_inner_pair_loc] = results[9];
		                input[current_idx_middle1_pair_loc] = results[10];
		                input[current_idx_middle1_inner_pair_loc] = results[11];
		                input[current_idx_middle2_pair_loc] = results[12];
		                input[current_idx_middle2_inner_pair_loc] = results[13];
		                input[current_idx_middle12_pair_loc] = results[14];
		                input[current_idx_middle12_inner_pair_loc] = results[15];
            	    }
                   });
                
               
                current_idx_inner = current_idx_inner +(index_step_inner << 1);
                }
            });
                current_idx_middle1 = current_idx_middle1 +(index_step_middle1 << 1);
            }
            });
                current_idx_middle2 = current_idx_middle2 +(index_step_middle2 << 1);
            }
            });
        current_idx = current_idx + (index_step_outer << 1);
        current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

    }
    });
}



/**
@brief Call to apply crot gate kernel on an input matrix using AVX
@param u3_1qbit1 The 2x2 kernel to be applied on target |1>
@param u3_1qbit2 The 2x2 kernel to be applied on target |0>
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit
@param control_qbit The control qubit
@param matrix_size The size of the input
*/
void
apply_crot_kernel_to_matrix_input_AVX(Matrix& u3_1qbit1, Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    input.ensure_aligned();

    int index_step_target = 1 << target_qbit;
    int current_idx       = 0;

    // load elements of the first U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit1[0].real);
    __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit1[0].imag);
    __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit1[1].real);
    __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit1[1].imag);
    __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit1[2].real);
    __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit1[2].imag);
    __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit1[3].real);
    __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit1[3].imag);
    // load elements of the second U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit2_00r_vec = _mm256_broadcast_sd(&u3_1qbit2[0].real);
    __m256d u3_1bit2_00i_vec = _mm256_broadcast_sd(&u3_1qbit2[0].imag);
    __m256d u3_1bit2_01r_vec = _mm256_broadcast_sd(&u3_1qbit2[1].real);
    __m256d u3_1bit2_01i_vec = _mm256_broadcast_sd(&u3_1qbit2[1].imag);
    __m256d u3_1bit2_10r_vec = _mm256_broadcast_sd(&u3_1qbit2[2].real);
    __m256d u3_1bit2_10i_vec = _mm256_broadcast_sd(&u3_1qbit2[2].imag);
    __m256d u3_1bit2_11r_vec = _mm256_broadcast_sd(&u3_1qbit2[3].real);
    __m256d u3_1bit2_11i_vec = _mm256_broadcast_sd(&u3_1qbit2[3].imag);


    for ( int current_idx_pair=current_idx + index_step_target; current_idx_pair<matrix_size; current_idx_pair=current_idx_pair+(index_step_target << 1) ) {
           

        for (int idx = 0; idx < index_step_target; idx++) {


                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    int row_offset = current_idx_loc * input.stride;
                    int row_offset_pair = current_idx_pair_loc * input.stride;
                    for (int col_idx = 0; col_idx < 2 * (input.cols - 3); col_idx = col_idx + 8) {
                      double* element = (double*)input.get_data() + 2 * row_offset;
                      double* element_pair = (double*)input.get_data() + 2 * row_offset_pair;
                     if ((current_idx_loc >> control_qbit) & 1) {

    
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
                      else {

    
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
                }

                        int remainder = input.cols % 4;
              if (remainder != 0) {

                            for (int col_idx = input.cols-remainder; col_idx < input.cols; col_idx++) {
                    int index      = row_offset+col_idx;
                    int index_pair = row_offset_pair+col_idx;  
                    if ( (current_idx_loc >> control_qbit) & 1 ) {

              

                    QGD_Complex16 element      = input[index];
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

            else {
                    QGD_Complex16 element      = input[index];
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



            current_idx = current_idx + (index_step_target << 1);

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

