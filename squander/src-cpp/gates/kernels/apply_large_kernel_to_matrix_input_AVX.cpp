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


#include "apply_large_kernel_to_matrix_input_AVX.h"
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
@brief Call to apply kernel to apply two qubit gate kernel on an input matrix using AVX and TBB
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits Vector containing inner_qbit (index 0) and outer_qbit (index 1)
@param matrix_size The size of the input
*/
void apply_2qbit_kernel_to_matrix_input_parallel_AVX_TBB(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    int inner_qbit = involved_qbits[0];
    int outer_qbit = involved_qbits[1];
    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

    int parallel_outer_cycles = input.rows/(index_step_outer << 1);
    int outer_grain_size;
    if ( index_step_outer <= 2 ) {
        outer_grain_size = 32;
    }
    else if ( index_step_outer <= 4 ) {
        outer_grain_size = 16;
    }
    else if ( index_step_outer <= 8 ) {
        outer_grain_size = 8;
    }
    else if ( index_step_outer <= 16 ) {
        outer_grain_size = 4;
    }
    else {
        outer_grain_size = 2;
    }

    tbb::parallel_for( tbb::blocked_range<int>(0, parallel_outer_cycles, outer_grain_size), [&](tbb::blocked_range<int> r) {

        int current_idx = r.begin() * (index_step_outer << 1);
        int current_idx_pair_outer = current_idx + index_step_outer;

        for (int rdx = r.begin(); rdx < r.end(); rdx++) {

            int inner_cycles = index_step_outer / (index_step_inner << 1);

            tbb::parallel_for( tbb::blocked_range<int>(0, inner_cycles, 8), [&](tbb::blocked_range<int> r_inner) {

                for (int inner_rdx = r_inner.begin(); inner_rdx < r_inner.end(); inner_rdx++) {

                    int current_idx_inner = inner_rdx * (index_step_inner << 1);

                    for (int idx = 0; idx < index_step_inner; idx++) {

                        int current_idx_outer_loc = current_idx + current_idx_inner + idx;
                        int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
                        int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
                        int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;

                        int row_offset_outer = current_idx_outer_loc * input.cols;
                        int row_offset_inner = current_idx_inner_loc * input.cols;
                        int row_offset_outer_pair = current_idx_outer_pair_loc * input.cols;
                        int row_offset_inner_pair = current_idx_inner_pair_loc * input.cols;

                        tbb::parallel_for( tbb::blocked_range<int>(0, input.cols, 32), [&](tbb::blocked_range<int> r_col) {

                            for (int col_idx = r_col.begin(); col_idx < r_col.end(); col_idx++) {

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

                                for (int mult_idx = 0; mult_idx < 4; mult_idx++) {
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
                        });
                    }
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
apply_crot_kernel_to_matrix_input_AVX_TBB(Matrix& u3_1qbit1,Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


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

/**
@brief Call to apply crot gate kernel on an input matrix using AVX and OpenMP
@param u3_1qbit1 The 2x2 kernel to be applied on target |1>
@param u3_1qbit2 The 2x2 kernel to be applied on target |0>
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit
@param control_qbit The control qubit
@param matrix_size The size of the input
*/
void
apply_crot_kernel_to_matrix_input_AVX_OpenMP(Matrix& u3_1qbit1, Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

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

    #pragma omp parallel for schedule(dynamic)
    for (int rdx = 0; rdx < parallel_outer_cycles; rdx++) {

        int current_idx = rdx * (index_step_target << 1);
        int current_idx_pair = index_step_target + rdx * (index_step_target << 1);

        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < index_step_target; idx++) {

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
        }
    }
}

