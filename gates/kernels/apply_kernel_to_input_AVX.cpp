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


#include "apply_kernel_to_input_AVX.h"
#include <immintrin.h>
#include "tbb/tbb.h"

/**
@brief AVX kernel to apply single qubit gate kernel on an input matrix (efficient for small inputs)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_input_AVX_small(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

    input.ensure_aligned();

    int index_step_target = 1 << target_qbit;
    int current_idx = 0;

    // load elements of the U3 unitary into 256bit registers (4 registers)
    __m128d* u3_1qubit_tmp = (__m128d*) & u3_1qbit[0];
    __m256d u3_1qbit_00_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

    u3_1qubit_tmp = (__m128d*) & u3_1qbit[1];
    __m256d u3_1qbit_01_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

    u3_1qubit_tmp = (__m128d*) & u3_1qbit[2];
    __m256d u3_1qbit_10_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

    u3_1qubit_tmp = (__m128d*) & u3_1qbit[3];
    __m256d u3_1qbit_11_vec = _mm256_broadcast_pd(u3_1qubit_tmp);


    for ( int current_idx_pair=current_idx + index_step_target; current_idx_pair<matrix_size; current_idx_pair=current_idx_pair+(index_step_target << 1) ) {

        for (int idx = 0; idx < index_step_target; idx++) {
            //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            int row_offset = current_idx_loc * input.stride;
            int row_offset_pair = current_idx_pair_loc * input.stride;

            if (control_qbit < 0 || ((current_idx_loc >> control_qbit) & 1)) {


                double* element = (double*)input.get_data() + 2 * row_offset;
                double* element_pair = (double*)input.get_data() + 2 * row_offset_pair;


                __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0); // 5th register


                for (int col_idx = 0; col_idx < 2 * (input.cols - 1); col_idx = col_idx + 4) {

                    // extract successive elements from arrays element, element_pair
                    __m256d element_vec = _mm256_load_pd(element + col_idx); // 6th register
                    __m256d element_pair_vec = _mm256_load_pd(element_pair + col_idx); // 7th register

                    //// u3_1qbit_00*element_vec ////

                    // 1 calculate the multiplications  u3_1qbit_00*element_vec
                    __m256d vec3 = _mm256_mul_pd(u3_1qbit_00_vec, element_vec); // 8th register

                    // 2 Switch the real and imaginary elements of element_vec
                    __m256d element_vec_permuted = _mm256_permute_pd(element_vec, 0x5);   // 9th register

                    // 3 Negate the imaginary elements of element_vec_permuted
                    element_vec_permuted = _mm256_mul_pd(element_vec_permuted, neg);

                    // 4 Multiply elements of u3_1qbit_00*element_vec_permuted
                    __m256d vec4 = _mm256_mul_pd(u3_1qbit_00_vec, element_vec_permuted);

                    // 5 Horizontally subtract the elements in vec3 and vec4
                    vec3 = _mm256_hsub_pd(vec3, vec4);


                    //// u3_1qbit_01*element_vec_pair ////

                    // 1 calculate the multiplications  u3_1qbit_01*element_pair_vec
                    __m256d vec5 = _mm256_mul_pd(u3_1qbit_01_vec, element_pair_vec); // 10th register

                    // 2 Switch the real and imaginary elements of element_vec
                    __m256d element_pair_vec_permuted = _mm256_permute_pd(element_pair_vec, 0x5);   // 11th register

                    // 3 Negate the imaginary elements of element_vec_permuted
                    element_pair_vec_permuted = _mm256_mul_pd(element_pair_vec_permuted, neg);

                    // 4 Multiply elements of u3_1qbit_01*element_vec_pair_permuted
                    vec4 = _mm256_mul_pd(u3_1qbit_01_vec, element_pair_vec_permuted);

                    // 5 Horizontally subtract the elements in vec5 and vec4
                    vec5 = _mm256_hsub_pd(vec5, vec4);

                    //// u3_1qbit_00*element_vec + u3_1qbit_01*element_vec_pair ////
                    vec3 = _mm256_add_pd(vec3, vec5);


                    // 6 store the transformed elements in vec3
                    _mm256_store_pd(element + col_idx, vec3);


                    //// u3_1qbit_10*element_vec ////

                    // 1 calculate the multiplications  u3_1qbit_10*element_vec
                    vec3 = _mm256_mul_pd(u3_1qbit_10_vec, element_vec);

                    // 4 Multiply elements of u3_1qbit_10*element_vec_permuted
                    vec4 = _mm256_mul_pd(u3_1qbit_10_vec, element_vec_permuted);

                    // 5 Horizontally subtract the elements in vec3 and vec4
                    vec3 = _mm256_hsub_pd(vec3, vec4);


                    //// u3_1qbit_01*element_vec_pair ////

                    // 1 calculate the multiplications  u3_1qbit_01*element_pair_vec
                    vec5 = _mm256_mul_pd(u3_1qbit_11_vec, element_pair_vec);

                    // 4 Multiply elements of u3_1qbit_01*element_vec_pair_permuted
                    vec4 = _mm256_mul_pd(u3_1qbit_11_vec, element_pair_vec_permuted);

                    // 5 Horizontally subtract the elements in vec5 and vec4
                    vec5 = _mm256_hsub_pd(vec5, vec4);

                    //// u3_1qbit_10*element_vec + u3_1qbit_11*element_vec_pair ////
                    vec3 = _mm256_add_pd(vec3, vec5);

                    // 6 store the transformed elements in vec3
                    _mm256_store_pd(element_pair + col_idx, vec3);

                }

                if (input.cols % 2 == 1) {

                    int col_idx = input.cols - 1;

                    int index = row_offset + col_idx;
                    int index_pair = row_offset_pair + col_idx;

                    QGD_Complex16 element = input[index];
                    QGD_Complex16 element_pair = input[index_pair];

                    QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit[2], element);
                    tmp2 = mult(u3_1qbit[3], element_pair);

                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;


                }

            }
            else if (deriv) {
                // when calculating derivatives, the constant element should be zeros
                memset(input.get_data() + row_offset, 0.0, input.cols * sizeof(QGD_Complex16));
                memset(input.get_data() + row_offset_pair, 0.0, input.cols * sizeof(QGD_Complex16));
            }
            else {
                // leave the state as it is
                continue;
            }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


                    //});
        }


        current_idx = current_idx + (index_step_target << 1);


    }




}



/**
@brief AVX kernel to apply single qubit gate kernel on an input matrix (single threaded)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_input_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

    input.ensure_aligned();

    int index_step_target = 1 << target_qbit;
    int current_idx       = 0;

    // load elements of the U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit[0].real);
    __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit[0].imag);
    __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit[1].real);
    __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit[1].imag);
    __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit[2].real);
    __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit[2].imag);
    __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit[3].real);
    __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit[3].imag);


    for ( int current_idx_pair=current_idx + index_step_target; current_idx_pair<matrix_size; current_idx_pair=current_idx_pair+(index_step_target << 1) ) {
           

        for (int idx = 0; idx < index_step_target; idx++) {


                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    int row_offset = current_idx_loc * input.stride;
                    int row_offset_pair = current_idx_pair_loc * input.stride;

                    if (control_qbit < 0 || ((current_idx_loc >> control_qbit) & 1)) {

    
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

                                QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                                QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                                input[index].real = tmp1.real + tmp2.real;
                                input[index].imag = tmp1.imag + tmp2.imag;

                                tmp1 = mult(u3_1qbit[2], element);
                                tmp2 = mult(u3_1qbit[3], element_pair);

                                input[index_pair].real = tmp1.real + tmp2.real;
                                input[index_pair].imag = tmp1.imag + tmp2.imag;
                            }
        
                        }

                    }
                    else if (deriv) {
                        // when calculating derivatives, the constant element should be zeros
                        memset(input.get_data() + row_offset, 0.0, input.cols * sizeof(QGD_Complex16));
                        memset(input.get_data() + row_offset_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                    }
                    else {
                        // leave the state as it is
                        continue;
                    }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;

                 
            }



            current_idx = current_idx + (index_step_target << 1);

    }



}


/**
@brief Parallel AVX kernel to apply single qubit gate kernel on an input matrix
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_input_AVX_parallel(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

    input.ensure_aligned();
    
    int index_step_target = 1 << target_qbit;

    // load elements of the U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit[0].real);
    __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit[0].imag);
    __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit[1].real);
    __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit[1].imag);
    __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit[2].real);
    __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit[2].imag);
    __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit[3].real);
    __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit[3].imag);



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

                    if (control_qbit < 0 || ((current_idx_loc >> control_qbit) & 1)) {

    
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

                                QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                                QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                                input[index].real = tmp1.real + tmp2.real;
                                input[index].imag = tmp1.imag + tmp2.imag;

                                tmp1 = mult(u3_1qbit[2], element);
                                tmp2 = mult(u3_1qbit[3], element_pair);

                                input[index_pair].real = tmp1.real + tmp2.real;
                                input[index_pair].imag = tmp1.imag + tmp2.imag;
                            }
        
                        }

                    }
                    else if (deriv) {
                        // when calculating derivatives, the constant element should be zeros
                        memset(input.get_data() + row_offset, 0.0, input.cols * sizeof(QGD_Complex16));
                        memset(input.get_data() + row_offset_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                    }
                    else {
                        // leave the state as it is
                        continue;
                    }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;

                 
                }
            });
            


            current_idx = current_idx + (index_step_target << 1);
            current_idx_pair = current_idx_pair + (index_step_target << 1);

        }
    });
    

}




