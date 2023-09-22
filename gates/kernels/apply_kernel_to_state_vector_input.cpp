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


#include "apply_kernel_to_state_vector_input.h"
#include <immintrin.h>
#include "tbb/tbb.h"

/**
@brief kernel to apply single qubit gate kernel on a state vector
*/
void
apply_kernel_to_state_vector_input(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int index_step_target = 1 << target_qbit;
    int current_idx = 0;


    for ( int current_idx_pair=current_idx + index_step_target; current_idx_pair<matrix_size; current_idx_pair=current_idx_pair+(index_step_target << 1) ) {

        for (int idx = 0; idx < index_step_target; idx++) {
            //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            int row_offset = current_idx_loc * input.stride;
            int row_offset_pair = current_idx_pair_loc * input.stride;

            if (control_qbit < 0 || ((current_idx_loc >> control_qbit) & 1)) {

                QGD_Complex16 element = input[row_offset];
                QGD_Complex16 element_pair = input[row_offset_pair];

                QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                input[row_offset].real = tmp1.real + tmp2.real;
                input[row_offset].imag = tmp1.imag + tmp2.imag;

                tmp1 = mult(u3_1qbit[2], element);
                tmp2 = mult(u3_1qbit[3], element_pair);

                input[row_offset_pair].real = tmp1.real + tmp2.real;
                input[row_offset_pair].imag = tmp1.imag + tmp2.imag;



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
@brief kernel to apply single qubit gate kernel on a state vector
*/
void
apply_kernel_to_state_vector_input_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int index_step_target = 1 << target_qbit;
    int current_idx = 0;

    unsigned int bitmask_low = (1 << target_qbit) - 1;
    unsigned int bitmask_high = ~bitmask_low;

    int control_qbit_step_index = (1<<control_qbit);

    if ( target_qbit == 0 || control_qbit == 0 ) {

        for (int idx=0; idx<matrix_size/2; idx++ ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {

                    QGD_Complex16 element = input[current_idx];
                    QGD_Complex16 element_pair = input[current_idx_pair];

                    QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                    input[current_idx].real = tmp1.real + tmp2.real;
                    input[current_idx].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit[2], element);
                    tmp2 = mult(u3_1qbit[3], element_pair);

                    input[current_idx_pair].real = tmp1.real + tmp2.real;
                    input[current_idx_pair].imag = tmp1.imag + tmp2.imag;



                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * sizeof(QGD_Complex16));
                    memset(input.get_data() + current_idx_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }

        
        }

    }
    else if (target_qbit == 1 || control_qbit == 1) {



        // load elements of the U3 unitary into 256bit registers (4 registers)
        __m128d* u3_1qubit_tmp = (__m128d*) & u3_1qbit[0];
        __m256d u3_1qbit_00_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        u3_1qubit_tmp = (__m128d*) & u3_1qbit[1];
        __m256d u3_1qbit_01_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        u3_1qubit_tmp = (__m128d*) & u3_1qbit[2];
        __m256d u3_1qbit_10_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        u3_1qubit_tmp = (__m128d*) & u3_1qbit[3];
        __m256d u3_1qbit_11_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0); // 5th register


        for (int idx=0; idx<matrix_size/2; idx=idx+2 ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;


                    // extract successive elements from arrays element, element_pair
                    __m256d element_vec = _mm256_loadu_pd(element); // 6th register
                    __m256d element_pair_vec = _mm256_loadu_pd(element_pair); // 7th register

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
                    _mm256_storeu_pd(element, vec3);


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
                    _mm256_storeu_pd(element_pair, vec3);

                    


                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * sizeof(QGD_Complex16));
                    memset(input.get_data() + current_idx_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;



        }



    } // else if
    else {



        // load elements of the U3 unitary into 256bit registers (8 registers)
        __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit[0].real);
        __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit[0].imag);
        __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit[1].real);
        __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit[1].imag);
        __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit[2].real);
        __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit[2].imag);
        __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit[3].real);
        __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit[3].imag);


        for (int idx=0; idx<matrix_size/2; idx=idx+4 ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;

                    // extract successive elements from arrays element, element_pair
                    __m256d element_vec = _mm256_loadu_pd(element);
                    __m256d element_vec2 = _mm256_loadu_pd(element + 4);
                    __m256d tmp = _mm256_shuffle_pd(element_vec, element_vec2, 0);
                    element_vec2 = _mm256_shuffle_pd(element_vec, element_vec2, 0xf);
                    element_vec = tmp;

                    __m256d element_pair_vec = _mm256_loadu_pd(element_pair);
                    __m256d element_pair_vec2 = _mm256_loadu_pd(element_pair + 4);
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
                    _mm256_storeu_pd(element, vec3);
                    _mm256_storeu_pd(element + 4, vec5);

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
                    _mm256_storeu_pd(element_pair, vec7);
                    _mm256_storeu_pd(element_pair + 4, vec9);
                        
                    


                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * sizeof(QGD_Complex16));
                    memset(input.get_data() + current_idx_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }



        }


    }// else

}





/**
@brief kernel to apply single qubit gate kernel on a state vector
*/
void
apply_kernel_to_state_vector_input_parallel(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int index_step_target = 1 << target_qbit;

    int parallel_outer_cycles = matrix_size/(index_step_target << 1);
    int outer_grain_size;
    if ( index_step_target <= 2 ) {
        outer_grain_size = 64;
    }
    else if ( index_step_target <= 4 ) {
        outer_grain_size = 32;
    }
    else if ( index_step_target <= 8 ) {
        outer_grain_size = 16;
    }
    else if ( index_step_target <= 16 ) {
        outer_grain_size = 8;
    }
    else {
        outer_grain_size = 2;
    }

    int inner_grain_size = 64;

    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 

        int current_idx      = r.begin()*(index_step_target << 1);
        int current_idx_pair = index_step_target + r.begin()*(index_step_target << 1);

        for (int rdx=r.begin(); rdx<r.end(); rdx++) {
            

            tbb::parallel_for( tbb::blocked_range<int>(0,index_step_target,inner_grain_size), [&](tbb::blocked_range<int> r) {
	        for (int idx=r.begin(); idx<r.end(); ++idx) {



                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    int row_offset = current_idx_loc * input.stride;
                    int row_offset_pair = current_idx_pair_loc * input.stride;

                    if (control_qbit < 0 || ((current_idx_loc >> control_qbit) & 1)) {

                        QGD_Complex16 element = input[row_offset];
                        QGD_Complex16 element_pair = input[row_offset_pair];

                        QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                        QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                        input[row_offset].real = tmp1.real + tmp2.real;
                        input[row_offset].imag = tmp1.imag + tmp2.imag;

                        tmp1 = mult(u3_1qbit[2], element);
                        tmp2 = mult(u3_1qbit[3], element_pair);

                        input[row_offset_pair].real = tmp1.real + tmp2.real;
                        input[row_offset_pair].imag = tmp1.imag + tmp2.imag;



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





/**
@brief kernel to apply single qubit gate kernel on a state vector
*/
void
apply_kernel_to_state_vector_input_parallel_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int grain_size = 64;

    unsigned int bitmask_low = (1 << target_qbit) - 1;
    unsigned int bitmask_high = ~bitmask_low;

    int control_qbit_step_index = (1<<control_qbit);

    if ( target_qbit == 0 || control_qbit == 0 ) {
        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/2,grain_size), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx++) {
            

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {

                    QGD_Complex16 element = input[current_idx];
                    QGD_Complex16 element_pair = input[current_idx_pair];

                    QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit[1], element_pair);

                    input[current_idx].real = tmp1.real + tmp2.real;
                    input[current_idx].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit[2], element);
                    tmp2 = mult(u3_1qbit[3], element_pair);

                    input[current_idx_pair].real = tmp1.real + tmp2.real;
                    input[current_idx_pair].imag = tmp1.imag + tmp2.imag;



                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * sizeof(QGD_Complex16));
                    memset(input.get_data() + current_idx_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


            }
        });


    }
    else if (target_qbit == 1 || control_qbit == 1) {

        // load elements of the U3 unitary into 256bit registers (4 registers)
        __m128d* u3_1qubit_tmp = (__m128d*) & u3_1qbit[0];
        __m256d u3_1qbit_00_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        u3_1qubit_tmp = (__m128d*) & u3_1qbit[1];
        __m256d u3_1qbit_01_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        u3_1qubit_tmp = (__m128d*) & u3_1qbit[2];
        __m256d u3_1qbit_10_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        u3_1qubit_tmp = (__m128d*) & u3_1qbit[3];
        __m256d u3_1qbit_11_vec = _mm256_broadcast_pd(u3_1qubit_tmp);

        __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0); // 5th register



        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/2,grain_size), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx=idx+2) {
            
                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);


                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;


                    // extract successive elements from arrays element, element_pair
                    __m256d element_vec = _mm256_loadu_pd(element); // 6th register
                    __m256d element_pair_vec = _mm256_loadu_pd(element_pair); // 7th register

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
                    _mm256_storeu_pd(element, vec3);


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
                    _mm256_storeu_pd(element_pair, vec3);

                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * sizeof(QGD_Complex16));
                    memset(input.get_data() + current_idx_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }


            }
        });



    } // else if
    else {


        // load elements of the U3 unitary into 256bit registers (8 registers)
        __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit[0].real);
        __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit[0].imag);
        __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit[1].real);
        __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit[1].imag);
        __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit[2].real);
        __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit[2].imag);
        __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit[3].real);    
        __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit[3].imag);




        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/2,grain_size), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx=idx+4) {
            

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;


                    // extract successive elements from arrays element, element_pair
                    __m256d element_vec = _mm256_loadu_pd(element);
                    __m256d element_vec2 = _mm256_loadu_pd(element + 4);
                    __m256d tmp = _mm256_shuffle_pd(element_vec, element_vec2, 0);
                    element_vec2 = _mm256_shuffle_pd(element_vec, element_vec2, 0xf);
                    element_vec = tmp;

                    __m256d element_pair_vec = _mm256_loadu_pd(element_pair);
                    __m256d element_pair_vec2 = _mm256_loadu_pd(element_pair + 4);
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
                    _mm256_storeu_pd(element, vec3);
                    _mm256_storeu_pd(element + 4, vec5);

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
                    _mm256_storeu_pd(element_pair, vec7);
                    _mm256_storeu_pd(element_pair + 4, vec9);

                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * sizeof(QGD_Complex16));
                    memset(input.get_data() + current_idx_pair, 0.0, input.cols * sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }



            }
        });



    } // else




}

void apply_large_kernel_to_state_vector_input(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<matrix_size; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
        
        	for (int idx=0; idx<index_step_inner; idx++){
        	

			int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
	    	int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			int indexes[4] = {current_idx_outer_loc,current_idx_inner_loc,current_idx_outer_pair_loc,current_idx_inner_pair_loc};
			//input.print_matrix();
			QGD_Complex16 element_outer = input[current_idx_outer_loc];
			QGD_Complex16 element_outer_pair = input[current_idx_outer_pair_loc];
			QGD_Complex16 element_inner = input[current_idx_inner_loc];
			QGD_Complex16 element_inner_pair = input[current_idx_inner_pair_loc];
			
			QGD_Complex16 tmp1;
			QGD_Complex16 tmp2;
			QGD_Complex16 tmp3;
			QGD_Complex16 tmp4;
			for (int mult_idx=0; mult_idx<4; mult_idx++){
			
				tmp1 = mult(two_qbit_unitary[mult_idx*4], element_outer);
				tmp2 = mult(two_qbit_unitary[mult_idx*4 + 1], element_inner);
				tmp3 = mult(two_qbit_unitary[mult_idx*4 + 2], element_outer_pair);
				tmp4 = mult(two_qbit_unitary[mult_idx*4 + 3], element_inner_pair);
				input[indexes[mult_idx]].real = tmp1.real + tmp2.real + tmp3.real + tmp4.real;
				input[indexes[mult_idx]].imag = tmp1.imag + tmp2.imag + tmp3.imag + tmp4.imag;
			}
        	}
        }
        current_idx = current_idx + (index_step_outer << 1);
    }

}


void apply_large_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<matrix_size; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
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
                    
                    __m256d vec3_upper              = _mm256_mul_pd(outer_inner_vec, unitary_row_01_vec);
                    __m256d unitary_row_01_switched = _mm256_permute_pd(unitary_row_01_vec, 0x5);
                    unitary_row_01_switched         = _mm256_mul_pd( unitary_row_01_switched, neg);
                    __m256d vec4_upper              = _mm256_mul_pd( outer_inner_vec, unitary_row_01_switched);
                    __m256d result_upper_vec        = _mm256_hsub_pd( vec3_upper, vec4_upper);
                    result_upper_vec                = _mm256_permute4x64_pd( result_upper_vec, 0b11011000);
                    
                    __m256d vec3_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_vec);
                    __m256d unitary_row_23_switched = _mm256_permute_pd(unitary_row_23_vec, 0x5);
                    unitary_row_23_switched = _mm256_mul_pd(unitary_row_23_switched, neg);
                    __m256d vec4_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_switched);
                    __m256d result_lower_vec = _mm256_hsub_pd(vec3_lower, vec4_lower);
                    result_lower_vec = _mm256_permute4x64_pd(result_lower_vec,0b11011000);
                    
                    
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


void apply_large_kernel_to_state_vector_input_parallel_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_outer = 1 << outer_qbit;
    
    int index_step_inner = 1 << inner_qbit;
    
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_outer/(index_step_inner << 1);
    
    int outer_grain_size;
    int inner_grain_size;
    
    if ( index_step_outer <= 4 ) {
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
    
    if (index_step_inner <=2){
        inner_grain_size = 64;
    }
    else if ( index_step_inner <= 4 ) {
        inner_grain_size = 32;
    }
    else if ( index_step_inner <= 8 ) {
        inner_grain_size = 16;
    }
    else if ( index_step_inner <= 16 ) {
        inner_grain_size = 8;
    }
    else {
        inner_grain_size = 2;
    }

    
    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx = r.begin()*(index_step_outer<<1);
        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_inner_cycles,inner_grain_size), [&](tbb::blocked_range<int> r) {
                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){
                    if (inner_qbit<2){
                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner,64),[&](tbb::blocked_range<int> r){
                    
        	            for (int idx=r.begin(); idx<r.end(); ++idx){
        	
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


			                __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

			                for (int mult_idx=0; mult_idx<4; mult_idx++){
			                    double* unitary_row_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
			                    double* unitary_row_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
			                    
                                __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                                __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                                
                                __m256d vec3_upper = _mm256_mul_pd(outer_inner_vec, unitary_row_01_vec);
                                __m256d unitary_row_01_switched = _mm256_permute_pd(unitary_row_01_vec, 0x5);
                                unitary_row_01_switched = _mm256_mul_pd(unitary_row_01_switched, neg);
                                __m256d vec4_upper = _mm256_mul_pd(outer_inner_vec, unitary_row_01_switched);
                                __m256d result_upper_vec = _mm256_hsub_pd(vec3_upper, vec4_upper);
                                result_upper_vec = _mm256_permute4x64_pd(result_upper_vec,0b11011000);
                                
                                __m256d vec3_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_vec);
                                __m256d unitary_row_23_switched = _mm256_permute_pd(unitary_row_23_vec, 0x5);
                                unitary_row_23_switched = _mm256_mul_pd(unitary_row_23_switched, neg);
                                __m256d vec4_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_switched);
                                __m256d result_lower_vec = _mm256_hsub_pd(vec3_lower, vec4_lower);
                                result_lower_vec = _mm256_permute4x64_pd(result_lower_vec,0b11011000);
                                
                                
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
                   });
                }
                else{
                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner/2,32),[&](tbb::blocked_range<int> r){
                    for (int alt_idx=r.begin(); alt_idx<r.end(); ++alt_idx){
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
                    });
                }
                current_idx_inner = current_idx_inner +(index_step_inner << 1);
                }
            });
        current_idx = current_idx + (index_step_outer << 1);
        current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

    }
    });
    
}
