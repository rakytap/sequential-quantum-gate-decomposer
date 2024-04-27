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
/*! \file apply_kerel_to_state_vector_input_AVX.cpp
    \brief ????????????????
*/


#include "apply_kernel_to_state_vector_input_AVX.h"
#include <immintrin.h>
#include "tbb/tbb.h"




/**
@brief AVX kernel on a state vector
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_state_vector_input_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int index_step_target = 1 << target_qbit;
    int current_idx = 0;

    unsigned int bitmask_low = (1 << target_qbit) - 1;
    unsigned int bitmask_high = ~bitmask_low;

    int control_qbit_step_index = (1<<control_qbit);

    if ( control_qbit == 0 ) {

        for (int idx=0; idx<matrix_size/2; idx++ ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if ( current_idx & control_qbit_step_index ) {

                    QGD_Complex16 element      = input[current_idx];
                    QGD_Complex16 element_pair = input[current_idx_pair];


                    QGD_Complex16&& tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16&& tmp2 = mult(u3_1qbit[1], element_pair);

                    input[current_idx].real = tmp1.real + tmp2.real;
                    input[current_idx].imag = tmp1.imag + tmp2.imag;

                    QGD_Complex16&& tmp3 = mult(u3_1qbit[2], element);
                    QGD_Complex16&& tmp4 = mult(u3_1qbit[3], element_pair);

                    input[current_idx_pair].real = tmp3.real + tmp4.real;
                    input[current_idx_pair].imag = tmp3.imag + tmp4.imag;



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
    else if ( target_qbit == 0 ) {

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

        __m256d mv00 = _mm256_set_pd(-u3_1qbit[1].imag, u3_1qbit[1].real, -u3_1qbit[0].imag, u3_1qbit[0].real);
        __m256d mv01 = _mm256_set_pd( u3_1qbit[1].real, u3_1qbit[1].imag,  u3_1qbit[0].real, u3_1qbit[0].imag);
        __m256d mv20 = _mm256_set_pd(-u3_1qbit[3].imag, u3_1qbit[3].real, -u3_1qbit[2].imag, u3_1qbit[2].real);
        __m256d mv21 = _mm256_set_pd( u3_1qbit[3].real, u3_1qbit[3].imag,  u3_1qbit[2].real, u3_1qbit[2].imag);

        for (int idx=0; idx<matrix_size/2; idx++ ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double *ptr = (double*)input.get_data() + 2 * current_idx;
                    __m256d data = _mm256_loadu_pd(ptr);

                    __m256d data_u0 = _mm256_mul_pd(data, mv00);
                    __m256d data_u1 = _mm256_mul_pd(data, mv01);
                    __m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
                    data_u2 = _mm256_permute4x64_pd(data_u2, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

                   __m256d data_d0 = _mm256_mul_pd(data, mv20);
                   __m256d data_d1 = _mm256_mul_pd(data, mv21);
                   __m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
                   data_d2 = _mm256_permute4x64_pd(data_d2, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

                   __m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

                   data_r = _mm256_permute4x64_pd(data_r, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
                   _mm256_storeu_pd(ptr, data_r);

                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * 2 *sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }

        
        }

    }
    else {


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

        __m256d mv00 = _mm256_set_pd(-u3_1qbit[0].imag, u3_1qbit[0].real, -u3_1qbit[0].imag, u3_1qbit[0].real);
        __m256d mv01 = _mm256_set_pd( u3_1qbit[0].real, u3_1qbit[0].imag,  u3_1qbit[0].real, u3_1qbit[0].imag);
        __m256d mv10 = _mm256_set_pd(-u3_1qbit[1].imag, u3_1qbit[1].real, -u3_1qbit[1].imag, u3_1qbit[1].real);
        __m256d mv11 = _mm256_set_pd( u3_1qbit[1].real, u3_1qbit[1].imag,  u3_1qbit[1].real, u3_1qbit[1].imag);
        __m256d mv20 = _mm256_set_pd(-u3_1qbit[2].imag, u3_1qbit[2].real, -u3_1qbit[2].imag, u3_1qbit[2].real);
        __m256d mv21 = _mm256_set_pd( u3_1qbit[2].real, u3_1qbit[2].imag,  u3_1qbit[2].real, u3_1qbit[2].imag);
        __m256d mv30 = _mm256_set_pd(-u3_1qbit[3].imag, u3_1qbit[3].real, -u3_1qbit[3].imag, u3_1qbit[3].real);
        __m256d mv31 = _mm256_set_pd( u3_1qbit[3].real, u3_1qbit[3].imag,  u3_1qbit[3].real, u3_1qbit[3].imag);


        for (int idx=0; idx<matrix_size/2; idx=idx+2 ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;

                   
                    __m256d data0 = _mm256_loadu_pd(element);
                    __m256d data1 = _mm256_loadu_pd(element_pair);

                    __m256d data_u2 = _mm256_mul_pd(data0, mv00);
                    __m256d data_u3 = _mm256_mul_pd(data1, mv10);
                    __m256d data_u4 = _mm256_mul_pd(data0, mv01);
                    __m256d data_u5 = _mm256_mul_pd(data1, mv11);

                    __m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
                    __m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

                    __m256d data_d2 = _mm256_mul_pd(data0, mv20);
                    __m256d data_d3 = _mm256_mul_pd(data1, mv30);
                    __m256d data_d4 = _mm256_mul_pd(data0, mv21);
                    __m256d data_d5 = _mm256_mul_pd(data1, mv31);

                    __m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
                    __m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

                    __m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
                    __m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

                    _mm256_storeu_pd(element, data_r0);
                    _mm256_storeu_pd(element_pair, data_r1);


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



    } // else

}





/**
@brief Parallel AVX kernel on a state vector (parallelized with OpenMP)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_state_vector_input_parallel_OpenMP_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int index_step_target = 1 << target_qbit;
    int current_idx = 0;

    unsigned int bitmask_low = (1 << target_qbit) - 1;
    unsigned int bitmask_high = ~bitmask_low;

    int control_qbit_step_index = (1<<control_qbit);

    if ( control_qbit == 0 ) {
#pragma omp parallel for
        for (int idx=0; idx<matrix_size/2; idx++ ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if ( current_idx & control_qbit_step_index ) {

                    QGD_Complex16 element      = input[current_idx];
                    QGD_Complex16 element_pair = input[current_idx_pair];


                    QGD_Complex16&& tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16&& tmp2 = mult(u3_1qbit[1], element_pair);

                    input[current_idx].real = tmp1.real + tmp2.real;
                    input[current_idx].imag = tmp1.imag + tmp2.imag;

                    QGD_Complex16&& tmp3 = mult(u3_1qbit[2], element);
                    QGD_Complex16&& tmp4 = mult(u3_1qbit[3], element_pair);

                    input[current_idx_pair].real = tmp3.real + tmp4.real;
                    input[current_idx_pair].imag = tmp3.imag + tmp4.imag;



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
    else if ( target_qbit == 0 ) {

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

        __m256d mv00 = _mm256_set_pd(-u3_1qbit[1].imag, u3_1qbit[1].real, -u3_1qbit[0].imag, u3_1qbit[0].real);
        __m256d mv01 = _mm256_set_pd( u3_1qbit[1].real, u3_1qbit[1].imag,  u3_1qbit[0].real, u3_1qbit[0].imag);
        __m256d mv20 = _mm256_set_pd(-u3_1qbit[3].imag, u3_1qbit[3].real, -u3_1qbit[2].imag, u3_1qbit[2].real);
        __m256d mv21 = _mm256_set_pd( u3_1qbit[3].real, u3_1qbit[3].imag,  u3_1qbit[2].real, u3_1qbit[2].imag);

#pragma omp parallel for
        for (int idx=0; idx<matrix_size/2; idx++ ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double *ptr = (double*)input.get_data() + 2 * current_idx;
                    __m256d data = _mm256_loadu_pd(ptr);

                    __m256d data_u0 = _mm256_mul_pd(data, mv00);
                    __m256d data_u1 = _mm256_mul_pd(data, mv01);
                    __m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
                    data_u2 = _mm256_permute4x64_pd(data_u2, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

                   __m256d data_d0 = _mm256_mul_pd(data, mv20);
                   __m256d data_d1 = _mm256_mul_pd(data, mv21);
                   __m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
                   data_d2 = _mm256_permute4x64_pd(data_d2, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

                   __m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

                   data_r = _mm256_permute4x64_pd(data_r, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
                   _mm256_storeu_pd(ptr, data_r);

                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * 2 *sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }

        
        }

    }
    else {


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

        __m256d mv00 = _mm256_set_pd(-u3_1qbit[0].imag, u3_1qbit[0].real, -u3_1qbit[0].imag, u3_1qbit[0].real);
        __m256d mv01 = _mm256_set_pd( u3_1qbit[0].real, u3_1qbit[0].imag,  u3_1qbit[0].real, u3_1qbit[0].imag);
        __m256d mv10 = _mm256_set_pd(-u3_1qbit[1].imag, u3_1qbit[1].real, -u3_1qbit[1].imag, u3_1qbit[1].real);
        __m256d mv11 = _mm256_set_pd( u3_1qbit[1].real, u3_1qbit[1].imag,  u3_1qbit[1].real, u3_1qbit[1].imag);
        __m256d mv20 = _mm256_set_pd(-u3_1qbit[2].imag, u3_1qbit[2].real, -u3_1qbit[2].imag, u3_1qbit[2].real);
        __m256d mv21 = _mm256_set_pd( u3_1qbit[2].real, u3_1qbit[2].imag,  u3_1qbit[2].real, u3_1qbit[2].imag);
        __m256d mv30 = _mm256_set_pd(-u3_1qbit[3].imag, u3_1qbit[3].real, -u3_1qbit[3].imag, u3_1qbit[3].real);
        __m256d mv31 = _mm256_set_pd( u3_1qbit[3].real, u3_1qbit[3].imag,  u3_1qbit[3].real, u3_1qbit[3].imag);

#pragma omp parallel for
        for (int idx=0; idx<matrix_size/2; idx=idx+2 ) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;

                   
                    __m256d data0 = _mm256_loadu_pd(element);
                    __m256d data1 = _mm256_loadu_pd(element_pair);

                    __m256d data_u2 = _mm256_mul_pd(data0, mv00);
                    __m256d data_u3 = _mm256_mul_pd(data1, mv10);
                    __m256d data_u4 = _mm256_mul_pd(data0, mv01);
                    __m256d data_u5 = _mm256_mul_pd(data1, mv11);

                    __m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
                    __m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

                    __m256d data_d2 = _mm256_mul_pd(data0, mv20);
                    __m256d data_d3 = _mm256_mul_pd(data1, mv30);
                    __m256d data_d4 = _mm256_mul_pd(data0, mv21);
                    __m256d data_d5 = _mm256_mul_pd(data1, mv31);

                    __m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
                    __m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

                    __m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
                    __m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

                    _mm256_storeu_pd(element, data_r0);
                    _mm256_storeu_pd(element_pair, data_r1);


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



    } // else

}





/**
@brief Parallel AVX kernel on a state vector (parallelized with Intel TBB)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_state_vector_input_parallel_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {


    int grain_size = 64;

    unsigned int bitmask_low = (1 << target_qbit) - 1;
    unsigned int bitmask_high = ~bitmask_low;

    int control_qbit_step_index = (1<<control_qbit);

    tbb::affinity_partitioner aff_p;

    if ( control_qbit == 0 ) {
        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/2,grain_size), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx++) {
            

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if ( current_idx & control_qbit_step_index ) {

                    QGD_Complex16 element      = input[current_idx];
                    QGD_Complex16 element_pair = input[current_idx_pair];


                    QGD_Complex16&& tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16&& tmp2 = mult(u3_1qbit[1], element_pair);

                    input[current_idx].real = tmp1.real + tmp2.real;
                    input[current_idx].imag = tmp1.imag + tmp2.imag;

                    QGD_Complex16&& tmp3 = mult(u3_1qbit[2], element);
                    QGD_Complex16&& tmp4 = mult(u3_1qbit[3], element_pair);

                    input[current_idx_pair].real = tmp3.real + tmp4.real;
                    input[current_idx_pair].imag = tmp3.imag + tmp4.imag;

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
        }, aff_p);


    }
    else if ( target_qbit == 0 ) {

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

        __m256d mv00 = _mm256_set_pd(-u3_1qbit[1].imag, u3_1qbit[1].real, -u3_1qbit[0].imag, u3_1qbit[0].real);
        __m256d mv01 = _mm256_set_pd( u3_1qbit[1].real, u3_1qbit[1].imag,  u3_1qbit[0].real, u3_1qbit[0].imag);
        __m256d mv20 = _mm256_set_pd(-u3_1qbit[3].imag, u3_1qbit[3].real, -u3_1qbit[2].imag, u3_1qbit[2].real);
        __m256d mv21 = _mm256_set_pd( u3_1qbit[3].real, u3_1qbit[3].imag,  u3_1qbit[2].real, u3_1qbit[2].imag);

        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/2,grain_size), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx++) {

                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double *ptr = (double*)input.get_data() + 2 * current_idx;
                    __m256d data = _mm256_loadu_pd(ptr);

                    __m256d data_u0 = _mm256_mul_pd(data, mv00);
                    __m256d data_u1 = _mm256_mul_pd(data, mv01);
                    __m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
                    data_u2 = _mm256_permute4x64_pd(data_u2, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

                   __m256d data_d0 = _mm256_mul_pd(data, mv20);
                   __m256d data_d1 = _mm256_mul_pd(data, mv21);
                   __m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
                   data_d2 = _mm256_permute4x64_pd(data_d2, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

                   __m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

                   data_r = _mm256_permute4x64_pd(data_r, 216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
                   _mm256_storeu_pd(ptr, data_r);

                }
                else if (deriv) {
                    // when calculating derivatives, the constant element should be zeros
                    memset(input.get_data() + current_idx, 0.0, input.cols * 2 *sizeof(QGD_Complex16));
                }
                else {
                    // leave the state as it is
                    continue;
                }

                
            }
        }, aff_p);

    }
    else {

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
        __m256d mv00 = _mm256_set_pd(-u3_1qbit[0].imag, u3_1qbit[0].real, -u3_1qbit[0].imag, u3_1qbit[0].real);
        __m256d mv01 = _mm256_set_pd( u3_1qbit[0].real, u3_1qbit[0].imag,  u3_1qbit[0].real, u3_1qbit[0].imag);
        __m256d mv10 = _mm256_set_pd(-u3_1qbit[1].imag, u3_1qbit[1].real, -u3_1qbit[1].imag, u3_1qbit[1].real);
        __m256d mv11 = _mm256_set_pd( u3_1qbit[1].real, u3_1qbit[1].imag,  u3_1qbit[1].real, u3_1qbit[1].imag);
        __m256d mv20 = _mm256_set_pd(-u3_1qbit[2].imag, u3_1qbit[2].real, -u3_1qbit[2].imag, u3_1qbit[2].real);
        __m256d mv21 = _mm256_set_pd( u3_1qbit[2].real, u3_1qbit[2].imag,  u3_1qbit[2].real, u3_1qbit[2].imag);
        __m256d mv30 = _mm256_set_pd(-u3_1qbit[3].imag, u3_1qbit[3].real, -u3_1qbit[3].imag, u3_1qbit[3].real);
        __m256d mv31 = _mm256_set_pd( u3_1qbit[3].real, u3_1qbit[3].imag,  u3_1qbit[3].real, u3_1qbit[3].imag);


        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/2,grain_size), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx=idx+2) {
                // generate index by inserting state 0 into the place of the target qbit while pushing high bits left by one
                int current_idx = ((idx & bitmask_high) << 1) | (idx & bitmask_low);
		
                // the index pair with target qubit state 1
                int current_idx_pair = current_idx | (1<<target_qbit);

                if (control_qbit < 0 || (current_idx & control_qbit_step_index) ) {


                    double* element = (double*)input.get_data() + 2 * current_idx;
                    double* element_pair = (double*)input.get_data() + 2 * current_idx_pair;

                   
                    __m256d data0 = _mm256_loadu_pd(element);
                    __m256d data1 = _mm256_loadu_pd(element_pair);

                    __m256d data_u2 = _mm256_mul_pd(data0, mv00);
                    __m256d data_u3 = _mm256_mul_pd(data1, mv10);
                    __m256d data_u4 = _mm256_mul_pd(data0, mv01);
                    __m256d data_u5 = _mm256_mul_pd(data1, mv11);

                    __m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
                    __m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

                    __m256d data_d2 = _mm256_mul_pd(data0, mv20);
                    __m256d data_d3 = _mm256_mul_pd(data1, mv30);
                    __m256d data_d4 = _mm256_mul_pd(data0, mv21);
                    __m256d data_d5 = _mm256_mul_pd(data1, mv31);

                    __m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
                    __m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

                    __m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
                    __m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

                    _mm256_storeu_pd(element, data_r0);
                    _mm256_storeu_pd(element_pair, data_r1);


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
        }, aff_p);



    } // else




}







