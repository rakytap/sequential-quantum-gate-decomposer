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
//#include <immintrin.h>
#include "tbb/tbb.h"

/**
@brief Call to apply a gate kernel on a state vector
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
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
@brief Call to apply a gate kernel on a state vector. Parallel version
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
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

    tbb::parallel_for( tbb::blocked_range<int>(0, parallel_outer_cycles, outer_grain_size), [&](tbb::blocked_range<int> r) { 

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



