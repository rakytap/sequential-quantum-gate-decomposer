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
#include <type_traits>
#include <utility>

template<typename MatrixT>
using StateVecComplexT = typename std::remove_reference<decltype(std::declval<MatrixT&>()[0])>::type;

template<typename MatrixT>
void
apply_kernel_to_state_vector_input_impl(MatrixT& u3_1qbit, MatrixT& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

/**
@brief Call to apply a gate kernel on a state vector
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
    using ComplexT = StateVecComplexT<MatrixT>;


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

                ComplexT element = input[row_offset];
                ComplexT element_pair = input[row_offset_pair];

                ComplexT tmp1 = mult(u3_1qbit[0], element);
                ComplexT tmp2 = mult(u3_1qbit[1], element_pair);

                input[row_offset].real = tmp1.real + tmp2.real;
                input[row_offset].imag = tmp1.imag + tmp2.imag;

                tmp1 = mult(u3_1qbit[2], element);
                tmp2 = mult(u3_1qbit[3], element_pair);

                input[row_offset_pair].real = tmp1.real + tmp2.real;
                input[row_offset_pair].imag = tmp1.imag + tmp2.imag;



            }
            else if (deriv) {
                // when calculating derivatives, the constant element should be zeros
                memset(input.get_data() + row_offset, 0, input.cols * sizeof(ComplexT));
                memset(input.get_data() + row_offset_pair, 0, input.cols * sizeof(ComplexT));
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


void
apply_kernel_to_state_vector_input(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    apply_kernel_to_state_vector_input_impl(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
}


void
apply_kernel_to_state_vector_input(Matrix_float& u3_1qbit, Matrix_float& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    apply_kernel_to_state_vector_input_impl(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
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
template<typename MatrixT>
void
apply_kernel_to_state_vector_input_parallel_impl(MatrixT& u3_1qbit, MatrixT& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

    using ComplexT = StateVecComplexT<MatrixT>;


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

                        ComplexT element = input[row_offset];
                        ComplexT element_pair = input[row_offset_pair];

                        ComplexT tmp1 = mult(u3_1qbit[0], element);
                        ComplexT tmp2 = mult(u3_1qbit[1], element_pair);

                        input[row_offset].real = tmp1.real + tmp2.real;
                        input[row_offset].imag = tmp1.imag + tmp2.imag;

                        tmp1 = mult(u3_1qbit[2], element);
                        tmp2 = mult(u3_1qbit[3], element_pair);

                        input[row_offset_pair].real = tmp1.real + tmp2.real;
                        input[row_offset_pair].imag = tmp1.imag + tmp2.imag;



                    }
                    else if (deriv) {
                        // when calculating derivatives, the constant element should be zeros
                        memset(input.get_data() + row_offset, 0, input.cols * sizeof(ComplexT));
                        memset(input.get_data() + row_offset_pair, 0, input.cols * sizeof(ComplexT));
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


void
apply_kernel_to_state_vector_input_parallel(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    apply_kernel_to_state_vector_input_parallel_impl(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
}


void
apply_kernel_to_state_vector_input_parallel(Matrix_float& u3_1qbit, Matrix_float& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    apply_kernel_to_state_vector_input_parallel_impl(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
}


template<typename MatrixT>
void apply_large_state_vector_2q_impl(MatrixT& two_qbit_unitary, MatrixT& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size) {

    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;

    for (int current_idx_pair_outer = current_idx + index_step_outer; current_idx_pair_outer < input.rows; current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1)) {

        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner = current_idx_inner + (index_step_inner << 1)) {

            for (int idx = 0; idx < index_step_inner; idx++) {

                int current_idx_outer_loc = current_idx + current_idx_inner + idx;
                int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
                int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
                int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
                int indexes[4] = {current_idx_outer_loc, current_idx_inner_loc, current_idx_outer_pair_loc, current_idx_inner_pair_loc};

                StateVecComplexT<MatrixT> element_outer = input[current_idx_outer_loc];
                StateVecComplexT<MatrixT> element_outer_pair = input[current_idx_outer_pair_loc];
                StateVecComplexT<MatrixT> element_inner = input[current_idx_inner_loc];
                StateVecComplexT<MatrixT> element_inner_pair = input[current_idx_inner_pair_loc];

                StateVecComplexT<MatrixT> tmp1;
                StateVecComplexT<MatrixT> tmp2;
                StateVecComplexT<MatrixT> tmp3;
                StateVecComplexT<MatrixT> tmp4;
                for (int mult_idx = 0; mult_idx < 4; mult_idx++) {

                    tmp1 = mult(two_qbit_unitary[mult_idx * 4], element_outer);
                    tmp2 = mult(two_qbit_unitary[mult_idx * 4 + 1], element_inner);
                    tmp3 = mult(two_qbit_unitary[mult_idx * 4 + 2], element_outer_pair);
                    tmp4 = mult(two_qbit_unitary[mult_idx * 4 + 3], element_inner_pair);
                    input[indexes[mult_idx]].real = tmp1.real + tmp2.real + tmp3.real + tmp4.real;
                    input[indexes[mult_idx]].imag = tmp1.imag + tmp2.imag + tmp3.imag + tmp4.imag;
                }
            }
        }
        current_idx = current_idx + (index_step_outer << 1);
    }

    (void)matrix_size;
}

template<typename MatrixT>
void apply_large_state_vector_3q_impl(MatrixT& unitary, MatrixT& input, std::vector<int> involved_qbits, const int& matrix_size) {

    int index_step_inner = 1 << involved_qbits[0];
    int index_step_middle = 1 << involved_qbits[1];
    int index_step_outer = 1 << involved_qbits[2];
    int current_idx = 0;

    for (int current_idx_pair_outer = current_idx + index_step_outer; current_idx_pair_outer < input.rows; current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1)) {

        for (int current_idx_middle = 0; current_idx_middle < index_step_outer; current_idx_middle = current_idx_middle + (index_step_middle << 1)) {

            for (int current_idx_inner = 0; current_idx_inner < index_step_middle; current_idx_inner = current_idx_inner + (index_step_inner << 1)) {

                for (int idx = 0; idx < index_step_inner; idx++) {

                    int current_idx_loc = current_idx + current_idx_middle + current_idx_inner + idx;
                    int current_idx_pair_loc = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle;

                    int current_idx_outer_loc = current_idx_loc;
                    int current_idx_inner_loc = current_idx_loc + index_step_inner;

                    int current_idx_middle_loc = current_idx_loc + index_step_middle;
                    int current_idx_middle_inner_loc = current_idx_loc + index_step_middle + index_step_inner;

                    int current_idx_outer_pair_loc = current_idx_pair_loc;
                    int current_idx_inner_pair_loc = current_idx_pair_loc + index_step_inner;

                    int current_idx_middle_pair_loc = current_idx_pair_loc + index_step_middle;
                    int current_idx_middle_inner_pair_loc = current_idx_pair_loc + index_step_middle + index_step_inner;

                    int indexes[8] = {current_idx_outer_loc, current_idx_inner_loc, current_idx_middle_loc, current_idx_middle_inner_loc, current_idx_outer_pair_loc, current_idx_inner_pair_loc, current_idx_middle_pair_loc, current_idx_middle_inner_pair_loc};
                    StateVecComplexT<MatrixT> element_outer = input[current_idx_outer_loc];
                    StateVecComplexT<MatrixT> element_outer_pair = input[current_idx_outer_pair_loc];

                    StateVecComplexT<MatrixT> element_inner = input[current_idx_inner_loc];
                    StateVecComplexT<MatrixT> element_inner_pair = input[current_idx_inner_pair_loc];

                    StateVecComplexT<MatrixT> element_middle = input[current_idx_middle_loc];
                    StateVecComplexT<MatrixT> element_middle_pair = input[current_idx_middle_pair_loc];

                    StateVecComplexT<MatrixT> element_middle_inner = input[current_idx_middle_inner_loc];
                    StateVecComplexT<MatrixT> element_middle_inner_pair = input[current_idx_middle_inner_pair_loc];

                    StateVecComplexT<MatrixT> tmp1;
                    StateVecComplexT<MatrixT> tmp2;
                    StateVecComplexT<MatrixT> tmp3;
                    StateVecComplexT<MatrixT> tmp4;
                    StateVecComplexT<MatrixT> tmp5;
                    StateVecComplexT<MatrixT> tmp6;
                    StateVecComplexT<MatrixT> tmp7;
                    StateVecComplexT<MatrixT> tmp8;
                    for (int mult_idx = 0; mult_idx < 8; mult_idx++) {
                        tmp1 = mult(unitary[mult_idx * 8], element_outer);
                        tmp2 = mult(unitary[mult_idx * 8 + 1], element_inner);
                        tmp3 = mult(unitary[mult_idx * 8 + 2], element_middle);
                        tmp4 = mult(unitary[mult_idx * 8 + 3], element_middle_inner);
                        tmp5 = mult(unitary[mult_idx * 8 + 4], element_outer_pair);
                        tmp6 = mult(unitary[mult_idx * 8 + 5], element_inner_pair);
                        tmp7 = mult(unitary[mult_idx * 8 + 6], element_middle_pair);
                        tmp8 = mult(unitary[mult_idx * 8 + 7], element_middle_inner_pair);
                        input[indexes[mult_idx]].real = tmp1.real + tmp2.real + tmp3.real + tmp4.real + tmp5.real + tmp6.real + tmp7.real + tmp8.real;
                        input[indexes[mult_idx]].imag = tmp1.imag + tmp2.imag + tmp3.imag + tmp4.imag + tmp5.imag + tmp6.imag + tmp7.imag + tmp8.imag;
                    }
                }
            }
        }
        current_idx = current_idx + (index_step_outer << 1);
    }

    (void)matrix_size;
}

template<typename MatrixT>
void apply_large_state_vector_4q_impl(MatrixT& unitary, MatrixT& input, std::vector<int> involved_qbits, const int& matrix_size) {

    int index_step_q0 = 1 << involved_qbits[0];
    int index_step_q1 = 1 << involved_qbits[1];
    int index_step_q2 = 1 << involved_qbits[2];
    int index_step_q3 = 1 << involved_qbits[3];

    int current_idx = 0;

    for (int current_idx_pair_q3 = current_idx + index_step_q3; current_idx_pair_q3 < input.rows; current_idx_pair_q3 += (index_step_q3 << 1)) {
        for (int current_idx_q2 = 0; current_idx_q2 < index_step_q3; current_idx_q2 += (index_step_q2 << 1)) {
            for (int current_idx_q1 = 0; current_idx_q1 < index_step_q2; current_idx_q1 += (index_step_q1 << 1)) {
                for (int current_idx_q0 = 0; current_idx_q0 < index_step_q1; current_idx_q0 += (index_step_q0 << 1)) {
                    for (int idx = 0; idx < index_step_q0; idx++) {

                        int current_idx_loc = current_idx + current_idx_q2 + current_idx_q1 + current_idx_q0 + idx;
                        int current_idx_pair_loc = current_idx_pair_q3 + idx + current_idx_q1 + current_idx_q2 + current_idx_q0;

                        int current_idx_q0_0_loc = current_idx_loc;
                        int current_idx_q0_1_loc = current_idx_loc + index_step_q0;
                        int current_idx_q1_0_loc = current_idx_loc + index_step_q1;
                        int current_idx_q1_1_loc = current_idx_loc + index_step_q1 + index_step_q0;
                        int current_idx_q2_0_loc = current_idx_loc + index_step_q2;
                        int current_idx_q2_1_loc = current_idx_loc + index_step_q2 + index_step_q0;
                        int current_idx_q2_q1_0_loc = current_idx_loc + index_step_q2 + index_step_q1;
                        int current_idx_q2_q1_1_loc = current_idx_loc + index_step_q2 + index_step_q1 + index_step_q0;

                        int current_idx_q3_q0_0_pair_loc = current_idx_pair_loc;
                        int current_idx_q3_q0_1_pair_loc = current_idx_pair_loc + index_step_q0;
                        int current_idx_q3_q1_0_pair_loc = current_idx_pair_loc + index_step_q1;
                        int current_idx_q3_q1_1_pair_loc = current_idx_pair_loc + index_step_q1 + index_step_q0;
                        int current_idx_q3_q2_0_pair_loc = current_idx_pair_loc + index_step_q2;
                        int current_idx_q3_q2_1_pair_loc = current_idx_pair_loc + index_step_q2 + index_step_q0;
                        int current_idx_q3_q2_q1_0_pair_loc = current_idx_pair_loc + index_step_q2 + index_step_q1;
                        int current_idx_q3_q2_q1_1_pair_loc = current_idx_pair_loc + index_step_q2 + index_step_q1 + index_step_q0;

                        int indexes[16] = {
                            current_idx_q0_0_loc, current_idx_q0_1_loc, current_idx_q1_0_loc, current_idx_q1_1_loc,
                            current_idx_q2_0_loc, current_idx_q2_1_loc, current_idx_q2_q1_0_loc, current_idx_q2_q1_1_loc,
                            current_idx_q3_q0_0_pair_loc, current_idx_q3_q0_1_pair_loc, current_idx_q3_q1_0_pair_loc, current_idx_q3_q1_1_pair_loc,
                            current_idx_q3_q2_0_pair_loc, current_idx_q3_q2_1_pair_loc, current_idx_q3_q2_q1_0_pair_loc, current_idx_q3_q2_q1_1_pair_loc
                        };

                        StateVecComplexT<MatrixT> element_0 = input[current_idx_q0_0_loc];
                        StateVecComplexT<MatrixT> element_1 = input[current_idx_q0_1_loc];
                        StateVecComplexT<MatrixT> element_2 = input[current_idx_q1_0_loc];
                        StateVecComplexT<MatrixT> element_3 = input[current_idx_q1_1_loc];
                        StateVecComplexT<MatrixT> element_4 = input[current_idx_q2_0_loc];
                        StateVecComplexT<MatrixT> element_5 = input[current_idx_q2_1_loc];
                        StateVecComplexT<MatrixT> element_6 = input[current_idx_q2_q1_0_loc];
                        StateVecComplexT<MatrixT> element_7 = input[current_idx_q2_q1_1_loc];
                        StateVecComplexT<MatrixT> element_8 = input[current_idx_q3_q0_0_pair_loc];
                        StateVecComplexT<MatrixT> element_9 = input[current_idx_q3_q0_1_pair_loc];
                        StateVecComplexT<MatrixT> element_10 = input[current_idx_q3_q1_0_pair_loc];
                        StateVecComplexT<MatrixT> element_11 = input[current_idx_q3_q1_1_pair_loc];
                        StateVecComplexT<MatrixT> element_12 = input[current_idx_q3_q2_0_pair_loc];
                        StateVecComplexT<MatrixT> element_13 = input[current_idx_q3_q2_1_pair_loc];
                        StateVecComplexT<MatrixT> element_14 = input[current_idx_q3_q2_q1_0_pair_loc];
                        StateVecComplexT<MatrixT> element_15 = input[current_idx_q3_q2_q1_1_pair_loc];

                        for (int mult_idx = 0; mult_idx < 16; mult_idx++) {
                            StateVecComplexT<MatrixT> tmp0 = mult(unitary[mult_idx * 16], element_0);
                            StateVecComplexT<MatrixT> tmp1 = mult(unitary[mult_idx * 16 + 1], element_1);
                            StateVecComplexT<MatrixT> tmp2 = mult(unitary[mult_idx * 16 + 2], element_2);
                            StateVecComplexT<MatrixT> tmp3 = mult(unitary[mult_idx * 16 + 3], element_3);
                            StateVecComplexT<MatrixT> tmp4 = mult(unitary[mult_idx * 16 + 4], element_4);
                            StateVecComplexT<MatrixT> tmp5 = mult(unitary[mult_idx * 16 + 5], element_5);
                            StateVecComplexT<MatrixT> tmp6 = mult(unitary[mult_idx * 16 + 6], element_6);
                            StateVecComplexT<MatrixT> tmp7 = mult(unitary[mult_idx * 16 + 7], element_7);
                            StateVecComplexT<MatrixT> tmp8 = mult(unitary[mult_idx * 16 + 8], element_8);
                            StateVecComplexT<MatrixT> tmp9 = mult(unitary[mult_idx * 16 + 9], element_9);
                            StateVecComplexT<MatrixT> tmp10 = mult(unitary[mult_idx * 16 + 10], element_10);
                            StateVecComplexT<MatrixT> tmp11 = mult(unitary[mult_idx * 16 + 11], element_11);
                            StateVecComplexT<MatrixT> tmp12 = mult(unitary[mult_idx * 16 + 12], element_12);
                            StateVecComplexT<MatrixT> tmp13 = mult(unitary[mult_idx * 16 + 13], element_13);
                            StateVecComplexT<MatrixT> tmp14 = mult(unitary[mult_idx * 16 + 14], element_14);
                            StateVecComplexT<MatrixT> tmp15 = mult(unitary[mult_idx * 16 + 15], element_15);

                            input[indexes[mult_idx]].real = tmp0.real + tmp1.real + tmp2.real + tmp3.real
                                + tmp4.real + tmp5.real + tmp6.real + tmp7.real
                                + tmp8.real + tmp9.real + tmp10.real + tmp11.real
                                + tmp12.real + tmp13.real + tmp14.real + tmp15.real;

                            input[indexes[mult_idx]].imag = tmp0.imag + tmp1.imag + tmp2.imag + tmp3.imag
                                + tmp4.imag + tmp5.imag + tmp6.imag + tmp7.imag
                                + tmp8.imag + tmp9.imag + tmp10.imag + tmp11.imag
                                + tmp12.imag + tmp13.imag + tmp14.imag + tmp15.imag;
                        }
                    }
                }
            }
        }
        current_idx += (index_step_q3 << 1);
    }

    (void)matrix_size;
}

template<typename MatrixT>
void apply_large_state_vector_5q_impl(MatrixT& unitary, MatrixT& input, std::vector<int> involved_qbits, const int& matrix_size) {

    int index_step_q0 = 1 << involved_qbits[0];
    int index_step_q1 = 1 << involved_qbits[1];
    int index_step_q2 = 1 << involved_qbits[2];
    int index_step_q3 = 1 << involved_qbits[3];
    int index_step_q4 = 1 << involved_qbits[4];

    int current_idx = 0;

    for (int current_idx_pair_q4 = current_idx + index_step_q4; current_idx_pair_q4 < input.rows; current_idx_pair_q4 += (index_step_q4 << 1)) {
        for (int current_idx_q3 = 0; current_idx_q3 < index_step_q4; current_idx_q3 += (index_step_q3 << 1)) {
            for (int current_idx_q2 = 0; current_idx_q2 < index_step_q3; current_idx_q2 += (index_step_q2 << 1)) {
                for (int current_idx_q1 = 0; current_idx_q1 < index_step_q2; current_idx_q1 += (index_step_q1 << 1)) {
                    for (int current_idx_q0 = 0; current_idx_q0 < index_step_q1; current_idx_q0 += (index_step_q0 << 1)) {
                        for (int idx = 0; idx < index_step_q0; idx++) {

                            int current_idx_loc = current_idx + current_idx_q3 + current_idx_q2 + current_idx_q1 + current_idx_q0 + idx;
                            int current_idx_pair_q4_loc = current_idx_pair_q4 + idx + current_idx_q1 + current_idx_q2 + current_idx_q3 + current_idx_q0;

                            int current_idx_q0_0_loc = current_idx_loc;
                            int current_idx_q0_1_loc = current_idx_loc + index_step_q0;
                            int current_idx_q1_0_loc = current_idx_loc + index_step_q1;
                            int current_idx_q1_1_loc = current_idx_loc + index_step_q1 + index_step_q0;
                            int current_idx_q2_0_loc = current_idx_loc + index_step_q2;
                            int current_idx_q2_1_loc = current_idx_loc + index_step_q2 + index_step_q0;
                            int current_idx_q2_q1_0_loc = current_idx_loc + index_step_q2 + index_step_q1;
                            int current_idx_q2_q1_1_loc = current_idx_loc + index_step_q2 + index_step_q1 + index_step_q0;
                            int current_idx_q3_0_loc = current_idx_loc + index_step_q3;
                            int current_idx_q3_1_loc = current_idx_loc + index_step_q3 + index_step_q0;
                            int current_idx_q3_q1_0_loc = current_idx_loc + index_step_q3 + index_step_q1;
                            int current_idx_q3_q1_1_loc = current_idx_loc + index_step_q3 + index_step_q1 + index_step_q0;
                            int current_idx_q3_q2_0_loc = current_idx_loc + index_step_q3 + index_step_q2;
                            int current_idx_q3_q2_1_loc = current_idx_loc + index_step_q3 + index_step_q2 + index_step_q0;
                            int current_idx_q3_q2_q1_0_loc = current_idx_loc + index_step_q3 + index_step_q2 + index_step_q1;
                            int current_idx_q3_q2_q1_1_loc = current_idx_loc + index_step_q3 + index_step_q2 + index_step_q1 + index_step_q0;

                            int current_idx_q4_q0_0_pair_loc = current_idx_pair_q4_loc;
                            int current_idx_q4_q0_1_pair_loc = current_idx_pair_q4_loc + index_step_q0;
                            int current_idx_q4_q1_0_pair_loc = current_idx_pair_q4_loc + index_step_q1;
                            int current_idx_q4_q1_1_pair_loc = current_idx_pair_q4_loc + index_step_q1 + index_step_q0;
                            int current_idx_q4_q2_0_pair_loc = current_idx_pair_q4_loc + index_step_q2;
                            int current_idx_q4_q2_1_pair_loc = current_idx_pair_q4_loc + index_step_q2 + index_step_q0;
                            int current_idx_q4_q2_q1_0_pair_loc = current_idx_pair_q4_loc + index_step_q2 + index_step_q1;
                            int current_idx_q4_q2_q1_1_pair_loc = current_idx_pair_q4_loc + index_step_q2 + index_step_q1 + index_step_q0;
                            int current_idx_q4_q3_0_pair_loc = current_idx_pair_q4_loc + index_step_q3;
                            int current_idx_q4_q3_1_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q0;
                            int current_idx_q4_q3_q1_0_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q1;
                            int current_idx_q4_q3_q1_1_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q1 + index_step_q0;
                            int current_idx_q4_q3_q2_0_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q2;
                            int current_idx_q4_q3_q2_1_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q2 + index_step_q0;
                            int current_idx_q4_q3_q2_q1_0_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q2 + index_step_q1;
                            int current_idx_q4_q3_q2_q1_1_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q2 + index_step_q1 + index_step_q0;

                            int indexes[32] = {
                                current_idx_q0_0_loc, current_idx_q0_1_loc, current_idx_q1_0_loc, current_idx_q1_1_loc,
                                current_idx_q2_0_loc, current_idx_q2_1_loc, current_idx_q2_q1_0_loc, current_idx_q2_q1_1_loc,
                                current_idx_q3_0_loc, current_idx_q3_1_loc, current_idx_q3_q1_0_loc, current_idx_q3_q1_1_loc,
                                current_idx_q3_q2_0_loc, current_idx_q3_q2_1_loc, current_idx_q3_q2_q1_0_loc, current_idx_q3_q2_q1_1_loc,
                                current_idx_q4_q0_0_pair_loc, current_idx_q4_q0_1_pair_loc, current_idx_q4_q1_0_pair_loc, current_idx_q4_q1_1_pair_loc,
                                current_idx_q4_q2_0_pair_loc, current_idx_q4_q2_1_pair_loc, current_idx_q4_q2_q1_0_pair_loc, current_idx_q4_q2_q1_1_pair_loc,
                                current_idx_q4_q3_0_pair_loc, current_idx_q4_q3_1_pair_loc, current_idx_q4_q3_q1_0_pair_loc, current_idx_q4_q3_q1_1_pair_loc,
                                current_idx_q4_q3_q2_0_pair_loc, current_idx_q4_q3_q2_1_pair_loc, current_idx_q4_q3_q2_q1_0_pair_loc, current_idx_q4_q3_q2_q1_1_pair_loc
                            };

                            StateVecComplexT<MatrixT> element_0 = input[current_idx_q0_0_loc];
                            StateVecComplexT<MatrixT> element_1 = input[current_idx_q0_1_loc];
                            StateVecComplexT<MatrixT> element_2 = input[current_idx_q1_0_loc];
                            StateVecComplexT<MatrixT> element_3 = input[current_idx_q1_1_loc];
                            StateVecComplexT<MatrixT> element_4 = input[current_idx_q2_0_loc];
                            StateVecComplexT<MatrixT> element_5 = input[current_idx_q2_1_loc];
                            StateVecComplexT<MatrixT> element_6 = input[current_idx_q2_q1_0_loc];
                            StateVecComplexT<MatrixT> element_7 = input[current_idx_q2_q1_1_loc];
                            StateVecComplexT<MatrixT> element_8 = input[current_idx_q3_0_loc];
                            StateVecComplexT<MatrixT> element_9 = input[current_idx_q3_1_loc];
                            StateVecComplexT<MatrixT> element_10 = input[current_idx_q3_q1_0_loc];
                            StateVecComplexT<MatrixT> element_11 = input[current_idx_q3_q1_1_loc];
                            StateVecComplexT<MatrixT> element_12 = input[current_idx_q3_q2_0_loc];
                            StateVecComplexT<MatrixT> element_13 = input[current_idx_q3_q2_1_loc];
                            StateVecComplexT<MatrixT> element_14 = input[current_idx_q3_q2_q1_0_loc];
                            StateVecComplexT<MatrixT> element_15 = input[current_idx_q3_q2_q1_1_loc];
                            StateVecComplexT<MatrixT> element_16 = input[current_idx_q4_q0_0_pair_loc];
                            StateVecComplexT<MatrixT> element_17 = input[current_idx_q4_q0_1_pair_loc];
                            StateVecComplexT<MatrixT> element_18 = input[current_idx_q4_q1_0_pair_loc];
                            StateVecComplexT<MatrixT> element_19 = input[current_idx_q4_q1_1_pair_loc];
                            StateVecComplexT<MatrixT> element_20 = input[current_idx_q4_q2_0_pair_loc];
                            StateVecComplexT<MatrixT> element_21 = input[current_idx_q4_q2_1_pair_loc];
                            StateVecComplexT<MatrixT> element_22 = input[current_idx_q4_q2_q1_0_pair_loc];
                            StateVecComplexT<MatrixT> element_23 = input[current_idx_q4_q2_q1_1_pair_loc];
                            StateVecComplexT<MatrixT> element_24 = input[current_idx_q4_q3_0_pair_loc];
                            StateVecComplexT<MatrixT> element_25 = input[current_idx_q4_q3_1_pair_loc];
                            StateVecComplexT<MatrixT> element_26 = input[current_idx_q4_q3_q1_0_pair_loc];
                            StateVecComplexT<MatrixT> element_27 = input[current_idx_q4_q3_q1_1_pair_loc];
                            StateVecComplexT<MatrixT> element_28 = input[current_idx_q4_q3_q2_0_pair_loc];
                            StateVecComplexT<MatrixT> element_29 = input[current_idx_q4_q3_q2_1_pair_loc];
                            StateVecComplexT<MatrixT> element_30 = input[current_idx_q4_q3_q2_q1_0_pair_loc];
                            StateVecComplexT<MatrixT> element_31 = input[current_idx_q4_q3_q2_q1_1_pair_loc];

                            for (int mult_idx = 0; mult_idx < 32; mult_idx++) {
                                StateVecComplexT<MatrixT> tmp1 = mult(unitary[mult_idx * 32], element_0);
                                StateVecComplexT<MatrixT> tmp2 = mult(unitary[mult_idx * 32 + 1], element_1);
                                StateVecComplexT<MatrixT> tmp3 = mult(unitary[mult_idx * 32 + 2], element_2);
                                StateVecComplexT<MatrixT> tmp4 = mult(unitary[mult_idx * 32 + 3], element_3);
                                StateVecComplexT<MatrixT> tmp5 = mult(unitary[mult_idx * 32 + 4], element_4);
                                StateVecComplexT<MatrixT> tmp6 = mult(unitary[mult_idx * 32 + 5], element_5);
                                StateVecComplexT<MatrixT> tmp7 = mult(unitary[mult_idx * 32 + 6], element_6);
                                StateVecComplexT<MatrixT> tmp8 = mult(unitary[mult_idx * 32 + 7], element_7);
                                StateVecComplexT<MatrixT> tmp9 = mult(unitary[mult_idx * 32 + 8], element_8);
                                StateVecComplexT<MatrixT> tmp10 = mult(unitary[mult_idx * 32 + 9], element_9);
                                StateVecComplexT<MatrixT> tmp11 = mult(unitary[mult_idx * 32 + 10], element_10);
                                StateVecComplexT<MatrixT> tmp12 = mult(unitary[mult_idx * 32 + 11], element_11);
                                StateVecComplexT<MatrixT> tmp13 = mult(unitary[mult_idx * 32 + 12], element_12);
                                StateVecComplexT<MatrixT> tmp14 = mult(unitary[mult_idx * 32 + 13], element_13);
                                StateVecComplexT<MatrixT> tmp15 = mult(unitary[mult_idx * 32 + 14], element_14);
                                StateVecComplexT<MatrixT> tmp16 = mult(unitary[mult_idx * 32 + 15], element_15);
                                StateVecComplexT<MatrixT> tmp17 = mult(unitary[mult_idx * 32 + 16], element_16);
                                StateVecComplexT<MatrixT> tmp18 = mult(unitary[mult_idx * 32 + 17], element_17);
                                StateVecComplexT<MatrixT> tmp19 = mult(unitary[mult_idx * 32 + 18], element_18);
                                StateVecComplexT<MatrixT> tmp20 = mult(unitary[mult_idx * 32 + 19], element_19);
                                StateVecComplexT<MatrixT> tmp21 = mult(unitary[mult_idx * 32 + 20], element_20);
                                StateVecComplexT<MatrixT> tmp22 = mult(unitary[mult_idx * 32 + 21], element_21);
                                StateVecComplexT<MatrixT> tmp23 = mult(unitary[mult_idx * 32 + 22], element_22);
                                StateVecComplexT<MatrixT> tmp24 = mult(unitary[mult_idx * 32 + 23], element_23);
                                StateVecComplexT<MatrixT> tmp25 = mult(unitary[mult_idx * 32 + 24], element_24);
                                StateVecComplexT<MatrixT> tmp26 = mult(unitary[mult_idx * 32 + 25], element_25);
                                StateVecComplexT<MatrixT> tmp27 = mult(unitary[mult_idx * 32 + 26], element_26);
                                StateVecComplexT<MatrixT> tmp28 = mult(unitary[mult_idx * 32 + 27], element_27);
                                StateVecComplexT<MatrixT> tmp29 = mult(unitary[mult_idx * 32 + 28], element_28);
                                StateVecComplexT<MatrixT> tmp30 = mult(unitary[mult_idx * 32 + 29], element_29);
                                StateVecComplexT<MatrixT> tmp31 = mult(unitary[mult_idx * 32 + 30], element_30);
                                StateVecComplexT<MatrixT> tmp32 = mult(unitary[mult_idx * 32 + 31], element_31);

                                input[indexes[mult_idx]].real = tmp1.real + tmp2.real + tmp3.real + tmp4.real
                                    + tmp5.real + tmp6.real + tmp7.real + tmp8.real + tmp9.real + tmp10.real
                                    + tmp11.real + tmp12.real + tmp13.real + tmp14.real + tmp15.real + tmp16.real
                                    + tmp17.real + tmp18.real + tmp19.real + tmp20.real + tmp21.real + tmp22.real
                                    + tmp23.real + tmp24.real + tmp25.real + tmp26.real + tmp27.real + tmp28.real
                                    + tmp29.real + tmp30.real + tmp31.real + tmp32.real;

                                input[indexes[mult_idx]].imag = tmp1.imag + tmp2.imag + tmp3.imag + tmp4.imag
                                    + tmp5.imag + tmp6.imag + tmp7.imag + tmp8.imag + tmp9.imag + tmp10.imag
                                    + tmp11.imag + tmp12.imag + tmp13.imag + tmp14.imag + tmp15.imag + tmp16.imag
                                    + tmp17.imag + tmp18.imag + tmp19.imag + tmp20.imag + tmp21.imag + tmp22.imag
                                    + tmp23.imag + tmp24.imag + tmp25.imag + tmp26.imag + tmp27.imag + tmp28.imag
                                    + tmp29.imag + tmp30.imag + tmp31.imag + tmp32.imag;
                            }
                        }
                    }
                }
            }
        }
        current_idx += (index_step_q4 << 1);
    }

    (void)matrix_size;
}

void apply_2qbit_kernel_to_state_vector_input(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size) {
    apply_large_state_vector_2q_impl(two_qbit_unitary, input, inner_qbit, outer_qbit, matrix_size);
}

void apply_2qbit_kernel_to_state_vector_input(Matrix_float& two_qbit_unitary, Matrix_float& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size) {
    apply_large_state_vector_2q_impl(two_qbit_unitary, input, inner_qbit, outer_qbit, matrix_size);
}

void apply_3qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size) {
    apply_large_state_vector_3q_impl(unitary, input, involved_qbits, matrix_size);
}

void apply_3qbit_kernel_to_state_vector_input(Matrix_float& unitary, Matrix_float& input, std::vector<int> involved_qbits, const int& matrix_size) {
    apply_large_state_vector_3q_impl(unitary, input, involved_qbits, matrix_size);
}

void apply_4qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size) {
    apply_large_state_vector_4q_impl(unitary, input, involved_qbits, matrix_size);
}

void apply_4qbit_kernel_to_state_vector_input(Matrix_float& unitary, Matrix_float& input, std::vector<int> involved_qbits, const int& matrix_size) {
    apply_large_state_vector_4q_impl(unitary, input, involved_qbits, matrix_size);
}

void apply_5qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size) {
    apply_large_state_vector_5q_impl(unitary, input, involved_qbits, matrix_size);
}

void apply_5qbit_kernel_to_state_vector_input(Matrix_float& unitary, Matrix_float& input, std::vector<int> involved_qbits, const int& matrix_size) {
    apply_large_state_vector_5q_impl(unitary, input, involved_qbits, matrix_size);
}



