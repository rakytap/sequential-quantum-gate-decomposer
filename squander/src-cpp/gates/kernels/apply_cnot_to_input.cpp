#include "apply_cnot_to_input.h"
#include "tbb/tbb.h"
#include <immintrin.h>

void apply_cnot_kernel_to_state_vector_input_OpenMP(Matrix& input, const int& control_qbit, const int& target_qbit){
/*    int matrix_size = input.size();
    int larger = (ctrl_qbit>trgt_qbit) ? ctrl_qbit:trgt_qbit;
    int smaller = (ctrl_qbit<trgt_qbit) ? ctrl_qbit:trgt_qbit;
    
    int index_step_larger = 1 << larger;
    int index_step_smaller= 1 << smaller;
    int current_idx = 0;
    int ratio = (int) index_step_larger/(index_step_smaller<<1);
#pragma omp parallel for
    for (int idx=0; idx<matrix_size/4; idx++){
                current_idx = (int) (index_step_larger<<1)*(idx/(index_step_smaller*ratio)) + (int) ((index_step_smaller<<1)*((idx%(index_step_smaller*ratio))/index_step_smaller)) + (int) ((idx%(index_step_smaller))); 
                int current_idx_trgt_loc = current_idx + (1<<ctrl_qbit);
                int current_idx_ctrl_loc = current_idx + (1<<ctrl_qbit) + (1<<trgt_qbit);
                double placeholder = input[current_idx_trgt_loc].real;
                input[current_idx_trgt_loc].real = input[current_idx_ctrl_loc].real;
                input[current_idx_ctrl_loc].real = placeholder;
                placeholder = input[current_idx_trgt_loc].imag;
                input[current_idx_trgt_loc].imag = input[current_idx_ctrl_loc].imag;
                input[current_idx_ctrl_loc].imag = placeholder;
            }*/

    int matrix_size = input.size();
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

                    input[current_idx].real = input[current_idx_pair].real;
                    input[current_idx].imag = input[current_idx_pair].imag;

                    input[current_idx_pair].real = element.real;
                    input[current_idx_pair].imag = element.imag;



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

                    _mm256_storeu_pd(element, data1);
                    _mm256_storeu_pd(element_pair, data0);


                }
                else {
                    // leave the state as it is
                    continue;
                }


            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;



        }



    } // else


}

void apply_cnot_kernel_to_state_vector_input(Matrix& input,const int& ctrl_qbit, const int& trgt_qbit){
 int matrix_size = input.size();
    int larger = (ctrl_qbit>trgt_qbit) ? ctrl_qbit:trgt_qbit;
    int smaller = (ctrl_qbit<trgt_qbit) ? ctrl_qbit:trgt_qbit;
    
    int index_step_larger = 1 << larger;
    int index_step_smaller= 1 << smaller;
    int current_idx = 0;
    int ratio = (int) index_step_larger/(index_step_smaller<<1);
    int kernel_size = (40>matrix_size/4) ? 10:2;
    tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/4,kernel_size), [&](tbb::blocked_range<int> r) { 

        for (int idx=r.begin(); idx<r.end(); idx++) {

        
        //Matrix placeholder( 1, placeholder_size);
current_idx = (int) (index_step_larger<<1)*(idx/(index_step_smaller*ratio)) + (int) ((index_step_smaller<<1)*((idx%(index_step_smaller*ratio))/index_step_smaller)) + (int) ((idx%(index_step_smaller))); 
                int current_idx_trgt_loc = current_idx + (1<<ctrl_qbit);
                int current_idx_ctrl_loc = current_idx + (1<<ctrl_qbit) + (1<<trgt_qbit);
                double placeholder = input[current_idx_trgt_loc].real;
                input[current_idx_trgt_loc].real = input[current_idx_ctrl_loc].real;
                input[current_idx_ctrl_loc].real = placeholder;
                placeholder = input[current_idx_trgt_loc].imag;
                input[current_idx_trgt_loc].imag = input[current_idx_ctrl_loc].imag;
                input[current_idx_ctrl_loc].imag = placeholder;
    } });
        
}
