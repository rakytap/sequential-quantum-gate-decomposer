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
/*! \file apply_dedicated_gate_kernel_to_input.cpp
    \brief ????????????????
*/


#include "apply_dedicated_gate_kernel_to_input.h"
//#include <immintrin.h>
#include "tbb/tbb.h"





/**
@brief Call to apply kernel to apply single qubit gate kernel on an input matrix
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void apply_X_kernel_to_input(Matrix& input, const int& target_qbit,
                           const int& control_qbit1, const int& control_qbit2,
                           const int& matrix_size) {


    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    
    for (int current_idx_pair = current_idx + index_step_target; 
         current_idx_pair < matrix_size; 
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {
        
        for(int idx = 0; idx < index_step_target; idx++) {
            
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;
            
            // FIXED: Check each control qubit separately
            bool control1_active = (control_qbit1 < 0) || ((current_idx_loc >> control_qbit1) & 1);
            bool control2_active = (control_qbit2 < 0) || ((current_idx_loc >> control_qbit2) & 1);


            // Apply X gate only when BOTH controls are active
            if (control1_active && control2_active) {
                
                int row_offset = current_idx_loc * input.stride;
                int row_offset_pair = current_idx_pair_loc * input.stride;
                
                std::swap_ranges(
                    input.get_data() + row_offset,
                    input.get_data() + row_offset + input.cols,
                    input.get_data() + row_offset_pair
                );
            }
        }
        current_idx = current_idx + (index_step_target << 1);
    }
}

void apply_Y_kernel_to_input(Matrix& input, const int& target_qbit, 
                           const int& control_qbit, 
                           const int& matrix_size) {
    
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    
    for (int current_idx_pair = current_idx + index_step_target; 
         current_idx_pair < matrix_size; 
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {
        
        for(int idx = 0; idx < index_step_target; idx++) {
            
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;
                        
            // Apply Y gate only when BOTH controls are active
            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                
                int row_offset = current_idx_loc * input.stride;
                int row_offset_pair = current_idx_pair_loc * input.stride;
                
                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index = row_offset + col_idx;
                    int index_pair = row_offset_pair + col_idx;
                    
                    QGD_Complex16 element = input[index];
                    QGD_Complex16 element_pair = input[index_pair];
                    
                    // Y gate transformation
                    input[index].real = element_pair.imag;
                    input[index].imag = -element_pair.real;
                    
                    input[index_pair].real = -element.imag;
                    input[index_pair].imag = element.real;
                }
            }
        }
        current_idx = current_idx + (index_step_target << 1);
    }
}

void apply_Z_kernel_to_input(Matrix& input, const int& target_qbit, 
                           const int& control_qbit, 
                           const int& matrix_size) {
    
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    
    for (int current_idx_pair = current_idx + index_step_target; 
         current_idx_pair < matrix_size; 
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {
        
        for(int idx = 0; idx < index_step_target; idx++) {
            
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;
                        
            // Apply Z gate only when BOTH controls are active
            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                
                int row_offset = current_idx_loc * input.stride;
                int row_offset_pair = current_idx_pair_loc * input.stride;
                
                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index_pair = row_offset_pair + col_idx;
                    
                    // Z gate transformation
                    input[index_pair].real = -input[index_pair].real;
                    input[index_pair].imag = -input[index_pair].imag;
                }
            }
        }
        current_idx = current_idx + (index_step_target << 1);
    }
}

void apply_H_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int current_idx_pair = current_idx + index_step_target;
         current_idx_pair < matrix_size;
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {

        for (int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset = current_idx_loc * input.stride;
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index = row_offset + col_idx;
                    int index_pair = row_offset_pair + col_idx;

                    QGD_Complex16 element = input[index];
                    QGD_Complex16 element_pair = input[index_pair];

                    // H gate transformation
                    input[index].real = inv_sqrt2 * (element.real + element_pair.real);
                    input[index].imag = inv_sqrt2 * (element.imag + element_pair.imag);

                    input[index_pair].real = inv_sqrt2 * (element.real - element_pair.real);
                    input[index_pair].imag = inv_sqrt2 * (element.imag - element_pair.imag);
                }
            }
        }
        current_idx = current_idx + (index_step_target << 1);
    }
}

void apply_S_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;

    for (int current_idx_pair = current_idx + index_step_target;
         current_idx_pair < matrix_size;
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {

        for (int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index_pair = row_offset_pair + col_idx;

                    // S gate transformation (multiply by i)
                    double real = input[index_pair].real;
                    double imag = input[index_pair].imag;
                    input[index_pair].real = -imag;
                    input[index_pair].imag = real;
                }
            }
        }
        current_idx = current_idx + (index_step_target << 1);
    }
}

void apply_T_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    for (int current_idx_pair = current_idx + index_step_target;
         current_idx_pair < matrix_size;
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {

        for (int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index_pair = row_offset_pair + col_idx;

                    // T gate transformation (multiply by exp(i*pi/4) = 1/sqrt(2) + i/sqrt(2))
                    double real = input[index_pair].real;
                    double imag = input[index_pair].imag;
                    input[index_pair].real = inv_sqrt2 * (real - imag);
                    input[index_pair].imag = inv_sqrt2 * (real + imag);
                }
            }
        }
        current_idx = current_idx + (index_step_target << 1);
    }
}

void apply_SWAP_kernel_to_input(Matrix& input, const int& target_qbit1, const int& target_qbit2, const int& control_qbit, const int& matrix_size){

    std::vector<int> non_involved_qbits;
    int qbit_num = (int)std::log2(matrix_size);
    for (int idx=0; idx<qbit_num; idx++){
        if ( (idx != target_qbit1 && idx != target_qbit2) && idx != control_qbit){
            non_involved_qbits.push_back(idx);
        }
    }
    int is_control_involved = control_qbit == -1 ? 0 : 1<<control_qbit;
    for (int block_idx=0; block_idx < matrix_size >> (qbit_num - non_involved_qbits.size()); block_idx++){
        int base = 0;
        for (int qdx=0; qdx<non_involved_qbits.size();qdx++){
            if ((block_idx >> qdx) & 1) {
                base |= (1<<non_involved_qbits[qdx]);
            }
        }
        base |= is_control_involved;
        int swap_idx = base|(1<<target_qbit1);
        int swap_idx_pair = base|(1<<target_qbit2);

        // Debug output for CSWAP
        if (control_qbit >= 0) {
            std::cout << "SWAP kernel: block_idx=" << block_idx
                      << ", base=" << base
                      << ", swap_idx=" << swap_idx
                      << ", swap_idx_pair=" << swap_idx_pair << std::endl;
        }

        std::swap_ranges(
            input.get_data() + swap_idx*input.stride,
            input.get_data() + swap_idx*input.stride + input.cols,
            input.get_data() + swap_idx_pair*input.stride
        );
    }
}