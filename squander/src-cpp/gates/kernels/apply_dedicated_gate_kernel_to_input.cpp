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
#include <omp.h>
#include <type_traits>
#include <utility>

template<typename MatrixT>
using KernelDedicatedComplexT = typename std::remove_reference<decltype(std::declval<MatrixT&>()[0])>::type;





/**
@brief Call to apply X gate kernel on an input matrix
@param input The input matrix on which the transformation is applied
@param target_qbits The target qubits (must contain exactly 1 element)
@param control_qbits The control qubits (empty vector if no control qubits)
@param matrix_size The size of the input
*/
template<typename MatrixT>
void apply_X_kernel_to_input_impl(MatrixT& input, const std::vector<int>& target_qbits,
                           const std::vector<int>& control_qbits,
                           const int& matrix_size) {

    // Validate target qubits - X gate requires exactly 1 target qubit
    if (target_qbits.size() != 1) {
        throw std::runtime_error("X gate kernel requires exactly 1 target qubit, got " +
                                std::to_string(target_qbits.size()));
    }

    int target_qbit = target_qbits[0];
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;

    for (int current_idx_pair = current_idx + index_step_target;
         current_idx_pair < matrix_size;
         current_idx_pair = current_idx_pair + (index_step_target << 1)) {

        for(int idx = 0; idx < index_step_target; idx++) {

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            // Check all control qubits are active
            bool all_controls_active = true;
            for (int control_qbit : control_qbits) {
                if (!((current_idx_loc >> control_qbit) & 1)) {
                    all_controls_active = false;
                    break;
                }
            }

            // Apply X gate only when ALL controls are active
            if (all_controls_active) {

                long long row_offset = (long long)current_idx_loc * input.stride;
                long long row_offset_pair = (long long)current_idx_pair_loc * input.stride;

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

template<typename MatrixT>
void apply_Y_kernel_to_input_impl(MatrixT& input, const int& target_qbit, 
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
                    
                    KernelDedicatedComplexT<MatrixT> element = input[index];
                    KernelDedicatedComplexT<MatrixT> element_pair = input[index_pair];
                    
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

template<typename MatrixT>
void apply_Z_kernel_to_input_impl(MatrixT& input, const int& target_qbit, 
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

template<typename MatrixT>
void apply_H_kernel_to_input_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
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

                    KernelDedicatedComplexT<MatrixT> element = input[index];
                    KernelDedicatedComplexT<MatrixT> element_pair = input[index_pair];

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

template<typename MatrixT>
void apply_S_kernel_to_input_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
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

template<typename MatrixT>
void apply_T_kernel_to_input_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
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

template<typename MatrixT>
void apply_SWAP_kernel_to_input_impl(MatrixT& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size){

    // Validate target qubits - SWAP gate requires exactly 2 target qubits
    if (target_qbits.size() != 2) {
        throw std::runtime_error("SWAP gate kernel requires exactly 2 target qubits, got " +
                                std::to_string(target_qbits.size()));
    }

    int target_qbit1 = target_qbits[0];
    int target_qbit2 = target_qbits[1];

    std::vector<int> non_involved_qbits;
    int qbit_num = (int)std::log2(matrix_size);
    for (int idx=0; idx<qbit_num; idx++){
        bool is_target = (idx == target_qbit1 || idx == target_qbit2);
        bool is_control = std::find(control_qbits.begin(), control_qbits.end(), idx) != control_qbits.end();
        if (!is_target && !is_control){
            non_involved_qbits.push_back(idx);
        }
    }

    // Build control qubit mask
    int control_mask = 0;
    for (int control_qbit : control_qbits) {
        control_mask |= (1 << control_qbit);
    }

    for (int block_idx=0; block_idx < matrix_size >> (qbit_num - non_involved_qbits.size()); block_idx++){
        int base = 0;
        for (size_t qdx=0; qdx<non_involved_qbits.size();qdx++){
            if ((block_idx >> qdx) & 1) {
                base |= (1<<non_involved_qbits[qdx]);
            }
        }
        base |= control_mask;
        int swap_idx = base|(1<<target_qbit1);
        int swap_idx_pair = base|(1<<target_qbit2);

        std::swap_ranges(
            input.get_data() + swap_idx*input.stride,
            input.get_data() + swap_idx*input.stride + input.cols,
            input.get_data() + swap_idx_pair*input.stride
        );
    }
}


template<typename MatrixT>
void apply_SYC_kernel_to_input_impl(MatrixT& input, const int& target_qbit,
                             const int& control_qbit,
                             const int& matrix_size) {

    int index_step_target = 1 << target_qbit;
    int index_step_control = 1 << control_qbit;

    int loop_size = index_step_target < index_step_control ? index_step_target : index_step_control;
    int iterations = control_qbit < target_qbit ? Power_of_2(target_qbit - control_qbit - 1) : Power_of_2(control_qbit - target_qbit - 1);

    int idx00 = 0;
    int idx01 = index_step_target;
    int idx10 = index_step_control;
    int idx11 = index_step_target + index_step_control;

    const double phase_real = std::sqrt(3.0) / 2.0;
    const double phase_imag = -0.5;

    while (idx11 < matrix_size) {
        for (int jdx = 0; jdx < iterations; ++jdx) {
            for (int idx = 0; idx < loop_size; ++idx) {
                int idx01_loc = idx01 + idx;
                int idx10_loc = idx10 + idx;
                int idx11_loc = idx11 + idx;

                int offset01 = idx01_loc * input.stride;
                int offset10 = idx10_loc * input.stride;
                int offset11 = idx11_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; ++col_idx) {
                    KernelDedicatedComplexT<MatrixT> element01 = input[offset01 + col_idx];
                    KernelDedicatedComplexT<MatrixT> element10 = input[offset10 + col_idx];

                    input[offset01 + col_idx].real = element10.imag;
                    input[offset01 + col_idx].imag = -element10.real;

                    input[offset10 + col_idx].real = element01.imag;
                    input[offset10 + col_idx].imag = -element01.real;

                    KernelDedicatedComplexT<MatrixT> element11 = input[offset11 + col_idx];
                    input[offset11 + col_idx].real = phase_real * element11.real - phase_imag * element11.imag;
                    input[offset11 + col_idx].imag = phase_real * element11.imag + phase_imag * element11.real;
                }
            }

            idx00 += 2 * loop_size;
            idx01 += 2 * loop_size;
            idx10 += 2 * loop_size;
            idx11 += 2 * loop_size;
        }

        idx00 += 2 * loop_size * iterations;
        idx01 += 2 * loop_size * iterations;
        idx10 += 2 * loop_size * iterations;
        idx11 += 2 * loop_size * iterations;
    }
}


template<typename MatrixT>
void apply_SYC_kernel_from_right_impl(MatrixT& input, const int& target_qbit,
                               const int& control_qbit,
                               const int& matrix_size) {

    int index_step_target = 1 << target_qbit;
    int index_step_control = 1 << control_qbit;

    int loop_size = index_step_target < index_step_control ? index_step_target : index_step_control;
    int iterations = control_qbit < target_qbit ? Power_of_2(target_qbit - control_qbit - 1) : Power_of_2(control_qbit - target_qbit - 1);

    const double phase_real = std::sqrt(3.0) / 2.0;
    const double phase_imag = -0.5;

    tbb::parallel_for(0, input.rows, 1, [&](int row_idx) {
        int offset = row_idx * input.stride;

        int idx00 = 0;
        int idx01 = index_step_target;
        int idx10 = index_step_control;
        int idx11 = index_step_target + index_step_control;

        while (idx11 < matrix_size) {
            for (int jdx = 0; jdx < iterations; ++jdx) {
                for (int idx = 0; idx < loop_size; ++idx) {
                    int idx01_loc = idx01 + idx;
                    int idx10_loc = idx10 + idx;
                    int idx11_loc = idx11 + idx;

                    KernelDedicatedComplexT<MatrixT> element01 = input[offset + idx01_loc];
                    KernelDedicatedComplexT<MatrixT> element10 = input[offset + idx10_loc];
                    input[offset + idx01_loc].real = element10.imag;
                    input[offset + idx01_loc].imag = -element10.real;

                    input[offset + idx10_loc].real = element01.imag;
                    input[offset + idx10_loc].imag = -element01.real;

                    KernelDedicatedComplexT<MatrixT> element11 = input[offset + idx11_loc];
                    input[offset + idx11_loc].real = phase_real * element11.real - phase_imag * element11.imag;
                    input[offset + idx11_loc].imag = phase_real * element11.imag + phase_imag * element11.real;
                }

                idx00 += 2 * loop_size;
                idx01 += 2 * loop_size;
                idx10 += 2 * loop_size;
                idx11 += 2 * loop_size;
            }

            idx00 += 2 * loop_size * iterations;
            idx01 += 2 * loop_size * iterations;
            idx10 += 2 * loop_size * iterations;
            idx11 += 2 * loop_size * iterations;
        }
    });
}

// TBB Parallelized versions

template<typename MatrixT>
void apply_X_kernel_to_input_tbb_impl(MatrixT& input, const std::vector<int>& target_qbits,
                                const std::vector<int>& control_qbits,
                                const int& matrix_size) {
    // Validate target qubits - X gate requires exactly 1 target qubit
    if (target_qbits.size() != 1) {
        throw std::runtime_error("X gate kernel requires exactly 1 target qubit, got " +
                                std::to_string(target_qbits.size()));
    }

    int target_qbit = target_qbits[0];
    int index_step_target = 1 << target_qbit;
    int total_blocks = matrix_size >> (target_qbit + 1);

    tbb::parallel_for(tbb::blocked_range<int>(0, total_blocks, 1024),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int current_idx = block_idx * (index_step_target << 1);
                int current_idx_pair = current_idx + index_step_target;

                for(int idx = 0; idx < index_step_target; idx++) {
                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    // Check all control qubits are active
                    bool all_controls_active = true;
                    for (int control_qbit : control_qbits) {
                        if (!((current_idx_loc >> control_qbit) & 1)) {
                            all_controls_active = false;
                            break;
                        }
                    }

                    if (all_controls_active) {
                        long long row_offset = (long long)current_idx_loc * input.stride;
                        long long row_offset_pair = (long long)current_idx_pair_loc * input.stride;

                        std::swap_ranges(
                            input.get_data() + row_offset,
                            input.get_data() + row_offset + input.cols,
                            input.get_data() + row_offset_pair
                        );
                    }
                }
            }
        }
    );
}

template<typename MatrixT>
void apply_Y_kernel_to_input_tbb_impl(MatrixT& input, const int& target_qbit,
                                const int& control_qbit,
                                const int& matrix_size) {
    int index_step_target = 1 << target_qbit;

    tbb::parallel_for(tbb::blocked_range<int>(0, matrix_size >> 1, 1024),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int current_idx = block_idx * (index_step_target << 1);
                int current_idx_pair = current_idx + index_step_target;

                if (current_idx_pair >= matrix_size) continue;

                for(int idx = 0; idx < index_step_target; idx++) {
                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                        int row_offset = current_idx_loc * input.stride;
                        int row_offset_pair = current_idx_pair_loc * input.stride;

                        for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                            int index = row_offset + col_idx;
                            int index_pair = row_offset_pair + col_idx;

                            KernelDedicatedComplexT<MatrixT> element = input[index];
                            KernelDedicatedComplexT<MatrixT> element_pair = input[index_pair];

                            input[index].real = element_pair.imag;
                            input[index].imag = -element_pair.real;

                            input[index_pair].real = -element.imag;
                            input[index_pair].imag = element.real;
                        }
                    }
                }
            }
        }
    );
}

template<typename MatrixT>
void apply_Z_kernel_to_input_tbb_impl(MatrixT& input, const int& target_qbit,
                                const int& control_qbit,
                                const int& matrix_size) {
    int index_step_target = 1 << target_qbit;

    tbb::parallel_for(tbb::blocked_range<int>(0, matrix_size >> 1, 1024),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int current_idx = block_idx * (index_step_target << 1);
                int current_idx_pair = current_idx + index_step_target;

                if (current_idx_pair >= matrix_size) continue;

                for(int idx = 0; idx < index_step_target; idx++) {
                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                        int row_offset_pair = current_idx_pair_loc * input.stride;

                        for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                            int index_pair = row_offset_pair + col_idx;

                            input[index_pair].real = -input[index_pair].real;
                            input[index_pair].imag = -input[index_pair].imag;
                        }
                    }
                }
            }
        }
    );
}

template<typename MatrixT>
void apply_H_kernel_to_input_tbb_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    tbb::parallel_for(tbb::blocked_range<int>(0, matrix_size >> 1, 1024),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int current_idx = block_idx * (index_step_target << 1);
                int current_idx_pair = current_idx + index_step_target;

                if (current_idx_pair >= matrix_size) continue;

                for (int idx = 0; idx < index_step_target; idx++) {
                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                        int row_offset = current_idx_loc * input.stride;
                        int row_offset_pair = current_idx_pair_loc * input.stride;

                        for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                            int index = row_offset + col_idx;
                            int index_pair = row_offset_pair + col_idx;

                            KernelDedicatedComplexT<MatrixT> element = input[index];
                            KernelDedicatedComplexT<MatrixT> element_pair = input[index_pair];

                            input[index].real = inv_sqrt2 * (element.real + element_pair.real);
                            input[index].imag = inv_sqrt2 * (element.imag + element_pair.imag);

                            input[index_pair].real = inv_sqrt2 * (element.real - element_pair.real);
                            input[index_pair].imag = inv_sqrt2 * (element.imag - element_pair.imag);
                        }
                    }
                }
            }
        }
    );
}

template<typename MatrixT>
void apply_S_kernel_to_input_tbb_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;

    tbb::parallel_for(tbb::blocked_range<int>(0, matrix_size >> 1, 1024),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int current_idx = block_idx * (index_step_target << 1);
                int current_idx_pair = current_idx + index_step_target;

                if (current_idx_pair >= matrix_size) continue;

                for (int idx = 0; idx < index_step_target; idx++) {
                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                        int row_offset_pair = current_idx_pair_loc * input.stride;

                        for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                            int index_pair = row_offset_pair + col_idx;

                            double real = input[index_pair].real;
                            double imag = input[index_pair].imag;
                            input[index_pair].real = -imag;
                            input[index_pair].imag = real;
                        }
                    }
                }
            }
        }
    );
}

template<typename MatrixT>
void apply_T_kernel_to_input_tbb_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    tbb::parallel_for(tbb::blocked_range<int>(0, matrix_size >> 1, 1024),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int current_idx = block_idx * (index_step_target << 1);
                int current_idx_pair = current_idx + index_step_target;

                if (current_idx_pair >= matrix_size) continue;

                for (int idx = 0; idx < index_step_target; idx++) {
                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                        int row_offset_pair = current_idx_pair_loc * input.stride;

                        for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                            int index_pair = row_offset_pair + col_idx;

                            double real = input[index_pair].real;
                            double imag = input[index_pair].imag;
                            input[index_pair].real = inv_sqrt2 * (real - imag);
                            input[index_pair].imag = inv_sqrt2 * (real + imag);
                        }
                    }
                }
            }
        }
    );
}

template<typename MatrixT>
void apply_SWAP_kernel_to_input_tbb_impl(MatrixT& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) {
    // Validate target qubits - SWAP gate requires exactly 2 target qubits
    if (target_qbits.size() != 2) {
        throw std::runtime_error("SWAP gate kernel requires exactly 2 target qubits, got " +
                                std::to_string(target_qbits.size()));
    }

    int target_qbit1 = target_qbits[0];
    int target_qbit2 = target_qbits[1];

    std::vector<int> non_involved_qbits;
    int qbit_num = (int)std::log2(matrix_size);
    for (int idx=0; idx<qbit_num; idx++){
        bool is_target = (idx == target_qbit1 || idx == target_qbit2);
        bool is_control = std::find(control_qbits.begin(), control_qbits.end(), idx) != control_qbits.end();
        if (!is_target && !is_control){
            non_involved_qbits.push_back(idx);
        }
    }

    // Build control qubit mask
    int control_mask = 0;
    for (int control_qbit : control_qbits) {
        control_mask |= (1 << control_qbit);
    }

    int total_blocks = matrix_size >> (qbit_num - non_involved_qbits.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, total_blocks, 64),
        [&](const tbb::blocked_range<int>& range) {
            for (int block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                int base = 0;
                for (size_t qdx=0; qdx<non_involved_qbits.size();qdx++){
                    if ((block_idx >> qdx) & 1) {
                        base |= (1<<non_involved_qbits[qdx]);
                    }
                }
                base |= control_mask;
                int swap_idx = base|(1<<target_qbit1);
                int swap_idx_pair = base|(1<<target_qbit2);

                std::swap_ranges(
                    input.get_data() + swap_idx*input.stride,
                    input.get_data() + swap_idx*input.stride + input.cols,
                    input.get_data() + swap_idx_pair*input.stride
                );
            }
        }
    );
}


template<typename MatrixT>
void apply_SYC_kernel_to_input_tbb_impl(MatrixT& input, const int& target_qbit,
                                 const int& control_qbit,
                                 const int& matrix_size) {

    int index_step_target = 1 << target_qbit;
    int index_step_control = 1 << control_qbit;

    int loop_size = index_step_target < index_step_control ? index_step_target : index_step_control;
    int iterations = control_qbit < target_qbit ? Power_of_2(target_qbit - control_qbit - 1) : Power_of_2(control_qbit - target_qbit - 1);

    int idx00 = 0;
    int idx01 = index_step_target;
    int idx10 = index_step_control;
    int idx11 = index_step_target + index_step_control;

    const double phase_real = std::sqrt(3.0) / 2.0;
    const double phase_imag = -0.5;

    while (idx11 < matrix_size) {
        for (int jdx = 0; jdx < iterations; ++jdx) {
            tbb::parallel_for(0, loop_size, 1, [&](int idx) {
                int idx01_loc = idx01 + idx;
                int idx10_loc = idx10 + idx;
                int idx11_loc = idx11 + idx;

                int offset01 = idx01_loc * input.stride;
                int offset10 = idx10_loc * input.stride;
                int offset11 = idx11_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; ++col_idx) {
                    KernelDedicatedComplexT<MatrixT> element01 = input[offset01 + col_idx];
                    KernelDedicatedComplexT<MatrixT> element10 = input[offset10 + col_idx];

                    input[offset01 + col_idx].real = element10.imag;
                    input[offset01 + col_idx].imag = -element10.real;

                    input[offset10 + col_idx].real = element01.imag;
                    input[offset10 + col_idx].imag = -element01.real;

                    KernelDedicatedComplexT<MatrixT> element11 = input[offset11 + col_idx];
                    input[offset11 + col_idx].real = phase_real * element11.real - phase_imag * element11.imag;
                    input[offset11 + col_idx].imag = phase_real * element11.imag + phase_imag * element11.real;
                }
            });

            idx00 += 2 * loop_size;
            idx01 += 2 * loop_size;
            idx10 += 2 * loop_size;
            idx11 += 2 * loop_size;
        }

        idx00 += 2 * loop_size * iterations;
        idx01 += 2 * loop_size * iterations;
        idx10 += 2 * loop_size * iterations;
        idx11 += 2 * loop_size * iterations;
    }
}

// OpenMP Parallelized versions

template<typename MatrixT>
void apply_X_kernel_to_input_omp_impl(MatrixT& input, const std::vector<int>& target_qbits,
                                 const std::vector<int>& control_qbits,
                                 const int& matrix_size) {
    // Validate target qubits - X gate requires exactly 1 target qubit
    if (target_qbits.size() != 1) {
        throw std::runtime_error("X gate kernel requires exactly 1 target qubit, got " +
                                std::to_string(target_qbits.size()));
    }

    int target_qbit = target_qbits[0];
    int index_step_target = 1 << target_qbit;
    int total_blocks = matrix_size >> (target_qbit + 1);

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int current_idx = block_idx * (index_step_target << 1);
        int current_idx_pair = current_idx + index_step_target;

        for(int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            // Check all control qubits are active
            bool all_controls_active = true;
            for (int control_qbit : control_qbits) {
                if (!((current_idx_loc >> control_qbit) & 1)) {
                    all_controls_active = false;
                    break;
                }
            }

            if (all_controls_active) {
                long long row_offset = (long long)current_idx_loc * input.stride;
                long long row_offset_pair = (long long)current_idx_pair_loc * input.stride;

                std::swap_ranges(
                    input.get_data() + row_offset,
                    input.get_data() + row_offset + input.cols,
                    input.get_data() + row_offset_pair
                );
            }
        }
    }
}

template<typename MatrixT>
void apply_Y_kernel_to_input_omp_impl(MatrixT& input, const int& target_qbit,
                                 const int& control_qbit,
                                 const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    int total_blocks = matrix_size >> 1;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int current_idx = block_idx * (index_step_target << 1);
        int current_idx_pair = current_idx + index_step_target;

        if (current_idx_pair >= matrix_size) continue;

        for(int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset = current_idx_loc * input.stride;
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index = row_offset + col_idx;
                    int index_pair = row_offset_pair + col_idx;

                    KernelDedicatedComplexT<MatrixT> element = input[index];
                    KernelDedicatedComplexT<MatrixT> element_pair = input[index_pair];

                    input[index].real = element_pair.imag;
                    input[index].imag = -element_pair.real;

                    input[index_pair].real = -element.imag;
                    input[index_pair].imag = element.real;
                }
            }
        }
    }
}

template<typename MatrixT>
void apply_Z_kernel_to_input_omp_impl(MatrixT& input, const int& target_qbit,
                                 const int& control_qbit,
                                 const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    int total_blocks = matrix_size >> 1;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int current_idx = block_idx * (index_step_target << 1);
        int current_idx_pair = current_idx + index_step_target;

        if (current_idx_pair >= matrix_size) continue;

        for(int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index_pair = row_offset_pair + col_idx;

                    input[index_pair].real = -input[index_pair].real;
                    input[index_pair].imag = -input[index_pair].imag;
                }
            }
        }
    }
}

template<typename MatrixT>
void apply_H_kernel_to_input_omp_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    int total_blocks = matrix_size >> 1;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int current_idx = block_idx * (index_step_target << 1);
        int current_idx_pair = current_idx + index_step_target;

        if (current_idx_pair >= matrix_size) continue;

        for (int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset = current_idx_loc * input.stride;
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index = row_offset + col_idx;
                    int index_pair = row_offset_pair + col_idx;

                    KernelDedicatedComplexT<MatrixT> element = input[index];
                    KernelDedicatedComplexT<MatrixT> element_pair = input[index_pair];

                    input[index].real = inv_sqrt2 * (element.real + element_pair.real);
                    input[index].imag = inv_sqrt2 * (element.imag + element_pair.imag);

                    input[index_pair].real = inv_sqrt2 * (element.real - element_pair.real);
                    input[index_pair].imag = inv_sqrt2 * (element.imag - element_pair.imag);
                }
            }
        }
    }
}

template<typename MatrixT>
void apply_S_kernel_to_input_omp_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    int total_blocks = matrix_size >> 1;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int current_idx = block_idx * (index_step_target << 1);
        int current_idx_pair = current_idx + index_step_target;

        if (current_idx_pair >= matrix_size) continue;

        for (int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index_pair = row_offset_pair + col_idx;

                    double real = input[index_pair].real;
                    double imag = input[index_pair].imag;
                    input[index_pair].real = -imag;
                    input[index_pair].imag = real;
                }
            }
        }
    }
}

template<typename MatrixT>
void apply_T_kernel_to_input_omp_impl(MatrixT& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    int index_step_target = 1 << target_qbit;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    int total_blocks = matrix_size >> 1;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int current_idx = block_idx * (index_step_target << 1);
        int current_idx_pair = current_idx + index_step_target;

        if (current_idx_pair >= matrix_size) continue;

        for (int idx = 0; idx < index_step_target; idx++) {
            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            if ((control_qbit < 0) || ((current_idx_loc >> control_qbit) & 1)) {
                int row_offset_pair = current_idx_pair_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; col_idx++) {
                    int index_pair = row_offset_pair + col_idx;

                    double real = input[index_pair].real;
                    double imag = input[index_pair].imag;
                    input[index_pair].real = inv_sqrt2 * (real - imag);
                    input[index_pair].imag = inv_sqrt2 * (real + imag);
                }
            }
        }
    }
}

template<typename MatrixT>
void apply_SWAP_kernel_to_input_omp_impl(MatrixT& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) {
    // Validate target qubits - SWAP gate requires exactly 2 target qubits
    if (target_qbits.size() != 2) {
        throw std::runtime_error("SWAP gate kernel requires exactly 2 target qubits, got " +
                                std::to_string(target_qbits.size()));
    }

    int target_qbit1 = target_qbits[0];
    int target_qbit2 = target_qbits[1];

    std::vector<int> non_involved_qbits;
    int qbit_num = (int)std::log2(matrix_size);
    for (int idx=0; idx<qbit_num; idx++){
        bool is_target = (idx == target_qbit1 || idx == target_qbit2);
        bool is_control = std::find(control_qbits.begin(), control_qbits.end(), idx) != control_qbits.end();
        if (!is_target && !is_control){
            non_involved_qbits.push_back(idx);
        }
    }

    // Build control qubit mask
    int control_mask = 0;
    for (int control_qbit : control_qbits) {
        control_mask |= (1 << control_qbit);
    }

    int total_blocks = matrix_size >> (qbit_num - non_involved_qbits.size());

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; block_idx++) {
        int base = 0;
        for (size_t qdx=0; qdx<non_involved_qbits.size();qdx++){
            if ((block_idx >> qdx) & 1) {
                base |= (1<<non_involved_qbits[qdx]);
            }
        }
        base |= control_mask;
        int swap_idx = base|(1<<target_qbit1);
        int swap_idx_pair = base|(1<<target_qbit2);

        std::swap_ranges(
            input.get_data() + swap_idx*input.stride,
            input.get_data() + swap_idx*input.stride + input.cols,
            input.get_data() + swap_idx_pair*input.stride
        );
    }
}


template<typename MatrixT>
void apply_SYC_kernel_to_input_omp_impl(MatrixT& input, const int& target_qbit,
                                 const int& control_qbit,
                                 const int& matrix_size) {

    int index_step_target = 1 << target_qbit;
    int index_step_control = 1 << control_qbit;

    int loop_size = index_step_target < index_step_control ? index_step_target : index_step_control;
    int iterations = control_qbit < target_qbit ? Power_of_2(target_qbit - control_qbit - 1) : Power_of_2(control_qbit - target_qbit - 1);

    int idx00 = 0;
    int idx01 = index_step_target;
    int idx10 = index_step_control;
    int idx11 = index_step_target + index_step_control;

    const double phase_real = std::sqrt(3.0) / 2.0;
    const double phase_imag = -0.5;

    while (idx11 < matrix_size) {
        for (int jdx = 0; jdx < iterations; ++jdx) {
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < loop_size; ++idx) {
                int idx01_loc = idx01 + idx;
                int idx10_loc = idx10 + idx;
                int idx11_loc = idx11 + idx;

                int offset01 = idx01_loc * input.stride;
                int offset10 = idx10_loc * input.stride;
                int offset11 = idx11_loc * input.stride;

                for (int col_idx = 0; col_idx < input.cols; ++col_idx) {
                    KernelDedicatedComplexT<MatrixT> element01 = input[offset01 + col_idx];
                    KernelDedicatedComplexT<MatrixT> element10 = input[offset10 + col_idx];

                    input[offset01 + col_idx].real = element10.imag;
                    input[offset01 + col_idx].imag = -element10.real;

                    input[offset10 + col_idx].real = element01.imag;
                    input[offset10 + col_idx].imag = -element01.real;

                    KernelDedicatedComplexT<MatrixT> element11 = input[offset11 + col_idx];
                    input[offset11 + col_idx].real = phase_real * element11.real - phase_imag * element11.imag;
                    input[offset11 + col_idx].imag = phase_real * element11.imag + phase_imag * element11.real;
                }
            }

            idx00 += 2 * loop_size;
            idx01 += 2 * loop_size;
            idx10 += 2 * loop_size;
            idx11 += 2 * loop_size;
        }

        idx00 += 2 * loop_size * iterations;
        idx01 += 2 * loop_size * iterations;
        idx10 += 2 * loop_size * iterations;
        idx11 += 2 * loop_size * iterations;
    }
}
void apply_X_kernel_to_input(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_X_kernel_to_input_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_X_kernel_to_input(Matrix_float& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_X_kernel_to_input_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_Y_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Y_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Y_kernel_to_input(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Y_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Z_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Z_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Z_kernel_to_input(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Z_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_H_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_H_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_H_kernel_to_input(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_H_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_S_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_S_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_S_kernel_to_input(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_S_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_T_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_T_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_T_kernel_to_input(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_T_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SWAP_kernel_to_input(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_SWAP_kernel_to_input_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_SWAP_kernel_to_input(Matrix_float& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_SWAP_kernel_to_input_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_SYC_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SYC_kernel_to_input(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_to_input_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SYC_kernel_from_right(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_from_right_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SYC_kernel_from_right(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_from_right_impl(input, target_qbit, control_qbit, matrix_size); }

void apply_X_kernel_to_input_tbb(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_X_kernel_to_input_tbb_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_X_kernel_to_input_tbb(Matrix_float& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_X_kernel_to_input_tbb_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_Y_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Y_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Y_kernel_to_input_tbb(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Y_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Z_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Z_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Z_kernel_to_input_tbb(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Z_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_H_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_H_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_H_kernel_to_input_tbb(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_H_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_S_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_S_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_S_kernel_to_input_tbb(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_S_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_T_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_T_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_T_kernel_to_input_tbb(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_T_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SWAP_kernel_to_input_tbb(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_SWAP_kernel_to_input_tbb_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_SWAP_kernel_to_input_tbb(Matrix_float& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_SWAP_kernel_to_input_tbb_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_SYC_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SYC_kernel_to_input_tbb(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_to_input_tbb_impl(input, target_qbit, control_qbit, matrix_size); }

void apply_X_kernel_to_input_omp(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_X_kernel_to_input_omp_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_X_kernel_to_input_omp(Matrix_float& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_X_kernel_to_input_omp_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_Y_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Y_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Y_kernel_to_input_omp(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Y_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Z_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Z_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_Z_kernel_to_input_omp(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_Z_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_H_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_H_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_H_kernel_to_input_omp(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_H_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_S_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_S_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_S_kernel_to_input_omp(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_S_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_T_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_T_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_T_kernel_to_input_omp(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_T_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SWAP_kernel_to_input_omp(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_SWAP_kernel_to_input_omp_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_SWAP_kernel_to_input_omp(Matrix_float& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size) { apply_SWAP_kernel_to_input_omp_impl(input, target_qbits, control_qbits, matrix_size); }
void apply_SYC_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
void apply_SYC_kernel_to_input_omp(Matrix_float& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) { apply_SYC_kernel_to_input_omp_impl(input, target_qbit, control_qbit, matrix_size); }
