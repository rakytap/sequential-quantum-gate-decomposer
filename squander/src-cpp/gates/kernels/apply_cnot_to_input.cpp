#include "apply_cnot_to_input.h"
#include "tbb/tbb.h"


void apply_cnot_kernel_to_state_vector_input_OpenMP(Matrix& input,const int& ctrl_qbit, const int& trgt_qbit){
    int matrix_size = input.size();
    int index_step_ctrl = 1 << ctrl_qbit;
    int index_step_trgt = 1 << trgt_qbit;
    if (trgt_qbit>ctrl_qbit){
        int current_idx = 0;
        int ratio2 = (int) index_step_trgt/(index_step_ctrl<<1);
#pragma omp parallel for
        for (int idx=0; idx<matrix_size/4; idx++){
                current_idx = (int) (index_step_trgt<<1)*(idx/(index_step_ctrl*ratio2)) + (int) ((index_step_ctrl<<1)*((idx%(index_step_ctrl*ratio2))/index_step_ctrl)) + (int) ((1)*(idx%(index_step_ctrl))) ; 
                int current_idx_trgt_loc = current_idx + index_step_ctrl;
                int current_idx_ctrl_loc = current_idx + index_step_ctrl + index_step_trgt;
                double placeholder = input[current_idx_trgt_loc].real;
                input[current_idx_trgt_loc].real = input[current_idx_ctrl_loc].real;
                input[current_idx_ctrl_loc].real = placeholder;
                placeholder = input[current_idx_trgt_loc].imag;
                input[current_idx_trgt_loc].imag = input[current_idx_ctrl_loc].imag;
                input[current_idx_ctrl_loc].imag = placeholder;
            }
        }
        else{
            int current_idx = 0;
            int ratio2 = (int) index_step_ctrl/(index_step_trgt<<1);
#pragma omp parallel for
            for (int idx=0; idx<matrix_size/4; idx++){
            current_idx = (int) (index_step_ctrl<<1)*(idx/(index_step_trgt*ratio2)) + (int) ((index_step_trgt<<1)*((idx%(index_step_trgt*ratio2))/index_step_trgt)) + (int) ((idx%(index_step_trgt))) ; 
                int current_idx_trgt_loc = current_idx + index_step_ctrl + index_step_trgt;
                int current_idx_ctrl_loc = current_idx + index_step_ctrl;
                double placeholder = input[current_idx_trgt_loc].real;
                input[current_idx_trgt_loc].real = input[current_idx_ctrl_loc].real;
                input[current_idx_ctrl_loc].real = placeholder;
                placeholder = input[current_idx_trgt_loc].imag;
                input[current_idx_trgt_loc].imag = input[current_idx_ctrl_loc].imag;
                input[current_idx_ctrl_loc].imag = placeholder;
            }
        }
}

void apply_cnot_kernel_to_state_vector_input(Matrix& input,const int& ctrl_qbit, const int& trgt_qbit){
    int matrix_size = input.size();
    int index_step_ctrl = 1 << ctrl_qbit;
    int index_step_trgt = 1 << trgt_qbit;
    if (trgt_qbit>ctrl_qbit){
        int current_idx = 0;
        int ratio2 = (int) index_step_trgt/(index_step_ctrl<<1);
        tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/4,10), [&](tbb::blocked_range<int> r) { 

            for (int idx=r.begin(); idx<r.end(); idx++) {

            
            //Matrix placeholder( 1, placeholder_size);
            current_idx = (int) (index_step_trgt<<1)*(idx/(index_step_ctrl*ratio2)) + (int) ((index_step_ctrl<<1)*((idx%(index_step_ctrl*ratio2))/index_step_ctrl)) + (int) ((1)*(idx%(index_step_ctrl))) ; 
            int current_idx_trgt_loc = current_idx + index_step_ctrl ;
            int current_idx_ctrl_loc = current_idx + index_step_ctrl + index_step_trgt ;
            double placeholder = input[current_idx_trgt_loc].real;
            input[current_idx_trgt_loc].real = input[current_idx_ctrl_loc].real;
            input[current_idx_ctrl_loc].real = placeholder;
            placeholder = input[current_idx_trgt_loc].imag;
            input[current_idx_trgt_loc].imag = input[current_idx_ctrl_loc].imag;
            input[current_idx_ctrl_loc].imag = placeholder;
        } });
        }
        else{
            int current_idx = 0;
            int ratio2 = (int) index_step_ctrl/(index_step_trgt<<1);
            tbb::parallel_for( tbb::blocked_range<int>(0,matrix_size/4,10), [&](tbb::blocked_range<int> r) { 
            for (int idx=r.begin(); idx<r.end(); idx++) {
            current_idx = (int) (index_step_ctrl<<1)*(idx/(index_step_trgt*ratio2)) + (int) ((index_step_trgt<<1)*((idx%(index_step_trgt*ratio2))/index_step_trgt)) + (int) ((idx%(index_step_trgt))) ; 
                int current_idx_trgt_loc = current_idx + index_step_ctrl + index_step_trgt;
                int current_idx_ctrl_loc = current_idx + index_step_ctrl;
                double placeholder = input[current_idx_trgt_loc].real;
                input[current_idx_trgt_loc].real = input[current_idx_ctrl_loc].real;
                input[current_idx_ctrl_loc].real = placeholder;
                placeholder = input[current_idx_trgt_loc].imag;
                input[current_idx_trgt_loc].imag = input[current_idx_ctrl_loc].imag;
                input[current_idx_ctrl_loc].imag = placeholder;
            } });
        }
}
