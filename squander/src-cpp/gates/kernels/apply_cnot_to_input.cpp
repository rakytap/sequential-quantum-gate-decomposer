#include "apply_cnot_to_input.h"
#include "tbb/tbb.h"


void apply_cnot_kernel_to_state_vector_input_OpenMP(Matrix& input,const int& ctrl_qbit, const int& trgt_qbit){
    int matrix_size = input.size();
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
            }
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
