#include "apply_cnot_to_input.h"
#include "tbb/tbb.h"


void apply_cnot_kernel_to_state_vector_input(Matrix& input,const int& ctrl_qbit, const int& trgt_qbit){
    int matrix_size = input.size();
    int index_step_ctrl = 1 << ctrl_qbit;
    int index_step_trgt = 1 << trgt_qbit;
    if (trgt_qbit>ctrl_qbit){
        int placeholder_size = (ctrl_qbit > 18) ? 1<<16 : index_step_ctrl;
        Matrix placeholder( 1, placeholder_size);
        int current_idx = 0;
        for (int current_idx_trgt=current_idx+index_step_trgt; current_idx_trgt<matrix_size; current_idx_trgt=current_idx_trgt+(index_step_trgt << 1)){
                for (int current_idx_ctrl = 0; current_idx_ctrl < index_step_trgt; current_idx_ctrl=current_idx_ctrl+(index_step_ctrl<<1)){
                    for (int idx=0;idx<index_step_ctrl;idx=idx+placeholder_size){
                    
        			    int current_idx_trgt_loc = current_idx + current_idx_ctrl + index_step_ctrl+idx;
                        int current_idx_ctrl_loc = current_idx_trgt + current_idx_ctrl + index_step_ctrl+idx;
                        
                        memcpy( placeholder.get_data(), input.get_data()+current_idx_trgt_loc, placeholder_size*sizeof(QGD_Complex16) );
                        memcpy( input.get_data()+current_idx_trgt_loc, input.get_data()+current_idx_ctrl_loc, placeholder_size*sizeof(QGD_Complex16) );
                        memcpy(input.get_data()+current_idx_ctrl_loc, placeholder.get_data(), placeholder_size*sizeof(QGD_Complex16) );
                    }
            }
            current_idx = current_idx + (index_step_trgt << 1);
         }
        }
        else{
            int placeholder_size = (trgt_qbit > 18) ? 1<<16 : index_step_trgt;
            Matrix placeholder( 1, placeholder_size);
            for (int current_idx_ctrl=index_step_ctrl; current_idx_ctrl<matrix_size; current_idx_ctrl=current_idx_ctrl+(index_step_ctrl << 1)){
                    for (int current_idx_trgt = 0; current_idx_trgt < index_step_ctrl; current_idx_trgt=current_idx_trgt+(index_step_trgt<<1)){
                        for(int idx=0; idx<index_step_trgt;idx=idx+placeholder_size){
                        
                            int current_idx_trgt_loc = current_idx_ctrl + current_idx_trgt + idx;
                            int current_idx_ctrl_loc = current_idx_trgt_loc + index_step_trgt;
                            
                            memcpy( placeholder.get_data(), input.get_data()+current_idx_trgt_loc, placeholder_size*sizeof(QGD_Complex16) );
                            memcpy( input.get_data()+current_idx_trgt_loc, input.get_data()+current_idx_ctrl_loc, placeholder_size*sizeof(QGD_Complex16) );
                            memcpy(input.get_data()+current_idx_ctrl_loc, placeholder.get_data(), placeholder_size*sizeof(QGD_Complex16) );
                    }
                }
             }
        }
}
