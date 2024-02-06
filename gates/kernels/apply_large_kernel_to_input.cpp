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


#include "apply_large_kernel_to_input.h"
#include "tbb/tbb.h"


int get_grain_size(int index_step){
    int grain_size=2;
    for (int step=1; step<7; step++){
        if (index_step <= 1<<step){
            grain_size = 256/(1<<step);
        }
    }
    return grain_size;
}

void apply_large_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    switch((int)involved_qbits.size()){
    case 2:{
        apply_2qbit_kernel_to_state_vector_input(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size);
    }
    case 3:{
        apply_3qbit_kernel_to_state_vector_input(unitary,input,involved_qbits,matrix_size);
    }
    }

}

void apply_2qbit_kernel_to_state_vector_input(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

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


void apply_3qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    int index_step_inner = 1 << involved_qbits[0];
    int index_step_middle = 1 << involved_qbits[1];
    int index_step_outer = 1 << involved_qbits[2];
    int current_idx = 0;
    
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<matrix_size; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_middle = 0; current_idx_middle < index_step_outer; current_idx_middle=current_idx_middle+(index_step_middle<<1)){
        
                for (int current_idx_inner = 0; current_idx_inner < index_step_middle; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
                
    	        for (int idx=0; idx<index_step_inner; idx++){
    	        
        	    int current_idx_loc = current_idx + current_idx_middle + current_idx_inner + idx;
                int current_idx_pair_loc = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle;

			    int current_idx_outer_loc = current_idx_loc;
			    int current_idx_inner_loc = current_idx_loc + index_step_inner;
			    
			    int current_idx_middle_loc = current_idx_loc + index_step_middle;
			    int current_idx_middle_inner_loc = current_idx_loc + index_step_middle + index_step_inner;
			    
	        	int current_idx_outer_pair_loc = current_idx_pair_loc;
			    int current_idx_inner_pair_loc = current_idx_pair_loc + index_step_inner;
			    
			    int current_idx_middle_pair_loc =current_idx_pair_loc + index_step_middle;
			    int current_idx_middle_inner_pair_loc = current_idx_pair_loc + index_step_middle + index_step_inner;
			    
			    int indexes[8] = {current_idx_outer_loc,current_idx_inner_loc,current_idx_middle_loc,current_idx_middle_inner_loc,current_idx_outer_pair_loc,current_idx_inner_pair_loc,current_idx_middle_pair_loc,current_idx_middle_inner_pair_loc};
			    //input.print_matrix();
			    QGD_Complex16 element_outer = input[current_idx_outer_loc];
			    QGD_Complex16 element_outer_pair = input[current_idx_outer_pair_loc];
			    
			    QGD_Complex16 element_inner = input[current_idx_inner_loc];
			    QGD_Complex16 element_inner_pair = input[current_idx_inner_pair_loc];
			    
			    QGD_Complex16 element_middle = input[current_idx_middle_loc];
			    QGD_Complex16 element_middle_pair = input[current_idx_middle_pair_loc];
			    
			    QGD_Complex16 element_middle_inner = input[current_idx_middle_inner_loc];
			    QGD_Complex16 element_middle_inner_pair = input[current_idx_middle_inner_pair_loc];
			    
			    QGD_Complex16 tmp1;
			    QGD_Complex16 tmp2;
			    QGD_Complex16 tmp3;
			    QGD_Complex16 tmp4;
			    QGD_Complex16 tmp5;
			    QGD_Complex16 tmp6;
			    QGD_Complex16 tmp7;
			    QGD_Complex16 tmp8;
			   for (int mult_idx=0; mult_idx<8; mult_idx++){
				    tmp1 = mult(unitary[mult_idx*8], element_outer);
				    tmp2 = mult(unitary[mult_idx*8 + 1], element_inner);
				    tmp3 = mult(unitary[mult_idx*8 + 2], element_middle);
				    tmp4 = mult(unitary[mult_idx*8 + 3], element_middle_inner);
				    tmp5 = mult(unitary[mult_idx*8 + 4], element_outer_pair);
				    tmp6 = mult(unitary[mult_idx*8 + 5], element_inner_pair);
				    tmp7 = mult(unitary[mult_idx*8 + 6], element_middle_pair);
				    tmp8 = mult(unitary[mult_idx*8 + 7], element_middle_inner_pair);
				    input[indexes[mult_idx]].real = tmp1.real + tmp2.real + tmp3.real + tmp4.real + tmp5.real + tmp6.real + tmp7.real + tmp8.real;
				    input[indexes[mult_idx]].imag = tmp1.imag + tmp2.imag + tmp3.imag + tmp4.imag + tmp5.imag + tmp6.imag + tmp7.imag + tmp8.imag;
		        }
        	  }
            }
            current_idx = current_idx + (index_step_outer << 1);
        }
     }
}

