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

void apply_large_kernel_to_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    if (input.cols==1){
    //switch((int)involved_qbits.size()){
    //case 2:{
        apply_2qbit_kernel_to_state_vector_input(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size);
    /*}
    case 3:{
        apply_3qbit_kernel_to_state_vector_input(unitary,input,involved_qbits,matrix_size);
    }
    }*/
    }
    else
    {
        apply_2qbit_kernel_to_matrix_input(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size);
    }
}

void apply_2qbit_kernel_to_state_vector_input(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<input.rows; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
        
        	for (int idx=0; idx<index_step_inner; idx++){
        	

			int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
			int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			int indexes[4] = {current_idx_outer_loc,current_idx_inner_loc,current_idx_outer_pair_loc,current_idx_inner_pair_loc};
			
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

void apply_2qbit_kernel_to_matrix_input(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<input.rows; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
        
        	for (int idx=0; idx<index_step_inner; idx++){
        	
			int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
            int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			
            int row_offset_outer = current_idx_outer_loc*input.stride;
            int row_offset_outer_pair = current_idx_outer_pair_loc*input.stride;
            int row_offset_inner = current_idx_inner_loc*input.stride;
            int row_offset_inner_pair = current_idx_inner_pair_loc*input.stride;
			//input.print_matrix();
            for ( int col_idx=0; col_idx<input.cols; col_idx++) {
                int index_outer      = row_offset_outer+col_idx;
                int index_outer_pair = row_offset_outer_pair+col_idx;     
                int index_inner = row_offset_inner+col_idx;
                int index_inner_pair = row_offset_inner_pair + col_idx;
      			int indexes[4] = {index_outer,index_inner,index_outer_pair,index_inner_pair};
			QGD_Complex16 element_outer = input[index_outer];
			QGD_Complex16 element_outer_pair = input[index_outer_pair];
			QGD_Complex16 element_inner = input[index_inner];
			QGD_Complex16 element_inner_pair = input[index_inner_pair];
			
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

//u3_1qbit2 is for when the control qbit is 0 
void
apply_crot_kernel_to_matrix_input(Matrix& u3_1qbit1, Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {

    int index_step_target = 1 << target_qbit;
    int current_idx = 0;


    for ( int current_idx_pair=current_idx + index_step_target; current_idx_pair<matrix_size; current_idx_pair=current_idx_pair+(index_step_target << 1) ) {

        for(int idx=0; idx<index_step_target; idx++) {  
        //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            int row_offset = current_idx_loc*input.stride;
            int row_offset_pair = current_idx_pair_loc*input.stride;
            for ( int col_idx=0; col_idx<input.cols; col_idx++) {
   			
                    int index      = row_offset+col_idx;
                    int index_pair = row_offset_pair+col_idx;  
                    if ( (current_idx_loc >> control_qbit) & 1 ) {

              

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    QGD_Complex16 tmp1 = mult(u3_1qbit1[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit1[1], element_pair);
 
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit1[2], element);
                    tmp2 = mult(u3_1qbit1[3], element_pair);

                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;

                }

            else {
                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    QGD_Complex16 tmp1 = mult(u3_1qbit2[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit2[1], element_pair);
 
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit2[2], element);
                    tmp2 = mult(u3_1qbit2[3], element_pair);

                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;
            }
  }

        
        //});
        }


        current_idx = current_idx + (index_step_target << 1);


    }



}

void
apply_crot_kernel_to_matrix_input_AVX(Matrix& u3_1qbit1, Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
    input.ensure_aligned();

    int index_step_target = 1 << target_qbit;
    int current_idx       = 0;

    // load elements of the first U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit_00r_vec = _mm256_broadcast_sd(&u3_1qbit1[0].real);
    __m256d u3_1bit_00i_vec = _mm256_broadcast_sd(&u3_1qbit1[0].imag);
    __m256d u3_1bit_01r_vec = _mm256_broadcast_sd(&u3_1qbit1[1].real);
    __m256d u3_1bit_01i_vec = _mm256_broadcast_sd(&u3_1qbit1[1].imag);
    __m256d u3_1bit_10r_vec = _mm256_broadcast_sd(&u3_1qbit1[2].real);
    __m256d u3_1bit_10i_vec = _mm256_broadcast_sd(&u3_1qbit1[2].imag);
    __m256d u3_1bit_11r_vec = _mm256_broadcast_sd(&u3_1qbit1[3].real);
    __m256d u3_1bit_11i_vec = _mm256_broadcast_sd(&u3_1qbit1[3].imag);
    // load elements of the second U3 unitary into 256bit registers (8 registers)
    __m256d u3_1bit2_00r_vec = _mm256_broadcast_sd(&u3_1qbit2[0].real);
    __m256d u3_1bit2_00i_vec = _mm256_broadcast_sd(&u3_1qbit2[0].imag);
    __m256d u3_1bit2_01r_vec = _mm256_broadcast_sd(&u3_1qbit2[1].real);
    __m256d u3_1bit2_01i_vec = _mm256_broadcast_sd(&u3_1qbit2[1].imag);
    __m256d u3_1bit2_10r_vec = _mm256_broadcast_sd(&u3_1qbit2[2].real);
    __m256d u3_1bit2_10i_vec = _mm256_broadcast_sd(&u3_1qbit2[2].imag);
    __m256d u3_1bit2_11r_vec = _mm256_broadcast_sd(&u3_1qbit2[3].real);
    __m256d u3_1bit2_11i_vec = _mm256_broadcast_sd(&u3_1qbit2[3].imag);


    for ( int current_idx_pair=current_idx + index_step_target; current_idx_pair<matrix_size; current_idx_pair=current_idx_pair+(index_step_target << 1) ) {
           

        for (int idx = 0; idx < index_step_target; idx++) {


                    int current_idx_loc = current_idx + idx;
                    int current_idx_pair_loc = current_idx_pair + idx;

                    int row_offset = current_idx_loc * input.stride;
                    int row_offset_pair = current_idx_pair_loc * input.stride;
                    for (int col_idx = 0; col_idx < 2 * (input.cols - 3); col_idx = col_idx + 8) {
                      double* element = (double*)input.get_data() + 2 * row_offset;
                      double* element_pair = (double*)input.get_data() + 2 * row_offset_pair;
                     if ((current_idx_loc >> control_qbit) & 1) {

    
                              // extract successive elements from arrays element, element_pair
                              __m256d element_vec = _mm256_load_pd(element + col_idx);
                              __m256d element_vec2 = _mm256_load_pd(element + col_idx + 4);
                              __m256d tmp = _mm256_shuffle_pd(element_vec, element_vec2, 0);
                              element_vec2 = _mm256_shuffle_pd(element_vec, element_vec2, 0xf);
                              element_vec = tmp;

                              __m256d element_pair_vec = _mm256_load_pd(element_pair + col_idx);
                              __m256d element_pair_vec2 = _mm256_load_pd(element_pair + col_idx + 4);
                              tmp = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0);
                              element_pair_vec2 = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0xf);
                              element_pair_vec = tmp;

                              __m256d vec3 = _mm256_mul_pd(u3_1bit_00r_vec, element_vec);
                              vec3 = _mm256_fnmadd_pd(u3_1bit_00i_vec, element_vec2, vec3);
                              __m256d vec4 = _mm256_mul_pd(u3_1bit_01r_vec, element_pair_vec);
                              vec4 = _mm256_fnmadd_pd(u3_1bit_01i_vec, element_pair_vec2, vec4);
                              vec3 = _mm256_add_pd(vec3, vec4);
                              __m256d vec5 = _mm256_mul_pd(u3_1bit_00r_vec, element_vec2);
                              vec5 = _mm256_fmadd_pd(u3_1bit_00i_vec, element_vec, vec5);
                              __m256d vec6 = _mm256_mul_pd(u3_1bit_01r_vec, element_pair_vec2);
                              vec6 = _mm256_fmadd_pd(u3_1bit_01i_vec, element_pair_vec, vec6);
                              vec5 = _mm256_add_pd(vec5, vec6);    

                              // 6 store the transformed elements in vec3
                              tmp = _mm256_shuffle_pd(vec3, vec5, 0);
                              vec5 = _mm256_shuffle_pd(vec3, vec5, 0xf);
                              vec3 = tmp;
                              _mm256_store_pd(element + col_idx, vec3);
                              _mm256_store_pd(element + col_idx + 4, vec5);

                              __m256d vec7 = _mm256_mul_pd(u3_1bit_10r_vec, element_vec);
                              vec7 = _mm256_fnmadd_pd(u3_1bit_10i_vec, element_vec2, vec7);
                              __m256d vec8 = _mm256_mul_pd(u3_1bit_11r_vec, element_pair_vec);
                              vec8 = _mm256_fnmadd_pd(u3_1bit_11i_vec, element_pair_vec2, vec8);
                              vec7 = _mm256_add_pd(vec7, vec8);
                              __m256d vec9 = _mm256_mul_pd(u3_1bit_10r_vec, element_vec2);
                              vec9 = _mm256_fmadd_pd(u3_1bit_10i_vec, element_vec, vec9);
                              __m256d vec10 = _mm256_mul_pd(u3_1bit_11r_vec, element_pair_vec2);
                              vec10 = _mm256_fmadd_pd(u3_1bit_11i_vec, element_pair_vec, vec10);
                              vec9 = _mm256_add_pd(vec9, vec10);

                              // 6 store the transformed elements in vec3
                              tmp = _mm256_shuffle_pd(vec7, vec9, 0);
                              vec9 = _mm256_shuffle_pd(vec7, vec9, 0xf);
                              vec7 = tmp;
                              _mm256_store_pd(element_pair + col_idx, vec7);
                              _mm256_store_pd(element_pair + col_idx + 4, vec9);



                      }
                      else {

    
                              // extract successive elements from arrays element, element_pair
                              __m256d element_vec = _mm256_load_pd(element + col_idx);
                              __m256d element_vec2 = _mm256_load_pd(element + col_idx + 4);
                              __m256d tmp = _mm256_shuffle_pd(element_vec, element_vec2, 0);
                              element_vec2 = _mm256_shuffle_pd(element_vec, element_vec2, 0xf);
                              element_vec = tmp;

                              __m256d element_pair_vec = _mm256_load_pd(element_pair + col_idx);
                              __m256d element_pair_vec2 = _mm256_load_pd(element_pair + col_idx + 4);
                              tmp = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0);
                              element_pair_vec2 = _mm256_shuffle_pd(element_pair_vec, element_pair_vec2, 0xf);
                              element_pair_vec = tmp;

                              __m256d vec3 = _mm256_mul_pd(u3_1bit2_00r_vec, element_vec);
                              vec3 = _mm256_fnmadd_pd(u3_1bit2_00i_vec, element_vec2, vec3);
                              __m256d vec4 = _mm256_mul_pd(u3_1bit2_01r_vec, element_pair_vec);
                              vec4 = _mm256_fnmadd_pd(u3_1bit2_01i_vec, element_pair_vec2, vec4);
                              vec3 = _mm256_add_pd(vec3, vec4);
                              __m256d vec5 = _mm256_mul_pd(u3_1bit2_00r_vec, element_vec2);
                              vec5 = _mm256_fmadd_pd(u3_1bit2_00i_vec, element_vec, vec5);
                              __m256d vec6 = _mm256_mul_pd(u3_1bit2_01r_vec, element_pair_vec2);
                              vec6 = _mm256_fmadd_pd(u3_1bit2_01i_vec, element_pair_vec, vec6);
                              vec5 = _mm256_add_pd(vec5, vec6);    

                              // 6 store the transformed elements in vec3
                              tmp = _mm256_shuffle_pd(vec3, vec5, 0);
                              vec5 = _mm256_shuffle_pd(vec3, vec5, 0xf);
                              vec3 = tmp;
                              _mm256_store_pd(element + col_idx, vec3);
                              _mm256_store_pd(element + col_idx + 4, vec5);

                              __m256d vec7 = _mm256_mul_pd(u3_1bit2_10r_vec, element_vec);
                              vec7 = _mm256_fnmadd_pd(u3_1bit2_10i_vec, element_vec2, vec7);
                              __m256d vec8 = _mm256_mul_pd(u3_1bit2_11r_vec, element_pair_vec);
                              vec8 = _mm256_fnmadd_pd(u3_1bit2_11i_vec, element_pair_vec2, vec8);
                              vec7 = _mm256_add_pd(vec7, vec8);
                              __m256d vec9 = _mm256_mul_pd(u3_1bit2_10r_vec, element_vec2);
                              vec9 = _mm256_fmadd_pd(u3_1bit2_10i_vec, element_vec, vec9);
                              __m256d vec10 = _mm256_mul_pd(u3_1bit2_11r_vec, element_pair_vec2);
                              vec10 = _mm256_fmadd_pd(u3_1bit2_11i_vec, element_pair_vec, vec10);
                              vec9 = _mm256_add_pd(vec9, vec10);

                              // 6 store the transformed elements in vec3
                              tmp = _mm256_shuffle_pd(vec7, vec9, 0);
                              vec9 = _mm256_shuffle_pd(vec7, vec9, 0xf);
                              vec7 = tmp;
                              _mm256_store_pd(element_pair + col_idx, vec7);
                              _mm256_store_pd(element_pair + col_idx + 4, vec9);
                      }
                }

                        int remainder = input.cols % 4;
              if (remainder != 0) {

                            for (int col_idx = input.cols-remainder; col_idx < input.cols; col_idx++) {
                    int index      = row_offset+col_idx;
                    int index_pair = row_offset_pair+col_idx;  
                    if ( (current_idx_loc >> control_qbit) & 1 ) {

              

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    QGD_Complex16 tmp1 = mult(u3_1qbit1[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit1[1], element_pair);
 
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit1[2], element);
                    tmp2 = mult(u3_1qbit1[3], element_pair);

                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;

                }

            else {
                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];              

                    QGD_Complex16 tmp1 = mult(u3_1qbit2[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit2[1], element_pair);
 
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit2[2], element);
                    tmp2 = mult(u3_1qbit2[3], element_pair);

                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;
            }
                            }
        
                        }
            //std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;

                 
            }



            current_idx = current_idx + (index_step_target << 1);

    }


}
