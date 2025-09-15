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
        switch(involved_qbits.size()){
            case 2: apply_2qbit_kernel_to_state_vector_input(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size); break;
            case 3: apply_3qbit_kernel_to_state_vector_input(unitary,input,involved_qbits,matrix_size); break;
            case 4: apply_4qbit_kernel_to_state_vector_input(unitary,input,involved_qbits,matrix_size); break;
            case 5: apply_5qbit_kernel_to_state_vector_input(unitary,input,involved_qbits,matrix_size); break;
            default: throw std::invalid_argument("Unsupported number of qubits for state vector.");
        }
    }
    else 
    {
        apply_2qbit_kernel_to_matrix_input(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size);
    }
}

/**
@brief Call to apply kernel to apply two qubit gate kernel on a state vector
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param inner_qbit The lower significance qubit (little endian convention)
@param outer_qbit The higher significance qubit (little endian convention)
@param matrix_size The size of the input
*/
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
/**
@brief Call to apply kernel to apply two qubit gate kernel on an input matrix using AVX
@param two_qbit_unitary The 4x4 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param inner_qbit The lower significance qubit (little endian convention)
@param outer_qbit The higher significance qubit (little endian convention)
@param matrix_size The size of the input
*/
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

/**
@brief Call to apply kernel to apply three qubit gate kernel on a state vector
@param two_qbit_unitary The 8x8 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_3qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    int index_step_inner = 1 << involved_qbits[0];
    int index_step_middle = 1 << involved_qbits[1];
    int index_step_outer = 1 << involved_qbits[2];
    int current_idx = 0;
    
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<input.rows; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
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
        }
        current_idx = current_idx + (index_step_outer << 1);
     }
}


/**
@brief Call to apply kernel to apply four qubit gate kernel on a state vector
@param unitary The 16x16 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_4qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size) {

    int index_step_q0 = 1 << involved_qbits[0];
    int index_step_q1 = 1 << involved_qbits[1];
    int index_step_q2 = 1 << involved_qbits[2];
    int index_step_q3 = 1 << involved_qbits[3];

    int current_idx = 0;

    // q3 loop (outermost)
    for (int current_idx_pair_q3 = current_idx + index_step_q3; current_idx_pair_q3 < input.rows; current_idx_pair_q3 += (index_step_q3 << 1)) {
        
        // q2 loop
        for (int current_idx_q2 = 0; current_idx_q2 < index_step_q3; current_idx_q2 += (index_step_q2 << 1)) {

            // q1 loop
            for (int current_idx_q1 = 0; current_idx_q1 < index_step_q2; current_idx_q1 += (index_step_q1 << 1)) {

                for (int current_idx_q0 = 0; current_idx_q0 < index_step_q1; current_idx_q0 += (index_step_q0 << 1)) {
                
                // q0 loop (innermost)
                for (int idx = 0; idx < index_step_q0; idx++) {

                    // base indices for current iteration
                    int current_idx_loc = current_idx + current_idx_q2 + current_idx_q1 + current_idx_q0 + idx;
                    int current_idx_pair_loc = current_idx_pair_q3 + idx + current_idx_q1 + current_idx_q2 + current_idx_q0;

                    // q3=0 states (first 8 states)
                    int current_idx_q0_0_loc = current_idx_loc; // |0000>
                    int current_idx_q0_1_loc = current_idx_loc + index_step_q0; // |0001>
                    int current_idx_q1_0_loc = current_idx_loc + index_step_q1;
                    int current_idx_q1_1_loc = current_idx_loc + index_step_q1 + index_step_q0;
                    int current_idx_q2_0_loc = current_idx_loc + index_step_q2;
                    int current_idx_q2_1_loc = current_idx_loc + index_step_q2 + index_step_q0;
                    int current_idx_q2_q1_0_loc = current_idx_loc + index_step_q2 + index_step_q1; // |0110>
                    int current_idx_q2_q1_1_loc = current_idx_loc + index_step_q2 + index_step_q1 + index_step_q0; // |0111>

                    // q3=1 states (first 8 states)
                    int current_idx_q3_q0_0_pair_loc = current_idx_pair_loc; // |1000>
                    int current_idx_q3_q0_1_pair_loc = current_idx_pair_loc + index_step_q0; // |1001>
                    int current_idx_q3_q1_0_pair_loc = current_idx_pair_loc + index_step_q1;
                    int current_idx_q3_q1_1_pair_loc = current_idx_pair_loc + index_step_q1 + index_step_q0;
                    int current_idx_q3_q2_0_pair_loc = current_idx_pair_loc + index_step_q2;
                    int current_idx_q3_q2_1_pair_loc = current_idx_pair_loc + index_step_q2 + index_step_q0;
                    int current_idx_q3_q2_q1_0_pair_loc = current_idx_pair_loc + index_step_q2 + index_step_q1; // |1110>
                    int current_idx_q3_q2_q1_1_pair_loc = current_idx_pair_loc + index_step_q2 + index_step_q1 + index_step_q0; // |1111>

                    int indexes[16] = {
                        current_idx_q0_0_loc, current_idx_q0_1_loc, current_idx_q1_0_loc, current_idx_q1_1_loc,
                        current_idx_q2_0_loc, current_idx_q2_1_loc, current_idx_q2_q1_0_loc, current_idx_q2_q1_1_loc,
                        current_idx_q3_q0_0_pair_loc, current_idx_q3_q0_1_pair_loc, current_idx_q3_q1_0_pair_loc, current_idx_q3_q1_1_pair_loc,
                        current_idx_q3_q2_0_pair_loc, current_idx_q3_q2_1_pair_loc, current_idx_q3_q2_q1_0_pair_loc, current_idx_q3_q2_q1_1_pair_loc
                    };

                    QGD_Complex16 element_0  = input[current_idx_q0_0_loc];
                    QGD_Complex16 element_1  = input[current_idx_q0_1_loc];
                    QGD_Complex16 element_2  = input[current_idx_q1_0_loc];
                    QGD_Complex16 element_3  = input[current_idx_q1_1_loc];
                    QGD_Complex16 element_4  = input[current_idx_q2_0_loc];
                    QGD_Complex16 element_5  = input[current_idx_q2_1_loc];
                    QGD_Complex16 element_6  = input[current_idx_q2_q1_0_loc];
                    QGD_Complex16 element_7  = input[current_idx_q2_q1_1_loc];
                    QGD_Complex16 element_8  = input[current_idx_q3_q0_0_pair_loc];
                    QGD_Complex16 element_9  = input[current_idx_q3_q0_1_pair_loc];
                    QGD_Complex16 element_10 = input[current_idx_q3_q1_0_pair_loc];
                    QGD_Complex16 element_11 = input[current_idx_q3_q1_1_pair_loc];
                    QGD_Complex16 element_12 = input[current_idx_q3_q2_0_pair_loc];
                    QGD_Complex16 element_13 = input[current_idx_q3_q2_1_pair_loc];
                    QGD_Complex16 element_14 = input[current_idx_q3_q2_q1_0_pair_loc];
                    QGD_Complex16 element_15 = input[current_idx_q3_q2_q1_1_pair_loc];

                    for (int mult_idx = 0; mult_idx < 16; mult_idx++) {
                        QGD_Complex16 tmp0  = mult(unitary[mult_idx*16], element_0);
                        QGD_Complex16 tmp1  = mult(unitary[mult_idx*16 + 1], element_1);
                        QGD_Complex16 tmp2  = mult(unitary[mult_idx*16 + 2], element_2);
                        QGD_Complex16 tmp3  = mult(unitary[mult_idx*16 + 3], element_3);
                        QGD_Complex16 tmp4  = mult(unitary[mult_idx*16 + 4], element_4);
                        QGD_Complex16 tmp5  = mult(unitary[mult_idx*16 + 5], element_5);
                        QGD_Complex16 tmp6  = mult(unitary[mult_idx*16 + 6], element_6);
                        QGD_Complex16 tmp7  = mult(unitary[mult_idx*16 + 7], element_7);
                        QGD_Complex16 tmp8  = mult(unitary[mult_idx*16 + 8], element_8);
                        QGD_Complex16 tmp9  = mult(unitary[mult_idx*16 + 9], element_9);
                        QGD_Complex16 tmp10 = mult(unitary[mult_idx*16 + 10], element_10);
                        QGD_Complex16 tmp11 = mult(unitary[mult_idx*16 + 11], element_11);
                        QGD_Complex16 tmp12 = mult(unitary[mult_idx*16 + 12], element_12);
                        QGD_Complex16 tmp13 = mult(unitary[mult_idx*16 + 13], element_13);
                        QGD_Complex16 tmp14 = mult(unitary[mult_idx*16 + 14], element_14);
                        QGD_Complex16 tmp15 = mult(unitary[mult_idx*16 + 15], element_15);

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
}


/**
@brief Call to apply kernel to apply five qubit gate kernel on a state vector
@param unitary The 32x32 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits affected by the gate in order
@param matrix_size The size of the input
*/
void apply_5qbit_kernel_to_state_vector_input(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    int index_step_q0 = 1 << involved_qbits[0];
    int index_step_q1 = 1 << involved_qbits[1];
    int index_step_q2 = 1 << involved_qbits[2];
    int index_step_q3 = 1 << involved_qbits[3];
    int index_step_q4 = 1 << involved_qbits[4];  

    int current_idx = 0;

    // q4 loop (outermost)
    for (int current_idx_pair_q4 = current_idx + index_step_q4; current_idx_pair_q4 < input.rows; current_idx_pair_q4 += (index_step_q4 << 1)) {
        
        // q3 loop
        for (int current_idx_q3 = 0; current_idx_q3 < index_step_q4; current_idx_q3 += (index_step_q3 << 1)) {

            // q2 loop
            for (int current_idx_q2 = 0; current_idx_q2 < index_step_q3; current_idx_q2 += (index_step_q2 << 1)) {
                
                // q1 loop
                for (int current_idx_q1 = 0; current_idx_q1 < index_step_q2; current_idx_q1 += (index_step_q1 << 1)) {

                    for (int current_idx_q0 = 0; current_idx_q0 < index_step_q1; current_idx_q0 += (index_step_q0 << 1)) {

                    // q0 loop (innermost)
                    for (int idx = 0; idx < index_step_q0; idx++) {

                    // base indices for current iteration
                    int current_idx_loc = current_idx + current_idx_q3 + current_idx_q2 + current_idx_q1 + current_idx_q0 + idx;
                    int current_idx_pair_q4_loc = current_idx_pair_q4 + idx + current_idx_q1 + current_idx_q2 + current_idx_q3 + current_idx_q0;

                    // q4=0 states (first 16 states)
                    int current_idx_q0_0_loc = current_idx_loc; // |00000>
                    int current_idx_q0_1_loc = current_idx_loc + index_step_q0; // |00001>
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
                    int current_idx_q3_q2_q1_0_loc = current_idx_loc + index_step_q3 + index_step_q2 + index_step_q1; // |01110>
                    int current_idx_q3_q2_q1_1_loc = current_idx_loc + index_step_q3 + index_step_q2 + index_step_q1 + index_step_q0; // |01111>

                    // q4=1 states (last 16 states)
                    int current_idx_q4_q0_0_pair_loc = current_idx_pair_q4_loc; // |10000>
                    int current_idx_q4_q0_1_pair_loc = current_idx_pair_q4_loc + index_step_q0; // |10001>
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
                    int current_idx_q4_q3_q2_q1_0_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q2 + index_step_q1; // |11110>
                    int current_idx_q4_q3_q2_q1_1_pair_loc = current_idx_pair_q4_loc + index_step_q3 + index_step_q2 + index_step_q1 + index_step_q0; // |11111>

                    int indexes[32] = {
                        current_idx_q0_0_loc, // state 0: |00000>
                        current_idx_q0_1_loc, // state 1: |00001>
                        current_idx_q1_0_loc,
                        current_idx_q1_1_loc,
                        current_idx_q2_0_loc,
                        current_idx_q2_1_loc,
                        current_idx_q2_q1_0_loc,
                        current_idx_q2_q1_1_loc,
                        current_idx_q3_0_loc,
                        current_idx_q3_1_loc,
                        current_idx_q3_q1_0_loc,
                        current_idx_q3_q1_1_loc,
                        current_idx_q3_q2_0_loc,
                        current_idx_q3_q2_1_loc,
                        current_idx_q3_q2_q1_0_loc,
                        current_idx_q3_q2_q1_1_loc, // state 15: |01111>
                        current_idx_q4_q0_0_pair_loc, // state 16: |10000>
                        current_idx_q4_q0_1_pair_loc,
                        current_idx_q4_q1_0_pair_loc,
                        current_idx_q4_q1_1_pair_loc,
                        current_idx_q4_q2_0_pair_loc,
                        current_idx_q4_q2_1_pair_loc,
                        current_idx_q4_q2_q1_0_pair_loc,
                        current_idx_q4_q2_q1_1_pair_loc,
                        current_idx_q4_q3_0_pair_loc,
                        current_idx_q4_q3_1_pair_loc,
                        current_idx_q4_q3_q1_0_pair_loc,
                        current_idx_q4_q3_q1_1_pair_loc,
                        current_idx_q4_q3_q2_0_pair_loc,
                        current_idx_q4_q3_q2_1_pair_loc,
                        current_idx_q4_q3_q2_q1_0_pair_loc, // state 30: |11110>
                        current_idx_q4_q3_q2_q1_1_pair_loc // state 31: |11111>
                    };

                    QGD_Complex16 element_0 = input[current_idx_q0_0_loc];
                    QGD_Complex16 element_1 = input[current_idx_q0_1_loc];
                    QGD_Complex16 element_2 = input[current_idx_q1_0_loc];
                    QGD_Complex16 element_3 = input[current_idx_q1_1_loc];
                    QGD_Complex16 element_4 = input[current_idx_q2_0_loc];
                    QGD_Complex16 element_5 = input[current_idx_q2_1_loc];
                    QGD_Complex16 element_6 = input[current_idx_q2_q1_0_loc];
                    QGD_Complex16 element_7 = input[current_idx_q2_q1_1_loc];
                    QGD_Complex16 element_8 = input[current_idx_q3_0_loc];
                    QGD_Complex16 element_9 = input[current_idx_q3_1_loc];
                    QGD_Complex16 element_10 = input[current_idx_q3_q1_0_loc];
                    QGD_Complex16 element_11 = input[current_idx_q3_q1_1_loc];
                    QGD_Complex16 element_12 = input[current_idx_q3_q2_0_loc];
                    QGD_Complex16 element_13 = input[current_idx_q3_q2_1_loc];
                    QGD_Complex16 element_14 = input[current_idx_q3_q2_q1_0_loc];
                    QGD_Complex16 element_15 = input[current_idx_q3_q2_q1_1_loc];
                    QGD_Complex16 element_16 = input[current_idx_q4_q0_0_pair_loc];
                    QGD_Complex16 element_17 = input[current_idx_q4_q0_1_pair_loc];
                    QGD_Complex16 element_18 = input[current_idx_q4_q1_0_pair_loc];
                    QGD_Complex16 element_19 = input[current_idx_q4_q1_1_pair_loc];
                    QGD_Complex16 element_20 = input[current_idx_q4_q2_0_pair_loc];
                    QGD_Complex16 element_21 = input[current_idx_q4_q2_1_pair_loc];
                    QGD_Complex16 element_22 = input[current_idx_q4_q2_q1_0_pair_loc];
                    QGD_Complex16 element_23 = input[current_idx_q4_q2_q1_1_pair_loc];
                    QGD_Complex16 element_24 = input[current_idx_q4_q3_0_pair_loc];
                    QGD_Complex16 element_25 = input[current_idx_q4_q3_1_pair_loc];
                    QGD_Complex16 element_26 = input[current_idx_q4_q3_q1_0_pair_loc];
                    QGD_Complex16 element_27 = input[current_idx_q4_q3_q1_1_pair_loc];
                    QGD_Complex16 element_28 = input[current_idx_q4_q3_q2_0_pair_loc];
                    QGD_Complex16 element_29 = input[current_idx_q4_q3_q2_1_pair_loc];
                    QGD_Complex16 element_30 = input[current_idx_q4_q3_q2_q1_0_pair_loc];
                    QGD_Complex16 element_31 = input[current_idx_q4_q3_q2_q1_1_pair_loc];

                    for (int mult_idx = 0; mult_idx < 32; mult_idx++) {
                        QGD_Complex16 tmp1 = mult(unitary[mult_idx*32], element_0);
                        QGD_Complex16 tmp2 = mult(unitary[mult_idx*32 + 1], element_1);
                        QGD_Complex16 tmp3 = mult(unitary[mult_idx*32 + 2], element_2);
                        QGD_Complex16 tmp4 = mult(unitary[mult_idx*32 + 3], element_3);
                        QGD_Complex16 tmp5 = mult(unitary[mult_idx*32 + 4], element_4);
                        QGD_Complex16 tmp6 = mult(unitary[mult_idx*32 + 5], element_5);
                        QGD_Complex16 tmp7 = mult(unitary[mult_idx*32 + 6], element_6);
                        QGD_Complex16 tmp8 = mult(unitary[mult_idx*32 + 7], element_7);
                        QGD_Complex16 tmp9 = mult(unitary[mult_idx*32 + 8], element_8);
                        QGD_Complex16 tmp10 = mult(unitary[mult_idx*32 + 9], element_9);
                        QGD_Complex16 tmp11 = mult(unitary[mult_idx*32 + 10], element_10);
                        QGD_Complex16 tmp12 = mult(unitary[mult_idx*32 + 11], element_11);
                        QGD_Complex16 tmp13 = mult(unitary[mult_idx*32 + 12], element_12);
                        QGD_Complex16 tmp14 = mult(unitary[mult_idx*32 + 13], element_13);
                        QGD_Complex16 tmp15 = mult(unitary[mult_idx*32 + 14], element_14);
                        QGD_Complex16 tmp16 = mult(unitary[mult_idx*32 + 15], element_15);
                        QGD_Complex16 tmp17 = mult(unitary[mult_idx*32 + 16], element_16);
                        QGD_Complex16 tmp18 = mult(unitary[mult_idx*32 + 17], element_17);
                        QGD_Complex16 tmp19 = mult(unitary[mult_idx*32 + 18], element_18);
                        QGD_Complex16 tmp20 = mult(unitary[mult_idx*32 + 19], element_19);
                        QGD_Complex16 tmp21 = mult(unitary[mult_idx*32 + 20], element_20);
                        QGD_Complex16 tmp22 = mult(unitary[mult_idx*32 + 21], element_21);
                        QGD_Complex16 tmp23 = mult(unitary[mult_idx*32 + 22], element_22);
                        QGD_Complex16 tmp24 = mult(unitary[mult_idx*32 + 23], element_23);
                        QGD_Complex16 tmp25 = mult(unitary[mult_idx*32 + 24], element_24);
                        QGD_Complex16 tmp26 = mult(unitary[mult_idx*32 + 25], element_25);
                        QGD_Complex16 tmp27 = mult(unitary[mult_idx*32 + 26], element_26);
                        QGD_Complex16 tmp28 = mult(unitary[mult_idx*32 + 27], element_27);
                        QGD_Complex16 tmp29 = mult(unitary[mult_idx*32 + 28], element_28);
                        QGD_Complex16 tmp30 = mult(unitary[mult_idx*32 + 29], element_29);
                        QGD_Complex16 tmp31 = mult(unitary[mult_idx*32 + 30], element_30);
                        QGD_Complex16 tmp32 = mult(unitary[mult_idx*32 + 31], element_31);

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
}

/**
@brief Call to apply crot gate kernel on an input matrix
@param u3_1qbit1 The 2x2 kernel to be applied on target |1>
@param u3_1qbit2 The 2x2 kernel to be applied on target |0>
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit
@param control_qbit The control qubit
@param matrix_size The size of the input
*/
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
/**
@brief Call to apply crot gate kernel on an input matrix using AVX
@param u3_1qbit1 The 2x2 kernel to be applied on target |1>
@param u3_1qbit2 The 2x2 kernel to be applied on target |0>
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit
@param control_qbit The control qubit
@param matrix_size The size of the input
*/
void
apply_crot_kernel_to_matrix_input_AVX(Matrix& u3_1qbit1, Matrix& u3_1qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size) {
#ifdef USE_AVX
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

#endif
}
