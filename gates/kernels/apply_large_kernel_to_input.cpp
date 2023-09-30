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
#include <immintrin.h>
#include "tbb/tbb.h"

void apply_large_kernel_to_state_vector_input(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

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


void apply_large_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

    int index_step_outer = 1 << outer_qbit;
    int index_step_inner = 1 << inner_qbit;
    int current_idx = 0;
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    for (int current_idx_pair_outer=current_idx + index_step_outer; current_idx_pair_outer<matrix_size; current_idx_pair_outer=current_idx_pair_outer+(index_step_outer << 1)){
    
        for (int current_idx_inner = 0; current_idx_inner < index_step_outer; current_idx_inner=current_idx_inner+(index_step_inner<<1)){
            if (inner_qbit==0){
            	for (int idx=0; idx<index_step_inner; idx++){
            	

			    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			    int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
	        	int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			    int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			    double results[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
			    
                double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
			    
                __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                element_outer_vec = _mm256_permute4x64_pd(element_outer_vec,0b11011000);
                __m256d element_inner_vec = _mm256_loadu_pd(element_inner);
                element_inner_vec = _mm256_permute4x64_pd(element_inner_vec,0b11011000);
                __m256d outer_inner_vec = _mm256_shuffle_pd(element_outer_vec,element_inner_vec,0b0000);
                outer_inner_vec = _mm256_permute4x64_pd(outer_inner_vec,0b11011000);


                __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                element_outer_pair_vec = _mm256_permute4x64_pd(element_outer_pair_vec,0b11011000);
                __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                element_inner_pair_vec = _mm256_permute4x64_pd(element_inner_pair_vec,0b11011000);
                __m256d outer_inner_pair_vec = _mm256_shuffle_pd(element_outer_pair_vec,element_inner_pair_vec,0b0000);
                outer_inner_pair_vec = _mm256_permute4x64_pd(outer_inner_pair_vec,0b11011000);




			    for (int mult_idx=0; mult_idx<4; mult_idx++){
			        double* unitary_row_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
			        double* unitary_row_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
			        
                    __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                    __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                    
                    __m256d vec3_upper              = _mm256_mul_pd(outer_inner_vec, unitary_row_01_vec);
                    __m256d unitary_row_01_switched = _mm256_permute_pd(unitary_row_01_vec, 0x5);
                    unitary_row_01_switched         = _mm256_mul_pd( unitary_row_01_switched, neg);
                    __m256d vec4_upper              = _mm256_mul_pd( outer_inner_vec, unitary_row_01_switched);
                    __m256d result_upper_vec        = _mm256_hsub_pd( vec3_upper, vec4_upper);
                    result_upper_vec                = _mm256_permute4x64_pd( result_upper_vec, 0b11011000);
                    
                    __m256d vec3_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_vec);
                    __m256d unitary_row_23_switched = _mm256_permute_pd(unitary_row_23_vec, 0x5);
                    unitary_row_23_switched = _mm256_mul_pd(unitary_row_23_switched, neg);
                    __m256d vec4_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_switched);
                    __m256d result_lower_vec = _mm256_hsub_pd(vec3_lower, vec4_lower);
                    result_lower_vec = _mm256_permute4x64_pd(result_lower_vec,0b11011000);
                    
                    
                    __m256d result_vec = _mm256_hadd_pd(result_upper_vec,result_lower_vec);
                    result_vec = _mm256_hadd_pd(result_vec,result_vec);
                    double* result = (double*)&result_vec;
                    results[mult_idx*2] = result[0];
                    results[mult_idx*2+1] = result[2];
			    }
			    input[current_idx_outer_loc].real = results[0];
			    input[current_idx_outer_loc].imag = results[1];
			    input[current_idx_inner_loc].real = results[2];
			    input[current_idx_inner_loc].imag = results[3];
			    input[current_idx_outer_pair_loc].real = results[4];
			    input[current_idx_outer_pair_loc].imag = results[5];
			    input[current_idx_inner_pair_loc].real = results[6];
			    input[current_idx_inner_pair_loc].imag = results[7];
            	}
            }
            else {

                for (int idx=0; idx<index_step_inner; idx=idx+2){
                
    			    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
		            
                    double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                    double* element_inner = element_outer + 2 * index_step_inner;
                    
                    double* element_outer_pair = element_outer + 2 * index_step_outer;
                    double* element_inner_pair = element_outer_pair + 2 * index_step_inner;
                    
                    double* indices[4] = {element_outer,element_inner,element_outer_pair,element_inner_pair};
                    
                    __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                    __m256d element_inner_vec = _mm256_loadu_pd(element_inner);

                    __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                    __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                    
                    __m256d result_vecs[4];
                    
                    for (int mult_idx=0; mult_idx<4; mult_idx++){
		                double* unitary_col_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
		                double* unitary_col_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
		                
		                __m256d unitary_col_01_vec     = _mm256_loadu_pd(unitary_col_01); // a,b,c,d
		                unitary_col_01_vec             = _mm256_permute4x64_pd(unitary_col_01_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_0_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_1_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b1111); // c, c, d, d
		                unitary_col_0_vec              = _mm256_permute4x64_pd(unitary_col_0_vec,0b11011000); // a, b, a, b
		                unitary_col_1_vec              = _mm256_permute4x64_pd(unitary_col_1_vec,0b11011000); // c, d, c, d

		                __m256d unitary_col_23_vec     = _mm256_loadu_pd(unitary_col_23); // a,b,c,d
		                unitary_col_23_vec             = _mm256_permute4x64_pd(unitary_col_23_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_2_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_3_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b1111); // c, c, d, d
		                unitary_col_2_vec              = _mm256_permute4x64_pd(unitary_col_2_vec,0b11011000); // a, a, b, b
		                unitary_col_3_vec              = _mm256_permute4x64_pd(unitary_col_3_vec,0b11011000); // c, c, d, d
		                
                        __m256d outer_vec_3            = _mm256_mul_pd(element_outer_vec, unitary_col_0_vec);
                        __m256d unitary_col_0_switched = _mm256_permute_pd(unitary_col_0_vec, 0x5);
                        unitary_col_0_switched         = _mm256_mul_pd( unitary_col_0_switched, neg);
                        __m256d outer_vec_4            = _mm256_mul_pd( element_outer_vec, unitary_col_0_switched);
                        __m256d result_outer_vec       = _mm256_hsub_pd( outer_vec_3, outer_vec_4);

                        
                        __m256d inner_vec_3            = _mm256_mul_pd(element_inner_vec, unitary_col_1_vec);
                        __m256d unitary_col_1_switched = _mm256_permute_pd(unitary_col_1_vec, 0x5);
                        unitary_col_1_switched         = _mm256_mul_pd( unitary_col_1_switched, neg);
                        __m256d inner_vec_4            = _mm256_mul_pd( element_inner_vec, unitary_col_1_switched);
                        __m256d result_inner_vec       = _mm256_hsub_pd( inner_vec_3, inner_vec_4);
                        
                        __m256d outer_pair_vec_3       = _mm256_mul_pd(element_outer_pair_vec, unitary_col_2_vec);
                        __m256d unitary_col_2_switched = _mm256_permute_pd(unitary_col_2_vec, 0x5);
                        unitary_col_2_switched         = _mm256_mul_pd( unitary_col_2_switched, neg);
                        __m256d outer_pair_vec_4       = _mm256_mul_pd( element_outer_pair_vec, unitary_col_2_switched);
                        __m256d result_outer_pair_vec  = _mm256_hsub_pd( outer_pair_vec_3, outer_pair_vec_4);

                        
                        __m256d inner_pair_vec_3       = _mm256_mul_pd(element_inner_pair_vec, unitary_col_3_vec);
                        __m256d unitary_col_3_switched = _mm256_permute_pd(unitary_col_3_vec, 0x5);
                        unitary_col_3_switched         = _mm256_mul_pd( unitary_col_3_switched, neg);
                        __m256d inner_pair_vec_4       = _mm256_mul_pd( element_inner_pair_vec, unitary_col_3_switched);
                        __m256d result_inner_pair_vec  = _mm256_hsub_pd( inner_pair_vec_3, inner_pair_vec_4);
                        
                        __m256d result_vec    = _mm256_add_pd(result_outer_vec,result_inner_vec);
                        result_vec            = _mm256_add_pd(result_vec,result_outer_pair_vec);
                        result_vec            = _mm256_add_pd(result_vec,result_inner_pair_vec);
                        result_vecs[mult_idx] = result_vec;
                    }
                    for (int row_idx=0; row_idx<4;row_idx++){
                        _mm256_storeu_pd(indices[row_idx], result_vecs[row_idx]);
                    }

                }
            
            }
        }
        current_idx = current_idx + (index_step_outer << 1);
    }
    
}


void apply_large_kernel_to_state_vector_input_parallel_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_outer = 1 << outer_qbit;
    
    int index_step_inner = 1 << inner_qbit;
    
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_outer/(index_step_inner << 1);
    
    int outer_grain_size;
    int inner_grain_size;
    
    if ( index_step_outer <= 4 ) {
        outer_grain_size = 32;
    }
    else if ( index_step_outer <= 8 ) {
        outer_grain_size = 16;
    }
    else if ( index_step_outer <= 16 ) {
        outer_grain_size = 8;
    }
    else {
        outer_grain_size = 2;
    }
    
    if (index_step_inner <=2){
        inner_grain_size = 64;
    }
    else if ( index_step_inner <= 4 ) {
        inner_grain_size = 32;
    }
    else if ( index_step_inner <= 8 ) {
        inner_grain_size = 16;
    }
    else if ( index_step_inner <= 16 ) {
        inner_grain_size = 8;
    }
    else {
        inner_grain_size = 2;
    }

    
    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx = r.begin()*(index_step_outer<<1);
        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_inner_cycles,inner_grain_size), [&](tbb::blocked_range<int> r) {
                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){
                    if (inner_qbit<2){
                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner,64),[&](tbb::blocked_range<int> r){
                    
        	            for (int idx=r.begin(); idx<r.end(); ++idx){
        	
			                int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			                int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
	                    	int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			                int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			                double results[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
			                
                            double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                            double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                            double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                            double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
			                
                            __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                            element_outer_vec = _mm256_permute4x64_pd(element_outer_vec,0b11011000);
                            __m256d element_inner_vec = _mm256_loadu_pd(element_inner);
                            element_inner_vec = _mm256_permute4x64_pd(element_inner_vec,0b11011000);
                            __m256d outer_inner_vec = _mm256_shuffle_pd(element_outer_vec,element_inner_vec,0b0000);
                            outer_inner_vec = _mm256_permute4x64_pd(outer_inner_vec,0b11011000);


                            __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                            element_outer_pair_vec = _mm256_permute4x64_pd(element_outer_pair_vec,0b11011000);
                            __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                            element_inner_pair_vec = _mm256_permute4x64_pd(element_inner_pair_vec,0b11011000);
                            __m256d outer_inner_pair_vec = _mm256_shuffle_pd(element_outer_pair_vec,element_inner_pair_vec,0b0000);
                            outer_inner_pair_vec = _mm256_permute4x64_pd(outer_inner_pair_vec,0b11011000);


			                __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

			                for (int mult_idx=0; mult_idx<4; mult_idx++){
			                    double* unitary_row_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
			                    double* unitary_row_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
			                    
                                __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                                __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                                
                                __m256d vec3_upper = _mm256_mul_pd(outer_inner_vec, unitary_row_01_vec);
                                __m256d unitary_row_01_switched = _mm256_permute_pd(unitary_row_01_vec, 0x5);
                                unitary_row_01_switched = _mm256_mul_pd(unitary_row_01_switched, neg);
                                __m256d vec4_upper = _mm256_mul_pd(outer_inner_vec, unitary_row_01_switched);
                                __m256d result_upper_vec = _mm256_hsub_pd(vec3_upper, vec4_upper);
                                result_upper_vec = _mm256_permute4x64_pd(result_upper_vec,0b11011000);
                                
                                __m256d vec3_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_vec);
                                __m256d unitary_row_23_switched = _mm256_permute_pd(unitary_row_23_vec, 0x5);
                                unitary_row_23_switched = _mm256_mul_pd(unitary_row_23_switched, neg);
                                __m256d vec4_lower = _mm256_mul_pd(outer_inner_pair_vec, unitary_row_23_switched);
                                __m256d result_lower_vec = _mm256_hsub_pd(vec3_lower, vec4_lower);
                                result_lower_vec = _mm256_permute4x64_pd(result_lower_vec,0b11011000);
                                
                                
                                __m256d result_vec = _mm256_hadd_pd(result_upper_vec,result_lower_vec);
                                result_vec = _mm256_hadd_pd(result_vec,result_vec);
                                double* result = (double*)&result_vec;
                                results[mult_idx*2] = result[0];
                                results[mult_idx*2+1] = result[2];
			                }
		                input[current_idx_outer_loc].real = results[0];
		                input[current_idx_outer_loc].imag = results[1];
		                input[current_idx_inner_loc].real = results[2];
		                input[current_idx_inner_loc].imag = results[3];
		                input[current_idx_outer_pair_loc].real = results[4];
		                input[current_idx_outer_pair_loc].imag = results[5];
		                input[current_idx_inner_pair_loc].real = results[6];
		                input[current_idx_inner_pair_loc].imag = results[7];
            	    }
                   });
                }
                else{
                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner/2,32),[&](tbb::blocked_range<int> r){
                    for (int alt_idx=r.begin(); alt_idx<r.end(); ++alt_idx){
                    int idx = alt_idx*2;
    			    int current_idx_outer_loc = current_idx + current_idx_inner + idx;
		            
                    double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                    double* element_inner = element_outer + 2 * index_step_inner;
                    
                    double* element_outer_pair = element_outer + 2 * index_step_outer;
                    double* element_inner_pair = element_outer_pair + 2 * index_step_inner;
                    
                    double* indices[4] = {element_outer,element_inner,element_outer_pair,element_inner_pair};
                    
                    __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
                    __m256d element_inner_vec = _mm256_loadu_pd(element_inner);

                    __m256d element_outer_pair_vec = _mm256_loadu_pd(element_outer_pair);
                    __m256d element_inner_pair_vec = _mm256_loadu_pd(element_inner_pair);
                    
                    __m256d result_vecs[4];
                    
                    for (int mult_idx=0; mult_idx<4; mult_idx++){
		                double* unitary_col_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
		                double* unitary_col_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
		                
		                __m256d unitary_col_01_vec     = _mm256_loadu_pd(unitary_col_01); // a,b,c,d
		                unitary_col_01_vec             = _mm256_permute4x64_pd(unitary_col_01_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_0_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_1_vec      = _mm256_shuffle_pd(unitary_col_01_vec,unitary_col_01_vec,0b1111); // c, c, d, d
		                unitary_col_0_vec              = _mm256_permute4x64_pd(unitary_col_0_vec,0b11011000); // a, b, a, b
		                unitary_col_1_vec              = _mm256_permute4x64_pd(unitary_col_1_vec,0b11011000); // c, d, c, d

		                __m256d unitary_col_23_vec     = _mm256_loadu_pd(unitary_col_23); // a,b,c,d
		                unitary_col_23_vec             = _mm256_permute4x64_pd(unitary_col_23_vec,0b11011000); // a, c, b, d
		                __m256d unitary_col_2_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b0000); // a, a, b, b
		                __m256d unitary_col_3_vec      = _mm256_shuffle_pd(unitary_col_23_vec,unitary_col_23_vec,0b1111); // c, c, d, d
		                unitary_col_2_vec              = _mm256_permute4x64_pd(unitary_col_2_vec,0b11011000); // a, a, b, b
		                unitary_col_3_vec              = _mm256_permute4x64_pd(unitary_col_3_vec,0b11011000); // c, c, d, d
		                
                        __m256d outer_vec_3            = _mm256_mul_pd(element_outer_vec, unitary_col_0_vec);
                        __m256d unitary_col_0_switched = _mm256_permute_pd(unitary_col_0_vec, 0x5);
                        unitary_col_0_switched         = _mm256_mul_pd( unitary_col_0_switched, neg);
                        __m256d outer_vec_4            = _mm256_mul_pd( element_outer_vec, unitary_col_0_switched);
                        __m256d result_outer_vec       = _mm256_hsub_pd( outer_vec_3, outer_vec_4);

                        
                        __m256d inner_vec_3            = _mm256_mul_pd(element_inner_vec, unitary_col_1_vec);
                        __m256d unitary_col_1_switched = _mm256_permute_pd(unitary_col_1_vec, 0x5);
                        unitary_col_1_switched         = _mm256_mul_pd( unitary_col_1_switched, neg);
                        __m256d inner_vec_4            = _mm256_mul_pd( element_inner_vec, unitary_col_1_switched);
                        __m256d result_inner_vec       = _mm256_hsub_pd( inner_vec_3, inner_vec_4);
                        
                        __m256d outer_pair_vec_3       = _mm256_mul_pd(element_outer_pair_vec, unitary_col_2_vec);
                        __m256d unitary_col_2_switched = _mm256_permute_pd(unitary_col_2_vec, 0x5);
                        unitary_col_2_switched         = _mm256_mul_pd( unitary_col_2_switched, neg);
                        __m256d outer_pair_vec_4       = _mm256_mul_pd( element_outer_pair_vec, unitary_col_2_switched);
                        __m256d result_outer_pair_vec  = _mm256_hsub_pd( outer_pair_vec_3, outer_pair_vec_4);

                        
                        __m256d inner_pair_vec_3       = _mm256_mul_pd(element_inner_pair_vec, unitary_col_3_vec);
                        __m256d unitary_col_3_switched = _mm256_permute_pd(unitary_col_3_vec, 0x5);
                        unitary_col_3_switched         = _mm256_mul_pd( unitary_col_3_switched, neg);
                        __m256d inner_pair_vec_4       = _mm256_mul_pd( element_inner_pair_vec, unitary_col_3_switched);
                        __m256d result_inner_pair_vec  = _mm256_hsub_pd( inner_pair_vec_3, inner_pair_vec_4);
                        
                        __m256d result_vec    = _mm256_add_pd(result_outer_vec,result_inner_vec);
                        result_vec            = _mm256_add_pd(result_vec,result_outer_pair_vec);
                        result_vec            = _mm256_add_pd(result_vec,result_inner_pair_vec);
                        result_vecs[mult_idx] = result_vec;
                    }
                    for (int row_idx=0; row_idx<4;row_idx++){
                        _mm256_storeu_pd(indices[row_idx], result_vecs[row_idx]);
                    }

                }
                    });
                }
                current_idx_inner = current_idx_inner +(index_step_inner << 1);
                }
            });
        current_idx = current_idx + (index_step_outer << 1);
        current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

    }
    });
    
}
