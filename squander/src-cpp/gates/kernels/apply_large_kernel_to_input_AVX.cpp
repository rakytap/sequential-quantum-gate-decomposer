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


#include "apply_large_kernel_to_input_AVX.h"
#include "tbb/tbb.h"

inline __m256d get_AVX_vector(double* element_outer, double* element_inner){

    __m256d element_outer_vec = _mm256_loadu_pd(element_outer);
    element_outer_vec = _mm256_permute4x64_pd(element_outer_vec,0b11011000);
    __m256d element_inner_vec = _mm256_loadu_pd(element_inner);
    element_inner_vec = _mm256_permute4x64_pd(element_inner_vec,0b11011000);
    __m256d outer_inner_vec = _mm256_shuffle_pd(element_outer_vec,element_inner_vec,0b0000);
    outer_inner_vec = _mm256_permute4x64_pd(outer_inner_vec,0b11011000);
    
    return outer_inner_vec;
}

inline __m256d complex_mult_AVX(__m256d input_vec, __m256d unitary_row_vec, __m256d neg){

    __m256d vec3 = _mm256_mul_pd(input_vec, unitary_row_vec);
    __m256d unitary_row_switched = _mm256_permute_pd(unitary_row_vec, 0x5);
    unitary_row_switched = _mm256_mul_pd(unitary_row_switched, neg);
    __m256d vec4 = _mm256_mul_pd(input_vec, unitary_row_switched);
    __m256d result_vec = _mm256_hsub_pd(vec3, vec4);
    result_vec = _mm256_permute4x64_pd(result_vec,0b11011000);
    
    return result_vec;
}


void apply_large_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){

    switch((int)involved_qbits.size()){
    case 2:{
            //apply_2qbit_kernel_to_state_vector_input_parallel_AVX(unitary,input,involved_qbits,matrix_size);
            apply_2qbit_kernel_to_state_vector_input_AVX(unitary, input, involved_qbits[0], involved_qbits[1], matrix_size);
    }
    case 3:{
        apply_3qbit_kernel_to_state_vector_input_parallel_AVX(unitary,input,involved_qbits,matrix_size);
    }
    case 4:{
            apply_4qbit_kernel_to_state_vector_input_parallel_AVX(unitary,input,involved_qbits,matrix_size);
    }
    }

}



void apply_2qbit_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, const int& inner_qbit, const int& outer_qbit, const int& matrix_size){

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

                    __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_01_vec,neg);
                                
                    __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_23_vec,neg);                   
                    
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


void apply_2qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int inner_qbit = involved_qbits[0];
    
    int outer_qbit = involved_qbits[1];
    
    int index_step_outer = 1 << outer_qbit;
    
    int index_step_inner = 1 << inner_qbit;
    
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_outer/(index_step_inner << 1);
    
    //int outer_grain_size = get_grain_size(index_step_outer);
    //int inner_grain_size = get_grain_size(index_step_inner);

    int outer_grain_size;
    if ( index_step_outer <= 2 ) {
        outer_grain_size = 64;
    }
    else if ( index_step_outer <= 4 ) {
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
outer_grain_size = 6000000;
    int inner_grain_size = 64000000;

    
    tbb::parallel_for( tbb::blocked_range<int>(0, parallel_outer_cycles, outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx            = r.begin()*(index_step_outer<<1);        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
            tbb::parallel_for( tbb::blocked_range<int>(0, parallel_inner_cycles, inner_grain_size), [&](tbb::blocked_range<int> r) {

                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){

                    if (inner_qbit<2){
                            for (int idx=0; idx<index_step_inner; ++idx){
        	
			        int current_idx_outer_loc = current_idx + current_idx_inner + idx;
			        int current_idx_inner_loc = current_idx + current_idx_inner + idx + index_step_inner;
	                        int current_idx_outer_pair_loc = current_idx_pair_outer + idx + current_idx_inner;
			        int current_idx_inner_pair_loc = current_idx_pair_outer + idx + current_idx_inner + index_step_inner;
			        double results[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
			                
                                double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                                double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                                double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                                double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
			                
                                __m256d outer_inner_vec = get_AVX_vector(element_outer, element_inner);



                                __m256d outer_inner_pair_vec =  get_AVX_vector(element_outer_pair, element_inner_pair);
                            

			        __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

			        for (int mult_idx=0; mult_idx<4; mult_idx++){
			            double* unitary_row_01 = (double*)two_qbit_unitary.get_data() + 8*mult_idx;
			            double* unitary_row_23 = (double*)two_qbit_unitary.get_data() + 8*mult_idx + 4;
			                    
                                    __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                                    __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                                
                                    __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_01_vec,neg);
                                            
                                    __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_23_vec,neg);                   
                                
                                
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
                    else{

                        for (int alt_idx=0; alt_idx<index_step_inner/2; ++alt_idx){

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
                    }

                    current_idx_inner = current_idx_inner +(index_step_inner << 1);

                }
            });

            current_idx = current_idx + (index_step_outer << 1);
            current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

        }   
    });
    
}


void apply_3qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_inner = 1 << involved_qbits[0];
    
    int index_step_middle = 1 << involved_qbits[1];
    
    int index_step_outer = 1 << involved_qbits[2];
    
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_middle/(index_step_inner << 1);
    
    int parallel_middle_cycles = index_step_outer/(index_step_middle << 1);
   
    
    int outer_grain_size = get_grain_size(index_step_outer);
    int inner_grain_size = get_grain_size(index_step_outer);
    int middle_grain_size = get_grain_size(index_step_middle);

    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx = r.begin()*(index_step_outer<<1);
        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
        tbb::parallel_for( tbb::blocked_range<int>(0,parallel_middle_cycles,middle_grain_size), [&](tbb::blocked_range<int> r) {
            
            int current_idx_middle = r.begin()*(index_step_middle<<1);
            for (int middle_rdx=r.begin(); middle_rdx<r.end(); middle_rdx++){
            
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_inner_cycles,inner_grain_size), [&](tbb::blocked_range<int> r) {
                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){

                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner,64),[&](tbb::blocked_range<int> r){
                    
        	            for (int idx=r.begin(); idx<r.end(); ++idx){
        	
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
			                
			                QGD_Complex16 results[8];
			                
                            double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                            double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                            
                            double* element_middle = (double*)input.get_data() + 2 * current_idx_middle_loc;
                            double* element_middle_inner = (double*)input.get_data() + 2 * current_idx_middle_inner_loc;
                            
                            double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                            double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
                                                        
                            double* element_middle_pair = (double*)input.get_data() + 2 * current_idx_middle_pair_loc;
                            double* element_middle_inner_pair = (double*)input.get_data() + 2 * current_idx_middle_inner_pair_loc;
			                
                            __m256d outer_inner_vec = get_AVX_vector(element_outer, element_inner);

                           __m256d middle_inner_vec = get_AVX_vector(element_middle,element_middle_inner);

                            __m256d outer_inner_pair_vec =  get_AVX_vector(element_outer_pair, element_inner_pair);
                            
                            __m256d middle_inner_pair_vec =  get_AVX_vector(element_middle_pair, element_middle_inner_pair);



			                for (int mult_idx=0; mult_idx<8; mult_idx++){
			                
			                    double* unitary_row_01 = (double*)unitary.get_data() + 16*mult_idx;
			                    double* unitary_row_23 = (double*)unitary.get_data() + 16*mult_idx + 4;
			                    double* unitary_row_45 = (double*)unitary.get_data() + 16*mult_idx + 8;
			                    double* unitary_row_67 = (double*)unitary.get_data() + 16*mult_idx + 12;
			                    
                                __m256d unitary_row_01_vec = _mm256_loadu_pd(unitary_row_01);
                                __m256d unitary_row_23_vec = _mm256_loadu_pd(unitary_row_23);
                                __m256d unitary_row_45_vec = _mm256_loadu_pd(unitary_row_45);
                                __m256d unitary_row_67_vec = _mm256_loadu_pd(unitary_row_67);
                                
                                
                                __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_01_vec,neg);
                                
                                __m256d result_upper_middle_vec = complex_mult_AVX(middle_inner_vec,unitary_row_23_vec,neg);
                                
                                __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_45_vec,neg);
                                
                                __m256d result_lower_middle_vec = complex_mult_AVX(middle_inner_pair_vec,unitary_row_67_vec,neg);
                                
                                
                                __m256d result1_vec = _mm256_hadd_pd(result_upper_vec,result_upper_middle_vec);
                                __m256d result2_vec = _mm256_hadd_pd(result_lower_vec,result_lower_middle_vec);
                                __m256d result_vec = _mm256_hadd_pd(result1_vec,result2_vec);
                                result_vec = _mm256_hadd_pd(result_vec,result_vec);
                                double* result = (double*)&result_vec;
                                results[mult_idx].real = result[0];
                                results[mult_idx].imag = result[2];
			                }
		                input[current_idx_outer_loc] = results[0];
		                input[current_idx_inner_loc] = results[1];
		                input[current_idx_middle_loc]  = results[2];
		                input[current_idx_middle_inner_loc] = results[3];
		                input[current_idx_outer_pair_loc] = results[4];
		                input[current_idx_inner_pair_loc] = results[5];
		                input[current_idx_middle_pair_loc] = results[6];
		                input[current_idx_middle_inner_pair_loc] = results[7];
            	    }
                   });
                
               
                current_idx_inner = current_idx_inner +(index_step_inner << 1);
                }
            });
                current_idx_middle = current_idx_middle +(index_step_middle << 1);
            }
            });
        current_idx = current_idx + (index_step_outer << 1);
        current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

    }
    });
}

void apply_4qbit_kernel_to_state_vector_input_parallel_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size){
    
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    
    int index_step_inner = 1 << involved_qbits[0];
    
    int index_step_middle1 = 1 << involved_qbits[1];
    
    int index_step_middle2 = 1 << involved_qbits[2];
    
    int index_step_outer = 1 << involved_qbits[3];
    
    int parallel_outer_cycles = matrix_size/(index_step_outer << 1);
    
    int parallel_inner_cycles = index_step_middle1/(index_step_inner << 1);
    
    int parallel_middle1_cycles = index_step_middle2/(index_step_middle1 << 1);
    
    int parallel_middle2_cycles = index_step_outer/(index_step_middle2 << 1);
    
    int outer_grain_size = get_grain_size(index_step_outer);
    int inner_grain_size = get_grain_size(index_step_outer);
    int middle1_grain_size = get_grain_size(index_step_middle1);
    int middle2_grain_size = get_grain_size(index_step_middle2);
    
    tbb::parallel_for( tbb::blocked_range<int>(0,parallel_outer_cycles,outer_grain_size), [&](tbb::blocked_range<int> r) { 
    
        int current_idx = r.begin()*(index_step_outer<<1);
        
        int current_idx_pair_outer = current_idx + index_step_outer;
        

        for (int outer_rdx=r.begin(); outer_rdx<r.end(); outer_rdx++){
        
        tbb::parallel_for( tbb::blocked_range<int>(0,parallel_middle2_cycles,middle2_grain_size), [&](tbb::blocked_range<int> r) {
            
            int current_idx_middle2 = r.begin()*(index_step_middle2<<1);
            
            for (int middle2_rdx=r.begin(); middle2_rdx<r.end(); middle2_rdx++){
            
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_middle1_cycles,middle1_grain_size), [&](tbb::blocked_range<int> r) {
            
            int current_idx_middle1 = r.begin()*(index_step_middle1<<1);
            
            for (int middle1_rdx=r.begin(); middle1_rdx<r.end(); middle1_rdx++){
            
            tbb::parallel_for( tbb::blocked_range<int>(0,parallel_inner_cycles,inner_grain_size), [&](tbb::blocked_range<int> r) {
                
                int current_idx_inner = r.begin()*(index_step_inner<<1);

                for (int inner_rdx=r.begin(); inner_rdx<r.end(); inner_rdx++){

                    tbb::parallel_for(tbb::blocked_range<int>(0,index_step_inner,64),[&](tbb::blocked_range<int> r){
                    
        	            for (int idx=r.begin(); idx<r.end(); ++idx){
        	
                    	    int current_idx_loc = current_idx + idx + current_idx_inner + current_idx_middle1 + current_idx_middle2  ;
                            int current_idx_pair_loc = current_idx_pair_outer + idx + current_idx_inner + current_idx_middle1 + current_idx_middle2;

		                    int current_idx_outer_loc = current_idx_loc;
		                    int current_idx_inner_loc = current_idx_loc + index_step_inner;
		                    
		                    int current_idx_middle1_loc = current_idx_loc + index_step_middle1;
		                    int current_idx_middle1_inner_loc = current_idx_loc + index_step_middle1 + index_step_inner;
		                    
		                    int current_idx_middle2_loc = current_idx_loc + index_step_middle2;
		                    int current_idx_middle2_inner_loc = current_idx_loc + index_step_middle2 + index_step_inner;
		                    
		                    int current_idx_middle12_loc = current_idx_loc + index_step_middle1 + index_step_middle2;
		                    int current_idx_middle12_inner_loc = current_idx_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
		                    
                        	int current_idx_outer_pair_loc = current_idx_pair_loc;
		                    int current_idx_inner_pair_loc = current_idx_pair_loc + index_step_inner;
		                    
		                    int current_idx_middle1_pair_loc =current_idx_pair_loc + index_step_middle1;
		                    int current_idx_middle1_inner_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_inner;
		                    
		                    int current_idx_middle2_pair_loc = current_idx_pair_loc + index_step_middle2;
		                    int current_idx_middle2_inner_pair_loc = current_idx_pair_loc + index_step_middle2 + index_step_inner;
		                    
		                    int current_idx_middle12_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_middle2;
		                    int current_idx_middle12_inner_pair_loc = current_idx_pair_loc + index_step_middle1 + index_step_middle2 + index_step_inner;
			                
			                QGD_Complex16 results[16];
			                
                            double* element_outer = (double*)input.get_data() + 2 * current_idx_outer_loc;
                            double* element_inner = (double*)input.get_data() + 2 * current_idx_inner_loc;
                            
                            double* element_middle1 = (double*)input.get_data() + 2 * current_idx_middle1_loc;
                            double* element_middle1_inner = (double*)input.get_data() + 2 * current_idx_middle1_inner_loc;
                            
                            double* element_middle2 = (double*)input.get_data() + 2 * current_idx_middle2_loc;
                            double* element_middle2_inner = (double*)input.get_data() + 2 * current_idx_middle2_inner_loc;
                            
                            double* element_middle12 = (double*)input.get_data() + 2 * current_idx_middle12_loc;
                            double* element_middle12_inner = (double*)input.get_data() + 2 * current_idx_middle12_inner_loc;
                            
                            double* element_outer_pair = (double*)input.get_data() + 2 * current_idx_outer_pair_loc;
                            double* element_inner_pair = (double*)input.get_data() + 2 * current_idx_inner_pair_loc;
                                                        
                            double* element_middle1_pair = (double*)input.get_data() + 2 * current_idx_middle1_pair_loc;
                            double* element_middle1_inner_pair = (double*)input.get_data() + 2 * current_idx_middle1_inner_pair_loc;
                            
                            double* element_middle2_pair = (double*)input.get_data() + 2 * current_idx_middle2_pair_loc;
                            double* element_middle2_inner_pair = (double*)input.get_data() + 2 * current_idx_middle2_inner_pair_loc;
                            
                            double* element_middle12_pair = (double*)input.get_data() + 2 * current_idx_middle12_pair_loc;
                            double* element_middle12_inner_pair = (double*)input.get_data() + 2 * current_idx_middle12_inner_pair_loc;
                            
			                
                            __m256d outer_inner_vec = get_AVX_vector(element_outer, element_inner);

                           __m256d middle1_inner_vec = get_AVX_vector(element_middle1,element_middle1_inner);
                            
                            __m256d middle2_inner_vec = get_AVX_vector(element_middle2, element_middle2_inner);

                           __m256d middle12_inner_vec = get_AVX_vector(element_middle12,element_middle12_inner);

                            __m256d outer_inner_pair_vec =  get_AVX_vector(element_outer_pair, element_inner_pair);
                            
                            __m256d middle1_inner_pair_vec =  get_AVX_vector(element_middle1_pair, element_middle1_inner_pair);
                            
                            __m256d middle2_inner_pair_vec =  get_AVX_vector(element_middle2_pair, element_middle2_inner_pair);
                            
                            __m256d middle12_inner_pair_vec =  get_AVX_vector(element_middle12_pair, element_middle12_inner_pair);


			                for (int mult_idx=0; mult_idx<16; mult_idx++){
			                
			                    double* unitary_row_1 = (double*)unitary.get_data() + 32*mult_idx;
			                    double* unitary_row_2 = (double*)unitary.get_data() + 32*mult_idx + 4;
			                    double* unitary_row_3 = (double*)unitary.get_data() + 32*mult_idx + 8;
			                    double* unitary_row_4 = (double*)unitary.get_data() + 32*mult_idx + 12;
			                    double* unitary_row_5 = (double*)unitary.get_data() + 32*mult_idx + 16;
			                    double* unitary_row_6 = (double*)unitary.get_data() + 32*mult_idx + 20;
			                    double* unitary_row_7 = (double*)unitary.get_data() + 32*mult_idx + 24;
			                    double* unitary_row_8 = (double*)unitary.get_data() + 32*mult_idx + 28;
			                    
                                __m256d unitary_row_1_vec = _mm256_loadu_pd(unitary_row_1);
                                __m256d unitary_row_2_vec = _mm256_loadu_pd(unitary_row_2);
                                __m256d unitary_row_3_vec = _mm256_loadu_pd(unitary_row_3);
                                __m256d unitary_row_4_vec = _mm256_loadu_pd(unitary_row_4);
                                __m256d unitary_row_5_vec = _mm256_loadu_pd(unitary_row_5);
                                __m256d unitary_row_6_vec = _mm256_loadu_pd(unitary_row_6);
                                __m256d unitary_row_7_vec = _mm256_loadu_pd(unitary_row_7);
                                __m256d unitary_row_8_vec = _mm256_loadu_pd(unitary_row_8);
                                
                                __m256d result_upper_vec = complex_mult_AVX(outer_inner_vec,unitary_row_1_vec,neg);
                                
                                __m256d result_upper_middle1_vec = complex_mult_AVX(middle1_inner_vec,unitary_row_2_vec,neg);
                                
                                __m256d result_upper_middle2_vec = complex_mult_AVX(middle2_inner_vec,unitary_row_3_vec,neg);
                                
                                __m256d result_upper_middle12_vec = complex_mult_AVX(middle12_inner_vec,unitary_row_4_vec,neg);
                                
                                __m256d result_lower_vec = complex_mult_AVX(outer_inner_pair_vec,unitary_row_5_vec,neg);
                                
                                __m256d result_lower_middle1_vec = complex_mult_AVX(middle1_inner_pair_vec,unitary_row_6_vec,neg);
                                
                                __m256d result_lower_middle2_vec = complex_mult_AVX(middle2_inner_pair_vec,unitary_row_7_vec,neg);
                                
                                __m256d result_lower_middle12_vec = complex_mult_AVX(middle12_inner_pair_vec,unitary_row_8_vec,neg);
                                
                                
                                __m256d result1_vec = _mm256_hadd_pd(result_upper_vec,result_upper_middle1_vec);
                                __m256d result2_vec = _mm256_hadd_pd(result_upper_middle2_vec,result_upper_middle12_vec);
                                __m256d result3_vec = _mm256_hadd_pd(result_lower_vec,result_lower_middle1_vec);
                                __m256d result4_vec = _mm256_hadd_pd(result_lower_middle2_vec,result_lower_middle12_vec);
                                __m256d result5_vec = _mm256_hadd_pd(result1_vec,result2_vec);
                                __m256d result6_vec = _mm256_hadd_pd(result3_vec,result4_vec);
                                __m256d result_vec  = _mm256_hadd_pd(result5_vec,result6_vec);
                                result_vec          = _mm256_hadd_pd(result_vec,result_vec);
                                double* result = (double*)&result_vec;
                                results[mult_idx].real = result[0];
                                results[mult_idx].imag = result[2];
			                }
		                input[current_idx_outer_loc] = results[0];
		                input[current_idx_inner_loc] = results[1];
		                input[current_idx_middle1_loc]  = results[2];
		                input[current_idx_middle1_inner_loc] = results[3];
		                input[current_idx_middle2_loc]  = results[4];
		                input[current_idx_middle2_inner_loc] = results[5];
		                input[current_idx_middle12_loc]  = results[6];
		                input[current_idx_middle12_inner_loc] = results[7];
		                input[current_idx_outer_pair_loc] = results[8];
		                input[current_idx_inner_pair_loc] = results[9];
		                input[current_idx_middle1_pair_loc] = results[10];
		                input[current_idx_middle1_inner_pair_loc] = results[11];
		                input[current_idx_middle2_pair_loc] = results[12];
		                input[current_idx_middle2_inner_pair_loc] = results[13];
		                input[current_idx_middle12_pair_loc] = results[14];
		                input[current_idx_middle12_inner_pair_loc] = results[15];
            	    }
                   });
                
               
                current_idx_inner = current_idx_inner +(index_step_inner << 1);
                }
            });
                current_idx_middle1 = current_idx_middle1 +(index_step_middle1 << 1);
            }
            });
                current_idx_middle2 = current_idx_middle2 +(index_step_middle2 << 1);
            }
            });
        current_idx = current_idx + (index_step_outer << 1);
        current_idx_pair_outer = current_idx_pair_outer + (index_step_outer << 1);

    }
    });
}
