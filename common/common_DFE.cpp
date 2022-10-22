/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file common.cpp
    \brief Provides commonly used functions and wrappers to CBLAS functions.
*/

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space

#include "common_DFE.h"

/**
@brief ????????????
@return ??????????
*/
void uploadMatrix2DFE( Matrix& input ) {

    // first convert the input to float32
    matrix_base<Complex8> input32( input.rows, input.cols ); // number of columns needed to be made twice due to complex -> real tranformation

    size_t element_num = input.size();
    QGD_Complex16* input_data = input.get_data();
    Complex8* input32_data = input32.get_data();
    for ( size_t idx=0; idx<element_num; idx++) {
        input32_data[idx].real = (float)(input_data[idx].real);
        input32_data[idx].imag = (float)(input_data[idx].imag);
    }
    
std::cout << "size in bytes of uploading: " << element_num*sizeof(float) << std::endl;    

    // load the data to LMEM
    load2LMEM( input32_data, input.rows, input.cols );

}



/**
@brief ????????????
@return ??????????
*/
/*
void DownloadMatrixFromDFE( std::vector<Matrix>& output_vec ) {

    // first convert the input to float32
    size_t element_num = output_vec[0].size();

    std::vector<matrix_base<Complex8>> output32_vec;
    for( int idx=0; idx<4; idx++) {
        output32_vec.push_back(matrix_base<Complex8>( output_vec[0].rows, output_vec[0].cols ));
    }
    
    Complex8* output32_data[4];
    for( int idx=0; idx<4; idx++) {
        output32_data[idx] = output32_vec[idx].get_data();
    }    
    

    // load the data to LMEM
    downloadFromLMEM( output32_data, output_vec[0].rows );

    for( int idx=0; idx<4; idx++) {
	QGD_Complex16* output_data = output_vec[idx].get_data();    
	Complex8* output32_data_loc = output32_data[idx]; 	
        for ( size_t jdx=0; jdx<element_num; jdx++) {
            output_data[jdx].real = (double)(output32_data_loc[jdx].real);
            output_data[jdx].imag = (double)(output32_data_loc[jdx].imag);
        }
    }



}
*/
