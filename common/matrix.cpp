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
/*! \file matrix.cpp
    \brief Implementation of complex array storage array with automatic and thread safe reference counting.
*/

#include "matrix.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"
#include <math.h>



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Matrix::Matrix() : matrix_base<QGD_Complex16>() {

}

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
Matrix::Matrix( QGD_Complex16* data_in, int rows_in, int cols_in) : matrix_base<QGD_Complex16>(data_in, rows_in, cols_in) {

}


/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@param stride_in The column stride of the matrix array (The array elements in one row are a_0, a_1, ... a_{cols-1}, 0, 0, 0, 0. The number of zeros is stride-cols)
@return Returns with the instance of the class.
*/
Matrix::Matrix( QGD_Complex16* data_in, int rows_in, int cols_in, int stride_in) : matrix_base<QGD_Complex16>(data_in, rows_in, cols_in, stride_in) {

}


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
Matrix::Matrix( int rows_in, int cols_in) : matrix_base<QGD_Complex16>(rows_in, cols_in) {


}


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@param stride_in The column stride of the matrix array (The array elements in one row are a_0, a_1, ... a_{cols-1}, 0, 0, 0, 0. The number of zeros is stride-cols)
@return Returns with the instance of the class.
*/
Matrix::Matrix( int rows_in, int cols_in, int stride_in) : matrix_base<QGD_Complex16>(rows_in, cols_in, stride_in) {

}



/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
Matrix::Matrix(const Matrix &in) : matrix_base<QGD_Complex16>(in)  {

}



/**
@brief Call to create a copy of the matrix
@return Returns with the instance of the class.
*/
Matrix
Matrix::copy() {

  Matrix ret = Matrix(rows, cols, stride);

  // logical variable indicating whether the matrix needs to be conjugated in CBLAS operations
  ret.conjugated = conjugated;
  // logical variable indicating whether the matrix needs to be transposed in CBLAS operations
  ret.transposed = transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  memcpy( ret.data, data, rows*cols*sizeof(QGD_Complex16));

  return ret;

}


/**
@brief Call to check the array for NaN entries.
@return Returns with true if the array has at least one NaN entry.
*/
bool
Matrix::isnan() {

    for (int idx=0; idx < rows*cols; idx++) {
        if ( std::isnan(data[idx].real) || std::isnan(data[idx].imag) ) {
            return true;
        }
    }

    return false;


}




/**
@brief Call to prints the stored matrix on the standard output
*/
void 
Matrix::print_matrix() const {

    std::cout << std::endl << "The stored matrix:" << std::endl;
    
    for ( int row_idx=0; row_idx < rows; row_idx++ ) {
        for ( int col_idx=0; col_idx < cols; col_idx++ ) {
            int element_idx = row_idx*stride + col_idx;
	     std::cout << " (" << data[element_idx].real << ", " << data[element_idx].imag << "*i)";	              
        }

        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl << std::endl;

}




