#include "qgd/matrix.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"


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
Matrix::Matrix( QGD_Complex16* data_in, size_t rows_in, size_t cols_in) : matrix_base<QGD_Complex16>(data_in, rows_in, cols_in) {

}


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
Matrix::Matrix( size_t rows_in, size_t cols_in) : matrix_base<QGD_Complex16>(rows_in, cols_in) {


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

  Matrix ret = Matrix(rows, cols);

  // logical variable indicating whether the matrix needs to be conjugated in CBLAS operations
  ret.conjugated = conjugated;
  // logical variable indicating whether the matrix needs to be transposed in CBLAS operations
  ret.transposed = transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  memcpy( ret.data, data, rows*cols*sizeof(QGD_Complex16));

  return ret;

}




