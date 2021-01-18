#ifndef matrix_H
#define matrix_H

#include "qgd/matrix_base.h"




/**
@brief Class to store data of complex arrays and its properties. Compatible with the Picasso numpy interface.
*/
class Matrix : public matrix_base<QGD_Complex16> {

    /// padding class object to cache line borders
    char padding[CACHELINE-48];

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Matrix();

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
Matrix( QGD_Complex16* data_in, size_t rows_in, size_t cols_in);


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
Matrix( size_t rows_in, size_t cols_in);


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
Matrix(const Matrix &in);


/**
@brief Call to create a copy of the matrix
@return Returns with the instance of the class.
*/
Matrix copy();


}; //matrix






#endif
