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
/*! \file matrix_sparse.h
    \brief Header file of complex array storage array with automatic and thread safe reference counting.
*/


#ifndef matrix_sparse_H
#define matrix_sparse_H

#include "matrix.h"
#include <cmath>

/*! \file matrix.h
    \brief Header file matrix storing complex types.
*/

/**
@brief Class to store data of complex arrays and its properties. Compatible with the Picasso numpy interface.
*/
class Matrix_sparse {

public:
    int rows;
    
    int cols;

    int NNZ;

    QGD_Complex16* data;
    
    int* indices;
    
    int* indptr;
/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Matrix_sparse();

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
Matrix_sparse( QGD_Complex16* data_in, int rows_in, int cols_in, int NNZ_in, int* indices_in, int* indptr);

}; //matrix






#endif
