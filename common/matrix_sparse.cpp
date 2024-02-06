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
#include "matrix_sparse.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"
#include <math.h>    


Matrix_sparse::Matrix_sparse(){
    rows = 0;

    cols = 0;

    NNZ = 0;

    data = NULL;

    indices = NULL;

    indptr = NULL;
}

Matrix_sparse::Matrix_sparse(QGD_Complex16* data_in, int rows_in, int cols_in, int NNZ_in, int* indices_in, int* indptr_in){

    rows = rows_in;

    cols = cols_in;

    NNZ = NNZ_in;

    data = data_in;

    indices = indices_in;

    indptr = indptr_in;

}

