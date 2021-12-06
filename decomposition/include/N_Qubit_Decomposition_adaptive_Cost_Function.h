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
/*! \file Sub_Matrix_Decomposition_Cost_Function.h
    \brief Header file for the paralleized calculation of the cost function of the subdecomposition (supporting TBB and OpenMP).
*/

#ifndef N_Qubit_Decomposition_adaptive_Cost_Function_H
#define N_Qubit_Decomposition_adaptive_Cost_Function_H


#include "common.h"
#include <tbb/combinable.h>



/**
@brief Call to calculate the cost function of a given matrix during the submatrix decomposition process.
@param matrix The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_new
@return Returns with the calculated cost function.
*/
double get_adaptive_cost_function(Matrix& matrix);




#endif
