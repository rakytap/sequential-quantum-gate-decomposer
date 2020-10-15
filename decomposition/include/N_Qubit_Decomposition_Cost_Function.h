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
/*! \file qgd/N_Qubit_Decomposition_Cost_Function.h
    \brief Header file for the paralleized calculation of the cost function of the final optimization problem (supporting TBB and OpenMP).
*/

#include "qgd/common.h"


/**
@brief Call co calculate the cost funtion during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param matrix_size The number rows in the matrix
@return Returns with the calculated cost function.
*/
double get_cost_function(QGD_Complex16* matrix, int matrix_size);



/**
@brief Function operator class to calculate the partial cost function of the final optimization process.
*/
class functor_cost_fnc {

protected:

    /// Array stroing the matrix
    QGD_Complex16* matrix;
    /// Number of rows in the matrix
    int matrix_size;
    /// array storing the partial cost functions
    double* partial_cost_functions;
    /// The number of partial cost functions
    int partial_cost_fnc_num;

public:

/**
@brief Constructor of the class.
@param matrix_in Arry containing the input matrix
@param matrix_size The number rows in the matrix.
@param partial_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param partial_cost_fnc_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_cost_fnc( QGD_Complex16* matrix_in, int matrix_size_in,  double* partial_cost_functions_in, int partial_cost_fnc_num_in );

/**
@brief Operator to calculate the partial cost function derived from the row of the matrix labeled by row_idx
@param row_idx The index labeling the partial cost function to be calculated.
*/
void operator()( int row_idx ) const;

};








