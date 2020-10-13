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
/*! \file qgd/Sub_Matrix_Decomposition_Cost_Function.h
    \brief Header file for the paralleized calculation of the cost function of the subdecomposition (supporting TBB and OpenMP).
*/

#include "qgd/common.h"


/**
@brief Call to calculate the cost function of a given matrix during the submatrix decomposition process.
@param matrix The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_new
@return Returns with the calculated cost function.
*/
double get_submatrix_cost_function(QGD_Complex16* matrix, int matrix_size);




/**
@brief Function operator class to extract the submatrices from a unitary.
*/
class functor_extract_submatrices {

protected:

    /// Array stroing the matrix
    QGD_Complex16* matrix;
    /// NUmber of rows in the matrix
    int matrix_size;
    /// array storing the submatrices
    QGD_Complex16** submatrices;

public:

/**
@brief Constructor of the class.
@param matrix_in The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_in
@param submatrices_in Preallocated array for the submatrices
@return Returns with the instance of the class.
*/
functor_extract_submatrices( QGD_Complex16* matrix_in, int matrix_size_in, QGD_Complex16** submatrices_in );

/**
@brief Operator to extract the sumbatrix indexed by submtx_idx
@param submtx_idx The index labeling the given submatrix to be extracted
*/
void operator()( int submtx_idx ) const;

};




/**
@brief Function operator class to calculate the partial cost function derived from the individual products of the submatrices.
*/
class functor_submtx_cost_fnc {

protected:

    /// The number of rows in the submatrices
    int submatrix_size;
    /// number of distinct submatix products
    int prod_num;
    /// array storing the submatrices
    QGD_Complex16** submatrices;
    //// array storing the partial cost functions
    double* prod_cost_functions;

public:

/**
@brief Constructor of the class.
@param submatrices_in The array of the submatrices.
@param submatrix_size The number rows in the submatrices.
@param prod_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param prod_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_submtx_cost_fnc( QGD_Complex16** submatrices_in, int submatrix_size, double* prod_cost_functions_in, int prod_num_in );

/**
@brief Operator to calculate the partial cost function labeled by product_idx
@param product_idx The index labeling the partial cost function to be calculated.
*/
void operator()( int product_idx ) const;

};








