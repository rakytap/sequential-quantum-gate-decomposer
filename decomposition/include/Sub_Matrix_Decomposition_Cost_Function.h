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

#pragma once
#include "qgd/common.h"


/**
@brief Call to calculate the cost function of a given matrix during the submatrix decomposition process.
@param matrix The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_new
@return Returns with the calculated cost function.
*/
double get_submatrix_cost_function(Matrix& matrix);




/**
@brief Function operator class to extract the submatrices from a unitary.
*/
class functor_extract_submatrices {

protected:

    /// The matrix from which submatrices would be extracted
    Matrix matrix;
    /// preallocated container storing the submatrices
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices;

public:

/**
@brief Constructor of the class.
@param matrix_in The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param submatrices_in Preallocated arrays for the submatrices
@return Returns with the instance of the class.
*/
functor_extract_submatrices( Matrix& matrix_in, std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices_in );

/**
@brief Operator to extract the sumbatrix indexed by submtx_idx
@param r A range of indices labeling the given submatrix to be extracted
*/
void operator()( tbb::blocked_range<size_t> r ) const;

};




/**
@brief Function operator class to calculate the partial cost function derived from the individual products of the submatrices.
*/
class functor_submtx_cost_fnc {

protected:

    /// number of distinct submatix products
    int prod_num;
    /// container storing the submatrices
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices;
    //// array storing the thread local partial cost functions
    tbb::combinable<double>* prod_cost_functions;

public:

/**
@brief Constructor of the class.
@param submatrices_in The array of the submatrices.
@param prod_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param prod_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_submtx_cost_fnc( std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices_in, tbb::combinable<double>* prod_cost_functions_in, size_t prod_num_in );

/**
@brief Operator to calculate the partial cost function labeled by product_idx
@param product_idx The index labeling the partial cost function to be calculated.
*/
void operator()( int product_idx ) const;

};








