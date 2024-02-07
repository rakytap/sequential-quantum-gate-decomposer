/*
Created on Fri Jun 26 14:13:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Sub_Matrix_Decomposition_Cost_Function.h
    \brief Header file for the paralleized calculation of the cost function of the subdecomposition (supporting TBB and OpenMP).
*/

#ifndef Sub_Matrix_Decomposition_Cost_Function_H
#define Sub_Matrix_Decomposition_Cost_Function_H


#include "common.h"
#include <tbb/combinable.h>
#include "logging.h"


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
class functor_extract_submatrices : public logging {

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
void operator()( tbb::blocked_range<int> r ) const;

};




/**
@brief Function operator class to calculate the partial cost function derived from the individual products of the submatrices.
*/
class functor_submtx_cost_fnc : public logging {

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
functor_submtx_cost_fnc( std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices_in, tbb::combinable<double>* prod_cost_functions_in, int prod_num_in );

/**
@brief Operator to calculate the partial cost function labeled by product_idx
@param product_idx The index labeling the partial cost function to be calculated.
*/
void operator()( int product_idx ) const;

};








#endif
