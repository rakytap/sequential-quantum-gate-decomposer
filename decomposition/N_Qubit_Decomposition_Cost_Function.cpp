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
/*! \file N_Qubit_Decomposition_Cost_Function.cpp
    \brief Methods to calculate the cost function of the final optimization problem (supporting parallel computations).
*/

#include "qgd/N_Qubit_Decomposition_Cost_Function.h"




/**
@brief Call co calculate the cost funtion during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param matrix_size The number rows in the matrix
@return Returns with the calculated cost function.
*/
double get_cost_function(QGD_Complex16* matrix, int matrix_size) {

    double* partial_cost_functions = (double*)qgd_calloc( matrix_size, sizeof(double), 64);
#ifdef TBB
    tbb::parallel_for(0, matrix_size, 1, functor_cost_fnc( matrix, matrix_size, partial_cost_functions, matrix_size ));
#else
    functor_cost_fnc tmp = functor_cost_fnc( matrix, matrix_size, partial_cost_functions, matrix_size );
    #pragma omp parallel for
    for (int idx=0; idx<matrix_size; idx++) {
        tmp(idx);
    }
#endif // TBB

    // sum up the partial cost funtions
    double cost_function = 0;
    for (int idx=0; idx<matrix_size; idx++) {
        cost_function = cost_function + partial_cost_functions[idx];
    }
    qgd_free(partial_cost_functions);
    partial_cost_functions = NULL;


    return cost_function;

}




/**
@brief Constructor of the class.
@param matrix_in Arry containing the input matrix
@param matrix_size_in The number rows in the matrix.
@param partial_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param partial_cost_fnc_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_cost_fnc::functor_cost_fnc( QGD_Complex16* matrix_in, int matrix_size_in, double* partial_cost_functions_in, int partial_cost_fnc_num_in ) {

    matrix = matrix_in;
    matrix_size = matrix_size_in;
    partial_cost_functions = partial_cost_functions_in;
    partial_cost_fnc_num = partial_cost_fnc_num_in;
}

/**
@brief Operator to calculate the partial cost function derived from the row of the matrix labeled by row_idx
@param row_idx The index labeling the partial cost function to be calculated.
*/
void functor_cost_fnc::operator()( int row_idx ) const {

    if ( row_idx > partial_cost_fnc_num ) {
        printf("Error: row idx should be less than the number of roes in the matrix\n");
        exit(-1);
    }

    // getting the corner element
    QGD_Complex16 corner_element = matrix[0];


    // Calculate the |x|^2 value of the elements of the matrix and summing them up to calculate the partial cost function
    double partial_cost_function = 0;
    int idx_offset = row_idx*matrix_size;
    int idx_max = idx_offset + row_idx;
    for ( int idx=idx_offset; idx<idx_max; idx++ ) {
         partial_cost_function = partial_cost_function + matrix[idx].real*matrix[idx].real + matrix[idx].imag*matrix[idx].imag;
    }

    int diag_element_idx = row_idx*matrix_size + row_idx;
    double diag_real = matrix[diag_element_idx].real - corner_element.real;
    double diag_imag = matrix[diag_element_idx].imag - corner_element.imag;
    partial_cost_function = partial_cost_function + diag_real*diag_real + diag_imag*diag_imag;


    idx_offset = idx_max + 1;
    idx_max = row_idx*matrix_size + matrix_size;
    for ( int idx=idx_offset; idx<idx_max; idx++ ) {
         partial_cost_function = partial_cost_function + matrix[idx].real*matrix[idx].real + matrix[idx].imag*matrix[idx].imag;
    }

    // storing the calculated partial cost function
    partial_cost_functions[row_idx] = partial_cost_function;
}








