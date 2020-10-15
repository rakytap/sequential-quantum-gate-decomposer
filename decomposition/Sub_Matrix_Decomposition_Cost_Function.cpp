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
/*! \file Sub_Matrix_Decomposition_Cost_Function.cpp
    \brief Methods to calculate the cost function of the sub-disantenglement problem with TBB parallelization.
*/

#include "qgd/Sub_Matrix_Decomposition_Cost_Function.h"




/**
@brief Call to calculate the cost function of a given matrix during the submatrix decomposition process.
@param matrix The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_new
@param submatrices Pointer to the preallocated array of the submatrices of the matrix matrix_new.
@param submatrix_prod Preallocated array for the product of two submatrices.
@return Returns with the calculated cost function.
*/
double get_submatrix_cost_function(QGD_Complex16* matrix, int matrix_size) {

    // ********************************
    // Extract Submatrices
    // ********************************

    // number of submatrices
    int submatrices_num = 4;

    int submatrices_num_row = 2;

    // number ofcolumns in the submatrices
    int submatrix_size = matrix_size/2;


    // array to store the sumbatrices
    QGD_Complex16** submatrices = (QGD_Complex16**)qgd_calloc( submatrices_num, sizeof(QGD_Complex16*), 64);

#ifdef TBB
    tbb::parallel_for(0, submatrices_num, 1, functor_extract_submatrices( matrix, matrix_size, submatrices ));
#else
    functor_extract_submatrices tmp = functor_extract_submatrices( matrix, matrix_size, submatrices );
    #pragma omp parallel for
    for (int idx=0; idx<submatrices_num; idx++) {
        tmp(idx);
    }
#endif // TBB

    // ********************************
    // Calculate the partial cost functions
    // ********************************


    int prod_num = submatrices_num*submatrices_num_row;
    double* prod_cost_functions = (double*)qgd_calloc( prod_num, sizeof(double), 64);
#ifdef TBB
    tbb::parallel_for(0, prod_num, 1, functor_submtx_cost_fnc( submatrices, submatrix_size, prod_cost_functions, prod_num ));
#else
    functor_submtx_cost_fnc tmp2 = functor_submtx_cost_fnc( submatrices, submatrix_size, prod_cost_functions, prod_num );
    #pragma omp parallel for
    for (int idx=0; idx<prod_num; idx++) {
        tmp2(idx);
    }
#endif //TBB

    // ********************************
    // Calculate the total cost function
    // ********************************

    // calculate the final cost function
    double cost_function = 0;
    for (int idx=0; idx<prod_num; idx++) {
        cost_function = cost_function + prod_cost_functions[idx];
    }
    qgd_free( prod_cost_functions );
    prod_cost_functions = NULL;


    // release submatrices
    for (int idx=0; idx<submatrices_num; idx++) {
        if (submatrices[idx] != NULL ) {
            qgd_free( submatrices[idx] );
            submatrices[idx] = NULL;
        }
    }
    qgd_free( submatrices );
    submatrices = NULL;


    return cost_function;

}

/**
@brief Constructor of the class.
@param matrix_in The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_in
@param submatrices_in Preallocated array for the submatrices
@return Returns with the instance of the class.
*/
functor_extract_submatrices::functor_extract_submatrices( QGD_Complex16* matrix_in, int matrix_size_in, QGD_Complex16** submatrices_in ) {

    matrix = matrix_in;
    matrix_size = matrix_size_in;
    submatrices = submatrices_in;


}

/**
@brief Operator to extract the sumbatrix indexed by submtx_idx
@param submtx_idx The index labeling the given submatrix to be extracted
*/
void functor_extract_submatrices::operator()( int submtx_idx ) const {

    // number of submatrices
    int submatrices_num_row = 2;

    // number ofcolumns in the submatrices
    int submatrix_size = matrix_size/2;

    // number of elements in the matrix of submatrix products
    int element_num = submatrix_size*submatrix_size;

    // preallocate memory for the submatrix
    submatrices[ submtx_idx ] = (QGD_Complex16*)qgd_calloc(element_num,sizeof(QGD_Complex16), 64);


    // extract the submatrix
    int jdx = submtx_idx % submatrices_num_row;
    int idx = (int) (submtx_idx-jdx)/submatrices_num_row;

    // copy memory to submatrices
    for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {

        int matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
        int submatrix_offset = row_idx*submatrix_size;
        memcpy(submatrices[submtx_idx]+submatrix_offset, matrix+matrix_offset, submatrix_size*sizeof(QGD_Complex16));

    }


}




/**
@brief Constructor of the class.
@param submatrices_in The array of the submatrices.
@param submatrix_size The number rows in the submatrices.
@param prod_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param prod_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_submtx_cost_fnc::functor_submtx_cost_fnc( QGD_Complex16** submatrices_in, int submatrix_size_in, double* prod_cost_functions_in, int prod_num_in ) {

    submatrices = submatrices_in;
    submatrix_size = submatrix_size_in;
    prod_cost_functions = prod_cost_functions_in;
    prod_num = prod_num_in;
}

/**
@brief Operator to calculate the partial cost function labeled by product_idx
@param product_idx The index labeling the partial cost function to be calculated.
*/
void functor_submtx_cost_fnc::operator()( int product_idx ) const {

    // number of elements in the matrix of submatrix products
    int element_num = submatrix_size*submatrix_size;

    // preallocate memeory for submatrix product
    QGD_Complex16* submatrix_prod = (QGD_Complex16*)qgd_calloc(element_num,sizeof(QGD_Complex16), 64);

    // number of submatrices
    int submatrices_num = 4;
    int submatrices_num_row = 2;

    // select the given submatrices used to calculate the partial cost_function
    int jdx = product_idx % submatrices_num_row;
    int idx = (int) ( product_idx - jdx )/submatrices_num_row;

    // calculate the submatrix product
    zgemm3m_wrapper_adj( submatrices[idx], submatrices[jdx], submatrix_prod, submatrix_size);


    // subtract the corner element from the diagonal
    QGD_Complex16 corner_element = submatrix_prod[0];
    for ( int row_idx=0; row_idx<submatrix_size; row_idx++) {
        int element_idx = row_idx*submatrix_size+row_idx;
        submatrix_prod[element_idx].real = submatrix_prod[element_idx].real  - corner_element.real;
        submatrix_prod[element_idx].imag = submatrix_prod[element_idx].imag  - corner_element.imag;
    }

    // Calculate the |x|^2 value of the elements of the submatrixproducts
    for ( int idx=0; idx<element_num; idx++ ) {
        submatrix_prod[idx].real = submatrix_prod[idx].real*submatrix_prod[idx].real + submatrix_prod[idx].imag*submatrix_prod[idx].imag;
        // for performance reason we leave the imaginary part intact (we dont neet it anymore)
        //submatrix_prods[idx].imag = 0;
    }



    // summing up elements and calculate the final cost function
    double cost_function_partial = 0;
    for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {

        // calculate the sum for each row
        for (int col_idx=0; col_idx<submatrix_size; col_idx++) {
            int element_idx = row_idx*submatrix_size + col_idx;
            cost_function_partial = cost_function_partial + submatrix_prod[element_idx].real;
        }
    }


    // checking NaN
    if (std::isnan(cost_function_partial)) {
        printf("cost function NaN on thread %d: exiting\n", product_idx);
        exit(-1);
    }


    // store the calculated value for the given submatrix product
    prod_cost_functions[product_idx] = cost_function_partial;



    // release submatrix product
    if (submatrix_prod != NULL ) {
        qgd_free( submatrix_prod );
        submatrix_prod = NULL;
    }


}








