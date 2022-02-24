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

#include "Sub_Matrix_Decomposition_Cost_Function.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

//The stringstream input to store the output messages.
std::stringstream sstream;

//Integer value to set the verbosity level of the output messages.
int verbose_level;

/**
@brief Call to calculate the cost function of a given matrix during the submatrix decomposition process.
@param matrix The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_new
@return Returns with the calculated cost function.
*/
double get_submatrix_cost_function(Matrix& matrix) {

    // ********************************
    // Extract Submatrices
    // ********************************

    // number of submatrices
    size_t submatrices_num = 4;

    size_t submatrices_num_row = 2;


    // Extracting submatrices from the unitary
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> submatrices(submatrices_num);

    tbb::parallel_for( tbb::blocked_range<size_t>(0, submatrices_num, 1), functor_extract_submatrices( matrix, &submatrices ));


    

    // ********************************
    // Calculate the partial cost functions
    // ********************************


    size_t prod_num = submatrices_num*submatrices_num_row;
    tbb::combinable<double> priv_prod_cost_functions{[](){return 0;}};

    tbb::parallel_for(0, (int) prod_num, 1, functor_submtx_cost_fnc( &submatrices, &priv_prod_cost_functions, prod_num ));

    // ********************************
    // Calculate the total cost function
    // ********************************

    // calculate the final cost function
    double cost_function = 0;
    priv_prod_cost_functions.combine_each([&cost_function](double a) {
        cost_function = cost_function + a;
    });


    return cost_function;

}

/**
@brief Constructor of the class.
@param matrix_in The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size_in The number rows in the matrix matrix_in
@param submatrices_in Preallocated arrays for the submatrices
@return Returns with the instance of the class.
*/
functor_extract_submatrices::functor_extract_submatrices( Matrix& matrix_in, std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices_in ) {

    matrix = matrix_in;
    submatrices = submatrices_in;


}

/**
@brief Operator to extract the submatrix indexed by submtx_idx
@param r A range of indices labeling the given submatrix to be extracted
*/
void functor_extract_submatrices::operator()( tbb::blocked_range<size_t> r ) const {

    // number of submatrices
    size_t submatrices_num_row = 2;

    // number of columns in the input matrix
    size_t matrix_size = matrix.rows;

    // number of columns in the submatrices
    size_t submatrix_size = matrix_size/2;

    for ( size_t submtx_idx = r.begin(); submtx_idx != r.end(); submtx_idx++) {

        // The given Submatrix
        Matrix& submatrix = (*submatrices)[ submtx_idx ];


        // create submarix data by striding the original matrix
        size_t jdx = submtx_idx % submatrices_num_row;
        size_t idx = (size_t) (submtx_idx-jdx)/submatrices_num_row;
        size_t matrix_offset = idx*(matrix.stride*submatrix_size) + jdx*(submatrix_size);
        submatrix = Matrix(matrix.get_data()+matrix_offset, submatrix_size, submatrix_size, matrix.stride);


/*
        // preallocate memory for the submatrix
        submatrix = Matrix(submatrix_size, submatrix_size);


        // extract the submatrix
        size_t jdx = submtx_idx % submatrices_num_row;
        size_t idx = (size_t) (submtx_idx-jdx)/submatrices_num_row;

        // copy memory to submatrices
        for ( size_t row_idx=0; row_idx<submatrix_size; row_idx++ ) {

            size_t matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
            size_t submatrix_offset = row_idx*submatrix_size;
            memcpy(submatrix.get_data()+submatrix_offset, matrix.get_data()+matrix_offset, submatrix_size*sizeof(QGD_Complex16));

        }
*/
#ifdef DEBUG
        if (submatrix.isnan()) {

		sstream << "Submatrix contains NaN." << std::endl;
		verbose_level=1;
            	//print(sstream,verbose_level);	
	        
            
        }
#endif

    }

}




/**
@brief Constructor of the class.
@param submatrices_in The array of the submatrices.
@param prod_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param prod_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_submtx_cost_fnc::functor_submtx_cost_fnc( std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices_in, tbb::combinable<double>* prod_cost_functions_in, size_t prod_num_in ) {

    submatrices = submatrices_in;
    prod_cost_functions = prod_cost_functions_in;
    prod_num = prod_num_in;
}

/**
@brief Operator to calculate the partial cost function labeled by product_idx
@param product_idx The index labeling the partial cost function to be calculated.
*/
void functor_submtx_cost_fnc::operator()( int product_idx ) const {



    // number of submatrices
    size_t submatrices_num_row = 2;

    // select the given submatrices used to calculate the partial cost_function
    size_t jdx = product_idx % submatrices_num_row;
    size_t idx = (size_t) ( product_idx - jdx )/submatrices_num_row;

    // calculate the submatrix product
    Matrix tmp = (*submatrices)[jdx];
    tmp.transpose();
    tmp.conjugate();
    Matrix submatrix_prod = dot( (*submatrices)[idx], tmp );
//    Matrix submatrix_prod = dot( tmp, (*submatrices)[idx] );

#ifdef DEBUG
    if (submatrix_prod.isnan()) {

		sstream << "functor_submtx_cost_fnc::operator: Submatrix product contains NaN. Exiting" << std::endl;
		verbose_level=1;
            	//print(sstream,verbose_level);	
	        
        
    }
#endif


//std::cout << idx << " " << jdx << std::endl;
//submatrix_prod.print_matrix();

    // number of elements in the matrix of submatrix products
    size_t submatrix_size = submatrix_prod.rows;
    size_t element_num = submatrix_size*submatrix_size;

    // subtract the corner element from the diagonal
    QGD_Complex16 corner_element = submatrix_prod[0];
    //tbb::parallel_for( tbb::blocked_range<size_t>(0, submatrix_size, 1), [&](tbb::blocked_range<size_t> r){
//        for ( size_t row_idx=r.begin(); row_idx != r.end(); row_idx++) {
        for ( size_t row_idx=0; row_idx < submatrix_size; row_idx++) {
            size_t element_idx = row_idx*submatrix_prod.stride+row_idx;
            submatrix_prod[element_idx].real = submatrix_prod[element_idx].real  - corner_element.real;
            submatrix_prod[element_idx].imag = submatrix_prod[element_idx].imag  - corner_element.imag;
        }

    //});

    // Calculate the |x|^2 value of the elements of the submatrixproducts and add to the partial cost function
    //tbb::parallel_for( tbb::blocked_range<size_t>(0, element_num, 1), [&](tbb::blocked_range<size_t> r){
//        for (size_t idx = r.begin(); idx != r.end(); idx ++) {
        for (size_t idx = 0; idx < element_num; idx++) {

            // store the calculated value for the given submatrix product element
            double &prod_cost_function_priv = prod_cost_functions->local();
            prod_cost_function_priv = prod_cost_function_priv + submatrix_prod[idx].real*submatrix_prod[idx].real + submatrix_prod[idx].imag*submatrix_prod[idx].imag;
        }
    //});

    // checking NaN
    if (std::isnan(prod_cost_functions->local())) {

		sstream << "cost function NaN on cost function product "<< product_idx << std::endl;
		verbose_level=1;
            	//print(sstream,verbose_level);	
	        
        
    }


}








