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
/*! \file Sub_Matrix_Decomposition_Cost_Function.cpp
    \brief Methods to calculate the cost function of the sub-disantenglement problem with TBB parallelization.
*/

#include "Sub_Matrix_Decomposition_Cost_Function.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>



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
    int submatrices_num = 4;

    int submatrices_num_row = 2;


    // Extracting submatrices from the unitary
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> submatrices(submatrices_num);

    tbb::parallel_for( tbb::blocked_range<int>(0, submatrices_num, 1), functor_extract_submatrices( matrix, &submatrices ));


    

    // ********************************
    // Calculate the partial cost functions
    // ********************************


    int prod_num = submatrices_num*submatrices_num_row;
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
void functor_extract_submatrices::operator()( tbb::blocked_range<int> r ) const {
  
    // number of submatrices
    int submatrices_num_row = 2;

    // number of columns in the input matrix
    int matrix_size = matrix.rows;

    // number of columns in the submatrices
    int submatrix_size = matrix_size/2;

    for ( int submtx_idx = r.begin(); submtx_idx != r.end(); submtx_idx++) {

        // The given Submatrix
        Matrix& submatrix = (*submatrices)[ submtx_idx ];


        // create submarix data by striding the original matrix
        int jdx = submtx_idx % submatrices_num_row;
        int idx = (int) (submtx_idx-jdx)/submatrices_num_row;
        int matrix_offset = idx*(matrix.stride*submatrix_size) + jdx*(submatrix_size);
        submatrix = Matrix(matrix.get_data()+matrix_offset, submatrix_size, submatrix_size, matrix.stride);


/*
        // preallocate memory for the submatrix
        submatrix = Matrix(submatrix_size, submatrix_size);


        // extract the submatrix
        int jdx = submtx_idx % submatrices_num_row;
        int idx = (int) (submtx_idx-jdx)/submatrices_num_row;

        // copy memory to submatrices
        for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {

            int matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
            int submatrix_offset = row_idx*submatrix_size;
            memcpy(submatrix.get_data()+submatrix_offset, matrix.get_data()+matrix_offset, submatrix_size*sizeof(QGD_Complex16));

        }
*/
#ifdef DEBUG
        if (submatrix.isnan()) {
          std::stringstream sstream;
	  sstream << "Submatrix contains NaN." << std::endl;
          print(sstream, 1);		            
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
functor_submtx_cost_fnc::functor_submtx_cost_fnc( std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* submatrices_in, tbb::combinable<double>* prod_cost_functions_in, int prod_num_in ) {

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
    int submatrices_num_row = 2;

    // select the given submatrices used to calculate the partial cost_function
    int jdx = product_idx % submatrices_num_row;
    int idx = (int) ( product_idx - jdx )/submatrices_num_row;

    // calculate the submatrix product
    Matrix tmp = (*submatrices)[jdx];
    tmp.transpose();
    tmp.conjugate();
    Matrix submatrix_prod = dot( (*submatrices)[idx], tmp );
//    Matrix submatrix_prod = dot( tmp, (*submatrices)[idx] );

#ifdef DEBUG
    if (submatrix_prod.isnan()) {
       std::stringstream sstream;
       sstream << "functor_submtx_cost_fnc::operator: Submatrix product contains NaN. Exiting" << std::endl;
       print(sstream, 1);		               
    }
#endif


//std::cout << idx << " " << jdx << std::endl;
//submatrix_prod.print_matrix();

    // number of elements in the matrix of submatrix products
    int submatrix_size = submatrix_prod.rows;
    int element_num = submatrix_size*submatrix_size;

    // subtract the corner element from the diagonal
    QGD_Complex16 corner_element = submatrix_prod[0];
    //tbb::parallel_for( tbb::blocked_range<int>(0, submatrix_size, 1), [&](tbb::blocked_range<int> r){
//        for ( int row_idx=r.begin(); row_idx != r.end(); row_idx++) {
        for ( int row_idx=0; row_idx < submatrix_size; row_idx++) {
            int element_idx = row_idx*submatrix_prod.stride+row_idx;
            submatrix_prod[element_idx].real = submatrix_prod[element_idx].real  - corner_element.real;
            submatrix_prod[element_idx].imag = submatrix_prod[element_idx].imag  - corner_element.imag;
        }

    //});

    // Calculate the |x|^2 value of the elements of the submatrixproducts and add to the partial cost function
    //tbb::parallel_for( tbb::blocked_range<int>(0, element_num, 1), [&](tbb::blocked_range<int> r){
//        for (int idx = r.begin(); idx != r.end(); idx ++) {
        for (int idx = 0; idx < element_num; idx++) {

            // store the calculated value for the given submatrix product element
            double &prod_cost_function_priv = prod_cost_functions->local();
            prod_cost_function_priv = prod_cost_function_priv + submatrix_prod[idx].real*submatrix_prod[idx].real + submatrix_prod[idx].imag*submatrix_prod[idx].imag;
        }
    //});


    // checking NaN
    if (std::isnan(prod_cost_functions->local())) {
       std::stringstream sstream;
       sstream << "cost function NaN on cost function product "<< product_idx << std::endl;
       print(sstream, 1);	
    }


}








