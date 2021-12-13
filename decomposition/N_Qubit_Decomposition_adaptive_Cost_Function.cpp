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

#include "N_Qubit_Decomposition_adaptive_Cost_Function.h"
#include "Sub_Matrix_Decomposition_Cost_Function.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

//Power_of_2(int n)

/**
@brief Call to calculate the cost function of a given matrix during the submatrix decomposition process.
@param matrix The square shaped complex matrix from which the cost function is calculated during the submatrix decomposition process.
@param matrix_size The number rows in the matrix matrix_new
@return Returns with the calculated cost function.
*/
double get_adaptive_cost_function(Matrix& matrix) {


    // deterimine the number of qubits
    const size_t &dim = matrix.rows;
    const size_t dim_over_2 = dim/2;
    int qbit_num = 0;
    size_t dim_loc = 1;
    while ( dim_loc < dim ) {
        dim_loc *= 2;
        qbit_num++;
    }

    QGD_Complex16* mtx_data = matrix.get_data();

    // thread local storage for cost functions
    tbb::combinable<double> priv_addend{[](){return 0.0;}};

    // determine the partial cost functions
//    for (int qbit=0; qbit<qbit_num; qbit++) {
    tbb::parallel_for(0, qbit_num, 1, [&](int qbit) {

        // reorder the matrix elements
        Matrix mtx_new( dim, dim );
        QGD_Complex16* mtx_new_data = mtx_new.get_data();

        size_t dim_loc = (size_t)Power_of_2(qbit);
        size_t dim_loc2 = (size_t)dim_loc*2;


        for ( size_t row_idx=0; row_idx<dim_over_2; row_idx++ ) {

            int row_tmp  = row_idx % dim_loc;
            int row_tmp2 = (row_idx-row_tmp)/dim_loc;

            for ( size_t col_idx=0; col_idx<dim_over_2; col_idx+=dim_loc ) {

                memcpy( mtx_new_data + row_idx*dim + col_idx,                             mtx_data + (row_tmp2*dim_loc2+row_tmp)*dim + (col_idx*2),                   dim_loc*sizeof(QGD_Complex16) );
                memcpy( mtx_new_data + row_idx*dim + col_idx + dim_over_2,                mtx_data + (row_tmp2*dim_loc2+row_tmp)*dim + (col_idx*2 + dim_loc),         dim_loc*sizeof(QGD_Complex16) );
                memcpy( mtx_new_data + (row_idx + dim_over_2)*dim + col_idx,              mtx_data + (row_tmp2*dim_loc2+row_tmp+dim_loc)*dim + (col_idx*2),           dim_loc*sizeof(QGD_Complex16) );
                memcpy( mtx_new_data + (row_idx + dim_over_2)*dim + col_idx + dim_over_2, mtx_data + (row_tmp2*dim_loc2+row_tmp+dim_loc)*dim + (col_idx*2 + dim_loc), dim_loc*sizeof(QGD_Complex16) );


            }
        }


        double &partial_cost_function = priv_addend.local();
        partial_cost_function += get_submatrix_cost_function(mtx_new);
//std::cout << "qbit: " << qbit << " cost function: " << get_submatrix_cost_function(mtx_new) << std::endl;
    });
  


    //cost_function = get_submatrix_cost_function(matrix);
    double cost_function = 0.0;
    priv_addend.combine_each([&](double &a) {
        cost_function = cost_function + a;
    });


//double cost_function = get_submatrix_cost_function(matrix);
//std::cout << "qbit: " << qbit_num-1 << " cost function: " << cost_function << std::endl;

    return cost_function;

}





