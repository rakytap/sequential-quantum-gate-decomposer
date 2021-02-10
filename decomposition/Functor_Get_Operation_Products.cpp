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
/*! \file Functor_Get_Operation_Products.cpp
    \brief Methods and classes for the parallelized calculation of the vector containing the products of operations (supporting TBB and OpenMP).
*/

#include "Decomposition_Base.h"
#include "Functor_Get_Operation_Products.h"
#include <tbb/parallel_for.h>


/**
@brief Calculate the list of gate operation matrices such that the i>0-th element in the result list is the product of the operations of all 0<=n<i operations from the input list and the 0th element in the result list is the identity.
@param parameters An array containing the parameters of the operations.
@param operations_it An iterator pointing to the first operation.
@param num_of_operations The number of operations involved in the calculations
@return Returns with a vector of the product matrices.
*/
std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> Decomposition_Base::get_operation_products(double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations) {


    // construct the vector of matrix representation of the gates
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> operation_mtxs(num_of_operations);

    // creating identity operation if no operations were involved in the calculations
    if (num_of_operations==0) {
        operation_mtxs.push_back( create_identity(matrix_size) );
        return operation_mtxs;
    }

    // calculate the matrices of the individual block operations
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_of_operations, 1), functor_get_operation_matrices( parameters, operations_it, &operation_mtxs, num_of_operations ));

    // calculate the operations products
    Matrix operation_product_mtx = Matrix(matrix_size, matrix_size);
    for (int idx=1; idx<num_of_operations; idx++) {
        operation_product_mtx = apply_operation(operation_mtxs[idx-1], operation_mtxs[idx] );
        operation_mtxs[idx] = operation_product_mtx;
        operations_it++;
    }

    return operation_mtxs;

}



/**
@brief Constructor of the class.
@param parameters_in An array containing the parameters of the operations.
@param operations_it_in An iterator pointing to the first operation.
@param operation_mtxs_in Pointer to a vector containing the matrix representation of the operations/operation blocks.
@param num_of_operations_in The number of operations in the vector
@return Returns with the instance of the class.
*/
functor_get_operation_matrices::functor_get_operation_matrices( double* parameters_in, std::vector<Operation*>::iterator operations_it_in, std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* operation_mtxs_in, int num_of_operations_in ) {

    parameters = parameters_in;
    operations_it = operations_it_in;
    operation_mtxs = operation_mtxs_in;
    num_of_operations = num_of_operations_in;

}

/**
@brief Operator to calculate the matrix representation of operation labeled by i.
@param r Range of indexes labeling the operation in the vector of operations iterated by operations_it.
*/
void functor_get_operation_matrices::operator()( const tbb::blocked_range<size_t> r ) const {

    for ( size_t i = r.begin(); i!=r.end(); i++) {

        // determine the range parameters
        double* parameters_loc = parameters;
        for (size_t idx=0; idx<i; idx++) {
            Operation* operation = *(operations_it+idx);
            parameters_loc = parameters_loc + operation->get_parameter_num();
        }

        // get the matrix representation of th eoperation
        Operation* operation = *(operations_it+i);

        if (operation->get_type() == CNOT_OPERATION ) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            (*operation_mtxs)[i] = cnot_operation->get_matrix();
        }
        else if (operation->get_type() == GENERAL_OPERATION ) {
            (*operation_mtxs)[i] = operation->get_matrix();
        }
            else if (operation->get_type() == U3_OPERATION ) {
            U3* u3_operation = static_cast<U3*>(operation);
            (*operation_mtxs)[i] = u3_operation->get_matrix(parameters_loc);
        }
        else if (operation->get_type() == BLOCK_OPERATION ) {
            Operation_block* block_operation = static_cast<Operation_block*>(operation);
            (*operation_mtxs)[i] = block_operation->get_matrix(parameters_loc);
        }

    }
}
