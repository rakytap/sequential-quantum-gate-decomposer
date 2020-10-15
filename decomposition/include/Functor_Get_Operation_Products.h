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
/*! \file qgd/Functor_Get_Operation_Products.h
    \brief Header file for the parallelized calculation of the vector containing the products of operations (supporting TBB and OpenMP).
*/


#ifndef FUNCTOR_GET_OPERATION_PRODUCTS_INCLUDED
#define FUNCTOR_GET_OPERATION_PRODUCTS_INCLUDED

#include "qgd/common.h"
#include "qgd/Operation.h"
#include "qgd/Operation_block.h"
#include "qgd/CNOT.h"
#include "qgd/U3.h"

/**
@brief Function operator class to calculate the matrix representation of multiple operators in parallel
*/
class functor_get_operation_matrices {

protected:
    /// An array containing the parameters of the operations.
    double* parameters;
    /// An iterator pointing to the first operation.
    std::vector<Operation*>::iterator operations_it;
    /// vector containing the matrix representation of the operations/operation blocks.
    std::vector<QGD_Complex16*> operation_mtxs;
    ///  The number of operations in the vector
    int num_of_operations;

public:

/**
@brief Constructor of the class.
@param parameters_in An array containing the parameters of the operations.
@param operations_it_in An iterator pointing to the first operation.
@param operation_mtxs_in vector containing the matrix representation of the operations/operation blocks.
@param num_of_operations_in The number of operations in the vector
@return Returns with the instance of the class.
*/
functor_get_operation_matrices( double* parameters_in, std::vector<Operation*>::iterator operations_it_in, std::vector<QGD_Complex16*> operation_mtxs_in, int num_of_operations_in );

/**
@brief Operator to calculate th ematrix representation of operation labeled by i.
@param i The index labeling the operation in the vector of operations iterated by operations_it.
*/
void operator()( int i ) const;

};


#endif // FUNCTOR_GET_OPERATION_PRODUCTS_INCLUDED
