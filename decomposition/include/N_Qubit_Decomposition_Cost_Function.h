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
/*! \file N_Qubit_Decomposition_Cost_Function.h
    \brief Header file for the paralleized calculation of the cost function of the final optimization problem (supporting TBB and OpenMP).
*/

#ifndef N_Qubit_Decomposition_Cost_Function_H
#define N_Qubit_Decomposition_Cost_Function_H

#include "common.h"
#include <tbb/combinable.h>
#include "logging.h"


/**
@brief Call co calculate the cost function during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@return Returns with the calculated cost function.
*/
double get_cost_function(Matrix matrix);



/**
@brief Function operator class to calculate the partial cost function of the final optimization process.
*/
class functor_cost_fnc : public logging {

protected:

    /// Array stroing the matrix
    Matrix matrix;
    /// Pointer to the data stored in the matrix
    QGD_Complex16* data;
    /// array storing the partial cost functions
    tbb::combinable<double>* partial_cost_functions;

public:

/**
@brief Constructor of the class.
@param matrix_in Arry containing the input matrix
@param partial_cost_functions_in Preallocated array storing the calculated partial cost functions.
@return Returns with the instance of the class.
*/
functor_cost_fnc( Matrix matrix_in,  tbb::combinable<double>* partial_cost_functions_in );

/**
@brief Operator to calculate the partial cost function derived from the row of the matrix labeled by row_idx
@param r A TBB range labeling the partial cost function to be calculated.
*/
void operator()( tbb::blocked_range<size_t> r ) const;

};


#endif





