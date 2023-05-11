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
#include "matrix_real.h"
#include <tbb/combinable.h>
#include "logging.h"


/**
@brief Call co calculate the cost function during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@return Returns with the calculated cost function.
*/
double get_cost_function(Matrix matrix, int trace_offset=0);


/**
@brief Call co calculate the cost function of the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0) and the first correction (index 1).
*/
Matrix_real get_cost_function_with_correction(Matrix matrix, int qbit_num, int trace_offset=0);


/**
@brief Call co calculate the cost function of the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0), the first correction (index 1) and the second correction (index 2).
*/
Matrix_real get_cost_function_with_correction2(Matrix matrix, int qbit_num, int trace_offset=0);

double get_cost_function_sum_of_squares(Matrix& matrix);

/**
@brief Call to calculate the real and imaginary parts of the trace
@param matrix The square shaped complex matrix from which the trace is calculated.
@return Returns with the calculated trace
*/
QGD_Complex16 get_trace(Matrix& matrix);


/**
@brief Call co calculate the cost function of the optimization process according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns the cost function
*/
double get_hilbert_schmidt_test(Matrix& matrix);


/**
@brief Call co calculate the Hilbert Schmidt testof the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0) and the first correction (index 1).
*/
Matrix get_trace_with_correction(Matrix& matrix, int qbit_num);


/**
@brief Call co calculate the Hilbert Schmidt testof the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0), the first correction (index 1) and the second correction (index 3).
*/
Matrix get_trace_with_correction2(Matrix& matrix, int qbit_num);

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
void operator()( tbb::blocked_range<int> r ) const;

};


#endif





