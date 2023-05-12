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
/*! \file apply_kerel_to_input_AVX.cpp
    \brief ????????????????
*/


#ifndef apply_kerel_to_input_AVX_H
#define apply_kerel_to_input_AVX_H

#include "matrix.h"
#include "common.h"

/**
@brief AVX kernel to apply single qubit gate kernel on an input matrix -- optimal for small number of qubits
@param ????????
@param ?????????
*/
void apply_kernel_to_input_AVX_small(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);



/**
@brief AVX kernel to apply single qubit gate kernel on an input matrix
@param ????????
@param ?????????
*/
void apply_kernel_to_input_AVX(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);


/**
@brief parallel AVX kernel to apply single qubit gate kernel on an input matrix
@param ????????
@param ?????????
*/
void apply_kernel_to_input_AVX_parallel(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);


#endif
