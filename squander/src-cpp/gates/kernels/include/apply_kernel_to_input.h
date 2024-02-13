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
/*! \file apply_kerel_to_input.h
    \brief Collection of single- and multi-threaded implementation to apply a gate kernel on unitary inputs
*/


#ifndef apply_kerel_to_input_H
#define apply_kerel_to_input_H

#include "matrix.h"
#include "common.h"


/**
@brief Call to apply kernel to apply single qubit gate kernel on an input matrix
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
@param target_qbit The targer qubit on which the transformation should be applied
@param control_qbit The contron qubit (-1 if the is no control qubit)
@param matrix_size The size of the input
*/
void
apply_kernel_to_input(Matrix& u3_1qbit, Matrix& input, const bool& deriv, const int& target_qbit, const int& control_qbit, const int& matrix_size);

#endif
