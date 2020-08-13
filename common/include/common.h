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

#pragma once
#include <mkl_types.h>
#include <mkl.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>


int Power_of_2(int n);

// print the matrix
void print_mtx( MKL_Complex16* , int );

// print a CNOT
void print_CNOT( MKL_Complex16* , int );


// converts integer to string
std::string int_to_string( int input );

// converts double to string
std::string double_to_string( double input );

// @brief Structure type conatining gate numbers
struct gates_num {
  int u3;
  int cnot;
};

// @brief Add an integer to an integer vector if the integer is not already an element of the vector. The sorted order is kept during the process
void add_unique_elelement( std::vector<int>& involved_qbits, int qbit );


// @brief Calculate the product of complex matrices stored in a vector of matrices
MKL_Complex16* reduce_zgemm( std::vector<MKL_Complex16*>, int );


// @brief Calculate the product of complex matrices stored in a vector of matrices
MKL_Complex16* reduce_zgemm( std::vector<MKL_Complex16*>, int );





