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
//#include <mkl_types.h>
//#include <mkl.h>
#include <gsl/gsl_blas_types.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

// @brief Structure type representing complex number
struct MKL_Complex16 {
  double real;
  double imag;
};


void* mkl_malloc( size_t size, size_t alignment );
void* mkl_calloc( size_t element_num, size_t size, size_t alignment );
void mkl_free( void* ptr );

int Power_of_2(int n);

// print the matrix
void print_mtx( MKL_Complex16* , int, int );

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

// @brief Create an identity matrix
MKL_Complex16* create_identity( int );

// @brief Create an identity matrix
int create_identity( MKL_Complex16* matrix, int matrix_size );

// @brief Call to calculate the product of two matrices using cblas_zgemm3m
MKL_Complex16* zgemm3m_wrapper( MKL_Complex16* , MKL_Complex16*, int);

// @brief Call to calculate the product of two matrices using cblas_zgemm3m
int zgemm3m_wrapper( MKL_Complex16* , MKL_Complex16*, MKL_Complex16*, int);

// @brief Calculate the product of complex matrices stored in a vector of matrices
int reduce_zgemm( std::vector<MKL_Complex16*>, MKL_Complex16* C, int );


// @brief subtract a scalar from the diagonal of a matrix
void subtract_diag( MKL_Complex16* & , int, MKL_Complex16 ); 

// calculate the cost funtion from the submatrices of the given matrix 
double get_submatrix_cost_function(MKL_Complex16* matrix_new, int matrix_size);

double get_submatrix_cost_function_2(MKL_Complex16* matrix, int matrix_size);

// calculate the cost funtion for the final optimalization
double get_cost_function(MKL_Complex16* matrix, int matrix_size);


// calculate the product of two scalars
MKL_Complex16 mult( MKL_Complex16 a, MKL_Complex16 b );

// calculate the product of two scalars
MKL_Complex16 mult( double a, MKL_Complex16 b );

// Multiply the elements of matrix "b" by a scalar "a".
void mult( MKL_Complex16 a, MKL_Complex16* b, int matrix_size );

