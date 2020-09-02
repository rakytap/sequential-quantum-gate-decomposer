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
#ifdef __cplusplus
extern "C"
{
#endif
#include <gsl/gsl_blas_types.h>
#ifdef __cplusplus
}
#endif

#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <sstream>

// include MKL header if MKL package and intel compiler are present
#if CXX==icpc
#if HAVE_LIBMKL_INTEL_LP64
#include <mkl_service.h>

// headers of cblas of MKL and GSL are in conflict and GSL headers need to be inculed due to multimin package,
// thus function zgemm3m needs to be declared inline
#ifdef __cplusplus
extern "C"
{
#endif
typedef CBLAS_ORDER CBLAS_LAYOUT;


void cblas_zgemm3m(const  CBLAS_LAYOUT Layout, const  CBLAS_TRANSPOSE TransA,
                 const  CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *B, const MKL_INT ldb,
                 const void *beta, void *C, const MKL_INT ldc);

#ifdef __cplusplus
}
#endif

#endif
#endif



// @brief Structure type representing complex number in the QGD package (compatible with MKL, cblas and Fortran)
struct QGD_Complex16 {
  double real;
  double imag;
};


// define aligned memory allocation function if they are not present (in case of older gcc compilers)
#if HAVE_LIBMKL_INTEL_LP64
#ifndef ac_cv_func_aligned_alloc
/* Allocate aligned memory in a portable way.
 *
 * Memory allocated with aligned alloc *MUST* be freed using aligned_free.
 *
 * @param alignment The number of bytes to which memory must be aligned. This
 *  value *must* be <= 255.
 * @param bytes The number of bytes to allocate.
 * @param zero If true, the returned memory will be zeroed. If false, the
 *  contents of the returned memory are undefined.
 * @returns A pointer to `size` bytes of memory, aligned to an `alignment`-byte
 *  boundary.
 */
void *aligned_alloc(size_t alignment, size_t size, bool zero);

/* Free memory allocated with aligned_alloc */
void aligned_free(void* aligned_ptr);
#endif
#endif

// @brief custom defined memory allocation function. (Refers to corresponding MKL function if present, or use another aligned memory allocator otherwise)
void* qgd_calloc( size_t element_num, size_t size, size_t alignment );

// @brief custom defined memory release function. (Refers to corresponding MKL function if present, or use another aligned memory allocator otherwise)
void qgd_free( void* ptr );

// @brief Calculates the n-th power of 2.
int Power_of_2(int n);

// @brief Print a complex matrix on standard output
void print_mtx( QGD_Complex16* , int, int );

// @brief Print a CNOT matrix on standard output
void print_CNOT( QGD_Complex16* , int );


// @brief converts integer to string
std::string int_to_string( int input );

// @brief converts double to string
std::string double_to_string( double input );

// @brief Structure type conatining gate numbers
struct gates_num {
  int u3;
  int cnot;
};

// @brief Add an integer to an integer vector if the integer is not already an element of the vector. The sorted order is kept during the process
void add_unique_elelement( std::vector<int>& involved_qbits, int qbit );

// @brief Create an identity matrix
QGD_Complex16* create_identity( int );

// @brief Create an identity matrix
int create_identity( QGD_Complex16* matrix, int matrix_size );

// @brief Call to calculate the product of two matrices using cblas_zgemm3m
QGD_Complex16 scalar_product( QGD_Complex16* A, QGD_Complex16* B, int vector_size);

// @brief Call to calculate the product of two matrices using cblas_zgemm3m
QGD_Complex16* zgemm3m_wrapper_adj( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size);

// @brief Call to calculate the product of two matrices using cblas_zgemm3m
QGD_Complex16* zgemm3m_wrapper( QGD_Complex16* , QGD_Complex16*, int);

// @brief Call to calculate the product of two matrices using cblas_zgemm3m
int zgemm3m_wrapper( QGD_Complex16* , QGD_Complex16*, QGD_Complex16*, int);

// @brief Calculate the product of complex matrices stored in a vector of matrices
int reduce_zgemm( std::vector<QGD_Complex16*>, QGD_Complex16* C, int );


// @brief subtract a scalar from the diagonal of a matrix
void subtract_diag( QGD_Complex16* & , int, QGD_Complex16 ); 

// calculate the cost funtion from the submatrices of the given matrix 
double get_submatrix_cost_function(QGD_Complex16* matrix_new, int matrix_size, QGD_Complex16** submatrices, QGD_Complex16* submatrix_prod);

double get_submatrix_cost_function_2(QGD_Complex16* matrix, int matrix_size);

// calculate the cost funtion for the final optimalization
double get_cost_function(QGD_Complex16* matrix, int matrix_size);


// calculate the product of two scalars
QGD_Complex16 mult( QGD_Complex16 a, QGD_Complex16 b );

// calculate the product of two scalars
QGD_Complex16 mult( double a, QGD_Complex16 b );

// Multiply the elements of matrix "b" by a scalar "a".
void mult( QGD_Complex16 a, QGD_Complex16* b, int matrix_size );

