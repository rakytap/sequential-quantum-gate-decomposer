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
/*! \file common.h
    \brief Header file for commonly used functions and wrappers to CBLAS functions.
*/

#ifndef common_H
#define common_H

#define _USE_MATH_DEFINES
#include <cmath>
#define NOMINMAX
#include <algorithm>

#include <omp.h>
#include "QGDTypes.h"
#include "dot.h"
#include "matrix_sparse.h"
#include "matrix_real.h"

#include <string>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <sstream>

#ifdef __cplusplus
extern "C"
{
#endif

#if BLAS==0 // undefined blas
    /// Set the number of threads on runtime in MKL
    void omp_set_num_threads(int num_threads);
    /// get the number of threads in MKL
    int omp_get_max_threads();
#elif BLAS==1 // MKL
    /// Set the number of threads on runtime in MKL
    void MKL_Set_Num_Threads(int num_threads);
    /// get the number of threads in MKL
    int mkl_get_max_threads();
#elif BLAS==2 // OpenBLAS
    /// Set the number of threads on runtime in OpenBlas
    void openblas_set_num_threads(int num_threads);
    /// get the number of threads in OpenBlas
    int openblas_get_num_threads();
#endif

#ifdef __cplusplus
}
#endif



/**
@brief ?????
*/
double activation_function( double Phi, int limit );


/**
@brief custom defined memory allocation function.  Memory allocated with aligned realloc *MUST* be freed using qgd_free
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_calloc( int element_num, int size, int alignment );

/**
@brief custom defined memory reallocation function. Memory allocated with aligned realloc *MUST* be freed using qgd_free. The reallocation is done by either:
a) expanding or contracting the existing area pointed to by aligned_ptr, if possible. The contents of the area remain unchanged up to the lesser of the new and old sizes. If the area is expanded, the contents of the new part of the array is set to zero.
b) allocating a new memory block of size new_size bytes, copying memory area with size equal the lesser of the new and the old sizes, and freeing the old block.
@param aligned_ptr The aligned pointer created by qgd_calloc
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_realloc(void* aligned_ptr, int element_num, int size, int alignment );

/**
@brief custom defined memory release function.
*/
void qgd_free( void* ptr );

/**
@brief Calculates the n-th power of 2.
@param n An natural number
@return Returns with the n-th power of 2.
*/
int Power_of_2(int n);


/**
@brief Add an integer to a vector of integers if the integer is not already an element of the vector. The ascending order is kept during the process.
@param involved_qbits The vector of integer to be updated by the new integer. The result is returned via this vector.
@param qbit The integer to be added to the vector
*/
void add_unique_elelement( std::vector<int>& involved_qbits, int qbit );

/**
@brief Call to create an identity matrix
@param matrix_size The number of rows in the resulted identity matrix
@return Returns with an identity matrix.
*/
Matrix create_identity( int matrix_size );

/**
@brief Calculate the product of several square shaped complex matrices stored in a vector.
@param mtxs The vector of matrices.
@param matrix_size The number rows in the matrices
@return Returns with the calculated product matrix
*/
Matrix reduce_zgemm( std::vector<Matrix>& mtxs );

/**
@brief Call to subtract a scalar from the diagonal of a complex matrix.
@param mtx The input-output matrix
@param scalar The complex scalar to be subtracked from the diagonal elements of the matrix
*/
void subtract_diag( Matrix& mtx,  QGD_Complex16 scalar );


/**
@brief Call to calculate the product of two complex scalars
@param a The firs scalar
@param b The second scalar
@return Returns with the calculated product.
*/
QGD_Complex16 mult( QGD_Complex16& a, QGD_Complex16& b );

/**
@brief calculate the product of a real scalar and a complex scalar
@param a The real scalar.
@param b The complex scalar.
@return Returns with the calculated product.
*/
QGD_Complex16 mult( double a, QGD_Complex16 b );

/**
@brief Multiply the elements of matrix b by a scalar a.
@param a A complex scalar.
@param b A square shaped matrix.
*/
void mult( QGD_Complex16 a, Matrix& b );


/**
@brief Multiply the elements of a sparse matrix a and a dense vector b.
@param a A complex sparse matrix in CSR format.
@param b A complex dense vector.
*/
Matrix mult(Matrix_sparse a, Matrix& b);

/**
@brief Call to retrieve the phase of a complex number
@param a A complex numberr.
*/
double arg( const QGD_Complex16& a );






void conjugate_gradient(Matrix_real A, Matrix_real b, Matrix_real& x0, double tol);


#endif
