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
/*! \file common.h
    \brief Header file for commonly used functions and wrappers to CBLAS functions.
*/

#ifndef common_H
#define common_H


#include <omp.h>
#include "QGDTypes.h"
#include "dot.h"


#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
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
QGD_Complex16 mult( QGD_Complex16 a, QGD_Complex16 b );

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
@brief ???????????????????
@param a A complex numberr.
*/
double arg( const QGD_Complex16& a );




#endif
