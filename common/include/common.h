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
/*! \file qgd/common.h
    \brief Header file for commonly used functions and wrappers to CBLAS functions.
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

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>
#include <omp.h>
#include "qgd/QGDTypes.h"
#include "qgd/matrix.h"

#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <sstream>

#ifdef CBLAS
#ifdef __cplusplus
extern "C"
{
#endif

/// Definition of the zgemm3m function from CBLAS. (Since headers of GNU Scientific Library and other CBLAS libraries are not compatible, we must define function zgemm3m on our own.)
void cblas_zgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);

#if CBLAS==1 // MKL
    /// Set the number of threads on runtime in MKL
    void MKL_Set_Num_Threads(int num_threads);
    /// get the number of threads in MKL
    int mkl_get_max_threads();
#elif CBLAS==2 // OpenBLAS
    /// Set the number of threads on runtime in OpenBlas
    void openblas_set_num_threads(int num_threads);
    /// get the number of threads in OpenBlas
    int openblas_get_num_threads();
#endif

#ifdef __cplusplus
}
#endif
#endif




/**
@brief custom defined memory allocation function.  Memory allocated with aligned realloc *MUST* be freed using qgd_free
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_calloc( size_t element_num, size_t size, size_t alignment );

/**
@brief custom defined memory reallocation function. Memory allocated with aligned realloc *MUST* be freed using qgd_free. The reallocation is done by either:
a) expanding or contracting the existing area pointed to by aligned_ptr, if possible. The contents of the area remain unchanged up to the lesser of the new and old sizes. If the area is expanded, the contents of the new part of the array is set to zero.
b) allocating a new memory block of size new_size bytes, copying memory area with size equal the lesser of the new and the old sizes, and freeing the old block.
@param aligned_ptr The aligned pointer created by qgd_calloc
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_realloc(void* aligned_ptr, size_t element_num, size_t size, size_t alignment );

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
@brief Call to create an identity matrix --- OBSOLETE
@param matrix The pointer to the memory array allocated for the identity matrix. The result is returned via this pointer.
@param matrix_size The number of rows in the resulted identity matrix
@return Returns with zero on success.
*/
int create_identity( QGD_Complex16* matrix, int matrix_size );

/**
@brief Call to calculate the scalar product of two complex vectors using function cblas_zgemm3m or cblas_zgemm
@param A The first vector of the product
@param B The second vector of the product
@param vector_size The size of the vectors.
@return Returns the scalar product of the two vectors.
*/
QGD_Complex16 scalar_product( QGD_Complex16* A, QGD_Complex16* B, int vector_size);

/**
@brief Call to calculate the product of a square shaped complex matrix and a complex transpose of a second square shaped complex matrix using function cblas_zgemm3m or cblas_zgemm.
@param A The first matrix.
@param B The second matrix
@param C Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
*/
int zgemm3m_wrapper_adj( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size);


/**
@brief Call to calculate the product of two square shaped complex matrices using function cblas_zgemm3m or cblas_zgemm
@param A The first matrix.
@param B The second matrix
@return Returns with the resulted matrix.
*/
Matrix zgemm3m_wrapper( Matrix& A , Matrix& B );


/**
@brief Call to calculate the product of two square shaped complex matrices using function cblas_zgemm3m or cblas_zgemm
@param A The first matrix.
@param B The second matrix
@param matrix_size The number rows in the matrices
@return Returns with a pointer to the resulted matrix.
*/
QGD_Complex16* zgemm3m_wrapper( QGD_Complex16* A , QGD_Complex16* B, int matrix_size);

/**
@brief Call to calculate the product of two square shaped complex matrices using function cblas_zgemm3m or cblas_zgemm
@param A The first matrix.
@param B The second matrix
@param C Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
@return Returns with zero on success.
*/
int zgemm3m_wrapper( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size);

/**
@brief Calculate the product of several square shaped complex matrices stored in a vector.
@param mtxs The vector of matrices.
@param C Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
@return Returns with zero on success.
*/
int reduce_zgemm( std::vector<QGD_Complex16*> mtxs, QGD_Complex16* C, int matrix_size );

/**
@brief Calculate the product of several square shaped complex matrices stored in a vector.
@param mtxs The vector of matrices.
@param matrix_size The number rows in the matrices
@return Returns with the calculated product matrix
*/
Matrix reduce_zgemm( std::vector<Matrix>& mtxs );

/**
@brief Call to subtract a scalar from the diagonal of a complex matrix.
@param mtx A pointer to the matrix. The resulted matrix is returned via this pointer.
@param matrix_size The number rows in the matrix
@param scalar The complex scalar to be subtracked from the diagonal elements of the matrix
*/
void subtract_diag( QGD_Complex16* & mtx,  int matrix_size, QGD_Complex16 scalar );


/**
@brief Call co calculate the cost funtion during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param matrix_size The number rows in the matrix
@return Returns with the calculated cost function.
*/
double get_cost_function(QGD_Complex16* matrix, int matrix_size);

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
@param b Pointer to the square shaped matrix.
@param matrix_size The number rows in the matrix b
*/
void mult( QGD_Complex16 a, QGD_Complex16* b, int matrix_size );

