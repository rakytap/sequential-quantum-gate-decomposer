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

#ifdef TBB
#include <tbb/tbb.h>
#else
#include <omp.h>
#endif // TBB

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
#ifdef MKL
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



/// @brief Structure type representing complex numbers in the QGD package (compatible with cblas libraries)
struct QGD_Complex16 {
  /// the real part of a complex number
  double real;
  /// the imaginary part of a complex number
  double imag;
};

/// @brief Structure type conatining numbers of gates.
struct gates_num {
  /// The number of U3 gates
  int u3;
  /// The number of CNOT gates
  int cnot;
  /// The number of general gates
  int general;
};


/**
@brief Allocate aligned memory in a portable way. Memory allocated with aligned alloc *MUST* be freed using aligned_free. The allocated memory is initialized to zero.
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
@param size The number of bytes to allocate.
@param zero If true, the returned memory will be zeroed. If false, the contents of the returned memory are undefined.
@returns A pointer to `size` bytes of memory, aligned to an `alignment`-byte boundary.
*/
void *aligned_alloc(size_t alignment, size_t size, bool zero);

/**
@brief Reallocate aligned memory in a portable way. Memory allocated with aligned realloc *MUST* be freed using aligned_free. The reallocation is done by either:
a) expanding or contracting the existing area pointed to by aligned_ptr, if possible. The contents of the area remain unchanged up to the lesser of the new and old sizes. If the area is expanded, the contents of the new part of the array is set to zero.
b) allocating a new memory block of size new_size bytes, copying memory area with size equal the lesser of the new and the old sizes, and freeing the old block.
@param aligned_ptr The aligned pointer created by aigned_alloc
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
@param old_size The number of bytes to allocated in pointer aligned_ptr.
@param size The number of bytes to allocate.
@param zero If true, the returned memory will be zeroed. If false, the contents of the returned memory are undefined.
@returns A pointer to `size` bytes of memory, aligned to an `alignment`-byte boundary.
*/
void *aligned_realloc(void* aligned_ptr, size_t alignment, size_t old_size, size_t size, bool zero);

/**
@brief Free memory allocated with aligned_alloc
@param aligned_ptr The aligned pointer created by aigned_alloc
*/
void aligned_free(void* aligned_ptr);


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
@param element_num_old The number of elements in the old array to be reallocated.
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_realloc(void* aligned_ptr, size_t element_num_old, size_t element_num, size_t size, size_t alignment );

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
@brief Print a complex matrix on standard output
@param matrix A pointer pointing to the matrix to be printed
@param rows The number of rows in the matrix.
@param cols The number of columns in the matrix.
*/
void print_mtx( QGD_Complex16* matrix, int rows, int cols );

/**
@brief Print a CNOT matrix on standard output
@param matrix A pointer pointing to the matrix to be printed
@param size The number of rows in the matrix.
*/
void print_CNOT( QGD_Complex16* matrix, int size );

/**
@brief Add an integer to a vector of integers if the integer is not already an element of the vector. The ascending order is kept during the process.
@param involved_qbits The vector of integer to be updated by the new integer. The result is returned via this vector.
@param qbit The integer to be added to the vector
*/
void add_unique_elelement( std::vector<int>& involved_qbits, int qbit );

/**
@brief Call to create an identity matrix
@param matrix_size The number of rows in the resulted identity matrix
@return Returns with a pointer to the created identity matrix.
*/
QGD_Complex16* create_identity( int matrix_size );

/**
@brief Call to create an identity matrix
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

