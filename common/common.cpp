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
/*! \file common.cpp
    \brief Provides commonly used functions and wrappers to CBLAS functions.
*/

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space

#include "qgd/common.h"



/**
@brief custom defined memory allocation function.  Memory allocated with aligned realloc *MUST* be freed using qgd_free.
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_calloc( size_t element_num, size_t size, size_t alignment ) {

    void* ret = scalable_aligned_malloc( size*element_num, CACHELINE);
    memset(ret, 0, element_num*size);
    return ret;
}

/**
@brief custom defined memory reallocation function. Memory allocated with aligned realloc *MUST* be freed using qgd_free. The reallocation is done by either:
a) expanding or contracting the existing area pointed to by aligned_ptr, if possible. The contents of the area remain unchanged up to the lesser of the new and old sizes. If the area is expanded, the contents of the new part of the array is set to zero.
b) allocating a new memory block of size new_size bytes, copying memory area with size equal the lesser of the new and the old sizes, and freeing the old block.
@param aligned_ptr The aligned pointer created by qgd_calloc
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_realloc(void* aligned_ptr, size_t element_num, size_t size, size_t alignment ) {

    void* ret = scalable_aligned_realloc(aligned_ptr, size*element_num, CACHELINE);
    return ret;

}



/**
@brief custom defined memory release function.
*/
void qgd_free( void* ptr ) {

    // preventing double free corruption
    if (ptr != NULL ) {
        scalable_aligned_free(ptr);
    }
    ptr = NULL;
}


/**
@brief Calculates the n-th power of 2.
@param n An natural number
@return Returns with the n-th power of 2.
*/
int Power_of_2(int n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * Power_of_2(n-1);
}



/**
@brief Add an integer to a vector of integers if the integer is not already an element of the vector. The ascending order is kept during the process.
@param involved_qbits The vector of integer to be updated by the new integer. The result is returned via this vector.
@param qbit The integer to be added to the vector
*/
void add_unique_elelement( std::vector<int> &involved_qbits, int qbit ) {

    if ( involved_qbits.size() == 0 ) {
        involved_qbits.push_back( qbit );
    }

    for(std::vector<int>::iterator it = involved_qbits.begin(); it != involved_qbits.end(); ++it) {

        int current_val = *it;

        if (current_val == qbit) {
            return;
        }
        else if (current_val > qbit) {
            involved_qbits.insert( it, qbit );
            return;
        }

    }

    // add the qbit to the end if neither of the conditions were satisfied
    involved_qbits.push_back( qbit );

    return;

}


/**
@brief Call to create an identity matrix
@param matrix_size The number of rows in the resulted identity matrix
@return Returns with a pointer to the created identity matrix.
*/
QGD_Complex16* create_identity( int matrix_size ) {

    QGD_Complex16* matrix = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

    // setting the diagonal elelments to identity
    for(int idx = 0; idx < matrix_size; ++idx)
    {
        int element_index = idx*matrix_size + idx;
            matrix[element_index].real = 1.0;
    }

    return matrix;

}



/**
@brief Call to create an identity matrix
@param matrix The pointer to the memory array allocated for the identity matrix. The result is returned via this pointer.
@param matrix_size The number of rows in the resulted identity matrix
@return Returns with zero on success.
*/
int create_identity( QGD_Complex16* matrix, int matrix_size ) {

    memset( matrix, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // setting the diagonal elelments to identity
    for(int idx = 0; idx < matrix_size; ++idx)
    {
        int element_index = idx*matrix_size + idx;
            matrix[element_index].real = 1.0;
    }

    return 0;

}


/**
@brief Call to calculate the scalar product of two complex vectors using function cblas_zgemm3m or cblas_zgemm
@param A The first vector of the product
@param B The second vector of the product
@param vector_size The size of the vectors.
@return Returns the scalar product of the two vectors.
*/
QGD_Complex16 scalar_product( QGD_Complex16* A, QGD_Complex16* B, int vector_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // preallocate array for the result
    QGD_Complex16 C;

    // calculate the product of A and B
#ifdef CBLAS
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasConjTrans, 1, 1, vector_size, &alpha, A, vector_size, B, vector_size, &beta, &C, 1);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasConjTrans, 1, 1, vector_size, &alpha, A, vector_size, B, vector_size, &beta, &C, 1);
#endif

    return C;
}


/**
@brief Call to calculate the product of a square shaped complex matrix and a complex transpose of a second square shaped complex matrix using function cblas_zgemm3m or cblas_zgemm.
@param A The first matrix.
@param B The second matrix
@param C Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
*/
int zgemm3m_wrapper_adj( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // remove memory trash from the allocated memory of the results
    memset( C, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // calculate the product of A and B
#ifdef CBLAS
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasConjTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasConjTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#endif

    return 0;
}




/**
@brief Call to calculate the product of two square shaped complex matrices using function cblas_zgemm3m or cblas_zgemm
@param A The first matrix.
@param B The second matrix
@param matrix_size The number rows in the matrices
@return Returns with a pointer to the resulted matrix.
*/
QGD_Complex16* zgemm3m_wrapper( QGD_Complex16* A, QGD_Complex16* B, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // preallocate array for the result
    QGD_Complex16* C = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

    // remove memory trash from the allocated memory of the results
    memset( C, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // calculate the product of A and B
#ifdef CBLAS
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#endif

    return C;
}


/**
@brief Call to calculate the product of two square shaped complex matrices using function cblas_zgemm3m or cblas_zgemm
@param A The first matrix.
@param B The second matrix
@param C Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
@return Returns with zero on success.
*/
int zgemm3m_wrapper( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // remove memory trash from the allocated memory of the results
    memset( C, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // calculate the product of A and B
#ifdef CBLAS
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#endif

    return 0;
}



/**
@brief Calculate the product of several square shaped complex matrices stored in a vector.
@param mtxs The vector of matrices.
@param C Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
@return Returns with zero on success.
*/
int reduce_zgemm( std::vector<QGD_Complex16*> mtxs, QGD_Complex16* C, int matrix_size ) {


    if (mtxs.size() == 0 ) {
        return create_identity(C, matrix_size);
    }



    // pointers to matrices to be used in the multiplications
    QGD_Complex16* A = NULL;
    QGD_Complex16* B = NULL;

    // the iteration number
    int iteration = 0;


    QGD_Complex16* tmp = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size,sizeof(QGD_Complex16), 64);
    A = *mtxs.begin();

    // calculate the product of complex matrices
    for(std::vector<QGD_Complex16*>::iterator it=++mtxs.begin(); it != mtxs.end(); ++it) {

        iteration++;
        B = *it;

        if ( iteration>1 ) {
            A = tmp;
            memcpy(A, C, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        }



        // calculate the product of A and B
        zgemm3m_wrapper(A, B, C, matrix_size);
/*if (matrix_size == 4) {
printf("reduce_zgemm\n");
print_mtx( A, matrix_size, matrix_size);
print_mtx( B, matrix_size, matrix_size);
print_mtx( C, matrix_size, matrix_size);
}*/

    }

    qgd_free(tmp);
    tmp = NULL;

    return 0;

}


/**
@brief Call to subtract a scalar from the diagonal of a complex matrix.
@param mtx A pointer to the matrix. The resulted matrix is returned via this pointer.
@param matrix_size The number rows in the matrix
@param scalar The complex scalar to be subtracked from the diagonal elements of the matrix
*/
void subtract_diag( QGD_Complex16* & mtx,  int matrix_size, QGD_Complex16 scalar ) {

    for(int idx = 0; idx < matrix_size; idx++)   {
        int element_idx = idx*matrix_size+idx;
        mtx[element_idx].real = mtx[element_idx].real - scalar.real;
        mtx[element_idx].imag = mtx[element_idx].imag - scalar.imag;
    }

}




/**
@brief Call to calculate the product of two complex scalars
@param a The firs scalar
@param b The second scalar
@return Returns with the calculated product.
*/
QGD_Complex16 mult( QGD_Complex16 a, QGD_Complex16 b ) {

    QGD_Complex16 ret;
    ret.real = a.real*b.real - a.imag*b.imag;
    ret.imag = a.real*b.imag + a.imag*b.real;

    return ret;

}

/**
@brief calculate the product of a real scalar and a complex scalar
@param a The real scalar.
@param b The complex scalar.
@return Returns with the calculated product.
*/
QGD_Complex16 mult( double a, QGD_Complex16 b ) {

    QGD_Complex16 ret;
    ret.real = a*b.real;
    ret.imag = a*b.imag;

    return ret;

}



/**
@brief Multiply the elements of matrix b by a scalar a.
@param a A complex scalar.
@param b Pointer to the square shaped matrix.
@param matrix_size The number rows in the matrix b
*/
void mult( QGD_Complex16 a, QGD_Complex16* b, int matrix_size ) {

    for (int idx=0; idx<matrix_size*matrix_size; idx++) {
        QGD_Complex16 tmp = b[idx];
        b[idx].real = a.real*tmp.real - a.imag*tmp.imag;
        b[idx].imag = a.real*tmp.imag + a.imag*tmp.real;
    }

    return;

}



