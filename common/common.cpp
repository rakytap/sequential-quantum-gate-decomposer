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

#include "common.h"
#include <tbb/scalable_allocator.h>


/**
@brief ?????
*/
double activation_function( double Phi, int limit ) {


    return Phi;

/*
    while (Phi < 0 ) {
        Phi = Phi + 2*M_PI;
    }

    while (Phi > 2*M_PI ) {
        Phi = Phi - 2*M_PI;
    }
 

    double ret = Phi;


    for (int idx=0; idx<limit; idx++) {


        ret = 0.5*(1.0-std::cos(ret))*M_PI;
 
        if ( Phi > M_PI ) {
            ret = 2*M_PI - ret;
        }

    }

    return ret;
*/
}


/**
@brief custom defined memory allocation function.  Memory allocated with aligned realloc *MUST* be freed using qgd_free.
@param element_num The number of elements in the array to be allocated.
@param size A size of one element (such as sizeof(double) )
@param alignment The number of bytes to which memory must be aligned. This value *must* be <= 255.
*/
void* qgd_calloc( int element_num, int size, int alignment ) {

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
void* qgd_realloc(void* aligned_ptr, int element_num, int size, int alignment ) {

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
@return Returns with an identity matrix.
*/
Matrix create_identity( int matrix_size ) {

    Matrix mtx = Matrix(matrix_size, matrix_size);
    memset(mtx.get_data(), 0, mtx.size()*sizeof(QGD_Complex16) );

    // setting the diagonal elelments to identity
    for(int idx = 0; idx < matrix_size; ++idx)
    {
        int element_index = idx*matrix_size + idx;
            mtx[element_index].real = 1.0;
    }

    return mtx;

}



/**
@brief Calculate the product of several square shaped complex matrices stored in a vector.
@param mtxs The vector of matrices.
@param matrix_size The number rows in the matrices
@return Returns with the calculated product matrix
*/
Matrix
reduce_zgemm( std::vector<Matrix>& mtxs ) {



    if (mtxs.size() == 0 ) {
        return Matrix();
    }

    // pointers to matrices to be used in the multiplications
    Matrix A;
    Matrix B;
    Matrix C;

    // the iteration number
    int iteration = 0;


    A = *mtxs.begin();

    // calculate the product of complex matrices
    for(std::vector<Matrix>::iterator it=++mtxs.begin(); it != mtxs.end(); ++it) {

        iteration++;
        B = *it;

        if ( iteration>1 ) {
            A = C;
        }

        // calculate the product of A and B
        C = dot(A, B);
/*
A.print_matrix();
B.print_matrix();
C.print_matrix();
*/
    }

    //C.print_matrix();
    return C;

}


/**
@brief Call to subtract a scalar from the diagonal of a complex matrix.
@param mtx The input-output matrix
@param scalar The complex scalar to be subtracked from the diagonal elements of the matrix
*/
void subtract_diag( Matrix& mtx, QGD_Complex16 scalar ) {

    int matrix_size = mtx.rows;

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
@param b A square shaped matrix.
*/
void mult( QGD_Complex16 a, Matrix& b ) {

    int element_num = b.size();

    for (int idx=0; idx<element_num; idx++) {
        QGD_Complex16 tmp = b[idx];
        b[idx].real = a.real*tmp.real - a.imag*tmp.imag;
        b[idx].imag = a.real*tmp.imag + a.imag*tmp.real;
    }

    return;

}



/**
@brief ???????????????????
@param a A complex numberr.
*/
double arg( const QGD_Complex16& a ) {


    double angle;

    if ( a.real > 0 && a.imag > 0 ) {
        angle = std::atan(a.imag/a.real);
        return angle;
    }
    else if ( a.real > 0 && a.imag <= 0 ) {
        angle = std::atan(a.imag/a.real);
        return angle;
    }
    else if ( a.real < 0 && a.imag > 0 ) {
        angle = std::atan(a.imag/a.real) + M_PI;
        return angle;
    }
    else if ( a.real < 0 && a.imag <= 0 ) {
        angle = std::atan(a.imag/a.real) - M_PI;
        return angle;
    }
    else if ( std::abs(a.real) < 1e-8 && a.imag > 0 ) {
        angle = M_PI/2;
        return angle;
    }
    else {
        angle = -M_PI/2;
        return angle;
    }



}



