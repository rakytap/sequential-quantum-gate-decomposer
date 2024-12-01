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
QGD_Complex16 mult( QGD_Complex16& a, QGD_Complex16& b ) {

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
@param b A complex matrix.
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
@brief Multiply the elements of a sparse matrix a and a dense vector b.
@param a A complex sparse matrix in CSR format.
@param b A complex dense vector.
*/
Matrix mult(Matrix_sparse a, Matrix& b){

    if ( b.cols != 1 ) {
        std::string error("mult: The input argument b should be a single-column vector.");
        throw error;
    }
    
    int n_rows = a.rows;
    Matrix ret = Matrix(n_rows,1);

    tbb::parallel_for( tbb::blocked_range<int>(0, n_rows, 32), [&](tbb::blocked_range<int> r) { 

        for (int idx=r.begin(); idx<r.end(); ++idx){

            ret[idx].imag = 0.0;
            ret[idx].real = 0.0;
            int nz_start  = a.indptr[idx];
            int nz_end    = a.indptr[idx+1];
 
                for (int nz_idx=nz_start; nz_idx<nz_end; nz_idx++){

                    int jdx              = a.indices[nz_idx];
                    QGD_Complex16& state = b[jdx];
                    QGD_Complex16 result = mult(a.data[nz_idx], state);

                    ret[idx].real += result.real;
                    ret[idx].imag += result.imag;

                }
            }
    });

    return ret;
}



/**
@brief Call to retrieve the phase of a complex number
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








void conjugate_gradient(Matrix_real A, Matrix_real b, Matrix_real& x0, double tol){

    int samples = b.cols;
    Matrix_real d(1,samples);
    Matrix_real g(1,samples);
    double sk =0.0;

    for (int rdx=0; rdx<samples; rdx++){
        d[rdx] = b[rdx];

        for(int cdx=0; cdx<samples; cdx++){
            d[rdx] = d[rdx] - A[rdx*samples+cdx]*x0[cdx];
        }

        g[rdx] = -1.*d[rdx];
        sk = sk + d[rdx]*d[rdx];
    }

    int iter=0.0;
    while (std::sqrt(sk/b.cols) > tol && iter<1000){
    
    double dAd=0.0;
    Matrix_real Ad(1,b.cols);
    for (int rdx=0; rdx<samples; rdx++){

        Ad[rdx] = 0.;

        for(int cdx=0; cdx<samples; cdx++){
            Ad[rdx] = Ad[rdx] + A[rdx*samples+cdx]*d[cdx];
        }
        
        dAd = dAd + d[rdx]*Ad[rdx];
    }

    double mu_k = sk / dAd;
    double sk_new = 0.;

    for(int idx=0; idx<samples; idx++){

        x0[idx] = x0[idx] + mu_k*d[idx];
        g[idx] = g[idx] + mu_k*Ad[idx];
        sk_new = sk_new + g[idx]*g[idx];

    }

    for (int idx=0; idx<samples;idx++){
        d[idx] = (sk_new/sk)*d[idx] - g[idx];
    }

    sk = sk_new;

    iter++;
    }
    return;
    
}



