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
/*! \file dot.cpp
    \brief Provides multithreaded binding for CBLAS function zgemm to calculate matrix products
*/


#include "common.h"
#include "dot.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"
#include <tbb/scalable_allocator.h>


 // number of rows in matrix A, for which serialized multiplication is applied instead of parallel one
#define SERIAL_CUTOFF 16

//tbb::spin_mutex my_mutex;

/**
@brief Call to calculate the product of two complex matrices by calling method zgemm3m from the CBLAS library.
@param A The first matrix.
@param B The second matrix
@return Returns with the resulted matrix.
*/
Matrix
dot( Matrix &A, Matrix &B ) {

#if BLAS==0 // undefined BLAS
    int NumThreads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    int NumThreads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    int NumThreads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif


    // check the Matrix shapes in DEBUG mode
    assert( check_matrices( A, B ) );


#if BLAS==1 // MKL does not support option CblasConjNoTrans so the conjugation of the matrices are done in prior.
    if ( B.is_conjugated() && !B.is_transposed() ) {
        Matrix tmp = Matrix( B.rows, B.cols );
        vzConj( B.cols*B.rows, B.get_data(), tmp.get_data() );
        B = tmp;
    }

    if ( A.is_conjugated() && !A.is_transposed() ) {
        Matrix tmp = Matrix( A.rows, A.cols );
        vzConj( A.cols*A.rows, A.get_data(), tmp.get_data() );
         A = tmp;
    }
#endif


    // Preparing the output Matrix
    Matrix C;
    if ( A.is_transposed() ){
        if ( B.is_transposed() ) {
            C = Matrix(A.cols, B.rows);
        }
        else {
            C = Matrix(A.cols, B.cols);
        }
    }
    else {
        if ( B.is_transposed() ) {
            C = Matrix(A.rows, B.rows);
        }
        else {
            C = Matrix(A.rows, B.cols);
        }
    }

    // Calculating the matrix product
    if ( A.rows <= SERIAL_CUTOFF && B.cols <= SERIAL_CUTOFF ) {
        // creating the serial task object
        zgemm3m_Task_serial calc_task = zgemm3m_Task_serial( A, B, C );
        calc_task.zgemm3m_chunk();
    }
    else {
        // creating the task object
        zgemm3m_Task& calc_task = *new(tbb::task::allocate_root()) zgemm3m_Task( A, B, C );

        // starting parallel calculations
        tbb::task::spawn_root_and_wait(calc_task);
    }

#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

    return C;


}



/**
@brief Call to check the shape of the matrices for method dot. (Called in DEBUG mode)
@param A The first matrix in the product of type matrix.
@param B The second matrix in the product of type matrix
@return Returns with true if the test passed, false otherwise.
*/
bool
check_matrices( Matrix &A, Matrix &B ) {


    if (!A.is_transposed() & !B.is_transposed())  {
        if ( A.cols != B.rows ) {
            std::cout << "pic::dot:: Cols of matrix A does not match rows of matrix B!" << std::endl;
            return false;
        }
    }
    else if ( A.is_transposed() & !B.is_transposed() )  {
        if ( A.rows != B.rows ) {
            std::cout << "pic::dot:: Cols of matrix A.transpose does not match rows of matrix B!" << std::endl;
            return false;
        }
    }
    else if ( A.is_transposed() & B.is_transposed() )  {
        if ( A.rows != B.cols ) {
            std::cout << "pic::dot:: Cols of matrix A.transpose does not match rows of matrix B.transpose!" << std::endl;
            return false;
        }
    }
    else if ( !A.is_transposed() & B.is_transposed() )  {
        if ( A.cols != B.cols ) {
            std::cout << "pic::dot:: Cols of matrix A does not match rows of matrix B.transpose!" << std::endl;
            return false;
        }
    }


    // check the pointer of the matrices
    if ( A.get_data() == NULL ) {
        std::cout << "pic::dot:: No preallocated data in matrix A!" << std::endl;
        return false;
    }
    if ( B.get_data() == NULL ) {
        std::cout << "pic::dot:: No preallocated data in matrix B!" << std::endl;
        return false;
    }

    return true;

}


/**
@brief Call to get the transpose properties of the input matrix for CBLAS calculations
@param A The matrix of type matrix.
@param transpose The returned vale of CBLAS_TRANSPOSE.
*/
void
get_cblas_transpose( Matrix &A, CBLAS_TRANSPOSE &transpose ) {

    if ( A.is_conjugated() & A.is_transposed() ) {
        transpose = CblasConjTrans;
    }
    else if ( A.is_conjugated() & !A.is_transposed() ) {
        transpose = CblasConjNoTrans; // not present in MKL
    }
    else if ( !A.is_conjugated() & A.is_transposed() ) {
        transpose = CblasTrans;
    }
    else {
        transpose = CblasNoTrans;
    }

}






/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Cont_Task_rowsA::Cont_Task_rowsA() {
}


/**
@brief Overriden execute function of class tbb::task
@return Returns with a pointer to a tbb::task instance or with a null pointer.
*/
tbb::task*
Cont_Task_rowsA::execute() {
        return nullptr;
}




/**
@brief Constructor of the class. (In this case the row/col limits are extracted from matrices A,B,C).
@param A_in The object representing matrix A.
@param B_in The object representing matrix B.
@param C_in The object representing matrix C.
*/
zgemm3m_Task_serial::zgemm3m_Task_serial( Matrix &A_in, Matrix &B_in, Matrix &C_in ) {

    A = A_in;
    B = B_in;
    C = C_in;


    order = CblasRowMajor;

    rows.Arows_start = 0;
    rows.Arows_end = A.rows;
    rows.Arows = A.rows;
    rows.Brows_start = 0;
    rows.Brows_end = B.rows;
    rows.Brows = B.rows;
    rows.Crows_start = 0;
    rows.Crows_end = C.rows;
    rows.Crows = C.rows;

    cols.Acols_start = 0;
    cols.Acols_end = A.cols;
    cols.Acols = A.cols;
    cols.Bcols_start = 0;
    cols.Bcols_end = B.cols;
    cols.Bcols = B.cols;
    cols.Ccols_start = 0;
    cols.Ccols_end = C.cols;
    cols.Ccols = C.cols;

}


/**
@brief Constructor of the class.
@param A_in The object representing matrix A.
@param B_in The object representing matrix B.
@param C_in The object representing matrix C.
@param rows_in Structure containing row limits for the partitioning of the matrix product calculations.
@param cols_in Structure containing column limits for the partitioning of the matrix product calculations.
*/
zgemm3m_Task_serial::zgemm3m_Task_serial( Matrix &A_in, Matrix &B_in, Matrix &C_in, row_indices& rows_in, col_indices& cols_in ) {

    A = A_in;
    B = B_in;
    C = C_in;

    rows = rows_in;
    cols = cols_in;

}



/**
@brief Call to calculate the product of matrix chunks defined by attributes rows, cols. The result is stored in the corresponding chunk of matrix C.
*/
void
zgemm3m_Task_serial::zgemm3m_chunk() {

    // setting CBLAS transpose operations
    CBLAS_TRANSPOSE Atranspose, Btranspose;
    get_cblas_transpose( A, Atranspose );
    get_cblas_transpose( B, Btranspose );

    QGD_Complex16* A_zgemm_data = A.get_data()+rows.Arows_start*A.cols+cols.Acols_start;
    QGD_Complex16* B_zgemm_data = B.get_data()+rows.Brows_start*B.cols+cols.Bcols_start;
    QGD_Complex16* C_zgemm_data = C.get_data()+rows.Crows_start*C.cols+cols.Ccols_start;


    // zgemm parameters
    int m,n,k,lda,ldb,ldc;

    if ( A.is_transposed() ) {
        m = cols.Acols;
        k = rows.Arows;
        lda = A.cols;
    }
    else {
        m = rows.Arows;
        k = cols.Acols;
        lda = A.cols;
    }



    if (B.is_transposed()) {
        n = rows.Brows;
        ldb = B.cols;
    }
    else {
        n = cols.Bcols;
        ldb = B.cols;
    }

    ldc = C.cols;

    // parameters alpha and beta for the cblas_zgemm3m function (the input matrices are not scaled)
    QGD_Complex16 alpha;
    alpha.real = 1;
    alpha.imag = 0;
    QGD_Complex16 beta;
    beta.real = 0;
    beta.imag = 0;


#ifdef CBLAS
    cblas_zgemm3m(CblasRowMajor, Atranspose, Btranspose, m, n, k, (double*)&alpha, (double*)A_zgemm_data, lda, (double*)B_zgemm_data, ldb, (double*)&beta, (double*)C_zgemm_data, ldc);
#else
    cblas_zgemm(CblasRowMajor, Atranspose, Btranspose, m, n, k, (double*)&alpha, (double*)A_zgemm_data, lda, (double*)B_zgemm_data, ldb, (double*)&beta, (double*)C_zgemm_data, ldc);
#endif

}


/**
@brief This function is called when a task is spawned. It divides the work into chunks following the strategy of divide-and-conquer until the problem size meets a predefined treshold.
@return Returns with a pointer to a tbb::task instance or with a null pointer.
*/
tbb::task*
zgemm3m_Task::execute() {



    if ( !A.is_transposed() && rows.Arows > A.rows/8 && rows.Arows > SERIAL_CUTOFF ) {
        // *********** divide rows of A into sub-tasks*********

        size_t rows_start = rows.Arows_start;
        size_t rows_end = rows.Arows_end;
        size_t rows_mid = (rows_end+rows_start)/2;

        row_indices rows_child2 = rows;
        rows_child2.Arows_start = rows_mid;
        rows_child2.Arows = rows_end-rows_mid;
        rows_child2.Crows_start = rows_mid;
        rows_child2.Crows = rows_end-rows_mid;

        Cont_Task_rowsA& cont = *new(allocate_continuation()) Cont_Task_rowsA();
        zgemm3m_Task& calc_task = *new(cont.allocate_child()) zgemm3m_Task( A, B, C, rows_child2, cols );

        recycle_as_child_of(cont);

        rows.Arows_end = rows_mid;
        rows.Arows = rows_mid-rows_start;
        rows.Crows_end = rows_mid;
        rows.Crows = rows_mid-rows_start;

        cont.set_ref_count(2);
        tbb::task::spawn(calc_task);;
        return this;

    }


    else if ( A.is_transposed() && cols.Acols > A.cols/8 && cols.Acols > SERIAL_CUTOFF  ) {
    // *********** divide cols of B into sub-tasks*********

        size_t cols_start = cols.Acols_start;
        size_t cols_end = cols.Acols_end;
        size_t cols_mid = (cols_end+cols_start)/2;

        col_indices cols_child2 = cols;
        cols_child2.Acols_start = cols_mid;
        cols_child2.Acols = cols_end-cols_mid;

        row_indices rows_child2 = rows;
        rows_child2.Crows_start = cols_mid;
        rows_child2.Crows = cols_end-cols_mid;

        // creating continuation task
        Cont_Task_rowsA& cont = *new(allocate_continuation()) Cont_Task_rowsA();

        // creating child task 2
        zgemm3m_Task& calc_task = *new(cont.allocate_child()) zgemm3m_Task( A, B, C, rows_child2, cols_child2);

        // recycling the present task as task1
        recycle_as_child_of(cont);

        cols.Acols_end = cols_mid;
        cols.Acols = cols_mid-cols_start;

        rows.Crows_end = cols_mid;
        rows.Crows = cols_mid-cols_start;

        // spawning task2 and continue with task1 on the same thread
        cont.set_ref_count(2);
        tbb::task::spawn(calc_task);
        return this;

    }


    else if ( !B.is_transposed() && cols.Bcols > B.cols/8 && cols.Bcols > SERIAL_CUTOFF  ) {
    // *********** divide cols of B into sub-tasks*********

        size_t cols_start = cols.Bcols_start;
        size_t cols_end = cols.Bcols_end;
        size_t cols_mid = (cols_end+cols_start)/2;

        col_indices cols_child2 = cols;
        cols_child2.Bcols_start = cols_mid;
        cols_child2.Bcols = cols_end-cols_mid;
        cols_child2.Ccols_start = cols_mid;
        cols_child2.Ccols = cols_end-cols_mid;

        // creating continuation task
        Cont_Task_rowsA& cont = *new(allocate_continuation()) Cont_Task_rowsA();

        // creating child task 2
        zgemm3m_Task& calc_task = *new(cont.allocate_child()) zgemm3m_Task( A, B, C, rows, cols_child2);

        // recycling the present task as task1
        recycle_as_child_of(cont);

        cols.Bcols_end = cols_mid;
        cols.Bcols = cols_mid-cols_start;
        cols.Ccols_end = cols_mid;
        cols.Ccols = cols_mid-cols_start;

        // spawning task2 and continue with task1 on the same thread
        cont.set_ref_count(2);
        tbb::task::spawn(calc_task);
        return this;

    }


    if ( B.is_transposed() && rows.Brows > B.rows/8 && rows.Brows > SERIAL_CUTOFF ) {
        // *********** divide rows of B into sub-tasks*********

        size_t rows_start = rows.Brows_start;
        size_t rows_end = rows.Brows_end;
        size_t rows_mid = (rows_end+rows_start)/2;

        row_indices rows_child2 = rows;
        rows_child2.Brows_start = rows_mid;
        rows_child2.Brows = rows_end-rows_mid;

        col_indices cols_child2 = cols;
        cols_child2.Ccols_start = rows_mid;
        cols_child2.Ccols = rows_end-rows_mid;

        Cont_Task_rowsA& cont = *new(allocate_continuation()) Cont_Task_rowsA();
        zgemm3m_Task& calc_task = *new(cont.allocate_child()) zgemm3m_Task( A, B, C, rows_child2, cols_child2 );

        recycle_as_child_of(cont);

        rows.Brows_end = rows_mid;
        rows.Brows = rows_mid-rows_start;

        cols.Ccols_end = rows_mid;
        cols.Ccols = rows_mid-rows_start;

        cont.set_ref_count(2);
        tbb::task::spawn(calc_task);;
        return this;

    }
    else {
        zgemm3m_chunk();
        return nullptr;
    }


    return nullptr;



} //execute



