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

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space

#include "qgd/common.h"


using namespace std;

// define aligned memory allocation function if they are not present (in case of older gcc compilers)
//#ifndef ac_cv_func_aligned_alloc
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
void *aligned_alloc(size_t alignment, size_t size, bool zero) {
    size_t request_size = size + alignment;
    char* buf = (char*)(zero ? calloc(1, request_size) : malloc(request_size));

    size_t remainder = ((size_t)buf) % alignment;
    size_t offset = alignment - remainder;
    char* ret = buf + (unsigned char)offset;

    // store how many extra bytes we allocated in the byte just before the
    // pointer we return
    *(unsigned char*)(ret - 1) = offset;

    return (void*)ret;
}

/* Free memory allocated with aligned_alloc */
void aligned_free(void* aligned_ptr) {
    int offset = *(((char*)aligned_ptr) - 1);
    free(((char*)aligned_ptr) - offset);
}
//#endif


void* qgd_calloc( size_t element_num, size_t size, size_t alignment ) {

//#ifndef ACALLOC
    void* ret = aligned_alloc(alignment, size*element_num, true);
/*#else
    void* ret = aligned_alloc(alignment, size*element_num);
    memset(ret, 0, size*element_num );
#endif*/
    return ret;
}

void qgd_free( void* ptr ) {
//#ifndef ACALLOC
    aligned_free(ptr);
/*#else
    free(ptr);
#endif*/
    ptr = NULL;
}


// Calculates the power of 2
int Power_of_2(int n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * Power_of_2(n-1);
}


// @brief Print a complex matrix on standard output
void print_mtx( QGD_Complex16* matrix, int rows, int cols ) {

    for ( int row_idx=0; row_idx < rows; row_idx++ ) {
        for ( int col_idx=0; col_idx < cols; col_idx++ ) {
            int element_idx = row_idx*cols + col_idx;    
            printf("%1.3f + i*%1.3f,  ", matrix[element_idx].real, matrix[element_idx].imag);
//            printf("%1.3f,  ", matrix[element_idx].real);
        }
        printf("\n");
    }
    printf("\n\n\n");

}


// @brief Print a CNOT matrix on standard output
void print_CNOT( QGD_Complex16* matrix, int size ) {

    for ( int row_idx=0; row_idx < size; row_idx++ ) {
        for ( int col_idx=0; col_idx < size; col_idx++ ) {
            int element_idx = row_idx*size+col_idx;    
            printf("%d,  ", int(matrix[element_idx].real));
        }
        printf("\n");
    }
    printf("\n\n\n");
}


// @brief Add an integer to an integer vector if the integer is not already an element of the vector. The sorted order is kept during the process
void add_unique_elelement( vector<int> &involved_qbits, int qbit ) {

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


// @brief Create an identity matrix
QGD_Complex16* create_identity( int matrix_size ) {

    QGD_Complex16* matrix = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

    // setting the giagonal elelments to identity
    #pragma omp parallel for
    for(int idx = 0; idx < matrix_size; ++idx)
    {
        int element_index = idx*matrix_size + idx;
            matrix[element_index].real = 1;
    }

    return matrix;

}



// @brief Create an identity matrix
int create_identity( QGD_Complex16* matrix, int matrix_size ) {

    // setting the giagonal elelments to identity
    #pragma omp parallel for
    for(int idx = 0; idx < matrix_size*matrix_size; ++idx)
    {
        int col_idx = idx % matrix_size;
        int row_idx = int((idx-col_idx)/matrix_size);

        if ( row_idx == col_idx ) {
            matrix[idx].real = 1;
            matrix[idx].imag = 0;
        }
        else {
            matrix[idx].real = 0;
            matrix[idx].imag = 0;
        }
            
    }

    return 0;

}


// @brief Call to calculate the product of two matrices using cblas_zgemm3m
QGD_Complex16 scalar_product( QGD_Complex16* A, QGD_Complex16* B, int vector_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1;
    double beta = 0;

    // preallocate array for the result
    QGD_Complex16 C; 

    // calculate the product of A and B
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasConjTrans, 1, 1, vector_size, &alpha, A, vector_size, B, vector_size, &beta, &C, 1);

    return C;
}


// @brief Call to calculate the product of two matrices using cblas_zgemm3m
int zgemm3m_wrapper_adj( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // remove memory trash from the allocated memory of the results
    memset( C, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // calculate the product of A and B
#ifdef MKL
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasConjTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasConjTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#endif

    return 0;
}




// @brief Call to calculate the product of two matrices using cblas_zgemm3m
QGD_Complex16* zgemm3m_wrapper( QGD_Complex16* A, QGD_Complex16* B, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // preallocate array for the result
    QGD_Complex16* C = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64); 

    // remove memory trash from the allocated memory of the results
    memset( C, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // calculate the product of A and B
#ifdef MKL
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#endif

    return C;
}


// @brief Call to calculate the product of two matrices using cblas_zgemm3m
int zgemm3m_wrapper( QGD_Complex16* A, QGD_Complex16* B, QGD_Complex16* C, int matrix_size) {

    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1.0;
    double beta = 0.0;

    // remove memory trash from the allocated memory of the results
    memset( C, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // calculate the product of A and B
#ifdef MKL
    cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#else
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);
#endif

    return 0;
}



// @brief Calculate the product of complex matrices stored in a vector of matrices
int reduce_zgemm( vector<QGD_Complex16*> mtxs, QGD_Complex16* C, int matrix_size ) {
    

    if (mtxs.size() == 0 ) {
        return create_identity(C, matrix_size);
    }



    // pointers to matrices to be used in the multiplications
    QGD_Complex16* A = NULL;   
    QGD_Complex16* B = NULL;

    // the iteration number
    int iteration = 0;


    std::vector<QGD_Complex16*>::iterator it = mtxs.begin();
    QGD_Complex16* tmp = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size,sizeof(QGD_Complex16), 64); 
    A = *it;
    it++;   
 
    // calculate the product of complex matrices
    for(it; it != mtxs.end(); ++it) {

        iteration++;

        if ( iteration>1 ) {
            A = tmp;
            memcpy(A, C, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        }
        B = *it;


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

}


// @brief subtract a scalar from the diagonal of a matrix
void subtract_diag( QGD_Complex16* & mtx,  int matrix_size, QGD_Complex16 scalar ) {

    #pragma omp parallel for
    for(int idx = 0; idx < matrix_size; idx++)   {
        int element_idx = idx*matrix_size+idx;
        mtx[element_idx].real = mtx[element_idx].real - scalar.real;
        mtx[element_idx].imag = mtx[element_idx].imag - scalar.imag;
    }

}

// calculate the cost funtion from the submatrices of the given matrix 
double get_submatrix_cost_function(QGD_Complex16* matrix, int matrix_size, QGD_Complex16** submatrices, QGD_Complex16* submatrix_prod) {

    // ********************************
    // Calculate the submatrix products
    // ********************************

    // number ofcolumns in the submatrices
    int submatrix_size = matrix_size/2;
    // number of elements in the matrix of submatrix products
    int element_num = submatrix_size*submatrix_size;

    int submatrices_num_row = 2;
    int submatrices_num = 4;

    // fill up the submatrices
    for (int idx=0; idx<submatrices_num_row; idx++) { //in range(0,submatrices_num_row):
        for ( int jdx=0; jdx<submatrices_num_row; jdx++) { //in range(0,submatrices_num_row):

            int submatirx_index = idx*submatrices_num_row + jdx;

            // copy memory to submatrices
            #pragma omp parallel for
            for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {

                int matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
                int submatrix_offset = row_idx*submatrix_size;
                memcpy(submatrices[submatirx_index]+submatrix_offset, matrix+matrix_offset, submatrix_size*sizeof(QGD_Complex16));

            }
        }
    }

    
    // calculate the cost function
    double cost_function = 0;
    for (int idx=0; idx<submatrices_num; idx++) { //idx in range(0,submatrices_num):
        for ( int jdx=0; jdx<submatrices_num_row; jdx++) { //jdx in range(0,submatrices_num_row): 
            
            // calculate the submatrix product
            zgemm3m_wrapper_adj( submatrices[idx], submatrices[jdx], submatrix_prod, submatrix_size);

            // subtract the corner element from the diagonal
            QGD_Complex16 corner_element = submatrix_prod[0];
            #pragma omp parallel for
            for ( int row_idx=0; row_idx<submatrix_size; row_idx++) {
                int element_idx = row_idx*submatrix_size+row_idx;
                submatrix_prod[element_idx].real = submatrix_prod[element_idx].real  - corner_element.real;
                submatrix_prod[element_idx].imag = submatrix_prod[element_idx].imag  - corner_element.imag;
            }

            #pragma omp barrier
        
            // Calculate the |x|^2 value of the elements of the submatrixproducts
            #pragma omp parallel for
            for ( int idx=0; idx<element_num; idx++ ) {
                submatrix_prod[idx].real = submatrix_prod[idx].real*submatrix_prod[idx].real + submatrix_prod[idx].imag*submatrix_prod[idx].imag;
                // for performance reason we leave the imaginary part intact (we dont neet it anymore)
                //submatrix_prods[idx].imag = 0;
            }

            #pragma omp barrier
         

            // summing up elements and calculate the final cost function

            double cost_function_tmp = 0;
            #pragma omp parallel for reduction(+:cost_function_tmp)
            for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {

                // calculate the sum for each row
                for (int col_idx=0; col_idx<submatrix_size; col_idx++) {
                    int element_idx = row_idx*submatrix_size + col_idx;
                    cost_function_tmp = cost_function_tmp + submatrix_prod[element_idx].real;//*submatrix_prod[element_idx].real + submatrix_prod[element_idx].imag*submatrix_prod[element_idx].imag;
                    //cost_function_tmp = cost_function_tmp + submatrix_product_square[element_idx];
                } 
            }


            // checking NaN
            if (isnan(cost_function)) {
                printf("cost function NaN: exiting\n");
                exit(-1);
            }   
            cost_function = cost_function + cost_function_tmp;

        }
    }

//printf("%f\n",   cost_function );   


    

    

    return cost_function;

}





// calculate the cost funtion for the final optimalization
double get_cost_function(QGD_Complex16* matrix, int matrix_size) {

    QGD_Complex16* mtx = matrix;

    // subtract the corner element from the diagonal
    QGD_Complex16 corner_element = matrix[0];
    #pragma omp parallel for
    for ( int row_idx=0; row_idx<matrix_size; row_idx++) {
        int element_idx = row_idx*matrix_size+row_idx;
        mtx[element_idx].real = mtx[element_idx].real  - corner_element.real;
        mtx[element_idx].imag = mtx[element_idx].imag  - corner_element.imag;
    }

    #pragma omp barrier

    // Calculate the |x|^2 value of the elements of the submatrixproducts
    //double* matrix_product_square = (double*)qgd_calloc(matrix_size*matrix_size*sizeof(double), 64); 
    int element_num = matrix_size*matrix_size;
    #pragma omp parallel for
    for ( int idx=0; idx<element_num; idx++ ) {
         //matrix_product_square[idx] = matrix[idx].real*matrix[idx].real + matrix[idx].imag*matrix[idx].imag;
         mtx[idx].real = mtx[idx].real*mtx[idx].real + mtx[idx].imag*mtx[idx].imag;
         // for performance reason we leave the imaginary part intact (we dont neet it anymore)
         //mtx[idx].imag = 0;
    }

    #pragma omp barrier

    // summing up elements and calculate the final cost function

    double cost_function = 0;
    //#pragma omp parallel for reduction(+:cost_function)
    for ( int row_idx=0; row_idx<matrix_size; row_idx++ ) {

        // calculate the sum for each row
        for (int col_idx=0; col_idx<matrix_size; col_idx++) {
            int element_idx = row_idx*matrix_size + col_idx;
            //cost_function = cost_function + matrix_product_square[element_idx];
            cost_function = cost_function + mtx[element_idx].real;
        } 
    }

    return cost_function;

}




// calculate the product of two scalars
QGD_Complex16 mult( QGD_Complex16 a, QGD_Complex16 b ) {

    QGD_Complex16 ret;
    ret.real = a.real*b.real - a.imag*b.imag;
    ret.imag = a.real*b.imag + a.imag*b.real;

    return ret;

}

// calculate the product of two scalars
QGD_Complex16 mult( double a, QGD_Complex16 b ) {

    QGD_Complex16 ret;
    ret.real = a*b.real;
    ret.imag = a*b.imag;

    return ret;

}



// Multiply the elements of matrix "b" by a scalar "a".
void mult( QGD_Complex16 a, QGD_Complex16* b, int matrix_size ) {

    for (int idx=0; idx<matrix_size*matrix_size; idx++) {
        QGD_Complex16 tmp = b[idx];
        b[idx].real = a.real*tmp.real - a.imag*tmp.imag;
        b[idx].imag = a.real*tmp.imag + a.imag*tmp.real;
    }

    return;

}



