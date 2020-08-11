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

#include <common.h> 
#include <sstream>

using namespace std;


int Power_of_2(int n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * Power_of_2(n-1);
}


// print the matrix
void print_mtx( MKL_Complex16* matrix, int size ) {

    for ( int row_idx=0; row_idx < size; row_idx++ ) {
        for ( int col_idx=0; col_idx < size; col_idx++ ) {
            int element_idx = row_idx*size+col_idx;    
            printf("%f + i*%f,  ", matrix[element_idx].real, matrix[element_idx].imag);
        }
        printf("\n");
    }
    printf("\n\n\n");
}


// print a CNOT
void print_CNOT( MKL_Complex16* matrix, int size ) {

    for ( int row_idx=0; row_idx < size; row_idx++ ) {
        for ( int col_idx=0; col_idx < size; col_idx++ ) {
            int element_idx = row_idx*size+col_idx;    
            printf("%d,  ", int(matrix[element_idx].real));
        }
        printf("\n");
    }
    printf("\n\n\n");
}



// @brief Calculate the product of complex matrices stored in a vector of matrices
MKL_Complex16* reduce_zgemm( vector<MKL_Complex16*> mtxs, int matrix_size ) {
    

    if (mtxs.size() == 0 ) {
        
        MKL_Complex16* matrix = (MKL_Complex16*)mkl_calloc(matrix_size*matrix_size, sizeof(MKL_Complex16), 64);

        // setting the giagonal elelments to identity
        #pragma omp parallel for
        for(int idx = 0; idx < matrix_size; ++idx)
        {
            int element_index = idx*matrix_size + idx;

            matrix[element_index].real = 1;
        }

        return matrix;
    }


    // parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1;
    double beta = 0;


    // pointers to matrices to be used in the multiplications
    MKL_Complex16* A = NULL;
    MKL_Complex16* B = NULL;

    // the iteration number
    int iteration = 0;


    std::vector<MKL_Complex16*>::iterator it = mtxs.begin();
    MKL_Complex16* C = *it;
    it++;   
 
    // calculate the product of complex matrices
    for(it; it != mtxs.end(); ++it) {

        iteration++;

        A = C;
        B = *it;

/*printf("The matrix A:\n");
print_mtx( A, matrix_size );
printf("\n");
printf("The matrix B:\n");
print_mtx( B, matrix_size );
printf("\n");*/

        // preallocate array for the result
        C = (MKL_Complex16*)mkl_calloc(matrix_size*matrix_size, sizeof(MKL_Complex16), 64); 

        // calculate the product of A and B
        cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size, matrix_size, matrix_size, &alpha, A, matrix_size, B, matrix_size, &beta, C, matrix_size);

        // free the memory of previously allocated matrix A which is irrelevant for the further calculations
        if ( iteration>1 ) {
            mkl_free(A);
        }

/*printf("The product matrix C:\n");
print_mtx( C, matrix_size );
printf("\n");*/


    }


    return C;
}



