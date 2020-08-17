/*
Created on Fri Jun 26 14:14:12 2020
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

#include <iostream>

#include <stdio.h>
#include <complex.h>
#include <mkl.h>
#include <vector>

#include "Two_Qubit_Decomposition.h"
#include "N_Qubit_Decomposition.h"

using namespace std;


extern "C" {
//
// @brief Decomposition of general two-qubit matrix into U3 and CNOT gates
void two_qubit_decomposition( double* mtx_real, double* mtx_imag, int matrix_size) {
    
    printf("\n\n****************************************\n");
    printf("Test of two qubit decomposition\n");
    printf("****************************************\n\n\n");


    // combining real and imaginary parts of the matrix inti MKL complex matrix
    int element_num = matrix_size*matrix_size;
    MKL_Complex16* Umtx = (MKL_Complex16*)mkl_malloc(element_num*sizeof(MKL_Complex16), 64);

    #pragma omp parallel for    
    for(int idx = 0; idx < element_num; idx++) {
        Umtx[idx].real = mtx_real[idx];
        Umtx[idx].imag = mtx_imag[idx];
    }

    printf("The test matrix to be decomposed is:\n");
    print_mtx( Umtx, matrix_size, matrix_size );

    
    
    // Creating the class to decompose the 2-qubit unitary

    std::map<int,int> num_of_layers;
    num_of_layers[2] = 3;

    Two_Qubit_Decomposition cDecomposition = Two_Qubit_Decomposition( Umtx, 2, num_of_layers, false, "close_to_zero" );

    printf("Calculating the cost function for the input matrix:\n");    
    cDecomposition.test_indepency();


    printf("Starting the decompsition\n");
    cDecomposition.start_decomposition(true);
    
/*
    #start the decomposition
    cDecomposition.start_decomposition()
    
    print('')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operations()
    
    print(' ')
    print('Constructing quantum circuit:')
    print(' ')
    quantum_circuit = cDecomposition.get_quantum_circuit()
    
    print(quantum_circuit)
    
    # test the decomposition of the matrix
    #Changing the simulator 
    backend = Aer.get_backend('unitary_simulator')
    
    #job execution and getting the result as an object
    job = execute(quantum_circuit, backend)
    result = job.result()
    
    #get the unitary matrix from the result object
    decomposed_matrix = result.get_unitary(quantum_circuit)
    
    # get the error of the decomposition
    product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
    decomposition_error =  np.linalg.norm(product_matrix - np.identity(2**qbit_num)*product_matrix[0,0], 2)
    
    print('The error of the decomposition is ' + str(decomposition_error))
    
*/

};



//
// @brief Decomposition of general four-qubit matrix into U3 and CNOT gates
void four_qubit_decomposition( double* mtx_real, double* mtx_imag, int matrix_size) {
    
    printf("\n\n****************************************\n");
    printf("Test of four qubit decomposition\n");
    printf("****************************************\n\n\n");

    // combining real and imaginary parts of the matrix inti MKL complex matrix
    int element_num = matrix_size*matrix_size;
    MKL_Complex16* Umtx = (MKL_Complex16*)mkl_malloc(element_num*sizeof(MKL_Complex16), 64);

    #pragma omp parallel for    
    for(int idx = 0; idx < element_num; idx++) {
        Umtx[idx].real = mtx_real[idx];
        Umtx[idx].imag = mtx_imag[idx];
    }
    
    
    // Creating the class to decompose the four-qubit unitary

    std::map<int,int> num_of_layers;
    num_of_layers[2] = 3;
    num_of_layers[3] = 20;
    num_of_layers[4] = 60;

    std::map<int,int> identical_blocks;
    identical_blocks[2] = 1;
    identical_blocks[3] = 1;
    identical_blocks[4] = 1;

    N_Qubit_Decomposition cDecomposition = N_Qubit_Decomposition( Umtx, 4, num_of_layers, identical_blocks, false, "zeros" );



    printf("Starting the decompsition\n");
    cDecomposition.start_decomposition(true);

};

}
