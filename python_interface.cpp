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

#include "qgd/python_interface.h"
#include "qgd/N_Qubit_Decomposition.h"


extern "C" {


// @brief Creates an instance of class N_Qubit_Decomposition and return with a void pointer pointing to the class instance
void* iface_new_N_Qubit_Decomposition( double* mtx_real, double* mtx_imag, int qbit_num, bool optimize_layer_num, int initial_guess_num ) {

    int matrix_size = Power_of_2(qbit_num);

    // combining real and imaginary parts of the matrix inti MKL complex matrix
    int element_num = matrix_size*matrix_size;
    QGD_Complex16* Umtx = (QGD_Complex16*)qgd_calloc(element_num,sizeof(QGD_Complex16), 64);

    #pragma omp parallel for    
    for(int idx = 0; idx < element_num; idx++) {
        Umtx[idx].real = mtx_real[idx];
        Umtx[idx].imag = mtx_imag[idx];
    }
    
    
    // skeleton input variables to initialize the class
    std::map<int,int> num_of_layers;
    std::map<int,int> identical_blocks;

    // setting the initial guess type
    string initial_guess;
    if ( initial_guess_num==0 ) {
        initial_guess = "zeros";        
    }
    else if ( initial_guess_num==1 ) {
        initial_guess = "random";        
    }
    else if ( initial_guess_num==2 ) {
        initial_guess = "close_to_zero";        
    }


    // creating an instance of class N_Qubit_decomposition
    N_Qubit_Decomposition* instance = new N_Qubit_Decomposition( Umtx, qbit_num, num_of_layers, identical_blocks, optimize_layer_num, initial_guess );

    return (void*)instance;


}



// @brief Starts the decomposition of the unitary
int iface_start_decomposition( void* ptr ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);

    instance->start_decomposition(true);

    return 0;

}

// @brief Deallocate the N_Qubit_Decomposition class
int iface_delete_N_Qubit_Decomposition( void* ptr ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    delete instance;

    return 0;

}



// @brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
int iface_set_identical_blocks( void* ptr, int qbit, int identical_blocks ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->set_identical_blocks( qbit, identical_blocks );

}

// @brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
int iface_set_iteration_loops( void* ptr, int qbit, int iteration_loops ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->set_iteration_loops( qbit, iteration_loops );

}

// @brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
int iface_set_max_layer_num( void* ptr, int qbit, int max_layer_num ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->set_max_layer_num( qbit, max_layer_num );

}

}
