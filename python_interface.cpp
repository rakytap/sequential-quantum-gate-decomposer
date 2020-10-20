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
/*! \file python_interface.cpp
    \brief Methods of the @python_iface.
*/

#include "qgd/python_interface.h"
#include "qgd/N_Qubit_Decomposition.h"


extern "C" {


/**
@brief Creates an instance of class N_Qubit_Decomposition and return with a void pointer pointing to the class instance
@param mtx_real Pointer to the real part of the unitary to be decomposed
@param mtx_imag Pointer to the imaginary part of the unitary to be decomposed
@param qbit_num Number of qubits spanning the unitary
@param optimize_layer_num Logical value. Set true to optimize the number of decomposing layers during the decomposition procedure, or false otherwise.
@param initial_guess_num Integer encoding the method to guess initial values for the optimalization. Possible values: 'zeros=0','random=1', 'close_to_zero=2'
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
void* iface_new_N_Qubit_Decomposition( double* mtx_real, double* mtx_imag, int qbit_num, bool optimize_layer_num, int initial_guess_num ) {

    int matrix_size = Power_of_2(qbit_num);

    // combining real and imaginary parts of the matrix inti MKL complex matrix
    int element_num = matrix_size*matrix_size;
    QGD_Complex16* Umtx = (QGD_Complex16*)qgd_calloc(element_num,sizeof(QGD_Complex16), 64);

    //#pragma omp parallel for
    for(int idx = 0; idx < element_num; idx++) {
        Umtx[idx].real = mtx_real[idx];
        Umtx[idx].imag = mtx_imag[idx];
    }

    // setting the initial guess type
    guess_type initial_guess;
    if ( initial_guess_num==0 ) {
        initial_guess = ZEROS;
    }
    else if ( initial_guess_num==1 ) {
        initial_guess = RANDOM;
    }
    else if ( initial_guess_num==2 ) {
        initial_guess = CLOSE_TO_ZERO;
    }
    else {
        printf("Wrong initial guess\n");
        exit(-1);
    }


    // creating an instance of class N_Qubit_decomposition
    N_Qubit_Decomposition* instance = new N_Qubit_Decomposition( Umtx, qbit_num, optimize_layer_num, initial_guess );

    return (void*)instance;


}



/**
@brief Starts the decomposition of the unitary
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
int iface_start_decomposition( void* ptr ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);

    instance->start_decomposition(true, true);

    return 0;

}

/**
@brief Call to deallocate the N_Qubit_Decomposition class
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
int iface_delete_N_Qubit_Decomposition( void* ptr ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);

    QGD_Complex16* Umtx = instance->get_Umtx();
    qgd_free( Umtx );
    Umtx = NULL;

    delete instance;

    return 0;

}



/**
@brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param qbit The number of qubits for which the subdecomposition should contain identical_blocks successive identical blocks.
@param identical_blocks Number of successive identical blocks in the decomposition.
*/
int iface_set_identical_blocks( void* ptr, int qbit, int identical_blocks ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->set_identical_blocks( qbit, identical_blocks );

}

/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param qbit The number of qubits for which the subdecomposition should run iteration_loops number of iterations in each step of the optimization.
@param iteration_loops Number of iteration in each step of the subdecomposition.
*/
int iface_set_iteration_loops( void* ptr, int qbit, int iteration_loops ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->set_iteration_loops( qbit, iteration_loops );

}

/**
@brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param qbit The number of qubits for which the subdecomposition should contain maximum max_layer_num layers of operation blocks.
@param max_layer_num The number of maximal number of layers used in the subdecomposition.
*/
int iface_set_max_layer_num( void* ptr, int qbit, int max_layer_num ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->set_max_layer_num( qbit, max_layer_num );

}

/**
@brief Call to list the operations giving the decomposition of the unitary
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param start_index The starting number of the operations to be listed.
*/
void iface_list_operations( void* ptr, int start_index ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->list_operations( start_index );

}

/**
@brief Call to get the number of decomposing operations
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
int iface_get_operation_num( void* ptr ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    return instance->get_operation_num();

}



/**
@brief Call to get the n-th optimized operation. The values are returned via the input parameter references, and pointers
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param n Integer labeling the n-th oepration  (n>=0).
@param op_type The type of operation. (Possible values: CNOT_OPERATION=2, U3_OPERATION=3)
@param target_qbit The id of the target qubit.
@param control_qbit The id of the control qubit.
@param parameters A pointer pointing to the 3-component array conatining the parameters of the U3 operation.
@return Returns with 0 if the export of the n-th operation was successful. If the n-th operation does not exists, -1 is returned. If the operation is not allowed to be exported, i.e. it is not a CNOT or U3 operation, then -2 is returned.
*/
int iface_get_operation( void* ptr, int n, int &op_type, int &target_qbit, int &control_qbit, double* parameters ) {

    operation_type type;
//printf("a %f, %f, %f\n", parameters[0], parameters[1], parameters[2] );
    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    int ret =  instance->get_operation( n, type, target_qbit, control_qbit, parameters );
//printf("b %f, %f, %f\n", parameters[0], parameters[1], parameters[2] );
    op_type = type;

    return ret;


}


/**
@brief Call to set the verbosity of the N_Qubit_Decomposition class
@param ptr A void pointer pointing to an instance of N_Qubit_Decomposition class.
@param verbose Set False to suppress the output messages of the decompostion, or True (deafult) otherwise.
@return Returns with 0 on success
*/
int iface_set_verbose( void* ptr, bool verbose ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(ptr);
    instance->set_verbose( verbose );

    return 0;

}

} // end extern C
