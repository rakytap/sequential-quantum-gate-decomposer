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
/*! \file decomposition_test.cpp
    \brief A simple example file for a decomposition of general random unitaries.
*/

#include <iostream>
#include <stdio.h>
#include <map>


//! [include]
#include "qgd/common.h"
#include "qgd/N_Qubit_Decomposition.h"
#include "qgd/Random_Unitary.h"
//! [include]

using namespace std;


/**
@brief Decomposition of general random unitary matrix into U3 and CNOT gates
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test of N qubit decomposition\n");
    printf("****************************************\n\n\n");

//! [few CNOT]
    // The number of qubits spanning the random unitary
    int qbit_num = 4;

    // the number of rows of the random unitary
    int matrix_size = Power_of_2(qbit_num);

    // creating random unitary constructing from 6 CNOT gates.
    int cnot_num = 1;
    QGD_Complex16* Umtx_few_CNOT = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
    few_CNOT_unitary( qbit_num, cnot_num, Umtx_few_CNOT);
//! [few CNOT]

    // release the constructed random unitary
    qgd_free( Umtx_few_CNOT );
    Umtx_few_CNOT = NULL;


//! [general random]
    // creating class to generate general random unitary
    Random_Unitary ru = Random_Unitary(matrix_size);
    // create general random unitary
    QGD_Complex16* Umtx = ru.Construct_Unitary_Matrix();
//! [general random]


//! [creating decomp class]
    // construct the complex transpose of the random unitary
    QGD_Complex16* Umtx_adj = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
    for (int element_idx=0; element_idx<matrix_size*matrix_size; element_idx++) {
        // determine the row and column index of the element to be filled.
        int col_idx = element_idx % matrix_size;
        int row_idx = int((element_idx-col_idx)/matrix_size);

        // setting the complex conjugate of the element in the adjungate matrix
        QGD_Complex16 element = Umtx[col_idx*matrix_size + row_idx];
        Umtx_adj[element_idx].real = element.real;
        Umtx_adj[element_idx].imag = -element.imag;
    }

    // creating the class for the decomposition
    N_Qubit_Decomposition cDecomposition = N_Qubit_Decomposition( Umtx_adj, qbit_num, false, RANDOM );
//! [creating decomp class]

//! [set parameters]
    // setting the number of successive identical layers used in the decomposition
    std::map<int,int> identical_blocks;
    identical_blocks[3] = 2;
    identical_blocks[4] = 2;
    cDecomposition.set_identical_blocks( identical_blocks );

    // setting the maximal number of layers used in the decomposition
    std::map<int,int> num_of_layers;
    num_of_layers[2] = 3;
    num_of_layers[3] = 16;
    num_of_layers[4] = 60;
    num_of_layers[5] = 240;
    num_of_layers[6] = 960;
    num_of_layers[7] = 3775;
    cDecomposition.set_max_layer_num( num_of_layers );

    // setting the number of optimization iteration loops in each step of the decomposition
    std::map<int,int> num_of_iterations;
    num_of_iterations[2] = 3;
    num_of_iterations[3] = 1;
    num_of_iterations[4] = 1;
    cDecomposition.set_iteration_loops( num_of_iterations );

    // setting operation layer
    cDecomposition.set_optimization_blocks( 1 );

    // setting the verbosity of the decomposition
    cDecomposition.set_verbose( true );
//! [set parameters]

    printf("Starting the decompsition\n");
//! [performing decomposition]
    // starting the decomposition
    cDecomposition.start_decomposition(true, true);

    cDecomposition.list_operations(1);
//! [performing decomposition]


    qgd_free( Umtx );
    Umtx = NULL;

    qgd_free( Umtx_adj );
    Umtx_adj = NULL;



  return 0;

};

