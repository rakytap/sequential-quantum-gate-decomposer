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
#include "common.h"
#include "N_Qubit_Decomposition.h"
#include "Sub_Matrix_Decomposition_Custom.h"
#include "Random_Unitary.h"
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
    int qbit_num = 3;

    // the number of rows of the random unitary
    int matrix_size = Power_of_2(qbit_num);

    // creating random unitary constructing from 6 CNOT gates.
    int cnot_num = 6;
    Matrix Umtx_few_CNOT = few_CNOT_unitary( qbit_num, cnot_num);
//! [few CNOT]


//! [general random]
    // creating class to generate general random unitary
    Random_Unitary ru = Random_Unitary(matrix_size);
    // create general random unitary
    Matrix Umtx = ru.Construct_Unitary_Matrix();
//! [general random]



    // construct the complex transpose of the random unitary
    Matrix Umtx_adj = Matrix(matrix_size, matrix_size);
    for (int element_idx=0; element_idx<matrix_size*matrix_size; element_idx++) {
        // determine the row and column index of the element to be filled.
        int col_idx = element_idx % matrix_size;
        int row_idx = int((element_idx-col_idx)/matrix_size);

        // setting the complex conjugate of the element in the adjungate matrix
        QGD_Complex16 element = Umtx[col_idx*matrix_size + row_idx];
        Umtx_adj[element_idx].real = element.real;
        Umtx_adj[element_idx].imag = -element.imag;
    }
//! [creating decomp class]
    // creating the class for the decomposition. Here Umtx_adj is the complex transposition of unitary Umtx
    N_Qubit_Decomposition<Sub_Matrix_Decomposition> cDecomposition =
                   N_Qubit_Decomposition<Sub_Matrix_Decomposition>( Umtx_adj, qbit_num, /* optimize_layer_num= */ false, /* initial_guess= */ RANDOM );
//! [creating decomp class]

//! [set parameters]
    // setting the number of successive identical layers used in the decomposition
    std::map<int,int> identical_blocks;
    identical_blocks[3] = 1;
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
    cDecomposition.start_decomposition(/* finalize_decomposition = */ true, /* prepare_export= */ true);

    cDecomposition.list_operations(1);
//! [performing decomposition]




  return 0;

};

