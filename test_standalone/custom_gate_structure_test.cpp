
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
/*! \file custom_gate_structure_test.cpp
    \brief A simple example file for a decomposition of general random unitaries into custom structure of U3 and CNOT gates.
*/

#include <iostream>
#include <stdio.h>
#include <map>


//! [include]
#include "common.h"
#include "N_Qubit_Decomposition.h"
#include "Random_Unitary.h"
//! [include]

using namespace std;


/**
@brief Function to create custom gate structure for the decomposition
@param qbit_num The number of qubits for which the gate structure is constructed.
@return Returns with a gate structure to be used in the decomposition.
*/
Operation_block* create_custom_gate_structure( int qbit_num ) {

        // creating an instance of the wrapper class qgd_Operation_Block
        Operation_block* Operation_Block_ret = new Operation_block( qbit_num );

        int control_qbit = qbit_num - 1;

        for ( int target_qbit=0;  target_qbit< control_qbit; target_qbit++ ) {

            // creating an instance of the wrapper class qgd_Operation_Block
            Operation_block* Operation_Block_inner = new Operation_block( qbit_num );

            if (target_qbit == 1) {

                // add CNOT gate to the block
                Operation_Block_inner->add_cnot_to_end( 0, 1);

                // add U3 fate to the block
                bool Theta = true;
                bool Phi = false;
                bool Lambda = true;
                Operation_Block_inner->add_u3_to_end( 0, Theta, Phi, Lambda );
                Operation_Block_inner->add_u3_to_end( 1, Theta, Phi, Lambda );

            }
            else {

                // add CNOT gate to the block
                Operation_Block_inner->add_cnot_to_end( control_qbit, target_qbit );

                // add U3 fate to the block
                bool Theta = true;
                bool Phi = false;
                bool Lambda = true;
                Operation_Block_inner->add_u3_to_end( target_qbit, Theta, Phi, Lambda );
                Operation_Block_inner->add_u3_to_end( control_qbit, Theta, Phi, Lambda );
            }

            Operation_Block_ret->add_operation_to_end( (Operation*)Operation_Block_inner );

        }

        return Operation_Block_ret;


}

/**
@brief Decomposition of general random unitary matrix into a custom structure of U3 and CNOT gates
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test of N qubit decomposition with custom gate structure\n");
    printf("****************************************\n\n\n");




//! [general random]
    // The number of qubits spanning the random unitary
    int qbit_num = 3;
    // the number of rows of the random unitary
    int matrix_size = Power_of_2(qbit_num);
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
    N_Qubit_Decomposition cDecomposition =
                   N_Qubit_Decomposition( Umtx_adj, qbit_num, /* optimize_layer_num= */ false, /* initial_guess= */ RANDOM );
//! [creating decomp class]



//! [creating custom gate structure]
    // creating custom gate structure for the decomposition
    std::map<int, Operation_block*> gate_structure;
    gate_structure.insert( pair<int, Operation_block*>(qbit_num, create_custom_gate_structure( qbit_num ) ) );
    gate_structure.insert( pair<int, Operation_block*>(qbit_num-1, create_custom_gate_structure( qbit_num-1 ) ) );

    // setting the custom gate structure in the decomposition class
    cDecomposition.set_custom_gate_structure( gate_structure);

    // release the custom gate structure since it was cloned by cDecomposition
    delete gate_structure[qbit_num];
    delete gate_structure[qbit_num-1];
//! [creating custom gate structure]


    printf("Starting the decompsition\n");
//! [performing decomposition]
    // starting the decomposition
    cDecomposition.start_decomposition(/* finalize_decomposition = */ true, /* prepare_export= */ true);

    cDecomposition.list_operations(1);
//! [performing decomposition]




  return 0;

};

