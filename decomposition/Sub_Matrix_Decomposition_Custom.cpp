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
/*! \file Sub_Matrix_Decomposition_Custom.cpp
    \brief Class responsible for the disentanglement of one qubit from the others.
    This class enables to define custom gate structure for the decomposition.
*/

#include "Sub_Matrix_Decomposition_Custom.h"



/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/

void Sub_Matrix_Decomposition_Custom::add_operation_layers() {

    int control_qbit_loc = qbit_num-1;

if (qbit_num == 40 || qbit_num == 30) {

    for (int target_qbit_loc = 0; target_qbit_loc<control_qbit_loc; target_qbit_loc++ ) {

        // creating block of operations
        Operation_block* block = new Operation_block( qbit_num );

        if ( target_qbit_loc == 1 ) {

            // swap qubits 0 and 1
            block->add_cnot_to_end(0, 1);
            //block->add_cnot_to_end(1, 0);
            //block->add_cnot_to_end(0, 1);

            // add CNOT gate between qubits target_qbit_loc and 0 to the block
            //block->add_cnot_to_end(control_qbit_loc, 0);

            // swap qubits 0 and 1
            //block->add_cnot_to_end(0, 1);
            //block->add_cnot_to_end(1, 0);
            //block->add_cnot_to_end(0, 1);

            // adding U3 operation to the block
            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
            block->add_u3_to_end(target_qbit_loc, Theta, Phi, Lambda);
            //block->add_u3_to_end(control_qbit_loc, Theta, Phi, Lambda);
            block->add_u3_to_end(0, Theta, Phi, Lambda);

        }
        else {

            // add CNOT gate to the block
            block->add_cnot_to_end(control_qbit_loc, target_qbit_loc);

            // adding U3 operation to the block
            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
            block->add_u3_to_end(target_qbit_loc, Theta, Phi, Lambda);
            block->add_u3_to_end(control_qbit_loc, Theta, Phi, Lambda);



        }


        // adding the operation block to the operations
        add_operation_to_end( block );

    }

}
else {




    // the  number of succeeding identical layers in the subdecomposition
    int identical_blocks_loc;
    try {
        identical_blocks_loc = identical_blocks[qbit_num];
        if (identical_blocks_loc==0) {
            identical_blocks_loc = 1;
        }
    }
    catch (...) {
        identical_blocks_loc=1;
    }


    int control_qbit_loc = qbit_num-1;

    for (int target_qbit_loc = 0; target_qbit_loc<control_qbit_loc; target_qbit_loc++ ) {

        for (int idx=0;  idx<identical_blocks_loc; idx++) {

            // creating block of operations
            Operation_block* block = new Operation_block( qbit_num );

            // add CNOT gate to the block
            block->add_cnot_to_end(control_qbit_loc, target_qbit_loc);

            // adding U3 operation to the block
            bool Theta = true;
            bool Phi = false;
            bool Lambda = true;
            block->add_u3_to_end(target_qbit_loc, Theta, Phi, Lambda);
            block->add_u3_to_end(control_qbit_loc, Theta, Phi, Lambda);

            // adding the opeartion block to the operations
            add_operation_to_end( block );

        }
    }

}

}


