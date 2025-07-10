
/*
Created on Fri Jun 26 14:14:12 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
#include "logging.h"
//! [include]

using namespace std;



/**
@brief Function to create custom gate structure for the decomposition
@param qbit_num The number of qubits for which the gate structure is constructed.
@return Returns with a gate structure to be used in the decomposition.
*/
Gates_block* create_custom_gate_structure( int qbit_num ) {

//! [gates block]
        // creating an instance of the wrapper class Gates_block
        Gates_block* gates_block = new Gates_block( qbit_num );
//! [gates block]

//! [disentangle qubit]
        int disentangle_qubit = qbit_num - 1;
//! [disentangle qubit]


//! [custom structure]
        for ( int qbit=0;  qbit< disentangle_qubit; qbit++ ) {

            // creating an instance of the wrapper class Gates_block
            Gates_block* layer = new Gates_block( qbit_num );

            if (qbit == 1) {

                int target_qbit = 0;
		int control_qbit = 1;

                // add U3 gate to the block
                bool Theta = true;
                bool Phi = false;
                bool Lambda = true;
                layer->add_u3( 0, Theta, Phi, Lambda );
                layer->add_u3( 1, Theta, Phi, Lambda );

                // add CNOT gate to the block		
                layer->add_cnot( target_qbit, control_qbit );


            }
            else {

                int target_qbit = qbit;
		int control_qbit = disentangle_qubit;

                // add U3 gate to the block
                bool Theta = true;
                bool Phi = false;
                bool Lambda = true;
                layer->add_u3( qbit, Theta, Phi, Lambda );
                layer->add_u3( disentangle_qubit, Theta, Phi, Lambda );

                // add CNOT gate to the block
                layer->add_cnot( target_qbit, control_qbit );
            }

            gates_block->add_gate( (Gate*)layer );

        }
//! [custom structure]

        return gates_block;


}

/**
@brief Decomposition of general random unitary matrix into a custom structure of U3 and CNOT gates
*/
int main() {



    /// Stringstream input to store the output messages.
    std::stringstream sstream;

    /// Logging variable to set the verbosity level.
    logging output;

    /// Set the verbosity level of the output messages. 
    int verbose_level;

    //Setting the verbosity level of the Test of N qubit decomposition with custom gate structure
    verbose_level=1;
    sstream << std::endl << std::endl << "****************************************" << std::endl;
    sstream << "Test of N qubit decomposition with custom gate structure" << std::endl;
    sstream << "****************************************"<< std::endl << std::endl << std::endl;
    output.print(sstream,verbose_level);	    	
 
//! [general random]
    // The number of qubits spanning the random unitary
    int qbit_num = 3;
    // the number of rows of the random unitary
    int matrix_size = Power_of_2(qbit_num);
    // creating class to generate general random unitary
    Random_Unitary ru = Random_Unitary(matrix_size);
    // create general random unitary
    Matrix Umtx = ru.Construct_Unitary_Matrix();




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
//! [general random]

//! [creating decomp class]
    // creating the class for the decomposition. Here Umtx_adj is the complex transposition of unitary Umtx
    N_Qubit_Decomposition cDecomposition =
                   N_Qubit_Decomposition( Umtx_adj, qbit_num, /* optimize_layer_num= */ false, /* initial_guess= */ RANDOM );
//! [creating decomp class]



//! [creating custom gate structure]
    // creating custom gate structure for the decomposition
    std::map<int, Gates_block*> gate_structure;
    gate_structure.insert( pair<int, Gates_block*>(qbit_num, create_custom_gate_structure( qbit_num ) ) );
    gate_structure.insert( pair<int, Gates_block*>(qbit_num-1, create_custom_gate_structure( qbit_num-1 ) ) );

    // setting the custom gate structure in the decomposition class
    cDecomposition.set_custom_gate_structure( gate_structure);

    // release the custom gate structure since it was cloned by cDecomposition
    delete gate_structure[qbit_num];
    delete gate_structure[qbit_num-1];
//! [creating custom gate structure]


    // setting the verbosity level of the decomposition
    verbose_level=1;
    sstream << "Starting the decompsition" << std::endl;
    output.print(sstream,verbose_level);	    	
    
//! [performing decomposition]
    // starting the decomposition
    cDecomposition.start_decomposition(/* finalize_decomposition = */ true, /* prepare_export= */ true);

    cDecomposition.list_gates(1);
//! [performing decomposition]




  return 0;

};

