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
// @brief A base class responsible for constructing matrices of C-NOT gates
// gates acting on the N-qubit space

#include "N_Qubit_Decomposition.h"

 
//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param max_layer_num_in A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process at the subdecomposing of n-th qubits.
// @param identical_blocks_in A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
N_Qubit_Decomposition::N_Qubit_Decomposition( MKL_Complex16* Umtx_in, int qbit_num_in, std::map<int,int> max_layer_num_in, std::map<int,int> identical_blocks_in, bool optimize_layer_num_in=false, string initial_guess_in="close_to_zero" ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {
        
    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = "N_Qubit_Decomposition";
        
    // The global minimum of the optimalization problem
    global_target_minimum = 0;
        
    // number of iteratrion loops in the optimalization
    iteration_loops[2] = 3;
    iteration_loops[3] = 1;
    iteration_loops[4] = 1;
    iteration_loops[5] = 1;
    iteration_loops[6] = 1;
    iteration_loops[7] = 1;
    iteration_loops[8] = 1;
                
    // The number of successive identical blocks in one leyer
    identical_blocks = identical_blocks_in;

    // layer number used in the decomposition
    max_layer_num = max_layer_num_in;
   
    // filling in numbers that were not given in the input
    for ( std::map<int,int>::iterator it = max_layer_num_def.begin(); it!=max_layer_num_def.end(); it++) {      
        if ( max_layer_num.count( it->first ) == 0 ) {
            max_layer_num.insert( std::pair<int, int>(it->first,  it->second) );
        }
    }


}


////
// @brief Start the disentanglig process of the least significant two qubit unitary
// @param finalize_decomposition Optional logical parameter. If true (default), the decoupled qubits are rotated into
// state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
void N_Qubit_Decomposition::start_decomposition(bool finalize_decomposition=true) {
        
        
            
        
    printf("***************************************************************\n");
    printf("Starting to disentangle %d-qubit matrix\n", qbit_num);
    printf("***************************************************************\n\n\n");
        
    //measure the time for the decompositin       
    clock_t start_time = clock();
            

    // create an instance of class to disentangle the given qubit pair
    Sub_Matrix_Decomposition cSub_decomposition = Sub_Matrix_Decomposition(Umtx, qbit_num, max_layer_num, 
                          identical_blocks, optimize_layer_num, initial_guess);
        
    // The maximal error of the optimalization problem 
    //cSub_decomposition.optimalization_tolerance = self.optimalization_tolerance
        
    // setting the maximal number of iterations in the disentangling process
    cSub_decomposition.optimalization_block = optimalization_block;
        
    // setting the number of operators in one sub-layer of the disentangling process
    //cSub_decomposition.max_iterations = self.max_iterations
            
    //start to disentangle the qubit pair
    cSub_decomposition.disentangle_submatrices();
                                
    if ( !cSub_decomposition.subdisentaglement_done) {
        return;
    }


        
/*        
    // saving the subunitarization operations
    extract_subdecomposition_results( cSub_decomposition );
        
    // decompose the qubits in the disentangled submatrices
    decompose_submatrix();
            
    if finalize_decomposition:
        # finalizing the decompostition
        self.finalize_decomposition()
            
        # simplify layers
        self.simplify_layers()
            
        # final tuning of the decomposition parameters
        self.final_optimalization()
            
        matrix_new = self.get_transformed_matrix(self.optimized_parameters, self.operations )    

        # calculating the final error of the decomposition
        self.decomposition_error = LA.norm(matrix_new*np.exp(np.complex(0,-np.angle(matrix_new[0,0]))) - np.identity(len(matrix_new))*abs(matrix_new[0,0]), 2)
            
        # get the number of gates used in the decomposition
        gates_num = self.get_gate_nums()
        print( 'In the decomposition with error = ' + str(self.decomposition_error) + ' were used ' + str(self.layer_num) + ' layers with '  + str(gates_num['u3']) + ' U3 operations and ' + str(gates_num['cnot']) + ' CNOT gates.' )        
*/            
        
    printf("--- In total %f seconds elapsed during the decomposition ---\n", (clock() - start_time));

}

