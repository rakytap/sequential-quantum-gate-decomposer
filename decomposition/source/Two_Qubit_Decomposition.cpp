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

#include "Two_Qubit_Decomposition.h"

 
    


//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
Two_Qubit_Decomposition::Two_Qubit_Decomposition( MKL_Complex16* Umtx_in, int qbit_num_in, bool optimize_layer_num_in=false, string initial_guess_in= "zeros" ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {
        
    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;
        
    // The global minimum of the optimalization problem
    global_target_minimum = 0;
        
    // number of iteratrion loops in the optimalization
    iteration_loops[2] = 3;
    
    // number of operators in one sub-layer of the optimalization process
    optimalization_block = 1;


}



//// start_decomposition
// @brief Start the decompostion process of the two-qubit unitary
// @param finalize_decomposition Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
void  Two_Qubit_Decomposition::start_decomposition( bool to_finalize_decomposition=true ) {
        
    if ( decomposition_finalized ) {
        printf("Decomposition was already finalized");
        return;
    }
        
    //check whether the problem can be solved without optimalization
    if ( !test_indepency() ) {

        long max_layer_num_loc;
        try {
            max_layer_num_loc = max_layer_num[qbit_num];
        }
        catch (...) {
            max_layer_num_loc = 3;
        }
printf("max layer num: %d", max_layer_num_loc );
            
        // Do the optimalization of the parameters
        while (layer_num < max_layer_num_loc) {
                
                // creating block of operations
                Operation_block* block = new Operation_block( qbit_num );
                    
                // add CNOT gate to the block
                block->add_cnot_to_end(1, 0);
                    
                // adding U3 operation to the block
                bool Theta = true;
                bool Phi = false;
                bool Lambda = true;
                block->add_u3_to_end(1, Theta, Phi, Lambda); 
                block->add_u3_to_end(0, Theta, Phi, Lambda);
                    
                // adding the opeartion block to the operations
                add_operation_to_end( block );
                
                // set the number of layers in the optimalization
                optimalization_block = layer_num;
                
                // Do the optimalization
                if (optimize_layer_num || layer_num >= max_layer_num_loc) {
                    // solve the optzimalization problem to find the correct mninimum
                    solve_optimalization_problem();

                    if (check_optimalization_solution()) {
                        break;
                    }
                }
    
        }
                    
                
    }

        
    // check the solution
    if (check_optimalization_solution() ) {
                
        // logical value describing whether the first optimalization problem was solved or not
        optimalization_problem_solved = true;
    }        
    else {
        // setting the logical variable to true even if no optimalization was needed
        optimalization_problem_solved = false;
    }
        
       
    //finalize the decomposition
    if ( to_finalize_decomposition ) {
        finalize_decomposition();
    }


                
}       
    
        
        

    
        
//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double Two_Qubit_Decomposition::optimalization_problem( double* parameters ) {
               
        // get the transformed matrix with the operations in the list
        MKL_Complex16* matrix_new = get_transformed_matrix( parameters, operations.begin(), operations.size(), Umtx );
/*                
        submatrices = list()
        submatrices.append(matrix_new[0:2, 0:2])
        submatrices.append(matrix_new[0:2, 2:4])
        submatrices.append(matrix_new[2:4, 0:2])
        submatrices.append(matrix_new[2:4, 2:4])
        
        cost_function = 0
        
        for idx in range(0,4):
            for jdx in range(idx,4):
                
                submatrix_prod = np.dot(submatrices[idx],submatrices[jdx].conj().T)
                #print(submatrix_prod)
                #print(' ')
                submatrix_prod = submatrix_prod - np.identity(len(submatrix_prod))*submatrix_prod[0,0]
                cost_function = cost_function + np.sum( np.multiply(submatrix_prod, submatrix_prod.conj() ) )
                
                
        return np.real(cost_function)
 
*/               
}                                        
                    

    
//// 
// @brief Check whether qubits are indepent or not
// @returns Return with true if qubits are disentangled, or false otherwise.
bool Two_Qubit_Decomposition::test_indepency() {
       
        current_minimum = optimalization_problem( optimized_parameters );
        
        return check_optimalization_solution();     
        
}        
   
      

