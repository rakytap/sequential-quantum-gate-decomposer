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



/// 
// @brief Destructor of the class
N_Qubit_Decomposition::~N_Qubit_Decomposition() { 

}




////
// @brief Start the disentanglig process of the least significant two qubit unitary
// @param finalize_decomposition Optional logical parameter. If true (default), the decoupled qubits are rotated into
// state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
void N_Qubit_Decomposition::start_decomposition(bool finalize_decomp=true) {
        
        
            
        
    printf("***************************************************************\n");
    printf("Starting to disentangle %d-qubit matrix\n", qbit_num);
    printf("***************************************************************\n\n\n");
        
    //measure the time for the decompositin       
    clock_t start_time = time(NULL);
            

    // create an instance of class to disentangle the given qubit pair
    Sub_Matrix_Decomposition* cSub_decomposition = new Sub_Matrix_Decomposition(Umtx, qbit_num, max_layer_num, 
                          identical_blocks, optimize_layer_num, initial_guess);
        
    // The maximal error of the optimalization problem 
    //cSub_decomposition->optimalization_tolerance = self.optimalization_tolerance
        
    // setting the maximal number of iterations in the disentangling process
    cSub_decomposition->optimalization_block = optimalization_block;
        
    // setting the number of operators in one sub-layer of the disentangling process
    //cSub_decomposition->max_iterations = self.max_iterations
            
    //start to disentangle the qubit pair
    cSub_decomposition->disentangle_submatrices();
                                
    if ( !cSub_decomposition->subdisentaglement_done) {
        return;
    }


        
        
    // saving the subunitarization operations
    extract_subdecomposition_results( cSub_decomposition );

    delete cSub_decomposition;       
    cSub_decomposition = NULL;
        
    // decompose the qubits in the disentangled submatrices
    decompose_submatrix();
            
    if (finalize_decomp) {
        // finalizing the decompostition
        finalize_decomposition();
            
        // simplify layers
        //self.simplify_layers();
            
        // final tuning of the decomposition parameters
        //final_optimalization();
/*
        // calculating the final error of the decomposition
        self.decomposition_error = LA.norm(matrix_new*np.exp(np.complex(0,-np.angle(matrix_new[0,0]))) - np.identity(len(matrix_new))*abs(matrix_new[0,0]), 2)
            
        // get the number of gates used in the decomposition
        gates_num = self.get_gate_nums()
        print( 'In the decomposition with error = ' + str(self.decomposition_error) + ' were used ' + str(self.layer_num) + ' layers with '  + str(gates_num['u3']) + ' U3 operations and ' + str(gates_num['cnot']) + ' CNOT gates.' )        
  */   

        printf("--- In total %f seconds elapsed during the decomposition ---\n", float(time(NULL) - start_time));
    }


}


////
// @brief stores the calculated parameters and operations of the sub-decomposition processes
// @param cSub_decomposition An instance of class Sub_Two_Qubit_Decomposition used to disentangle qubit pairs from the others.
// @param qbits_reordered A permutation of qubits that was applied on the initial unitary in prior of the sub decomposition.
// (This is needed to restore the correct qubit indices.)
void  N_Qubit_Decomposition::extract_subdecomposition_results( Sub_Matrix_Decomposition* cSub_decomposition ) {
                        
        // get the unitarization parameters
        double* parameters_sub_decomp = cSub_decomposition->get_optimized_parameters();
        int parameter_num_sub_decomp = cSub_decomposition->get_parameter_num();

        // adding the unitarization parameters to the ones stored in the class
        double* optimized_parameters_tmp = (double*)mkl_malloc( (parameter_num_sub_decomp+parameter_num)*sizeof(double), 64 );
        memcpy(optimized_parameters_tmp, parameters_sub_decomp, parameter_num_sub_decomp*sizeof(double));
        if ( optimized_parameters != NULL ) {
            memcpy(optimized_parameters_tmp+parameter_num_sub_decomp, optimized_parameters, parameter_num*sizeof(double));
            mkl_free( optimized_parameters );
        }
        
        optimized_parameters = optimized_parameters_tmp;
        optimized_parameters_tmp = NULL;


        // cloning the operation list obtained during the subdecomposition
        std::vector<Operation*> sub_decomp_ops = cSub_decomposition->get_operations();
        int operation_num = cSub_decomposition->get_operation_num();

        for ( int idx = operation_num-1; idx >=0; idx--) {
            Operation* op = sub_decomp_ops[idx];

            if (op->get_type().compare("cnot")==0) {
                CNOT* cnot_op = static_cast<CNOT*>( op );
                CNOT* cnot_op_cloned = cnot_op->clone();
                cnot_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( cnot_op_cloned );
                add_operation_to_front( op_cloned );  
            }
            else if (op->get_type().compare("u3")==0) {
                U3* u3_op = static_cast<U3*>( op );
                U3* u3_op_cloned = u3_op->clone();
                u3_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( u3_op_cloned );
                add_operation_to_front( op_cloned ); 
            }
            else if (op->get_type().compare("block")==0) {
                Operation_block* block_op = static_cast<Operation_block*>( op );
                Operation_block* block_op_cloned = block_op->clone();
                block_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( block_op_cloned );
                add_operation_to_front( op_cloned );       
            }


        }

}


    
////
// @brief Start the decompostion process to disentangle the submatrices
void  N_Qubit_Decomposition::decompose_submatrix() {
        
        if (decomposition_finalized) {
            printf("Decomposition was already finalized\n");
            return;
        }

        if (qbit_num == 2) {
            return;
        }
                       
        
        // obtaining the subdecomposed submatrices
        MKL_Complex16* subdecomposed_mtx = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
//print_mtx( subdecomposed_mtx, matrix_size, matrix_size);
        
        // get the most unitary submatrix
        // get the number of 2qubit submatrices
        int submatrices_num_row = 2;
        
        // get the size of the submatrix
        int submatrix_size = int(matrix_size/2);
        
        // fill up the submatrices and select the most unitary submatrix
        
        MKL_Complex16* most_unitary_submatrix = (MKL_Complex16*)mkl_malloc( submatrix_size*submatrix_size*sizeof(MKL_Complex16), 64 );
        double unitary_error_min = 1e8;

        for (int idx=0; idx<submatrices_num_row; idx++) { // in range(0,submatrices_num_row):
            for (int jdx=0; jdx<submatrices_num_row; jdx++) { // in range(0,submatrices_num_row):

                MKL_Complex16* submatrix_prod = (MKL_Complex16*)mkl_malloc( submatrix_size*submatrix_size*sizeof(MKL_Complex16), 64 );
                MKL_Complex16* submatrix = (MKL_Complex16*)mkl_malloc( submatrix_size*submatrix_size*sizeof(MKL_Complex16), 64 );

                for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {
                    int matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
                    int submatrix_offset = row_idx*submatrix_size;
                    memcpy(submatrix+submatrix_offset, subdecomposed_mtx+matrix_offset, submatrix_size*sizeof(MKL_Complex16));
                }

                
//print_mtx( submatrix, submatrix_size, submatrix_size);


                // parameters alpha and beta for the cblas_zgemm3m function
                double alpha = 1;
                double beta = 0;

                // calculate the product of submatrix*submatrix'
                cblas_zgemm3m (CblasRowMajor, CblasNoTrans, CblasConjTrans, submatrix_size, submatrix_size, submatrix_size, &alpha, submatrix, submatrix_size, submatrix, submatrix_size, &beta, submatrix_prod, submatrix_size);

//print_mtx( submatrix_prod, submatrix_size, submatrix_size);
                // subtract corner element
                MKL_Complex16 corner_element = submatrix_prod[0];
                for (int row_idx=0; row_idx<submatrix_size; row_idx++) {
                    submatrix_prod[row_idx*submatrix_size+row_idx].real = submatrix_prod[row_idx*submatrix_size+row_idx].real - corner_element.real;
                    submatrix_prod[row_idx*submatrix_size+row_idx].imag = submatrix_prod[row_idx*submatrix_size+row_idx].imag - corner_element.imag;
                }
//print_mtx( submatrix_prod, submatrix_size, submatrix_size);



                double unitary_error = cblas_dznrm2( submatrix_size*submatrix_size, submatrix_prod, 1 );
//printf("%f\n", unitary_error);
                if (unitary_error < unitary_error_min) {
                    unitary_error_min = unitary_error;                    
                    memcpy(most_unitary_submatrix, submatrix, submatrix_size*submatrix_size*sizeof(MKL_Complex16));
                }

                mkl_free(submatrix);
                mkl_free(submatrix_prod);
            }
        }


//print_mtx( most_unitary_submatrix, submatrix_size, submatrix_size);
                
                    
        // if the qubit number in the submatirx is greater than 2 new N-qubit decomposition is started

        // use optimization of the layer numbers only for 3qubits
        bool optimize_layer_num_loc;
        if (submatrix_size==8) {
            optimize_layer_num_loc = true;
        }
        else {
            optimize_layer_num_loc = false;
        }

        N_Qubit_Decomposition* cdecomposition = new N_Qubit_Decomposition(most_unitary_submatrix, qbit_num-1, max_layer_num, identical_blocks, optimize_layer_num_loc, initial_guess);


        // Maximal number of iteartions in the optimalization process
        cdecomposition->set_max_iteration(max_iterations);
            
        // setting operation layer
        cdecomposition->set_optimalization_blocks( optimalization_block );

        // starting the decomposition of the random unitary
        cdecomposition->start_decomposition(false);
              
                
        // saving the decomposition operations
        extract_subdecomposition_results( reinterpret_cast<Sub_Matrix_Decomposition*>(cdecomposition) );

        delete cdecomposition;
        
}

