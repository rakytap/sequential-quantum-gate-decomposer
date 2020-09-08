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

#include "qgd/N_Qubit_Decomposition.h"

 
//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param max_layer_num_in A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process at the subdecomposing of n-th qubits.
// @param identical_blocks_in A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: ZEROS, RANDOM, CLOSE_TO_ZERO (default)
// @return An instance of the class
N_Qubit_Decomposition::N_Qubit_Decomposition( QGD_Complex16* Umtx_in, int qbit_num_in, std::map<int,int> max_layer_num_in, std::map<int,int> identical_blocks_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {
        
    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;
        
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

    // setting the iteration loops in each step of the optimization process
    cSub_decomposition->set_iteration_loops( iteration_loops );
        
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
        if (qbit_num>2) {
            simplify_layers();
        }
            
        // final tuning of the decomposition parameters
        final_optimalization();



        // calculating the final error of the decomposition
        QGD_Complex16* matrix_decomposed = get_transformed_matrix(optimized_parameters, operations.begin(), operations.size(), Umtx ); 
//print_mtx(matrix_decomposed, matrix_size, matrix_size);

        subtract_diag( matrix_decomposed, matrix_size, matrix_decomposed[0] );

        decomposition_error = cblas_dznrm2( matrix_size*matrix_size, matrix_decomposed, 1 );
            
        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();

        printf( "In the decomposition with error = %f were used %d layers with %d U3 operations and %d CNOT gates.\n", decomposition_error, layer_num, gates_num.u3, gates_num.cnot );
        printf("--- In total %f seconds elapsed during the decomposition ---\n", float(time(NULL) - start_time));

        // prepare operations to export
        //prepare_operations_to_export();
    }


}


////
// @brief stores the calculated parameters and operations of the sub-decomposition processes
// @param cSub_decomposition An instance of class Sub_Two_Qubit_Decomposition used to disentangle qubit pairs from the others.
// @param qbits_reordered A permutation of qubits that was applied on the initial unitary in prior of the sub decomposition.
// (This is needed to restore the correct qubit indices.)
void  N_Qubit_Decomposition::extract_subdecomposition_results( Sub_Matrix_Decomposition* cSub_decomposition ) {
                        
        // get the unitarization parameters
        int parameter_num_sub_decomp = cSub_decomposition->get_parameter_num();

        // adding the unitarization parameters to the ones stored in the class
        double* optimized_parameters_tmp = (double*)qgd_calloc( (parameter_num_sub_decomp+parameter_num),sizeof(double), 64 );
        cSub_decomposition->get_optimized_parameters(optimized_parameters_tmp);

        if ( optimized_parameters != NULL ) {
            memcpy(optimized_parameters_tmp+parameter_num_sub_decomp, optimized_parameters, parameter_num*sizeof(double));
            qgd_free( optimized_parameters );
        }
        
        optimized_parameters = optimized_parameters_tmp;
        optimized_parameters_tmp = NULL;


        // cloning the operation list obtained during the subdecomposition
        std::vector<Operation*> sub_decomp_ops = cSub_decomposition->get_operations();
        int operation_num = cSub_decomposition->get_operation_num();

        for ( int idx = operation_num-1; idx >=0; idx--) {
            Operation* op = sub_decomp_ops[idx];

            if (op->get_type() == CNOT_OPERATION) {
                CNOT* cnot_op = static_cast<CNOT*>( op );
                CNOT* cnot_op_cloned = cnot_op->clone();
                cnot_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( cnot_op_cloned );
                add_operation_to_front( op_cloned );  
            }
            else if (op->get_type() == U3_OPERATION) {
                U3* u3_op = static_cast<U3*>( op );
                U3* u3_op_cloned = u3_op->clone();
                u3_op_cloned->set_qbit_num( qbit_num );
                Operation* op_cloned = static_cast<Operation*>( u3_op_cloned );
                add_operation_to_front( op_cloned ); 
            }
            else if (op->get_type() == BLOCK_OPERATION) {
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
        QGD_Complex16* subdecomposed_mtx = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );

        // get the most unitary submatrix
        // get the number of 2qubit submatrices
        int submatrices_num_row = 2;
        
        // get the size of the submatrix
        int submatrix_size = int(matrix_size/2);
        
        // fill up the submatrices and select the most unitary submatrix
        
        QGD_Complex16* most_unitary_submatrix = (QGD_Complex16*)qgd_calloc( submatrix_size*submatrix_size,sizeof(QGD_Complex16), 64 );
        double unitary_error_min = 1e8;

        for (int idx=0; idx<submatrices_num_row; idx++) { // in range(0,submatrices_num_row):
            for (int jdx=0; jdx<submatrices_num_row; jdx++) { // in range(0,submatrices_num_row):

                QGD_Complex16* submatrix_prod = (QGD_Complex16*)qgd_calloc( submatrix_size*submatrix_size,sizeof(QGD_Complex16), 64 );
                QGD_Complex16* submatrix = (QGD_Complex16*)qgd_calloc( submatrix_size*submatrix_size,sizeof(QGD_Complex16), 64 );

                for ( int row_idx=0; row_idx<submatrix_size; row_idx++ ) {
                    int matrix_offset = idx*(matrix_size*submatrix_size) + jdx*(submatrix_size) + row_idx*matrix_size;
                    int submatrix_offset = row_idx*submatrix_size;
                    memcpy(submatrix+submatrix_offset, subdecomposed_mtx+matrix_offset, submatrix_size*sizeof(QGD_Complex16));
                }

                // calculate the product of submatrix*submatrix'
                zgemm3m_wrapper_adj( submatrix, submatrix, submatrix_prod, submatrix_size);

                // subtract corner element
                QGD_Complex16 corner_element = submatrix_prod[0];
                #pragma omp parallel for
                for (int row_idx=0; row_idx<submatrix_size; row_idx++) {
                    submatrix_prod[row_idx*submatrix_size+row_idx].real = submatrix_prod[row_idx*submatrix_size+row_idx].real - corner_element.real;
                    submatrix_prod[row_idx*submatrix_size+row_idx].imag = submatrix_prod[row_idx*submatrix_size+row_idx].imag - corner_element.imag;
                }
 
                double unitary_error = cblas_dznrm2( submatrix_size*submatrix_size, submatrix_prod, 1 );

                if (unitary_error < unitary_error_min) {
                    unitary_error_min = unitary_error;                    
                    memcpy(most_unitary_submatrix, submatrix, submatrix_size*submatrix_size*sizeof(QGD_Complex16));
                }

                qgd_free(submatrix);
                qgd_free(submatrix_prod);
            }
        }
                
                    
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

        // setting the iteration loops in each step of the optimization process
        cdecomposition->set_iteration_loops( iteration_loops );
            
        // setting operation layer
        cdecomposition->set_optimalization_blocks( optimalization_block );

        // starting the decomposition of the random unitary
        cdecomposition->start_decomposition(false);
              
                
        // saving the decomposition operations
        extract_subdecomposition_results( reinterpret_cast<Sub_Matrix_Decomposition*>(cdecomposition) );

        delete cdecomposition;
        qgd_free( most_unitary_submatrix );
        
}

////
// @brief final optimalization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
void  N_Qubit_Decomposition::final_optimalization() {

        printf("***************************************************************\n");
        printf("Final fine tuning of the parameters\n");
        printf("***************************************************************\n");

//printf("%f\n", optimalization_problem(optimized_parameters ) );
//QGD_Complex16* matrix_new = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
//print_mtx( matrix_new, matrix_size, matrix_size );

        //# setting the global minimum
        global_target_minimum = 0;
        solve_optimalization_problem( optimized_parameters, parameter_num) ;
//matrix_new = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
//print_mtx( matrix_new, matrix_size, matrix_size );
//printf("%f\n", optimalization_problem(optimized_parameters ) );
}   



////
// @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
// @param 'solution_guess' Array of guessed parameters
// @param 'num_of_parameters' NUmber of free parameters to be optimized
void N_Qubit_Decomposition::solve_layer_optimalization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) { 

        if (operations.size() == 0 ) {
            return;
        }
       
              
        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }
        
        if (optimized_parameters == NULL) {
            optimized_parameters = (double*)qgd_calloc(num_of_parameters,sizeof(double), 64);
            memcpy(optimized_parameters, solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }

        // maximal number of iteration loops
        int iteration_loops_max;
        try {
            iteration_loops_max = iteration_loops[qbit_num];
        }
        catch (...) {
            iteration_loops_max = 1;
        }

        // do the optimalization loops
        double* solution;

        for (int idx=0; idx<iteration_loops_max; idx++) {
            
            size_t iter = 0;
            int status;

            const gsl_multimin_fdfminimizer_type *T;
            gsl_multimin_fdfminimizer *s;

            N_Qubit_Decomposition* par = this;

            
            gsl_multimin_function_fdf my_func;


            my_func.n = num_of_parameters;
            my_func.f = optimalization_problem;
            my_func.df = optimalization_problem_grad;
            my_func.fdf = optimalization_problem_combined;
            my_func.params = par;

            
            T = gsl_multimin_fdfminimizer_vector_bfgs2;
            s = gsl_multimin_fdfminimizer_alloc (T, num_of_parameters);


            gsl_multimin_fdfminimizer_set (s, &my_func, solution_guess_gsl, 0.1, 0.1);

            do {
                iter++;

                status = gsl_multimin_fdfminimizer_iterate (s);

                if (status) {
                  break;
                }

                status = gsl_multimin_test_gradient (s->gradient, 1e-1);
                /*if (status == GSL_SUCCESS) {
                    printf ("Minimum found\n");
                }*/

            } while (status == GSL_CONTINUE && iter < 100);
       
//printf("s->f: %f\n", s->f);             
            if (current_minimum > s->f) {
                current_minimum = s->f;
                //#pragma omp parallel for
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = s->x->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
                }
                memcpy( optimized_parameters, s->x->data, num_of_parameters*sizeof(double) );
            }
            else {
                //#pragma omp parallel for
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
                }
            }

            gsl_multimin_fdfminimizer_free (s);
     
        }         

                
}      



////
// @brief The final optimalization problem to fine tune the optimized parameters obtained during the subdecomposition process
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the value representing the difference between the decomposed matrix and the unity.
double N_Qubit_Decomposition::optimalization_problem( const double* parameters ) {

        // get the transformed matrix with the operations in the list
        QGD_Complex16* matrix_new = get_transformed_matrix( parameters, operations.begin(), operations.size(), Umtx );

        double cost_function = get_cost_function(matrix_new, matrix_size); 

        return cost_function;
}               

////
// @brief The final optimalization problem to fine tune the optimized parameters obtained during the subdecomposition process
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the value representing the difference between the decomposed matrix and the unity.
double N_Qubit_Decomposition::optimalization_problem( const gsl_vector* parameters, void* params ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(params);
    std::vector<Operation*> operations_loc = instance->get_operations(); 

    QGD_Complex16* matrix_new = instance->get_transformed_matrix( parameters->data, operations_loc.begin(), operations_loc.size(), instance->get_Umtx() );

    double cost_function = get_cost_function(matrix_new, instance->get_Umtx_size()); 

//printf("%f, \n", cblas_dznrm2( instance->get_Umtx_size()*instance->get_Umtx_size(), matrix_new, 1) );
//printf("%f\n", cost_function);    

    return cost_function; 
}  

         

//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void N_Qubit_Decomposition::optimalization_problem_grad( const gsl_vector* parameters, void* params, gsl_vector* grad ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(params);

    double f0 = instance->optimalization_problem(parameters, params);

    optimalization_problem_grad( parameters, params, grad, f0 );

}


//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void N_Qubit_Decomposition::optimalization_problem_grad( const gsl_vector* parameters, void* params, gsl_vector* grad, double f0 ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(params);

    // the difference in one direction in the parameter for the gradient calculaiton
    double dparam = 1e-8;

    // the displaced parameters by dparam
    double* parameters_d;


    int parameter_num_loc = instance->get_parameter_num();

    parameters_d = parameters->data;

    // calculate the gradient components
//    if ( grad == NULL ) {
//        grad = gsl_vector_alloc(parameter_num_loc);
//    }
/*
//printf("Derivates:");
    for (int idx = 0; idx<parameter_num_loc; idx++) {
        gsl_function F;
        double result, abserr;
        deriv params_diff;
        params_diff.idx = idx;
        params_diff.parameters = parameters;
        params_diff.instance = params;

        F.function = instance->optimalization_problem_deriv;
        F.params = &params_diff;
        gsl_deriv_central (&F, parameters_d[idx], dparam, &result, &abserr);
//printf("%f, ", result);
        gsl_vector_set(grad, idx, result);
   }
//printf("\n");*/

    for (int idx = 0; idx<parameter_num_loc; idx++) {
        parameters_d[idx] = parameters_d[idx] + dparam;
        double f = instance->optimalization_problem(parameters, params);
        gsl_vector_set(grad, idx, (f-f0)/dparam);
        parameters_d[idx] = parameters_d[idx] - dparam;
    }
        

}     


////
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void N_Qubit_Decomposition::optimalization_problem_combined( const gsl_vector* parameters, void* params, double* cost_function, gsl_vector* grad ) {
    *cost_function = optimalization_problem(parameters, params);
    optimalization_problem_grad(parameters, params, grad, *cost_function);
}    


////  
// @brief Call to simplify the gate structure in each layers if possible (i.e. tries to reduce the number of CNOT gates)
void N_Qubit_Decomposition::simplify_layers() {

        printf("***************************************************************\n");
        printf("Try to simplify layers\n");
        printf("***************************************************************\n");
        
        // current starting index of the optimized parameters
        int parameter_idx = 0;
        
        std::vector<Operation*> operations_loc;
        double* optimized_parameters_loc = (double*)qgd_calloc(parameter_num, sizeof(double), 64); //????? = np.array([])
        int parameter_num_loc = 0;
        
        int layer_idx = 0;
        
        while (layer_idx < operations.size()) {
            
            // generate a block of operations to be simplified 
            // (containg only successive two-qubit operations)
            Operation_block* block_to_simplify = new Operation_block( qbit_num );
            
            std::vector<int> involved_qbits;
            // layers in the block to be simplified
            std::vector<Operation*> blocks_to_save; // = list()
            
            // get the successive operations involving the same qubits
            while (true) {
                
                if (layer_idx >= operations.size() ) {
                    break;
                }
                
                // get the current layer of operations
                Operation* operation = operations[layer_idx];
                layer_idx = layer_idx + 1;

                Operation_block* block_operation;
                if (operation->get_type() == BLOCK_OPERATION) {
                    block_operation = static_cast<Operation_block*>(operation);
                }
                else {
                    layer_idx = layer_idx + 1;
                    continue;
                }     
                
                
                //get the involved qubits
                std::vector<int> involved_qbits_op = block_operation->get_involved_qubits();
                
                // add involved qubits of the operation to the list of involved qubits in the layer
                for (std::vector<int>::iterator it=involved_qbits_op.begin(); it!=involved_qbits_op.end(); it++) {
                    add_unique_elelement( involved_qbits, *it );
                }

                if ( (involved_qbits.size())> 2 && blocks_to_save.size() > 0 ) { 
                    layer_idx = layer_idx -1;
                    involved_qbits.clear();
                    break;       
                }
                
                blocks_to_save.push_back(block_operation->clone());
                
                // adding the operations to teh block if they act on the same qubits
                block_to_simplify->combine(block_operation);                
                
                
            }                            
             
            //number of perations in the block
            int parameter_num_block = block_to_simplify->get_parameter_num();
            
            // get the number of CNOT gates and store the block operations if the number of CNOT gates cannot be reduced
            gates_num gate_nums = block_to_simplify->get_gate_nums();

            if (gate_nums.cnot < 2) {
                operations_loc.insert( operations_loc.end(), blocks_to_save.begin(), blocks_to_save.end() );
                memcpy(optimized_parameters_loc+parameter_num_loc, optimized_parameters+parameter_idx, parameter_num_block*sizeof(double) );
                parameter_idx = parameter_idx + parameter_num_block;                    
                parameter_num_loc = parameter_num_loc + parameter_num_block; 
                delete block_to_simplify;
                continue;
            }


            
            // simplify the given layer
            std::map<int,int> max_layer_num_loc;
            max_layer_num_loc.insert( std::pair<int, int>(2,  gate_nums.cnot-1 ) );
            Operation_block* simplified_layer = new Operation_block(qbit_num);
            double* simplified_parameters = (double*)qgd_calloc(parameter_num_block, sizeof(double), 64); ///optimized_parameters + parameter_idx;
            int simplified_parameter_num=0;

            // Try to simplify the sequence of 2-qubit operations
            int simplification_status = simplify_layer( block_to_simplify, optimized_parameters+parameter_idx, parameter_num_block, max_layer_num_loc, simplified_layer, simplified_parameters, simplified_parameter_num );
//int simplification_status = -1;

            
            // adding the simplified operations (or the non-simplified if the simplification was not successfull)
            if (simplification_status==0) {
                operations_loc.push_back( simplified_layer );
                memcpy(optimized_parameters_loc+parameter_num_loc, simplified_parameters, simplified_parameter_num*sizeof(double) );
                parameter_num_loc = parameter_num_loc + simplified_parameter_num; 
            }
            else {
                operations_loc.insert( operations_loc.end(), blocks_to_save.begin(), blocks_to_save.end() );
                memcpy(optimized_parameters_loc+parameter_num_loc, optimized_parameters+parameter_idx, parameter_num_block*sizeof(double) );
                parameter_num_loc = parameter_num_loc + parameter_num_block; 
                delete simplified_layer;
            }

            parameter_idx = parameter_idx + parameter_num_block;

            //optimized_parameters_loc = np.concatenate(( optimized_parameters_loc, simplified_parameters ))
            qgd_free( simplified_parameters );

            delete block_to_simplify;
        }

        // get the number of CNOT gates in the initial structure
        gates_num gate_num_initial = get_gate_nums();
        int cnot_num_initial = gate_num_initial.cnot;

        // clearing the original list of operations and parameters
        release_operations();
        qgd_free( optimized_parameters );

        // store the modified list of operations and parameters
        operations = operations_loc;
        layer_num = operations.size();

        
        optimized_parameters = optimized_parameters_loc;

        parameter_num = parameter_num_loc;

        gates_num gate_num_simplified = get_gate_nums();
        int cnot_num_simplified = gate_num_simplified.cnot;

        printf("\n\n************************************\n");
        printf("After some additional 2-qubit decompositions the initial gate structure with %d CNOT gates simplified to a structure containing %d CNOT gates.\n", cnot_num_initial, cnot_num_simplified);
        printf("************************************\n\n");

    
    
}    
    
////  
// @brief Call to simplify the gate structure (i.e. tries to reduce the number of CNOT gates) in one layer if possible
// @param layer An instance of class #Operation_block containing the 2-qubit gate structure to be simplified
// @parameters An array of parameters to calculate the matrix of the layer.
// @max_layer_num The maximal allowed number of layers containing CNOT gates
// @return An instance of class operation_block containing the simplified gate operations.       
int N_Qubit_Decomposition::simplify_layer( Operation_block* layer, double* parameters, int parameter_num_block, std::map<int,int> max_layer_num, Operation_block* simplified_layer, double* simplified_parameters, int &simplified_parameter_num) {

        printf("Try to simplify sub-structure \n");

        // get the target bit
        int target_qbit = -1;
        int control_qbit = -1;

        std::vector<Operation*> layer_operations = layer->get_operations();
        for (std::vector<Operation*>::iterator it = layer_operations.begin(); it!=layer_operations.end(); it++) {
            Operation* op = *it;
            if (op->get_type() == CNOT_OPERATION) {
                target_qbit = op->get_target_qbit();
                control_qbit = op->get_control_qbit();
                break;
            }
        }
            
        // if there are no target or control qubits, return the initial values
        if ( (target_qbit == -1) || ( control_qbit == -1 ) ) {
            return 1;            
        }
            
        // get the matrix of the two qubit space
            
        // reorder the control and target qubits to the end of the list
        vector<int> qbits_reordered;
        for (int qbit_idx=qbit_num-1; qbit_idx>-1; qbit_idx-- ) { // in range(self.qbit_num-1,-1,-1):
            if (  (qbit_idx != target_qbit) && (qbit_idx != control_qbit) )  {
                qbits_reordered.push_back(qbit_idx);
            }
        }
            
        qbits_reordered.push_back(control_qbit);
        qbits_reordered.push_back(target_qbit); 
            
        // construct abstarct operations correspond to the reordeerd qubits
        Operation_block* reordered_layer = layer->clone();
        reordered_layer->reorder_qubits( qbits_reordered );

        //  get the reordered N-qubit matrix of the layer
        QGD_Complex16* full_matrix_reordered = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64 );
        reordered_layer->matrix( parameters, full_matrix_reordered ); 
        delete reordered_layer;
 
        // construct the Two-qubit submatrix from the reordered matrix
        QGD_Complex16* submatrix = (QGD_Complex16*)qgd_calloc( 16, sizeof(QGD_Complex16), 64 );
        #pragma omp parallel for
        for ( int element_idx=0; element_idx<16; element_idx++) {
            int col_idx = element_idx % 4;
            int row_idx = int((element_idx-col_idx)/4);
            submatrix[element_idx].real = full_matrix_reordered[col_idx*matrix_size+row_idx].real;
            submatrix[element_idx].imag = -full_matrix_reordered[col_idx*matrix_size+row_idx].imag;
        }

        // decompose the chosen 2-qubit unitary
        std::map<int,int> identical_blocks_loc;
        N_Qubit_Decomposition* cdecomposition = new N_Qubit_Decomposition(submatrix, 2, max_layer_num, identical_blocks_loc, true, initial_guess); 
        
        // starting the decomposition of the random unitary
        cdecomposition->start_decomposition(true);
      
        // check whether simplification was succesfull
        if (!cdecomposition->check_optimalization_solution()) {
            // return with the original layer, if the simplification wa snot successfull
            printf("The simplification of the sub-structure was not possible\n");
            delete cdecomposition;
            qgd_free( submatrix );
            qgd_free( full_matrix_reordered );
            return 1;
        }
            
        // contruct the layer containing the simplified operations in the N-qubit space
        // but first get the inverse reordered qubit list 
        vector<int> qbits_inverse_reordered;
        for (int idx=0; idx<qbit_num; idx++) {
            qbits_inverse_reordered.push_back(-1);
        }

        for (int qbit_idx=qbit_num-1; qbit_idx>-1; qbit_idx-- ) { // in range(self.qbit_num-1,-1,-1):
            qbits_inverse_reordered[qbit_num-1-qbits_reordered[qbit_num-1-qbit_idx]] =  qbit_idx;
        }

        std::vector<Operation*> simplifying_operations = cdecomposition->get_operations();
        for ( std::vector<Operation*>::iterator it=simplifying_operations.begin(); it!=simplifying_operations.end(); it++) { //int operation_block_idx=0 in range(0,len(cdecomposition.operations)):
            Operation* op = *it;
            Operation_block* block_op = static_cast<Operation_block*>( op );
             block_op->set_qbit_num( qbit_num );
             block_op->reorder_qubits( qbits_inverse_reordered );
            simplified_layer->combine( block_op );
        }

        simplified_parameter_num = cdecomposition->get_parameter_num();
        cdecomposition->get_optimized_parameters( simplified_parameters );

//QGD_Complex16* tmp = simplified_layer->matrix(simplified_parameters);
//QGD_Complex16* tmp2 = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
//print_mtx( zgemm3m_wrapper_adj( full_matrix, tmp, tmp2, matrix_size), matrix_size, matrix_size );

        gates_num gate_nums_layer = layer->get_gate_nums();
        gates_num gate_nums_simplified = simplified_layer->get_gate_nums();
        printf("%d CNOT gates successfully simplified to %d CNOT gates\n", gate_nums_layer.cnot, gate_nums_simplified.cnot);


        //release allocated memory
        delete cdecomposition;
        qgd_free( submatrix );
        qgd_free( full_matrix_reordered );



        return 0;


}                              


////
// @brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
// @param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.
// @param identical_blocks_in The number of successive identical layers used in the subdecomposition.
int N_Qubit_Decomposition::set_identical_blocks( int qbit, int identical_blocks_in )  {

    std::map<int,int>::iterator key_it = identical_blocks.find( qbit );

    if ( key_it != identical_blocks.end() ) {
        identical_blocks.erase( key_it );
    }

    identical_blocks.insert( std::pair<int, int>(qbit,  identical_blocks_in) );    

}
