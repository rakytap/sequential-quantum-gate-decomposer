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

#include "qgd/Sub_Matrix_Decomposition.h"

 
    


//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param max_layer_num_in A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process at the subdecomposing of n-th qubits.
// @param identical_blocks_in A map of <int n: int num> indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
Sub_Matrix_Decomposition::Sub_Matrix_Decomposition( QGD_Complex16* Umtx_in, int qbit_num_in, std::map<int,int> max_layer_num_in, std::map<int,int> identical_blocks_in, bool optimize_layer_num_in=false, string initial_guess_in="close_to_zero" ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {
        
    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = "Sub_Matrix_Decomposition";
        
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

    // logical value indicating whether the quasi-unitarization of the submatrices was done or not 
    subdisentaglement_done = false;
        
    // The subunitarized matrix
    subdecomposed_mtx = NULL;
                
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
// @brief Start the optimalization process to disentangle the most significant qubit from the others. The optimized parameters and operations are stored in the attributes @optimized_parameters and @operations.
void  Sub_Matrix_Decomposition::disentangle_submatrices() {
        
    if (subdisentaglement_done) {
        printf("Sub-disentaglement already done.\n");
        return;
    }
        
     
    printf("\nDisentagling submatrices.\n");
        
    // setting the global target minimum
    global_target_minimum = 0;   
          
    // check if it needed to do the subunitarization
    if (optimalization_problem(NULL) < optimalization_tolerance) {
        printf("Disentanglig not needed\n");
        subdecomposed_mtx = Umtx;
        subdisentaglement_done = true;
        return;
    }
        
                       
        
    if ( !check_optimalization_solution() ) {
        // Adding the operations of the successive layers
            
        //measure the time for the decompositin       
        clock_t start_time = time(NULL);
            
        // the maximal number of layers in the subdeconposition
        int max_layer_num_loc;
        try {
            max_layer_num_loc = max_layer_num[qbit_num];
        }
        catch (...) {
            throw "Layer number not given";
        }


        // the  number of succeeding identicallayers in the subdeconposition
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
   
        while ( layer_num < max_layer_num_loc ) {
                
            int control_qbit_loc = qbit_num-1;
            int solution_guess_num = parameter_num;
             
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
                
            // get the number of blocks
            layer_num = operations.size();
                                
            // Do the optimalization
            if (optimize_layer_num || layer_num >= max_layer_num_loc ) {

                //solve_optimalization_problem( NULL, 0); 

                // solve the optzimalization problem to find the correct mninimum
                if ( optimized_parameters == NULL ) {
                    solve_optimalization_problem( optimized_parameters, 0);   
                }
                else {
                    solve_optimalization_problem( optimized_parameters, solution_guess_num);   
                }

                if (check_optimalization_solution()) {
                    break;
                }
            }
                    
        }
            
        printf("--- %f seconds elapsed during the decomposition ---\n\n", float(time(NULL) - start_time));
                
           
    }
                       
        
        
    if (check_optimalization_solution()) {            
        printf("Sub-disentaglement was succesfull.\n\n");
    }
    else {
        printf("Sub-disentaglement did not reach the tolerance limit.\n\n");
    }
        
        
    // indicate that the unitarization of the sumbatrices was done
    subdisentaglement_done = true;
        
    // The subunitarized matrix
    subdecomposed_mtx = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
}



   
        
////
// @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
// @param 'solution_guess' Array of guessed parameters
// @param 'num_of_parameters' NUmber of free parameters to be optimized
void Sub_Matrix_Decomposition::solve_layer_optimalization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) { 

////////////////////////////////////
//#ifdef MIC
/*
if (optimized_parameters == NULL) {
     optimized_parameters = (double*)qgd_calloc(num_of_parameters*sizeof(double), 64);
}


for (int idx =0; idx<num_of_parameters; idx++) {
    optimized_parameters[idx] = solution_guess_gsl->data[idx];
}

printf("hhhhhhhhhhhhhhhh\n");
clock_t start_time = clock();
for (int idx =0; idx<10; idx++) {
//optimalization_problem( solution_guess_gsl, this );
printf("cost function:%f\n", optimalization_problem( solution_guess_gsl->data )); 
fflush(stdout);
}
//printf("%f\n", float((clock()-start_time)));
printf("%e\n", float((clock()-start_time)/CLOCKS_PER_SEC));


start_time = clock();
for (int idx =0; idx<1; idx++) {
optimalization_problem( solution_guess_gsl, this ); 
//optimalization_problem( optimized_parameters);
}
//printf("%f\n", float((clock()-start_time)));
printf("%e\n", float((clock()-start_time)/CLOCKS_PER_SEC));

//printf("%f, diff: %e\n", optimalization_problem( solution_guess_gsl, this ), optimalization_problem( solution_guess_gsl, this )-optimalization_problem( solution_guess_gsl->data) );
printf("%f\n", optimalization_problem( solution_guess_gsl, this ) );
throw "klll";*/
//#endif
//return; 
///////////////////////////////////

        if (operations.size() == 0 ) {
            return;
        }
       
          
        if (solution_guess_gsl == NULL) {
            //solution_guess = (double*)qgd_calloc(parameter_num, sizeof(double), 64);
printf("solve_layer_optimalization_problem::Allocating solution guess\n");
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }
        
        if (optimized_parameters == NULL) {
            optimized_parameters = (double*)qgd_calloc(num_of_parameters,sizeof(double), 64);
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

            Sub_Matrix_Decomposition* par = this;

            
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
                    optimized_parameters[jdx] = s->x->data[jdx];
                }
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
       

    
        
//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double Sub_Matrix_Decomposition::optimalization_problem( const double* parameters ) {

        // get the transformed matrix with the operations in the list
        QGD_Complex16* matrix_new = get_transformed_matrix( parameters, operations.begin(), operations.size(), Umtx );
//printf("Sub_Matrix_Decomposition::optimalization_problem 1\n");
//print_mtx( matrix_new, matrix_size, matrix_size );
        double cost_function = get_submatrix_cost_function(matrix_new, matrix_size); //NEW METHOD
        //double cost_function = get_submatrix_cost_function_2(matrix_new, matrix_size); //OLD METHOD

        // free the allocated matrix and returning with the cost function
        if ( matrix_new != Umtx ) {
            qgd_free( matrix_new );              
        }

        return cost_function;
}               

//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double Sub_Matrix_Decomposition::optimalization_problem( const gsl_vector* parameters, void* params ) {

    Sub_Matrix_Decomposition* instance = reinterpret_cast<Sub_Matrix_Decomposition*>(params);
    std::vector<Operation*> operations_loc = instance->get_operations(); 

    QGD_Complex16* matrix_new = instance->get_transformed_matrix( parameters->data, operations_loc.begin(), operations_loc.size(), instance->get_Umtx() );

    double cost_function = get_submatrix_cost_function(matrix_new, instance->get_Umtx_size());  //NEW METHOD
    //double cost_function = get_submatrix_cost_function_2(matrix_new, instance->get_Umtx_size());  //OLD METHOD

    // free the allocated matrix and returning with the cost function
    if ( matrix_new != instance->get_Umtx() ) {
        qgd_free( matrix_new );              
    }     

    return cost_function; 
}  



//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double Sub_Matrix_Decomposition::optimalization_problem_deriv( double x, void* params  ) {
    
    deriv* params_diff = reinterpret_cast<deriv*>(params);
    Sub_Matrix_Decomposition* instance = reinterpret_cast<Sub_Matrix_Decomposition*>(params_diff->instance);


    double x_orig = params_diff->parameters->data[params_diff->idx];
    params_diff->parameters->data[params_diff->idx] = x;

    double fval = instance->optimalization_problem( params_diff->parameters, params_diff->instance );

    params_diff->parameters->data[params_diff->idx] = x_orig;

    return fval;
    

}                 

//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void Sub_Matrix_Decomposition::optimalization_problem_grad( const gsl_vector* parameters, void* params, gsl_vector* grad ) {

    Sub_Matrix_Decomposition* instance = reinterpret_cast<Sub_Matrix_Decomposition*>(params);

    double f0 = instance->optimalization_problem(parameters, params);

    optimalization_problem_grad( parameters, params, grad, f0 );

}


//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void Sub_Matrix_Decomposition::optimalization_problem_grad( const gsl_vector* parameters, void* params, gsl_vector* grad, double f0 ) {

    Sub_Matrix_Decomposition* instance = reinterpret_cast<Sub_Matrix_Decomposition*>(params);

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


//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void Sub_Matrix_Decomposition::optimalization_problem_combined( const gsl_vector* parameters, void* params, double* cost_function, gsl_vector* grad ) {
    *cost_function = optimalization_problem(parameters, params);
    optimalization_problem_grad(parameters, params, grad, *cost_function);
}                                        
   

