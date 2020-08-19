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
// @param max_layer_num_in A map of <int n: int num> indicating that how many layers should be used in the subdecomposition process at the subdecomposing of n-th qubits.
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
// @return An instance of the class
Two_Qubit_Decomposition::Two_Qubit_Decomposition( MKL_Complex16* Umtx_in, int qbit_num_in, std::map<int,int> max_layer_num_in, bool optimize_layer_num_in=false, string initial_guess_in="close_to_zero" ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {
        
    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = "Two_Qubit_Decomposition";
        
    // The global minimum of the optimalization problem
    global_target_minimum = 0;
        
    // number of iteratrion loops in the optimalization
    iteration_loops[2] = 1;
    
    // number of operators in one sub-layer of the optimalization process
    optimalization_block = 1;

    // layer number used in the decomposition
    max_layer_num = max_layer_num_in;
    if ( max_layer_num.count( 2 ) == 0 ) {
        max_layer_num.insert( std::pair<int, int>(2,  max_layer_num_def[2]) );
    }
    


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
                optimalization_block = 1;//layer_num;
                
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
    
        
////
// @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
// @param 'solution_guess' Array of guessed parameters
// @param 'num_of_parameters' NUmber of free parameters to be optimized
void Two_Qubit_Decomposition::solve_layer_optimalization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) { 


/////////////////////////////////////////
/*
        if (optimized_parameters == NULL) {
            optimized_parameters = (double*)mkl_malloc(num_of_parameters*sizeof(double), 64);
        }

for (int idx=0; idx<num_of_parameters; idx++  ) {
optimized_parameters[idx] = solution_guess_gsl->data[idx];

}

return;*/
///////////////////////////////////////


        if (operations.size() == 0 ) {
            return;
        }
       
          
        if (solution_guess_gsl == NULL) {
            //solution_guess = (double*)mkl_calloc(parameter_num, sizeof(double), 64);
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }
        
        if (optimized_parameters == NULL) {
            optimized_parameters = (double*)mkl_malloc(num_of_parameters*sizeof(double), 64);
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

            Two_Qubit_Decomposition* par = this;

            
            gsl_multimin_function_fdf my_func;


            my_func.n = num_of_parameters;
            my_func.f = optimalization_problem;
            my_func.df = optimalization_problem_grad;
            my_func.fdf = optimalization_problem_combined;
            my_func.params = par;

            
            T = gsl_multimin_fdfminimizer_vector_bfgs2;
            s = gsl_multimin_fdfminimizer_alloc (T, num_of_parameters);


            gsl_multimin_fdfminimizer_set (s, &my_func, solution_guess_gsl, 0.01, 0.1);

            do {
                iter++;
                status = gsl_multimin_fdfminimizer_iterate (s);
                if (status) {
                  break;
                }

                status = gsl_multimin_test_gradient (s->gradient, 1e-3);
                /*if (status == GSL_SUCCESS) {
                    printf ("Minimum found\n");
                }*/

            } while (status == GSL_CONTINUE && iter < 10000);
       
                        
            if (current_minimum > s->f) {
                current_minimum = s->f;
                #pragma omp parallel for
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = s->x->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
                    optimized_parameters[jdx] = s->x->data[jdx];
                }
            }
            else {
                #pragma omp parallel for
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
double Two_Qubit_Decomposition::optimalization_problem( const double* parameters ) {

        // get the transformed matrix with the operations in the list
        MKL_Complex16* matrix_new = get_transformed_matrix( parameters, operations.begin(), operations.size(), Umtx );
        

        double cost_function = get_submatrix_cost_function(matrix_new, matrix_size);

        // free the allocated matrix and returning with the cost function
        if ( matrix_new != Umtx ) {
            mkl_free( matrix_new );              
        }
        return cost_function;            
}               

//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double Two_Qubit_Decomposition::optimalization_problem( const gsl_vector* parameters, void* params ) {

    Two_Qubit_Decomposition* instance = reinterpret_cast<Two_Qubit_Decomposition*>(params);
    std::vector<Operation*> operations_loc = instance->get_operations(); 

    MKL_Complex16* matrix_new = instance->get_transformed_matrix( parameters->data, operations_loc.begin(), operations_loc.size(), instance->get_Umtx() );

    double cost_function = get_submatrix_cost_function(matrix_new, instance->get_Umtx_size());  

    // free the allocated matrix and returning with the cost function
    if ( matrix_new != instance->get_Umtx() ) {
        mkl_free( matrix_new );              
    }     

    return cost_function;        
}  



//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
double Two_Qubit_Decomposition::optimalization_problem_deriv( double x, void* params  ) {
    
    deriv* params_diff = reinterpret_cast<deriv*>(params);
    Two_Qubit_Decomposition* instance = reinterpret_cast<Two_Qubit_Decomposition*>(params_diff->instance);


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
void Two_Qubit_Decomposition::optimalization_problem_grad( const gsl_vector* parameters, void* params, gsl_vector* grad ) {

    Two_Qubit_Decomposition* instance = reinterpret_cast<Two_Qubit_Decomposition*>(params);

    double f0 = instance->optimalization_problem(parameters, params);

    // the difference in one direction in the parameter for the gradient calculaiton
    double dparam = 1e-8;

    // the displaced parameters by dparam
    double* parameters_d;


    int parameter_num_loc = instance->get_parameter_num();

    parameters_d = parameters->data;

    // calculate the gradient components
    if ( grad == NULL ) {
        grad = gsl_vector_alloc(parameter_num_loc);
    }

/*
    for (int idx = 0; idx<parameter_num_loc; idx++) {
        gsl_function F;
        double result, abserr;
        deriv params_diff;
        params_diff.idx = idx;
        params_diff.parameters = parameters;
        params_diff.instance = params;

        F.function = instance->optimalization_problem_deriv;
        F.params = &params_diff;
        gsl_deriv_central (&F, parameters_d[idx], 1e-8, &result, &abserr);
        gsl_vector_set(grad, idx, result);
   }
*/
/*
printf("f0: %f\n", f0);
f0 = instance->optimalization_problem(parameters, params);
printf("f0: %f\n", f0);

printf("%d parameters:", parameter_num_loc);
for (int idx = 0; idx<parameter_num_loc; idx++) {
printf("%f, ", parameters_d[idx]);
}
printf("\n");
printf("Derivates check:");*/
    for (int idx = 0; idx<parameter_num_loc; idx++) {
        parameters_d[idx] = parameters_d[idx] + dparam;
        double f = instance->optimalization_problem(parameters, params);
        gsl_vector_set(grad, idx, (f-f0)/dparam);
//printf("%f, ", (f-f0)/dparam);
        parameters_d[idx] = parameters_d[idx] - dparam;
    }
//printf("\n");

//f0 = instance->optimalization_problem(parameters, params);
//printf("f0: %f\n", f0);

//throw "jjjjjjjjjjj";
          

}     


//
// @brief The optimalization problem to be solved in order to disentangle the qubits
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
// @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
// @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
void Two_Qubit_Decomposition::optimalization_problem_combined( const gsl_vector* parameters, void* params, double* cost_function, gsl_vector* grad ) {
    *cost_function = optimalization_problem(parameters, params);
    optimalization_problem_grad(parameters, params, grad);
}                                        
                                        
                    

    
//// 
// @brief Check whether qubits are indepent or not
// @returns Return with true if qubits are disentangled, or false otherwise.
bool Two_Qubit_Decomposition::test_indepency() {
       
        current_minimum = optimalization_problem( optimized_parameters );
        
        return check_optimalization_solution();     
        
}        
   
/*      
double Two_Qubit_Decomposition::_evaluate( void *instance, const double *x, double *g, const int n, const double step ) {
    return reinterpret_cast<Two_Qubit_Decomposition*>(instance)->evaluate(x, g, n, step);
}


int Two_Qubit_Decomposition::_progress(void *instance, const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls) {
    return reinterpret_cast<Two_Qubit_Decomposition*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
}*/
