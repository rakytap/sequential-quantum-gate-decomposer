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
/*! \file N_Qubit_Decomposition.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of CNOT and U3 operations.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition.h"


/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_Base::N_Qubit_Decomposition_Base( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Decomposition_Base(Umtx_in, qbit_num_in, initial_guess_in) {

    // logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS_BASE;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // number of iteratrion loops in the optimization
    iteration_loops[2] = 3;

    // filling in numbers that were not given in the input
    for ( std::map<int,int>::iterator it = max_layer_num_def.begin(); it!=max_layer_num_def.end(); it++) {
        if ( max_layer_num.count( it->first ) == 0 ) {
            max_layer_num.insert( std::pair<int, int>(it->first,  it->second) );
        }
    }


}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Base::~N_Qubit_Decomposition_Base() {

}


/**
@brief final optimization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
*/
void  N_Qubit_Decomposition_Base::final_optimization() {

        if (verbose) {
            printf("***************************************************************\n");
            printf("Final fine tuning of the parameters in the %d-qubit decomposition\n", qbit_num);
            printf("***************************************************************\n");
        }


        //# setting the global minimum
        global_target_minimum = 0;
        solve_optimization_problem( optimized_parameters, parameter_num) ;
}



/**
// @brief Call to solve layer by layer the optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
// @param num_of_parameters Number of parameters to be optimized
// @param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) {

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
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }

        // do the optimization loops
        for (int idx=0; idx<iteration_loops_max; idx++) {

            size_t iter = 0;
            int status;

            const gsl_multimin_fdfminimizer_type *T;
            gsl_multimin_fdfminimizer *s;

            N_Qubit_Decomposition_Base* par = this;


            gsl_multimin_function_fdf my_func;


            my_func.n = num_of_parameters;
            my_func.f = optimization_problem;
            my_func.df = optimization_problem_grad;
            my_func.fdf = optimization_problem_combined;
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

            if (current_minimum > s->f) {
                current_minimum = s->f;
                memcpy( optimized_parameters, s->x->data, num_of_parameters*sizeof(double) );
                gsl_multimin_fdfminimizer_free (s);

                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
                }
            }
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
                }
                gsl_multimin_fdfminimizer_free (s);
            }



        }


}



/**
// @brief The optimization problem of the final optimization
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( const double* parameters ) {

        // get the transformed matrix with the operations in the list
        Matrix matrix_new = get_transformed_matrix( parameters, operations.begin(), operations.size(), Umtx );

        double cost_function = get_cost_function(matrix_new);

        return cost_function;
}


/**
// @brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( const gsl_vector* parameters, void* void_instance ) {

    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);
    std::vector<Operation*> operations_loc = instance->get_operations();

    // get the transformed matrix with the operations in the list
    Matrix Umtx_loc = instance->get_Umtx();
    Matrix matrix_new = instance->get_transformed_matrix( parameters->data, operations_loc.begin(), operations_loc.size(), Umtx_loc );

    double cost_function = get_cost_function(matrix_new);

    return cost_function;
}





/**
@brief Set the number of identical successive blocks during the subdecomposition of the n-th qubit.
@param n The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param identical_blocks_in The number of successive identical layers used in the subdecomposition.
@return Returns with zero in case of success.
*/
int N_Qubit_Decomposition_Base::set_identical_blocks( int n, int identical_blocks_in )  {

    std::map<int,int>::iterator key_it = identical_blocks.find( n );

    if ( key_it != identical_blocks.end() ) {
        identical_blocks.erase( key_it );
    }

    identical_blocks.insert( std::pair<int, int>(n,  identical_blocks_in) );

    return 0;

}



/**
@brief Set the number of identical successive blocks during the subdecomposition of the n-th qubit.
@param identical_blocks_in An <int,int> map containing the number of successive identical layers used in the subdecompositions.
@return Returns with zero in case of success.
*/
int N_Qubit_Decomposition_Base::set_identical_blocks( std::map<int, int> identical_blocks_in )  {

    for ( std::map<int,int>::iterator it=identical_blocks_in.begin(); it!= identical_blocks_in.end(); it++ ) {
        set_identical_blocks( it->first, it->second );
    }

    return 0;

}


