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
/*! \file Decomposition_Base.cpp
    \brief Class containing basic methods for the decomposition process.
*/

#include "Decomposition_Base.h"


// default layer numbers
std::map<int,int> Decomposition_Base::max_layer_num_def;


/** Nullary constructor of the class
@return An instance of the class
*/
Decomposition_Base::Decomposition_Base() {

    Init_max_layer_num();

    // Logical variable. Set true for verbose mode, or to false to suppress output messages.
    verbose = true;

    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;

    // A string describing the type of the class
    type = DECOMPOSITION_BASE_CLASS;

    // error of the unitarity of the final decomposition
    decomposition_error = -1;

    // number of finalizing (deterministic) opertaions counted from the top of the array of operations
    finalizing_operations_num = 0;

    // the number of the finalizing (deterministic) parameters counted from the top of the optimized_parameters list
    finalizing_parameter_num = 0;

    // The current minimum of the optimization problem
    current_minimum = 1e10;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;

    // number of iteratrion loops in the finale optimization
    //iteration_loops = dict()

    // The maximal allowed error of the optimization problem
    optimization_tolerance = 1e-7;

    // Maximal number of iteartions in the optimization process
    max_iterations = 1e8;

    // number of operators in one sub-layer of the optimization process
    optimization_block = 1;

    // method to guess initial values for the optimization. Possible values: ZEROS, RANDOM, CLOSE_TO_ZERO (default)
    initial_guess = ZEROS;

    // optimized parameters
    optimized_parameters = NULL;


#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif

}


/** Contructor of the class
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary to be decomposed.
@param initial_guess_in Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return An instance of the class
*/
Decomposition_Base::Decomposition_Base( Matrix Umtx_in, int qbit_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Operation_block(qbit_num_in) {

    Init_max_layer_num();

    // Logical variable. Set true for verbose mode, or to false to suppress output messages.
    verbose = true;

    // the unitary operator to be decomposed
    Umtx = Umtx_in;

    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;

    // A string describing the type of the class
    type = DECOMPOSITION_BASE_CLASS;

    // error of the unitarity of the final decomposition
    decomposition_error = -1;

    // number of finalizing (deterministic) opertaions counted from the top of the array of operations
    finalizing_operations_num = 0;

    // the number of the finalizing (deterministic) parameters counted from the top of the optimized_parameters list
    finalizing_parameter_num = 0;

    // The current minimum of the optimization problem
    current_minimum = 1e10;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;

    // number of iteratrion loops in the finale optimization
    //iteration_loops = dict()

    // The maximal allowed error of the optimization problem
    optimization_tolerance = 1e-7;

    // Maximal number of iteartions in the optimization process
    max_iterations = 1e8;

    // number of operators in one sub-layer of the optimization process
    optimization_block = 1;

    // method to guess initial values for the optimization. Possible values: ZEROS, RANDOM, CLOSE_TO_ZERO (default)
    initial_guess = initial_guess_in;

    // optimized parameters
    optimized_parameters = NULL;


#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif

}

/**
@brief Destructor of the class
*/
Decomposition_Base::~Decomposition_Base() {

    if (optimized_parameters != NULL ) {
        qgd_free( optimized_parameters );
        optimized_parameters = NULL;
    }

}


/**
@brief Call to set the number of operation blocks to be optimized in one shot
@param optimization_block_in The number of operation blocks to be optimized in one shot
*/
void Decomposition_Base::set_optimization_blocks( int optimization_block_in) {
    optimization_block = optimization_block_in;
}

/**
@brief Call to set the maximal number of the iterations in the optimization process
@param max_iterations_in maximal number of iteartions in the optimization process
*/
void Decomposition_Base::set_max_iteration( int max_iterations_in) {
    max_iterations = max_iterations_in;
}


/**
@brief After the main optimization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed operations are added to the array of operations needed to the decomposition of the input unitary.
*/
void Decomposition_Base::finalize_decomposition() {

        // get the transformed matrix resulted by the operations in the list
        Matrix transformed_matrix = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );

        // preallocate the storage for the finalizing parameters
        finalizing_parameter_num = 3*qbit_num;
        double* finalizing_parameters = (double*)qgd_calloc(finalizing_parameter_num,sizeof(double), 64);

        // obtaining the final operations of the decomposition
        Operation_block* finalizing_operations = new Operation_block( qbit_num );;
        Matrix final_matrix = get_finalizing_operations( transformed_matrix, finalizing_operations, finalizing_parameters );

        // adding the finalizing operations to the list of operations
        // adding the opeartion block to the operations
        add_operation_to_front( finalizing_operations );
// TODO: use memcpy
        double* optimized_parameters_tmp = (double*)qgd_calloc( (parameter_num),sizeof(double), 64 );
        for (int idx=0; idx < finalizing_parameter_num; idx++) {
            optimized_parameters_tmp[idx] = finalizing_parameters[idx];
        }
        for (unsigned int idx=0; idx < parameter_num-finalizing_parameter_num; idx++) {
            optimized_parameters_tmp[idx+finalizing_parameter_num] = optimized_parameters[idx];
        }
        qgd_free( optimized_parameters );
        qgd_free( finalizing_parameters);
        optimized_parameters = NULL;
        finalizing_parameters = NULL;
        optimized_parameters = optimized_parameters_tmp;
        optimized_parameters_tmp = NULL;

        finalizing_operations_num = finalizing_operations->get_operation_num();


        // indicate that the decomposition was finalized
        decomposition_finalized = true;

        // calculating the final error of the decomposition
        subtract_diag( final_matrix, final_matrix[0] );
        decomposition_error = cblas_dznrm2( matrix_size*matrix_size, (void*)final_matrix.get_data(), 1 );

        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();


        if (verbose) {
            printf( "The error of the decomposition after finalyzing operations is %f with %d layers containing %d U3 operations and %d CNOT gates.\n", decomposition_error, layer_num, gates_num.u3, gates_num.cnot );
        }

}


/**
@brief Call to print the operations decomposing the initial unitary. These operations brings the intial matrix into unity.
@param start_index The index of the first operation
*/
void Decomposition_Base::list_operations( int start_index ) {

        Operation_block::list_operations( optimized_parameters, start_index );

}



/**
@brief This method determine the operations needed to rotate the indepent qubits into the state |0>
@param mtx The unitary describing indepent qubits.  The resulting matrix is returned by this pointer
@param finalizing_operations Pointer pointig to a block of operations containing the final operations.
@param finalizing_parameters Parameters corresponding to the finalizing operations.
@return Returns with the finalized matrix
*/
Matrix Decomposition_Base::get_finalizing_operations( Matrix& mtx, Operation_block* finalizing_operations, double* finalizing_parameters  ) {


        int parameter_idx = finalizing_parameter_num-1;

        Matrix mtx_tmp = mtx.copy();


        double Theta, Lambda, Phi;
        for (int target_qbit=0;  target_qbit<qbit_num; target_qbit++ ) {

            // get the base indices of the taget qubit states |0>, where all other qubits are in state |0>
            int state_0 = 0;

            // get the base indices of the taget qubit states |1>, where all other qubits are in state |0>
            int state_1 = Power_of_2(target_qbit);

            // finalize the 2x2 submatrix with z-y-z rotation
            QGD_Complex16 element00 = mtx[state_0*matrix_size+state_0];
            QGD_Complex16 element01 = mtx[state_0*matrix_size+state_1];
            QGD_Complex16 element10 = mtx[state_1*matrix_size+state_0];
            QGD_Complex16 element11 = mtx[state_1*matrix_size+state_1];

            // finalize the 2x2 submatrix with z-y-z rotation
            double cos_theta_2 = sqrt(element00.real*element00.real + element00.imag*element00.imag)/sqrt(element00.real*element00.real + element00.imag*element00.imag + element01.real*element01.real + element01.imag*element01.imag);
            Theta = 2*acos( cos_theta_2 );

            if ( sqrt(element00.real*element00.real + element00.imag*element00.imag) < 1e-7 ) {
                Phi = atan2(element10.imag, element10.real); //np.angle( submatrix[1,0] )
                Lambda = atan2(-element01.imag, -element01.real); //np.angle( -submatrix[0,1] )
            }
            else if ( sqrt(element10.real*element10.real + element10.imag*element10.imag) < 1e-7 ) {
                Phi = 0;
                Lambda = atan2(element11.imag*element00.real - element11.real*element00.imag, element11.real*element00.real + element11.imag*element00.imag); //np.angle( element11*np.conj(element00))
            }
            else {
                Phi = atan2(element10.imag*element00.real - element10.real*element00.imag, element10.real*element00.real + element10.imag*element00.imag); //np.angle( element10*np.conj(element00))
                Lambda = atan2(-element01.imag*element00.real + element01.real*element00.imag, -element01.real*element00.real - element01.imag*element00.imag); //np.angle( -element01*np.conj(element00))
            }

            double parameters_loc[3];
            parameters_loc[0] = Theta;
            parameters_loc[1] = M_PI-Lambda;
            parameters_loc[2] = M_PI-Phi;

            U3* u3_loc = new U3( qbit_num, target_qbit, true, true, true);

            // adding the new operation to the list of finalizing operations
            finalizing_parameters[parameter_idx] = M_PI-Phi; //np.concatenate((parameters_loc, finalizing_parameters))
            parameter_idx--;
            finalizing_parameters[parameter_idx] = M_PI-Lambda; //np.concatenate((parameters_loc, finalizing_parameters))
            parameter_idx--;
            finalizing_parameters[parameter_idx] = Theta; //np.concatenate((parameters_loc, finalizing_parameters))
            parameter_idx--;

            finalizing_operations->add_operation_to_front( u3_loc );
            // get the new matrix


            Matrix u3_mtx = u3_loc->get_matrix(parameters_loc);

            Matrix tmp2 = apply_operation( u3_mtx, mtx_tmp);
            mtx_tmp = tmp2;

        }


        return mtx_tmp;


}




/**
@brief This method can be used to solve the main optimization problem which is divided into sub-layer optimization processes. (The aim of the optimization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute optimized_parameters.
@param solution_guess An array of the guessed parameters
@param solution_guess_num The number of guessed parameters. (not necessarily equal to the number of free parameters)
*/
void  Decomposition_Base::solve_optimization_problem( double* solution_guess, int solution_guess_num ) {


        if ( operations.size() == 0 ) {
            return;
        }

        // array containing minimums to check convergence of the solution
        int min_vec_num = 20;
        double minimum_vec[min_vec_num];
        for ( int idx=0; idx<min_vec_num; idx++) {
            minimum_vec[idx] = 0;
        }

        // setting the initial value for the current minimum
        current_minimum = 1e8;

        // store the operations
        std::vector<Operation*> operations_loc = operations;



        // store the number of parameters
        int parameter_num_loc = parameter_num;

        // store the initial unitary to be decomposed
        Matrix Umtx_loc = Umtx; // copy??

        // storing the initial computational parameters
        int optimization_block_loc = optimization_block;

        // initialize random seed:
        srand (time(NULL));

        // the array storing the optimized parameters
        gsl_vector* optimized_parameters_gsl = gsl_vector_alloc (parameter_num_loc);

        // preparing solution guess for the iterations
        if ( initial_guess == ZEROS ) {
            for(unsigned int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters_gsl->data[idx] = 0;
            }
        }
        else if ( initial_guess == RANDOM ) {
            for(unsigned int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters_gsl->data[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
            }
        }
        else if ( initial_guess == CLOSE_TO_ZERO ) {
            for(unsigned int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters_gsl->data[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
            }
        }
        else {
            printf("bad value for initial guess\n");
            exit(-1);
        }

        if ( solution_guess_num > 0) {
            memcpy(optimized_parameters_gsl->data + parameter_num-solution_guess_num, solution_guess, solution_guess_num*sizeof(double));
        }

        // starting number of operation block applied prior to the optimalized operation blocks
        int pre_operation_parameter_num = 0;

        // defining temporary variables for iteration cycles
        int block_idx_end;
        unsigned int block_idx_start = operations.size();
        operations.clear();
        int block_parameter_num;
        Matrix operations_mtx_pre;
        Operation* fixed_operation_post = new Operation( qbit_num );
        std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> operations_mtxs_post;

        // the identity matrix used in the calculations
        Matrix Identity =  create_identity( matrix_size );


        gsl_vector *solution_guess_gsl = NULL;


        //measure the time for the decomposition
        clock_t start_time = time(NULL);

        ////////////////////////////////////////
        // Start the iterations
        int iter_idx;
        for ( iter_idx=0;  iter_idx<max_iterations+1; iter_idx++) {

            //determine the range of blocks to be optimalized togedther
            block_idx_end = block_idx_start - optimization_block;
            if (block_idx_end < 0) {
                block_idx_end = 0;
            }

            // determine the number of free parameters to be optimized
            block_parameter_num = 0;
            for ( int block_idx=block_idx_start-1; block_idx>=block_idx_end; block_idx--) {
                block_parameter_num = block_parameter_num + operations_loc[block_idx]->get_parameter_num();
            }

            // ***** get the fixed operations applied before the optimized operations *****
            if (block_idx_start < operations_loc.size() ) {
                std::vector<Operation*>::iterator fixed_operations_pre_it = operations.begin() + 1;
                Matrix tmp = get_transformed_matrix(optimized_parameters, fixed_operations_pre_it, operations.size()-1, operations_mtx_pre);
                operations_mtx_pre = tmp; // copy?
            }
            else {
                operations_mtx_pre = Identity.copy();
            }

            // Transform the initial unitary upon the fixed pre-optimization operations
            Umtx = apply_operation(operations_mtx_pre, Umtx_loc);

            // clear the operation list used in the previous iterations
            operations.clear();

            if (optimized_parameters != NULL ) {
                qgd_free( optimized_parameters );
                optimized_parameters = NULL;
            }


            // ***** get the fixed operations applied after the optimized operations *****
            // create a list of post operations matrices
            if (block_idx_start == operations_loc.size() ) {
                // matrix of the fixed operations aplied after the operations to be varied
                double* fixed_parameters_post = optimized_parameters_gsl->data;
                std::vector<Operation*>::iterator fixed_operations_post_it = operations_loc.begin();

                operations_mtxs_post = get_operation_products(fixed_parameters_post, fixed_operations_post_it, block_idx_end);
            }

            // Create a general operation describing the cumulative effect of gates following the optimized operations
            if (block_idx_end > 0) {
                fixed_operation_post->set_matrix( operations_mtxs_post[block_idx_end-1] );
            }
            else {
                // release operation products
                operations_mtxs_post.clear();
                fixed_operation_post->set_matrix( Identity );
            }

            // create a list of operations for the optimization process
            operations.push_back( fixed_operation_post );
            for ( unsigned int idx=block_idx_end; idx<block_idx_start; idx++ ) {
                operations.push_back( operations_loc[idx] );
            }


            // constructing solution guess for the optimization
            parameter_num = block_parameter_num;
            if ( solution_guess_gsl == NULL ) {
                solution_guess_gsl = gsl_vector_alloc (parameter_num);
            }
            else if ( parameter_num != solution_guess_gsl->size ) {
                gsl_vector_free(solution_guess_gsl);
                solution_guess_gsl = gsl_vector_alloc (parameter_num);
            }
            memcpy( solution_guess_gsl->data, optimized_parameters_gsl->data+parameter_num_loc - pre_operation_parameter_num - block_parameter_num, parameter_num*sizeof(double) );

            // solve the optimization problem of the block
            solve_layer_optimization_problem( parameter_num, solution_guess_gsl  );

            // add the current minimum to the array of minimums and calculate the mean
            double minvec_mean = 0;
            for (int idx=min_vec_num-1; idx>0; idx--) {
                minimum_vec[idx] = minimum_vec[idx-1];
                minvec_mean = minvec_mean + minimum_vec[idx-1];
            }
            minimum_vec[0] = current_minimum;
            minvec_mean = minvec_mean + current_minimum;
            minvec_mean = minvec_mean/min_vec_num;

            // store the obtained optimalized parameters for the block
            memcpy( optimized_parameters_gsl->data+parameter_num_loc - pre_operation_parameter_num-block_parameter_num, optimized_parameters, parameter_num*sizeof(double) );

            if (block_idx_end == 0) {
                block_idx_start = operations_loc.size();
                pre_operation_parameter_num = 0;
            }
            else {
                block_idx_start = block_idx_start - optimization_block;
                pre_operation_parameter_num = pre_operation_parameter_num + block_parameter_num;
            }


            // optimization result is displayed in each 500th iteration
            if (iter_idx % 500 == 0) {
                if (verbose) {
                    printf("The minimum with %d layers after %d iterations is %e calculated in %f seconds\n", layer_num, iter_idx, current_minimum, float(time(NULL) - start_time));
                    fflush(stdout);
                }
                start_time = time(NULL);
            }


            // calculate the variance of the last 10 minimums
            double minvec_std = sqrt(gsl_stats_variance_m( minimum_vec, 1, min_vec_num, minvec_mean));

            // conditions to break the iteration cycles
            if (abs(minvec_std/minimum_vec[min_vec_num-1]) < optimization_tolerance ) {
                if (verbose) {
                    printf("The iterations converged to minimum %e after %d iterations with %d layers\n", current_minimum, iter_idx, layer_num  );
                    fflush(stdout);
                }
                break;
            }
            else if (check_optimization_solution()) {
                if (verbose) {
                    printf("The minimum with %d layers after %d iterations is %e\n", layer_num, iter_idx, current_minimum);
                }
                break;
            }


        }


        if (iter_idx == max_iterations ) {
            if (verbose) {
                printf("Reached maximal number of iterations\n\n");
            }
        }

        // restoring the parameters to originals
        optimization_block = optimization_block_loc;

        // store the result of the optimization
        operations.clear();
        operations = operations_loc;

        parameter_num = parameter_num_loc;
        if (optimized_parameters != NULL ) {
            qgd_free( optimized_parameters );
        }

        optimized_parameters = (double*)qgd_calloc(parameter_num,sizeof(double), CACHELINE);
        memcpy( optimized_parameters, optimized_parameters_gsl->data, parameter_num*sizeof(double) );


        // free unnecessary resources
        gsl_vector_free(optimized_parameters_gsl);
        gsl_vector_free(solution_guess_gsl);
        optimized_parameters_gsl = NULL;
        solution_guess_gsl = NULL;

        delete(fixed_operation_post);

        // restore the original unitary
        Umtx = Umtx_loc; // copy?


}



/**
@brief Abstarct function to be used to solve a single sub-layer optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
@param 'num_of_parameters' The number of free parameters to be optimized
@param solution_guess_gsl A GNU Scientific Libarary vector containing the free parameters to be optimized.
*/
void Decomposition_Base::solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) {
    return;
}




/**
@brief This is an abstact definition of function giving the cost functions measuring the entaglement of the qubits. When the qubits are indepent, teh cost function should be zero.
@param parameters An array of the free parameters to be optimized. (The number of the free paramaters should be equal to the number of parameters in one sub-layer)
*/
double Decomposition_Base::optimization_problem( const double* parameters ) {
        return current_minimum;
}





/** check_optimization_solution
@brief Checks the convergence of the optimization problem.
@return Returns with true if the target global minimum was reached during the optimization process, or false otherwise.
*/
bool Decomposition_Base::check_optimization_solution() {

        return (abs(current_minimum - global_target_minimum) < optimization_tolerance);

}


/**
@brief Call to retrive a pointer to the unitary to be transformed
@return Return with the unitary Umtx
*/
Matrix Decomposition_Base::get_Umtx() {
    return Umtx;
}


/**
@brief Call to get the size of the unitary to be transformed
@return Return with the size N of the unitary NxN
*/
int Decomposition_Base::get_Umtx_size() {
    return matrix_size;
}

/**
@brief Call to get the optimized parameters.
@return Return with the pointer pointing to the array storing the optimized parameters
*/
double* Decomposition_Base::get_optimized_parameters() {
    double *ret = (double*)qgd_calloc( parameter_num,sizeof(double), 64);
    get_optimized_parameters( ret );
    return ret;
}

/**
@brief Call to get the optimized parameters.
@param ret Preallocated array to store the optimized parameters.
*/
void Decomposition_Base::get_optimized_parameters( double* ret ) {
    memcpy(ret, optimized_parameters, parameter_num*sizeof(double));
    return;
}

/**
@brief Calculate the transformed matrix resulting by an array of operations on the matrix Umtx
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the first operation to be applied on the initial matrix.
@param num_of_operations The number of operations to be applied on the initial matrix
@return Returns with the transformed matrix.
*/
Matrix
Decomposition_Base::get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations ) {

    return get_transformed_matrix( parameters, operations_it, num_of_operations, Umtx );
}


/**
@brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the first operation to be applied on the initial matrix.
@param num_of_operations The number of operations to be applied on the initial matrix
@param initial_matrix The initial matrix wich is transformed by the given operations. (by deafult it is set to the attribute Umtx)
@return Returns with the transformed matrix.
*/
Matrix
Decomposition_Base::get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations, Matrix& initial_matrix ) {

    if (num_of_operations==0) {
        return initial_matrix.copy();
    }

    // The matrix of the transformed matrix
    Matrix Operation_product;

    // The matrix to be returned
    Matrix ret_matrix;

    for (int idx=0; idx<num_of_operations; idx++) {

        // The current operation
        Operation* operation = *operations_it;

        // The matrix of the current operation
        Matrix operation_mtx;

        if (operation->get_type() == CNOT_OPERATION ) {
            CNOT* cnot_operation = static_cast<CNOT*>( operation );
            operation_mtx = cnot_operation->get_matrix();
        }
        else if (operation->get_type() == GENERAL_OPERATION ) {
            operation_mtx = operation->get_matrix();
        }
        else if (operation->get_type() == U3_OPERATION ) {
            U3* u3_operation = static_cast<U3*>( operation );
            int parameters_num = u3_operation->get_parameter_num();
            operation_mtx = u3_operation->get_matrix( parameters );
            parameters = parameters + parameters_num;
        }
        else if (operation->get_type() == BLOCK_OPERATION ) {
            Operation_block* block_operation = static_cast<Operation_block*>( operation );
            int parameters_num = block_operation->get_parameter_num();
            operation_mtx = block_operation->get_matrix( parameters );
            parameters = parameters + parameters_num;
        }

        if ( idx == 0 ) {
            Operation_product = operation_mtx; //copy?
        }
        else {
            ret_matrix = dot( Operation_product, operation_mtx );
            Operation_product = ret_matrix; // copy?

        }

        operations_it++;
    }

    ret_matrix = apply_operation( Operation_product, initial_matrix );

    return ret_matrix;



}

/**
@brief Calculate the decomposed matrix resulted by the effect of the optimized operations on the unitary Umtx
@return Returns with the decomposed matrix.
*/
Matrix Decomposition_Base::get_decomposed_matrix() {

        return get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
}

/**
@brief Apply an operations on the input matrix
@param operation_mtx The matrix of the operation.
@param input_matrix The input matrix to be transformed.
@return Returns with the transformed matrix
*/
Matrix
Decomposition_Base::apply_operation( Matrix& operation_mtx, Matrix& input_matrix ) {

    // Getting the transformed state upon the transformation given by operation
    return dot( operation_mtx, input_matrix );

}

/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param n The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param max_layer_num_in The maximal number of the operation layers used in the subdecomposition.
@return Returns with 0 if succeded.
*/
int Decomposition_Base::set_max_layer_num( int n, int max_layer_num_in ) {

    std::map<int,int>::iterator key_it = max_layer_num.find( n );

    if ( key_it != max_layer_num.end() ) {
        max_layer_num.erase( key_it );
    }

    max_layer_num.insert( std::pair<int, int>(n,  max_layer_num_in) );

    return 0;

}


/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param max_layer_num_in An <int,int> map containing the maximal number of the operation layers used in the subdecomposition.
@return Returns with 0 if succeded.
*/
int Decomposition_Base::set_max_layer_num( std::map<int, int> max_layer_num_in ) {


    for ( std::map<int,int>::iterator it = max_layer_num_in.begin(); it!=max_layer_num_in.end(); it++) {
        set_max_layer_num( it->first, it->second );
    }

    return 0;

}


/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Decomposition_Base::reorder_qubits( std::vector<int>  qbit_list) {

    Operation_block::reorder_qubits( qbit_list );

    // now reorder the unitary to be decomposed

    // obtain the permutation indices of the matrix rows/cols
    std::vector<int> perm_indices;
    perm_indices.reserve(matrix_size);

    for (int idx=0; idx<matrix_size; idx++) {
        int row_idx=0;

        // get the binary representation of idx
        std::vector<int> bin_rep;
        bin_rep.reserve(qbit_num);
        for (int i = 1 << (qbit_num-1); i > 0; i = i / 2) {
            (idx & i) ? bin_rep.push_back(1) : bin_rep.push_back(0);
        }

        // determine the permutation row index
        for (int jdx=0; jdx<qbit_num; jdx++) {
            row_idx = row_idx + bin_rep[qbit_num-1-qbit_list[jdx]]*Power_of_2(qbit_num-1-jdx);
        }
        perm_indices.push_back(row_idx);
    }

/*
    for (auto it=qbit_list.begin(); it!=qbit_list.end(); it++) {
        std::cout << *it;
    }
    std::cout << std::endl;

    for (auto it=perm_indices.begin(); it!=perm_indices.end(); it++) {
        std::cout << *it << std::endl;
    }
*/

    // reordering the matrix elements
    Matrix reordered_mtx = Matrix(matrix_size, matrix_size);
    for (int row_idx = 0; row_idx<matrix_size; row_idx++) {
        for (int col_idx = 0; col_idx<matrix_size; col_idx++) {
            int index_reordered = perm_indices[row_idx]*Umtx.rows + perm_indices[col_idx];
            int index_umtx = row_idx*Umtx.rows + col_idx;
            reordered_mtx[index_reordered] = Umtx[index_umtx];
        }
    }

    Umtx = reordered_mtx;
}



/**
@brief Set the number of iteration loops during the subdecomposition of the n-th qubit.
@param n The number of qubits for which number of iteration loops should be used in the subdecomposition.,
@param iteration_loops_in The number of iteration loops in each sted of the subdecomposition.
@return Returns with 0 if succeded.
*/
int Decomposition_Base::set_iteration_loops( int n, int iteration_loops_in ) {

    std::map<int,int>::iterator key_it = iteration_loops.find( n );

    if ( key_it != iteration_loops.end() ) {
        iteration_loops.erase( key_it );
    }

    iteration_loops.insert( std::pair<int, int>(n,  iteration_loops_in) );

    return 0;

}


/**
@brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
@param iteration_loops_in An <int,int> map containing the number of iteration loops for the individual subdecomposition processes
@return Returns with 0 if succeded.
*/
int Decomposition_Base::set_iteration_loops( std::map<int, int> iteration_loops_in ) {

    for ( std::map<int,int>::iterator it=iteration_loops_in.begin(); it!= iteration_loops_in.end(); it++ ) {
        set_iteration_loops( it->first, it->second );
    }

    return 0;

}



/**
@brief Initializes default layer numbers
*/
void Decomposition_Base::Init_max_layer_num() {

    // default layer numbers
    max_layer_num_def[2] = 3;
    max_layer_num_def[3] = 16;
    max_layer_num_def[4] = 60;
    max_layer_num_def[5] = 240;
    max_layer_num_def[6] = 1350;
    max_layer_num_def[7] = 7000;//6180;

}



/**
@brief Call to prepare the optimized operations to export. The operations are stored in the attribute operations
*/
void Decomposition_Base::prepare_operations_to_export() {

    std::vector<Operation*> operations_tmp = prepare_operations_to_export( operations, optimized_parameters );

    // release the operations and replace them with the ones prepared to export
    operations.clear();
    operations = operations_tmp;

}



/**
@brief Call to prepare the optimized operations to export
@param ops A list of operations
@param parameters The parameters of the operations
@return Returns with a list of CNOT and U3 operations.
*/
std::vector<Operation*> Decomposition_Base::prepare_operations_to_export( std::vector<Operation*> ops, const double* parameters ) {

    std::vector<Operation*> ops_ret;
    int parameter_idx = 0;


    for(std::vector<Operation*>::iterator it = ops.begin(); it != ops.end(); it++) {

        Operation* operation = *it;

        if (operation->get_type() == CNOT_OPERATION) {
            ops_ret.push_back( operation );
        }
        else if (operation->get_type() == U3_OPERATION) {

            // definig the U3 parameters
            double vartheta;
            double varphi;
            double varlambda;

            // get the inverse parameters of the U3 rotation

            U3* u3_operation = static_cast<U3*>(operation);

            if ((u3_operation->get_parameter_num() == 1) && u3_operation->is_theta_parameter()) {
                vartheta = std::fmod( parameters[parameter_idx], 4*M_PI);
                varphi = 0;
                varlambda =0;
                parameter_idx = parameter_idx + 1;

            }
            else if ((u3_operation->get_parameter_num() == 1) && u3_operation->is_phi_parameter()) {
                vartheta = 0;
                varphi = std::fmod( parameters[ parameter_idx ], 2*M_PI);
                varlambda =0;
                parameter_idx = parameter_idx + 1;
            }
            else if ((u3_operation->get_parameter_num() == 1) && u3_operation->is_lambda_parameter()) {
                vartheta = 0;
                varphi =  0;
                varlambda = std::fmod( parameters[ parameter_idx ], 2*M_PI);
                parameter_idx = parameter_idx + 1;
            }
            else if ((u3_operation->get_parameter_num() == 2) && u3_operation->is_theta_parameter() && u3_operation->is_phi_parameter() ) {
                vartheta = std::fmod( parameters[ parameter_idx ], 4*M_PI);
                varphi = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                varlambda = 0;
                parameter_idx = parameter_idx + 2;
            }
            else if ((u3_operation->get_parameter_num() == 2) && u3_operation->is_theta_parameter() && u3_operation->is_lambda_parameter() ) {
                vartheta = std::fmod( parameters[ parameter_idx ], 4*M_PI);
                varphi = 0;
                varlambda = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                parameter_idx = parameter_idx + 2;
            }
            else if ((u3_operation->get_parameter_num() == 2) && u3_operation->is_phi_parameter() && u3_operation->is_lambda_parameter() ) {
                vartheta = 0;
                varphi = std::fmod( parameters[ parameter_idx], 2*M_PI);
                varlambda = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                parameter_idx = parameter_idx + 2;
            }
            else if ((u3_operation->get_parameter_num() == 3)) {
                vartheta = std::fmod( parameters[ parameter_idx ], 4*M_PI);
                varphi = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                varlambda = std::fmod( parameters[ parameter_idx+2 ], 2*M_PI);
                parameter_idx = parameter_idx + 3;
            }
            else {
                printf("wrong parameters in U3 class\n");
                exit(-1);
            }

            u3_operation->set_optimized_parameters( vartheta, varphi, varlambda );
            ops_ret.push_back( static_cast<Operation*>(u3_operation) );


        }
        else if (operation->get_type() == BLOCK_OPERATION) {
            Operation_block* block_operation = static_cast<Operation_block*>(operation);
            const double* parameters_layer = parameters + parameter_idx;

            std::vector<Operation*> ops_loc = prepare_operations_to_export(block_operation, parameters_layer);
            parameter_idx = parameter_idx + block_operation->get_parameter_num();

            ops_ret.insert( ops_ret.end(), ops_loc.begin(), ops_loc.end() );
        }

    }


    return ops_ret;


}


/**
@brief Call to prepare the operations of an operation block to export
@param block_op A pointer to a block of operations
@param parameters The parameters of the operations
@return Returns with a list of CNOT and U3 operations.
*/
std::vector<Operation*> Decomposition_Base::prepare_operations_to_export( Operation_block* block_op, const double* parameters ) {

    std::vector<Operation*> ops_tmp = block_op->get_operations();
    std::vector<Operation*> ops_ret = prepare_operations_to_export( ops_tmp, parameters );

    return ops_ret;

}


/**
@brief Call to prepare the optimized operations to export --- OBSOLETE
@param n Integer labeling the n-th oepration  (n>=0).
@param type The type of the operation from enumeration operation_type is returned via this parameter.
@param target_qbit The ID of the target qubit is returned via this input parameter.
@param control_qbit The ID of the control qubit is returned via this input parameter.
@param parameters The parameters of the operations
@return Returns with 0 if the export of the n-th operation was successful. If the n-th operation does not exists, -1 is returned. If the operation is not allowed to be exported, i.e. it is not a CNOT or U3 operation, then -2 is returned.
*/
int Decomposition_Base::get_operation( unsigned int n, operation_type &type, int &target_qbit, int &control_qbit, double* parameters ) {

//printf("n: %d\n", n);
    // get the n-th operation if exists
    if ( n >= operations.size() ) {
        return -1;
    }

    Operation* operation = operations[n];
//printf("operation type: %d\n", operation->get_type());


    if (operation->get_type() == CNOT_OPERATION) {
        type = operation->get_type();
        target_qbit = operation->get_target_qbit();
        control_qbit = operation->get_control_qbit();
        memset( parameters, 0, 3*sizeof(double) );
        return 0;
    }
    else if (operation->get_type() == U3_OPERATION) {
        U3* u3_operation = static_cast<U3*>(operation);
        type = u3_operation->get_type();
        target_qbit = u3_operation->get_target_qbit();
        control_qbit = operation->get_control_qbit();
        u3_operation->get_optimized_parameters(parameters);
//printf("c %f, %f, %f\n", parameters[0], parameters[1], parameters[2] );
        return 0;
    }
    else {
        return -2;
    }

}



/**
@brief Call to prepare the optimized operations to export
@param n Integer labeling the n-th oepration  (n>=0).
@return Returns with a pointer to the n-th Operation, or with MULL if the n-th operation cant be retrived.
*/
Operation* Decomposition_Base::get_operation( int n ) {

    // get the n-th operation if exists
    if ( (unsigned int) n >= operations.size() ) {
        return NULL;
    }

    return operations[n];

}


/**
@brief Call to set the verbose attribute to true or false.
@param verbose_in Logical variable. Set true for verbose mode, or to false to suppress output messages.
*/
void Decomposition_Base::set_verbose( bool verbose_in ) {

    verbose = verbose_in;

}


/**
@brief Call to set the tolerance of the optimization processes.
@param tolerance_in The value of the tolerance. The error of the decomposition would scale with the square root of this value.
*/
void Decomposition_Base::set_optimization_tolerance( double tolerance_in ) {

    optimization_tolerance = tolerance_in;
    return;
}


/**
@brief Call to get the error of the decomposition
@return Returns with the error of the decomposition
*/
double Decomposition_Base::get_decomposition_error( ) {

    return decomposition_error;

}




