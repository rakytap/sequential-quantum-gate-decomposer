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
#include "qgd/Decomposition_Base.h"

 
// default layer numbers
std::map<int,int> Decomposition_Base::max_layer_num_def;    


/** Contructor of the class
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary to be decomposed.
@param initial_guess_in Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return An instance of the class
*/
Decomposition_Base::Decomposition_Base( QGD_Complex16* Umtx_in, int qbit_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Operation_block(qbit_num_in) {
        
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

    // auxiliary variable storing the transformed matrix
    transformed_mtx = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

    // The number of threads for the parallel optimization (The remaining threads are used for nested parallelism at matrix multiplications)
    num_threads = 1;

}

/** 
@brief Destructor of the class
*/
Decomposition_Base::~Decomposition_Base() {

    if (optimized_parameters != NULL ) {
        qgd_free( optimized_parameters );
        optimized_parameters = NULL;
    }

    if (transformed_mtx != NULL ) {
        qgd_free( transformed_mtx );
        transformed_mtx = NULL;
    }

/*    if (m_x != NULL) {
        lbfgs_free(m_x);
        m_x = NULL;
    }*/
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
        QGD_Complex16* transformed_matrix = get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );

        // preallocate the storage for the finalizing parameters
        finalizing_parameter_num = 3*qbit_num;
        double* finalizing_parameters = (double*)qgd_calloc(finalizing_parameter_num,sizeof(double), 64);

        // obtaining the final operations of the decomposition
        Operation_block* finalizing_operations = new Operation_block( qbit_num );;
        get_finalizing_operations( transformed_matrix, finalizing_operations, finalizing_parameters );
            
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


        // indicat that the decomposition was finalized    
        decomposition_finalized = true;

        // calculating the final error of the decomposition
        //decomposition_error = LA.norm(matrix_new*np.exp(np.complex(0,-np.angle(matrix_new[0,0]))) - np.identity(len(matrix_new))*abs(matrix_new[0,0]), 2)
        subtract_diag( transformed_matrix, matrix_size, transformed_matrix[0] );
        decomposition_error = cblas_dznrm2( matrix_size*matrix_size, transformed_matrix, 1 );
        //qgd_free( finalized_matrix_new );
            
        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();

        qgd_free( transformed_matrix );
        transformed_matrix = NULL;

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
*/
void Decomposition_Base::get_finalizing_operations( QGD_Complex16* mtx, Operation_block* finalizing_operations, double* finalizing_parameters  ) {
              
        
        int parameter_idx = finalizing_parameter_num-1;   

        QGD_Complex16* mtx_tmp = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
        memcpy( mtx_tmp, mtx, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        QGD_Complex16* mtx_tmp2 = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
        QGD_Complex16* u3_mtx = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

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

//printf("Decomposition_Base::get_finalizing_operations 1\n");
//print_mtx(mtx, matrix_size, matrix_size );                  
            

            memset(u3_mtx, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );
            u3_loc->matrix(parameters_loc, u3_mtx);
            //QGD_Complex16* u3_mtx = u3_loc->matrix(parameters_loc);
//printf("Decomposition_Base::get_finalizing_operations umtx\n");
//print_mtx(u3_mtx, matrix_size, matrix_size );       
            apply_operation( u3_mtx, mtx_tmp, mtx_tmp2);
            //qgd_free( u3_mtx );
 

            memcpy( mtx_tmp, mtx_tmp2, matrix_size*matrix_size*sizeof(QGD_Complex16) );
//printf("Decomposition_Base::get_finalizing_operations 2\n");
//print_mtx(mtx, matrix_size, matrix_size );            
        }         

        memcpy( mtx, mtx_tmp, matrix_size*matrix_size*sizeof(QGD_Complex16) );

        qgd_free( mtx_tmp );
        qgd_free( mtx_tmp2 );
        qgd_free( u3_mtx );
        mtx_tmp = NULL;
        mtx_tmp2 = NULL;
        u3_mtx = NULL;



//printf("Decomposition_Base::get_finalizing_operations 3\n");
//print_mtx(mtx, matrix_size, matrix_size );   
        return;
            
        
}                
    
   

    
/** 
@brief This method can be used to solve the main optimization problem which is devidid into sub-layer optimization processes. (The aim of the optimization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute optimized_parameters.
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
        QGD_Complex16* Umtx_loc = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
        memcpy(Umtx_loc, Umtx, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        
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
        QGD_Complex16* operations_mtx_pre = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
        QGD_Complex16* tmp = NULL;
        Operation* fixed_operation_post = new Operation( qbit_num );
        std::vector<QGD_Complex16*> operations_mtxs_post;

        // the identity matrix used in the calcualtions
        QGD_Complex16* Identity =  (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
        create_identity( Identity, matrix_size );

        gsl_vector *solution_guess_gsl = NULL;

        
        //measure the time for the decompositin        
        clock_t start_time = time(NULL);

        int iter_idx;
        for ( iter_idx=0;  iter_idx<max_iterations+1; iter_idx++) {                        

            //determine the range of blocks to be optimalized togedther
            block_idx_end = block_idx_start - optimization_block;
            if (block_idx_end < 0) {
                block_idx_end = 0;
            }

            // determine the number of free parameters to be optimized
            block_parameter_num = 0;
            for ( int block_idx=block_idx_start-1; block_idx>=block_idx_end; block_idx--) { //for block_idx in range(block_idx_start-1,block_idx_end-1,-1):
                block_parameter_num = block_parameter_num + operations_loc[block_idx]->get_parameter_num();
            }

            // ***** get the fixed operations applied before the optimized operations *****
            if (block_idx_start < operations_loc.size() ) { //if block_idx_start < len(operations):
                std::vector<Operation*>::iterator fixed_operations_pre_it = operations.begin() + 1;
                tmp = get_transformed_matrix(optimized_parameters, fixed_operations_pre_it, operations.size()-1, operations_mtx_pre);  
                memcpy( operations_mtx_pre, tmp, matrix_size*matrix_size*sizeof(QGD_Complex16) );
                qgd_free(tmp); 
                tmp = NULL;
                //QGD_Complex16* operations_mtx_pre_tmp = get_transformed_matrix(optimized_parameters, fixed_operations_pre_it, operations.size()-1, operations_mtx_pre );  
                //qgd_free( operations_mtx_pre );
                //operations_mtx_pre = operations_mtx_pre_tmp;
            }
            else {
                memcpy( operations_mtx_pre, Identity, matrix_size*matrix_size*sizeof(QGD_Complex16) );
            }

//print_mtx(operations_mtx_pre, matrix_size, matrix_size );
//////////////////////////
//operations_mtx_pre = Identity;
/////////////////////

            // Transform the initial unitary upon the fixed pre-optimization operations
            apply_operation(operations_mtx_pre, Umtx_loc, Umtx);


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
//printf("operations_mtxs_post with prefixed:\n");
//print_mtx(operations_mtxs_post[block_idx_end-1], matrix_size, matrix_size); 
            }
            else {
                // release operation products
                for (std::vector<QGD_Complex16*>::iterator mtxs_it=operations_mtxs_post.begin(); mtxs_it != operations_mtxs_post.end(); mtxs_it++ ) {
                    qgd_free( *mtxs_it );
                    *mtxs_it = NULL;
                }
                operations_mtxs_post.clear();
                fixed_operation_post->set_matrix( Identity );
            }


////////////////////////
//fixed_operation_post->set_matrix( Identity );
////////////////////
            
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
            
            // the convergence at low minimums is much faster if only one layer is considered in the optimization at once
            if ( current_minimum < 1 ) {
                optimization_block = 1;
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
            optimized_parameters = (double*)qgd_calloc(parameter_num,sizeof(double), 64);
        }

        memcpy( optimized_parameters, optimized_parameters_gsl->data, parameter_num*sizeof(double) );
     
  
        // free unnecessary resources
        gsl_vector_free(optimized_parameters_gsl);
        gsl_vector_free(solution_guess_gsl);
        optimized_parameters_gsl = NULL;
        solution_guess_gsl = NULL;

        QGD_Complex16* operations_mtx_post = fixed_operation_post->matrix();

        // Identity can be freed if operations_mtx_post and operations_mtx_pre are not equal to identity
        if ( operations_mtx_post!=Identity ) {
            qgd_free(Identity);
        }
        Identity = NULL;
        

        // release preoperation matrix
        qgd_free( operations_mtx_pre );
        operations_mtx_pre = NULL;

        // release post operation products
        for (std::vector<QGD_Complex16*>::iterator mtxs_it=operations_mtxs_post.begin(); mtxs_it != operations_mtxs_post.end(); mtxs_it++ ) {
            qgd_free( *mtxs_it );
            *mtxs_it = NULL;
        }
        operations_mtxs_post.clear();
 

        delete(fixed_operation_post);

        // restore the original unitary
        memcpy(Umtx, Umtx_loc, matrix_size*matrix_size*sizeof(QGD_Complex16)) ;

        // free the allocated temporary Umtx
        qgd_free(Umtx_loc);
        Umtx_loc = NULL;

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
@brief Calculate the list of gate operation matrices such that the i>0-th element in the result list is the product of the operations of all 0<=n<i operations from the input list and the 0th element in the result list is the identity.
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the forst operation.
@param num_of_operations The number of operations involved in the calculations
@return Returns with a vector of the product matrices.
*/
std::vector<QGD_Complex16*> Decomposition_Base::get_operation_products(double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations) {


    // construct the list of matrix representation of the gates
    std::vector<QGD_Complex16*> operation_mtxs;
    // preallocate memory for the operation products
    operation_mtxs.reserve(num_of_operations);

    for ( int idx=0; idx<num_of_operations; idx++) {
        operation_mtxs.push_back( (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size,sizeof(QGD_Complex16), 64) );
    }

    QGD_Complex16* operation_mtx = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size,sizeof(QGD_Complex16), 64);

    for (int idx=0; idx<num_of_operations; idx++) {

        Operation* operation = *operations_it;

        if (operation->get_type() == CNOT_OPERATION ) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            cnot_operation->matrix(operation_mtx);
        }
        else if (operation->get_type() == GENERAL_OPERATION ) {
            operation->matrix(operation_mtx);
        }
        else if (operation->get_type() == U3_OPERATION ) {
            U3* u3_operation = static_cast<U3*>(operation);
            u3_operation->matrix(parameters, operation_mtx);
            parameters = parameters + u3_operation->get_parameter_num();
        }
        else if (operation->get_type() == BLOCK_OPERATION ) {
            Operation_block* block_operation = static_cast<Operation_block*>(operation);
            block_operation->matrix(parameters, operation_mtx);
            parameters = parameters + block_operation->get_parameter_num();
        }

        if (idx == 0) {          
            memcpy( operation_mtxs[idx], operation_mtx, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        }
        else {
            apply_operation(operation_mtxs[idx-1], operation_mtx, operation_mtxs[idx]);
        }

     
        operations_it++;
    }

    if (operation_mtxs.size()==0) {
        create_identity(operation_mtx, matrix_size);
        operation_mtxs.push_back( operation_mtx );
    }
    else {
        qgd_free(operation_mtx); 
        operation_mtx = NULL;
    }

    return operation_mtxs;

}

/**
@brief Call to retrive a pointer to the unitary to be transformed
@return Return with a pointer pointing to the unitary Umtx
*/
QGD_Complex16* Decomposition_Base::get_Umtx() {
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
@brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
@param parameters An array containing the parameters of the U3 operations.
@param operations_it An iterator pointing to the first operation to be applied on the initial matrix.
@param num_of_operations The number of operations to be applied on the initial matrix
@param initial_matrix The initial matrix wich is transformed by the given operations. (by deafult it is set to the attribute Umtx)
@return Returns with the transformed matrix (ehich is also stored in the attribute transformed_mtx).
*/
QGD_Complex16* Decomposition_Base::get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations, QGD_Complex16* initial_matrix=NULL  ) {

    //QGD_Complex16* ret_matrix = transformed_mtx;
    // auxiliary variable storing the transformed matrix
    QGD_Complex16* ret_matrix = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);


        if (initial_matrix == NULL) {
            initial_matrix = Umtx;
        }

        if (num_of_operations==0) {
            memcpy(ret_matrix, initial_matrix, matrix_size*matrix_size*sizeof(QGD_Complex16) );
            return ret_matrix;
        }
     
        // The matrix of the current operation
        QGD_Complex16* operation_mtx;    
        // The matrix of the transformed matrix
        QGD_Complex16* Operation_product = NULL;

        operation_mtx = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size,sizeof(QGD_Complex16), 64);
        Operation_product = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size,sizeof(QGD_Complex16), 64);

        for (int idx=0; idx<num_of_operations; idx++) {

            Operation* operation = *operations_it;     

            if (operation->get_type() == CNOT_OPERATION ) {
                CNOT* cnot_operation = static_cast<CNOT*>( operation );
                cnot_operation->matrix(operation_mtx);
            }
            else if (operation->get_type() == GENERAL_OPERATION ) {
                operation->matrix(operation_mtx);
            }                                
            else if (operation->get_type() == U3_OPERATION ) {
                U3* u3_operation = static_cast<U3*>( operation );
                int parameters_num = u3_operation->get_parameter_num();
                u3_operation->matrix( parameters, operation_mtx );
                parameters = parameters + parameters_num;
            }
            else if (operation->get_type() == BLOCK_OPERATION ) {
                Operation_block* block_operation = static_cast<Operation_block*>( operation );
                int parameters_num = block_operation->get_parameter_num();
                block_operation->matrix( parameters, operation_mtx );
                parameters = parameters + parameters_num;
            }
/*if (verbose) {
printf("Decomposition_Base::get_transformed_matrix 4\n");
print_mtx( operation_mtx, matrix_size, matrix_size );
print_mtx( Operation_product, matrix_size, matrix_size );
}*/

            if ( idx == 0 ) {
                //Operation_product = operation_mtx;
                memcpy( Operation_product, operation_mtx, matrix_size*matrix_size*sizeof(QGD_Complex16) );
            }
            else {
                zgemm3m_wrapper( Operation_product, operation_mtx, ret_matrix, matrix_size );
/*if (verbose) {
printf("Decomposition_Base::get_transformed_matrix 4b\n");
print_mtx( ret_matrix, matrix_size, matrix_size );
}*/
                memcpy( Operation_product, ret_matrix, matrix_size*matrix_size*sizeof(QGD_Complex16) );

            }

            operations_it++;
        }
/*if (verbose) {
printf("Decomposition_Base::get_transformed_matrix 5\n");
print_mtx( Operation_product, matrix_size, matrix_size );
printf("Decomposition_Base::get_transformed_matrix 5b\n");
print_mtx( ret_matrix, matrix_size, matrix_size );
printf("Decomposition_Base::get_transformed_matrix 5c\n");
print_mtx( initial_matrix, matrix_size, matrix_size );
//throw "jjj";
}*/
        apply_operation( Operation_product, initial_matrix, ret_matrix );
        qgd_free( Operation_product );
        qgd_free( operation_mtx );
        Operation_product = NULL;
        operation_mtx = NULL;

//printf("Decomposition_Base::get_transformed_matrix 6\n");
//print_mtx( ret_matrix, matrix_size, matrix_size );

        return ret_matrix;
}    
    
    
/**
@brief Calculate the decomposed matrix resulted by the effect of the optimized operations on the unitary Umtx
@return Returns with the decomposed matrix.
*/
QGD_Complex16* Decomposition_Base::get_decomposed_matrix() {
     
        return get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
}
        

/**
@brief Apply an operations on the input matrix
@param operation_mtx The matrix of the operation.
@param input_matrix The input matrix to be transformed.
@return Returns with the transformed matrix
*/
QGD_Complex16* Decomposition_Base::apply_operation( QGD_Complex16* operation_mtx, QGD_Complex16* input_matrix ) {

    // Getting the transformed state upon the transformation given by operation
    return zgemm3m_wrapper( operation_mtx, input_matrix, matrix_size);
}

/**
@brief Apply an operations on the input matrix
@param operation_mtx The matrix of the operation.
@param input_matrix The input matrix to be transformed.
@param result_matrix The result is returned via this matrix
*/
int Decomposition_Base::apply_operation( QGD_Complex16* operation_mtx, QGD_Complex16* input_matrix,  QGD_Complex16* result_matrix) {

    // Getting the transformed state upon the transformation given by operation
    return zgemm3m_wrapper( operation_mtx, input_matrix, result_matrix, matrix_size);
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
@brief Call to prepare the optimized operations to export
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
@brief Call to set the verbose attribute to true or false.
@param verbose_in Logical variable. Set true for verbose mode, or to false to suppress output messages.
*/
void Decomposition_Base::set_verbose( bool verbose_in ) {

    verbose = verbose_in;

}


/**
@brief Call to get the error of the decomposition
@return Returns with the error of the decomposition
*/
double Decomposition_Base::get_decomposition_error( ) {

    return decomposition_error;

}


/**
@brief Call to set the number of threads for the parallel optimization (The remaining threads are used for nested parallelism at matrix multiplications)
@param num_threads_in the number of threads for the parallel optimization
*/
void Decomposition_Base::set_num_threads_optimization( int num_threads_in ) {

    num_threads = num_threads_in;
}


/**
@brief Call to get the number of threads for the parallel optimization (The remaining threads are used for nested parallelism at matrix multiplications)
@return Returns with the number of threads for the parallel optimization
*/
int Decomposition_Base::get_num_threads_optimization() {

    return num_threads;
}
