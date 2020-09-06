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

#include "qgd/Decomposition_Base.h"

 
// default layer numbers
std::map<int,int> Decomposition_Base::max_layer_num_def;    


//// Contructor of the class
// @brief Constructor of the class.
// @param Umtx The unitary matrix to be decomposed
// @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
// @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: "zeros" (deafult),"random", "close_to_zero"
// @return An instance of the class
Decomposition_Base::Decomposition_Base( QGD_Complex16* Umtx_in, int qbit_num_in, string initial_guess_in= "close_to_zero" ) : Operation_block(qbit_num_in) {
        
    Init_max_layer_num();
        
    // the unitary operator to be decomposed
    Umtx = Umtx_in;
        
    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;

    // A string describing the type of the class
    type = "Decomposition_Base";
        
    // error of the unitarity of the final decomposition
    decomposition_error = -1;
        
    // number of finalizing (deterministic) opertaions counted from the top of the array of operations
    finalizing_operations_num = 0;
        
    // the number of the finalizing (deterministic) parameters counted from the top of the optimized_parameters list
    finalizing_parameter_num = 0;
        
    // The current minimum of the optimalization problem
    current_minimum = 1e10;                       
        
    // The global minimum of the optimalization problem
    global_target_minimum = 0;
        
    // logical value describing whether the optimalization problem was solved or not
    optimalization_problem_solved = false;
        
    // number of iteratrion loops in the finale optimalization
    //iteration_loops = dict()
        
    // The maximal allowed error of the optimalization problem
    optimalization_tolerance = 1e-7;
        
    // Maximal number of iteartions in the optimalization process
    max_iterations = 1e8;
  
    // number of operators in one sub-layer of the optimalization process
    optimalization_block = 1;
        
    // method to guess initial values for the optimalization. POssible values: 'zeros', 'random', 'close_to_zero'
    initial_guess = initial_guess_in;

    // optimized parameters
    optimized_parameters = NULL;

    // auxiliary variable storing the transformed matrix
    transformed_mtx = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

verbose = false;
}

//// 
// @brief Destructor of the class
Decomposition_Base::~Decomposition_Base() {

    if (optimized_parameters != NULL ) {
        qgd_free( optimized_parameters );
        optimized_parameters = NULL;
    }

    if (transformed_mtx != NULL ) {
        qgd_free( transformed_mtx );
    }

/*    if (m_x != NULL) {
        lbfgs_free(m_x);
        m_x = NULL;
    }*/
}    
               
     
////   
// @brief Call to set the number of operation layers to optimize in one shot
// @param optimalization_block The number of operation blocks to optimize in one shot 
void Decomposition_Base::set_optimalization_blocks( int optimalization_block_in) {
    optimalization_block = optimalization_block_in;
}
        
////   
// @brief Call to set the maximal number of the iterations in the optimalization process
// @param max_iterations aximal number of iteartions in the optimalization process
void Decomposition_Base::set_max_iteration( int max_iterations_in) {
    max_iterations = max_iterations_in;  
}
    
    
//// 
// @brief After the main optimalization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed operations are added to the array of operations needed to the decomposition of the input unitary.
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
        for (int idx=0; idx < parameter_num-finalizing_parameter_num; idx++) {
            optimized_parameters_tmp[idx+finalizing_parameter_num] = optimized_parameters[idx];
        }
        qgd_free( optimized_parameters );
        qgd_free( finalizing_parameters);
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

        printf( "The error of the decomposition after finalyzing operations is %f with %d layers containing %d U3 operations and %d CNOT gates.\n", decomposition_error, layer_num, gates_num.u3, gates_num.cnot );


}
            

////
// @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
// @param start_index The index of the first inverse operation
void Decomposition_Base::list_operations( int start_index = 1 ) {
       
        Operation_block::list_operations( optimized_parameters, start_index );

}
       

                
////
// @brief This method determine the operations needed to rotate the indepent qubits into the state |0>
// @param mtx The unitary describing indepent qubits.
// @return [1] The operations needed to rotate the qubits into the state |0>
// @return [2] The parameters of the U3 operations needed to rotate the qubits into the state |0>
// @return [3] The resulted diagonalized matrix.
void Decomposition_Base::get_finalizing_operations( QGD_Complex16* mtx, Operation_block* finalizing_operations, double* finalizing_parameters  ) {
              
        
        int parameter_idx = finalizing_parameter_num-1;   

        QGD_Complex16* mtx_tmp = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
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
            apply_operation( u3_mtx, mtx, mtx_tmp);
            //qgd_free( u3_mtx );
 

            memcpy( mtx, mtx_tmp, matrix_size*matrix_size*sizeof(QGD_Complex16) );
//printf("Decomposition_Base::get_finalizing_operations 2\n");
//print_mtx(mtx, matrix_size, matrix_size );            
        }         

        qgd_free( mtx_tmp );
        qgd_free( u3_mtx );
//printf("Decomposition_Base::get_finalizing_operations 3\n");
//print_mtx(mtx, matrix_size, matrix_size );   
        return;
            
        
}                
    
        
    
   

    
//// solve_optimalization_problem
// @brief This method can be used to solve the main optimalization problem which is devidid into sub-layer optimalization processes. (The aim of the optimalization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute @optimized_parameters.
// @param solution_guess ????????????
void  Decomposition_Base::solve_optimalization_problem( double* solution_guess, int solution_guess_num ) {

       
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
        int optimalization_block_loc = optimalization_block;

        // initialize random seed:
        srand (time(NULL));

        // the array storing the optimized parameters
        gsl_vector* optimized_parameters_gsl = gsl_vector_alloc (parameter_num_loc);

        // preparing solution guess for the iterations
        if ( initial_guess.compare("zeros")==0 ) {
            #pragma omp parallel for
            for(int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters_gsl->data[idx] = 0;
            }
        }
        else if ( initial_guess.compare("random")==0 ) {
            #pragma omp parallel for
            for(int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters_gsl->data[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
            }
        }
        else if ( initial_guess.compare("close_to_zero")==0 ) {
            #pragma omp parallel for
            for(int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters_gsl->data[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI/100;
            }
        }
        else {
            printf("bad value for initial guess");
            throw "bad value for initial guess";
        }

        if ( solution_guess_num > 0) {
            memcpy(optimized_parameters_gsl->data + parameter_num-solution_guess_num, solution_guess, solution_guess_num*sizeof(double));
        }  

        // starting number of operation block applied prior to the optimalized operation blocks
        int pre_operation_parameter_num = 0;

        // defining temporary variables for iteration cycles
        int block_idx_end;
        int block_idx_start = operations.size();
        operations.clear();
        int block_parameter_num;
        QGD_Complex16* operations_mtx_pre = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
        QGD_Complex16* tmp;
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
                        
            fflush(stdout);

            //determine the range of blocks to be optimalized togedther
            block_idx_end = block_idx_start - optimalization_block;
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
                //QGD_Complex16* operations_mtx_pre_tmp = get_transformed_matrix(optimized_parameters, fixed_operations_pre_it, operations.size()-1, operations_mtx_pre );  
                //qgd_free( operations_mtx_pre );
                //operations_mtx_pre = operations_mtx_pre_tmp;
            }
            else {
                create_identity( operations_mtx_pre, matrix_size );
            }

//print_mtx(operations_mtx_pre, matrix_size, matrix_size );
//////////////////////////
//operations_mtx_pre = Identity;
/////////////////////
            // clear the operation list used in the previous iterations
            operations.clear();

            if (optimized_parameters != NULL ) {
                qgd_free( optimized_parameters );
                optimized_parameters = NULL;
            }

            // Transform the initial unitary upon the fixed pre-optimalization operations
            apply_operation(operations_mtx_pre, Umtx_loc, Umtx);

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
                }
                operations_mtxs_post.clear();
                fixed_operation_post->set_matrix( Identity );
            }


////////////////////////
//fixed_operation_post->set_matrix( Identity );
////////////////////
            
            // create a list of operations for the optimalization process
            operations.push_back( fixed_operation_post );
            for ( int idx=block_idx_end; idx<block_idx_start; idx++ ) {
                operations.push_back( operations_loc[idx] );
            }        
            

            // constructing solution guess for the optimalization           
            parameter_num = block_parameter_num;
            if ( solution_guess_gsl == NULL ) {
                solution_guess_gsl = gsl_vector_alloc (parameter_num);
            }
            else if ( parameter_num != solution_guess_gsl->size ) {
                gsl_vector_free(solution_guess_gsl);
                solution_guess_gsl = gsl_vector_alloc (parameter_num);
            }
            memcpy( solution_guess_gsl->data, optimized_parameters_gsl->data+parameter_num_loc - pre_operation_parameter_num - block_parameter_num, parameter_num*sizeof(double) );
            
            // solve the optimalization problem of the block 
            solve_layer_optimalization_problem( parameter_num, solution_guess_gsl  );

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
                block_idx_start = block_idx_start - optimalization_block;
                pre_operation_parameter_num = pre_operation_parameter_num + block_parameter_num;
            }
                
            
            // optimalization result is displayed in each 500th iteration
            if (iter_idx % 500 == 0) {
                printf("The minimum with %d layers after %d iterations is %e calculated in %f seconds\n", layer_num, iter_idx, current_minimum, float(time(NULL) - start_time));
                fflush(stdout);
                start_time = time(NULL);
            }

            
            // calculate the variance of the last 10 minimums
            double minvec_std = sqrt(gsl_stats_variance_m( minimum_vec, 1, min_vec_num, minvec_mean));

            // conditions to break the iteration cycles
            if (abs(minvec_std/minimum_vec[min_vec_num-1]) < optimalization_tolerance ) {
                printf("The iterations converged to minimum %e after %d iterations with %d layers\n", current_minimum, iter_idx, layer_num  );
                fflush(stdout);
                break; 
            }
            else if (check_optimalization_solution()) {
                printf("The minimum with %d layers after %d iterations is %e\n", layer_num, iter_idx, current_minimum);
                break;
            }
            
            // the convergence at low minimums is much faster if only one layer is considered in the optimalization at once
            if ( current_minimum < 1 ) {
                optimalization_block = 1;
            }
            


          
        }


        if (iter_idx == max_iterations ) {
            printf("Reached maximal number of iterations\n\n");
        }
        
        // restoring the parameters to originals
        optimalization_block = optimalization_block_loc;
        
        // store the result of the optimalization
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
        else {
            Identity = NULL;
        }

        // release preoperation matrix
        qgd_free( operations_mtx_pre );

        // release post operation products
        for (std::vector<QGD_Complex16*>::iterator mtxs_it=operations_mtxs_post.begin(); mtxs_it != operations_mtxs_post.end(); mtxs_it++ ) {
            qgd_free( *mtxs_it );
        }
        operations_mtxs_post.clear();
 
        tmp = NULL;

        delete(fixed_operation_post);

        // restore the original unitary
        memcpy(Umtx, Umtx_loc, matrix_size*matrix_size*sizeof(QGD_Complex16)) ;

        // free the allocated temporary Umtx
        qgd_free(Umtx_loc);

}      
        

   
////
// @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
// @param 'solution_guess' Array of guessed parameters
// @param 'num_of_parameters' NUmber of free parameters to be optimized
void Decomposition_Base::solve_layer_optimalization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) { 
    return;
}
       
    
    
    
////
// @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
double Decomposition_Base::optimalization_problem( const double* parameters ) {
        return current_minimum;
}
        
        
       
     
    
//// check_optimalization_solution
// @brief Checks the convergence of the optimalization problem.
// @return Returns with true if the target global minimum was reached during the optimalization process, or false otherwise.
bool Decomposition_Base::check_optimalization_solution() {
        
        return (abs(current_minimum - global_target_minimum) < optimalization_tolerance);
        
}

////
// @brief Calculate the list of cumulated gate operation matrices such that the i>0-th element in the result list is the product of the operations of all 0<=n<i operations from the input list and the 0th element in the result list is the identity.
// @param parameters An array containing the parameters of the U3 operations.
// @param operations Iterator pointing to the first element in a vector of operations to be considered in the multiplications.
// @param num_of_operations The number of operations counted from the first element of the operations.
// @return Returns with a vector of the product matrices.
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

        if (operation->get_type().compare("cnot")==0 ) {
            CNOT* cnot_operation = static_cast<CNOT*>(operation);
            cnot_operation->matrix(operation_mtx);
        }
        else if (operation->get_type().compare("general")==0 ) {
            operation->matrix(operation_mtx);
        }
        else if (operation->get_type().compare("U3")==0 ) {
            U3* u3_operation = static_cast<U3*>(operation);
            u3_operation->matrix(parameters, operation_mtx);
            parameters = parameters + u3_operation->get_parameter_num();
        }
        else if (operation->get_type().compare("block")==0 ) {
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
    }

    return operation_mtxs;

}

//
// @brief Call to get the unitary to be transformed
// @return Return with a pointer pointing to the unitary
QGD_Complex16* Decomposition_Base::get_Umtx() {
    return Umtx;
}


//
// @brief Call to get the size of the unitary to be transformed
// @return Return with the size of the unitary
int Decomposition_Base::get_Umtx_size() {
    return matrix_size;
}

//
// @brief Call to get the optimized parameters
// @return Return with the pointer pointing to the array storing the (memory copied) optimized parameters 
double* Decomposition_Base::get_optimized_parameters() {
    double *ret = (double*)qgd_calloc( parameter_num,sizeof(double), 64);
    get_optimized_parameters( ret );
    return ret;
}

//
// @brief Call to get the optimized parameters
// @return Return with the pointer pointing to the array storing the (memory copied) optimized parameters 
void Decomposition_Base::get_optimized_parameters( double* ret ) {
    memcpy(ret, optimized_parameters, parameter_num*sizeof(double));
    return;
}



////
// @brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
// @param parameters An array containing the parameters of the U3 operations.
// @param operations_it An iterator pointing to the first operation.
// @param num_of_operations The number of operations
// @param initial_matrix The initial matrix wich is transformed by the given operations. (by deafult it is set to the attribute @Umtx)
// @return Returns with the transformed matrix.
QGD_Complex16* Decomposition_Base::get_transformed_matrix( const double* parameters, std::vector<Operation*>::iterator operations_it, int num_of_operations, QGD_Complex16* initial_matrix=NULL  ) {

    QGD_Complex16* ret_matrix = transformed_mtx;


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

            if (operation->get_type().compare("cnot") == 0 ) {
                CNOT* cnot_operation = static_cast<CNOT*>( operation );
                cnot_operation->matrix(operation_mtx);
            }
            else if (operation->get_type().compare("general") == 0 ) {
                operation->matrix(operation_mtx);
            }                                
            else if (operation->get_type().compare("U3") == 0 ) {
                U3* u3_operation = static_cast<U3*>( operation );
                int parameters_num = u3_operation->get_parameter_num();
                u3_operation->matrix( parameters, operation_mtx );
                parameters = parameters + parameters_num;
            }
            else if (operation->get_type().compare("block") == 0 ) {
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
                memcpy( Operation_product, ret_matrix, matrix_size*matrix_size*sizeof(QGD_Complex16) );

            }

            operations_it++;
        }
/*if (qbit_num == 2) {
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

//printf("Decomposition_Base::get_transformed_matrix 6\n");
//print_mtx( ret_matrix, matrix_size, matrix_size );

        return ret_matrix;
}    
    
    
////
// @brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
// @return Returns with the decomposed matrix.
QGD_Complex16* Decomposition_Base::get_decomposed_matrix() {
     
        return get_transformed_matrix( optimized_parameters, operations.begin(), operations.size(), Umtx );
}
        
            
    
////
// @brief Gives an array of permutation indexes that can be used to permute the basis in the N-qubit unitary according to the flip in the qubit order.
// @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
// @retrun Returns with the reordering indexes of the basis     
std::vector<int> Decomposition_Base::get_basis_of_reordered_qubits( vector<int> qbit_list) {
        
    std::vector<int> bases_reorder_indexes;
        
    // generate the reordered  basis set
    for (int idx=0; idx<matrix_size; idx++) {
//TODO
        /*reordered_state = bin(idx)
        reordered_state = reordered_state[2:].zfill(self.qbit_num)
        reordered_state = [int(i) for i in reordered_state ]
        bases_reorder_indexes.append(int(np.dot( [2**power for power in qbit_list], reordered_state)))*/
    }
        
    return bases_reorder_indexes;
}           
 
////
// @brief Call to reorder the qubits in the unitary to be decomposed (the qubits become reordeerd in the operations a well)        
// @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
void Decomposition_Base::reorder_qubits( vector<int> qbit_list) {
//TODO      
/*    // contruct the permutation to get the basis for the reordered qbit list
    bases_reorder_indexes = self.get_basis_of_reordered_qubits( qbit_list )
           
    // reordering the matrix elements
    self.Umtx = self.Umtx[:, bases_reorder_indexes][bases_reorder_indexes]
*/       
    // reordering the matrix eleemnts of the operations
    Operation_block::reorder_qubits( qbit_list );

}
    
/*       
////
// @brief Call to contruct Qiskit compatible quantum circuit from the operations
    def get_quantum_circuit_inverse(self, circuit=None):
        return Operations.get_quantum_circuit_inverse( self, self.optimized_parameters, circuit=circuit)
    
////
// @brief Call to contruct Qiskit compatible quantum circuit from the operations that brings the original unitary into identity
    def get_quantum_circuit(self, circuit=None):    
        return Operations.get_quantum_circuit( self, self.optimized_parameters, circuit=circuit)

*/
////
// @brief Apply an operations on the input matrix
// @param operation_mtx The matrix of the operation.
// @param input_matrix The input matrix to be transformed.
// @return Returns with the transformed matrix
QGD_Complex16* Decomposition_Base::apply_operation( QGD_Complex16* operation_mtx, QGD_Complex16* input_matrix ) {

    // Getting the transformed state upon the transformation given by operation
    return zgemm3m_wrapper( operation_mtx, input_matrix, matrix_size);
}

////
// @brief Apply an operations on the input matrix
// @param operation_mtx The matrix of the operation.
// @param input_matrix The input matrix to be transformed.
// @return Returns with the transformed matrix
int Decomposition_Base::apply_operation( QGD_Complex16* operation_mtx, QGD_Complex16* input_matrix,  QGD_Complex16* result_matrix) {

    // Getting the transformed state upon the transformation given by operation
    return zgemm3m_wrapper( operation_mtx, input_matrix, result_matrix, matrix_size);
}


////
// @brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
// @param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.
// @param max_layer_num_in The maximal number of the operation layers used in the subdecomposition.
int Decomposition_Base::set_max_layer_num( int qbit, int max_layer_num_in ) {

    std::map<int,int>::iterator key_it = max_layer_num.find( qbit );

    if ( key_it != max_layer_num.end() ) {
        max_layer_num.erase( key_it );
    }

    max_layer_num.insert( std::pair<int, int>(qbit,  max_layer_num_in) );    

}


////
// @brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
// @param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.,
// @param iteration_loops_in The number of iteration loops in each sted of the subdecomposition.
int Decomposition_Base::set_iteration_loops( int qbit, int iteration_loops_in ) {

    std::map<int,int>::iterator key_it = iteration_loops.find( qbit );

    if ( key_it != iteration_loops.end() ) {
        iteration_loops.erase( key_it );
    }

    iteration_loops.insert( std::pair<int, int>(qbit,  iteration_loops_in) );   

}


////
// @brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
// @param iteration_loops_in An <int,int> map contining the iteration loops for the individual subdecomposition processes
int Decomposition_Base::set_iteration_loops( std::map<int, int> iteration_loops_in ) {

    for ( std::map<int,int>::iterator it=iteration_loops_in.begin(); it!= iteration_loops_in.end(); it++ ) {
        set_iteration_loops( it->first, it->second );
    }

}



////
// @briefinitializes default layer numbers
void Decomposition_Base::Init_max_layer_num() {

    // default layer numbers
    max_layer_num_def[2] = 3;
    max_layer_num_def[3] = 20;
    max_layer_num_def[4] = 60;
    max_layer_num_def[5] = 240;
    max_layer_num_def[6] = 1350;
    max_layer_num_def[7] = 7000;//6180;    

}



/*
double Decomposition_Base::_evaluate( void *instance, const double *x, double *g, const int n, const double step ) {
    return reinterpret_cast<Decomposition_Base*>(instance)->evaluate(x, g, n, step);
}
*/



/*
double Decomposition_Base::evaluate(const double *parameters, double *g, const int parameter_num, const double step) {
    

    double cost_function = optimalization_problem( parameters );
    double step_loc = 0.000001;
    printf("step %f, step_loc %f, cost function: %f, param num %d\n", step, step_loc, cost_function, parameter_num);
    printf("parameters:\n");
    for (int idx=0; idx<parameter_num; idx++) {
        printf("%f, ", parameters[idx]);
    }
    printf("\n");

    // calculate the gradients

    // preallocate the storage for the gardients
    if (g==NULL) {
        g = lbfgs_malloc( parameter_num );
    }

    // modified parameters to calculate the gradients
    double *parameters_loc = lbfgs_malloc( parameter_num );


    for (int idx=0; idx<parameter_num; idx++) {
        parameters_loc[idx] = parameters[idx];
    }

    // determine the gradients
    printf("gradients:\n");
    for (int idx=0; idx<parameter_num; idx++) {
        if (step_loc > 0.0) {
            parameters_loc[idx] = parameters_loc[idx] + step_loc;
            double cost_function_tmp = optimalization_problem( parameters_loc );

            g[idx] = (cost_function_tmp-cost_function)/step_loc;
            printf("%f ", g[idx]);
            parameters_loc[idx] = parameters_loc[idx] - step_loc;
        }
        else {
            g[idx] = 0;
        }
    }
    printf("\n");
    //throw "hhh";

    return cost_function;


}

*/







/*
int Decomposition_Base::_progress(void *instance, const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls) {
    return reinterpret_cast<Decomposition_Base*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
}*/
/*
int Decomposition_Base::progress(const double *x, const double *g, const double fx, const double xnorm, const double gnorm, const double step, int n, int k, int ls) {
 
        printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("\n");
        return 0;
}*/
