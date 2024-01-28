/*
Created on Fri Jun 26 14:13:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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

    
    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;

    // A string describing the type of the class
    type = DECOMPOSITION_BASE_CLASS;

    // error of the unitarity of the final decomposition
    decomposition_error = -1;

    // number of finalizing (deterministic) opertaions counted from the top of the array of gates
    finalizing_gates_num = 0;

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
    max_outer_iterations = 1e8;

    // number of operators in one sub-layer of the optimization process
    optimization_block = -1;

    // method to guess initial values for the optimization. Possible values: ZEROS, RANDOM, CLOSE_TO_ZERO (default)
    initial_guess = ZEROS;

    // The convergence threshold in the optimization process
    convergence_threshold = 1e-5;
    
    //global phase of the unitary matrix
    global_phase_factor.real = 1;
    global_phase_factor.imag = 0;
    
    //the name of the SQUANDER project
    std::string projectname = "";


    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  

    // seedign the random generator
    gen = std::mt19937(rd());

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
Decomposition_Base::Decomposition_Base( Matrix Umtx_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : Gates_block(qbit_num_in) {

    Init_max_layer_num();

   
    // the unitary operator to be decomposed
    Umtx = Umtx_in;
   
    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;

    // A string describing the type of the class
    type = DECOMPOSITION_BASE_CLASS;

    // error of the unitarity of the final decomposition
    decomposition_error = -1;

    // number of finalizing (deterministic) opertaions counted from the top of the array of gates
    finalizing_gates_num = 0;

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
    max_outer_iterations = 1e8;

    // number of operators in one sub-layer of the optimization process
    optimization_block = -1;

    // method to guess initial values for the optimization. Possible values: ZEROS, RANDOM, CLOSE_TO_ZERO (default)
    initial_guess = initial_guess_in;

    // The convergence threshold in the optimization process
    convergence_threshold = 1e-5;
    
    //global phase of the unitary matrix
    global_phase_factor.real = 1;
    global_phase_factor.imag = 0;
    
    //name of the SQUANDER project
    std::string projectname = "";


    // config maps
    config   = config_in;


    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  

    // seedign the random generator
    gen = std::mt19937(rd());

#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif

}

/**
@brief Destructor of the class
*/
Decomposition_Base::~Decomposition_Base() {
/*
    if (optimized_parameters != NULL ) {
        qgd_free( optimized_parameters );
        optimized_parameters = NULL;
    }
*/
}


/**
@brief Call to set the number of gate blocks to be optimized in one shot
@param optimization_block_in The number of gate blocks to be optimized in one shot
*/
void Decomposition_Base::set_optimization_blocks( int optimization_block_in) {
    optimization_block = optimization_block_in;
}

/**
@brief Call to set the maximal number of the iterations in the optimization process
@param max_outer_iterations_in maximal number of iteartions in the optimization process
*/
void Decomposition_Base::set_max_iteration( int max_outer_iterations_in) {
    max_outer_iterations = max_outer_iterations_in;
}


/**
@brief After the main optimization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed gates are added to the array of gates needed to the decomposition of the input unitary.
*/
void Decomposition_Base::finalize_decomposition() {

        // get the transformed matrix resulted by the gates in the list
        Matrix transformed_matrix = get_transformed_matrix( optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );

        // preallocate the storage for the finalizing parameters
        finalizing_parameter_num = 3*qbit_num;
        double* finalizing_parameters = (double*)qgd_calloc(finalizing_parameter_num,sizeof(double), 64);

        // obtaining the final gates of the decomposition
        Gates_block* finalizing_gates = new Gates_block( qbit_num );;
        Matrix final_matrix = get_finalizing_gates( transformed_matrix, finalizing_gates, finalizing_parameters );

        // adding the finalizing gates to the list of gates
        // adding the opeartion block to the gates
        add_gate( finalizing_gates );
// TODO: use memcpy
        Matrix_real optimized_parameters_tmp(1, parameter_num);
        for (int idx=0; idx < finalizing_parameter_num; idx++) {
            optimized_parameters_tmp[idx] = finalizing_parameters[idx];
        }
        for (int idx=0; idx < parameter_num-finalizing_parameter_num; idx++) {
            optimized_parameters_tmp[idx+finalizing_parameter_num] = optimized_parameters_mtx[idx];
        }
        qgd_free( finalizing_parameters);
        finalizing_parameters = NULL;
        optimized_parameters_mtx = optimized_parameters_tmp;

        finalizing_gates_num = finalizing_gates->get_gate_num();


        // indicate that the decomposition was finalized
        decomposition_finalized = true;

        // calculating the final error of the decomposition
        subtract_diag( final_matrix, final_matrix[0] );
        decomposition_error = cblas_dznrm2( matrix_size*matrix_size, (void*)final_matrix.get_data(), 1 );

        // get the number of gates used in the decomposition
        gates_num gates_num = get_gate_nums();


        //The stringstream input to store the output messages.
	std::stringstream sstream;
	sstream << "The error of the decomposition after finalyzing gates is " << decomposition_error << " with " << layer_num << " layers containing " << gates_num.u3 << " U3 gates and " << gates_num.cnot <<  " CNOT gates" << std::endl;
	print(sstream, 1);	    	
	
            
        

}


/**
@brief Call to print the gates decomposing the initial unitary. These gates brings the intial matrix into unity.
@param start_index The index of the first gate
*/
void Decomposition_Base::list_gates( int start_index ) {

        Gates_block::list_gates( optimized_parameters_mtx, start_index );

}



/**
@brief This method determine the gates needed to rotate the indepent qubits into the state |0>
@param mtx The unitary describing indepent qubits.  The resulting matrix is returned by this pointer
@param finalizing_gates Pointer pointig to a block of gates containing the final gates.
@param finalizing_parameters Parameters corresponding to the finalizing gates.
@return Returns with the finalized matrix
*/
Matrix Decomposition_Base::get_finalizing_gates( Matrix& mtx, Gates_block* finalizing_gates, double* finalizing_parameters  ) {


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

            Matrix_real parameters_loc(1,3);
            parameters_loc[0] = Theta;
            parameters_loc[1] = M_PI-Lambda;
            parameters_loc[2] = M_PI-Phi;

            U3* u3_loc = new U3( qbit_num, target_qbit, true, true, true);

            // adding the new gate to the list of finalizing gates
            finalizing_parameters[parameter_idx] = M_PI-Phi; //np.concatenate((parameters_loc, finalizing_parameters))
            parameter_idx--;
            finalizing_parameters[parameter_idx] = M_PI-Lambda; //np.concatenate((parameters_loc, finalizing_parameters))
            parameter_idx--;
            finalizing_parameters[parameter_idx] = Theta; //np.concatenate((parameters_loc, finalizing_parameters))
            parameter_idx--;

            finalizing_gates->add_gate( u3_loc );
            // get the new matrix


            Matrix u3_mtx = u3_loc->get_matrix(parameters_loc);

            Matrix tmp2 = apply_gate( u3_mtx, mtx_tmp);
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
	

        if ( gates.size() == 0 ) {
            return;
        }

        // array containing minimums to check convergence of the solution
        const int min_vec_num = 20;
        double minimum_vec[min_vec_num];
        for ( int idx=0; idx<min_vec_num; idx++) {
            minimum_vec[idx] = 0;
        }

        // setting the initial value for the current minimum
        current_minimum = 1e8;

        // store the gates
        std::vector<Gate*> gates_loc = gates;



        // store the number of parameters
        int parameter_num_loc = parameter_num;

        // store the initial unitary to be decomposed
        Matrix Umtx_loc = Umtx;

        // storing the initial computational parameters
        int optimization_block_loc = optimization_block;
        
        if ( optimization_block == -1 ) {
            optimization_block = gates.size();
        }

        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        // the array storing the optimized parameters
        Matrix_real optimized_parameters(1, parameter_num_loc);

        // preparing solution guess for the iterations
        if ( initial_guess == ZEROS ) {
            for(int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters[idx] = 0;
            }
        }
        else if ( initial_guess == RANDOM ) {
            for(int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters[idx] = distrib_real(gen);
            }

#ifdef __MPI__        
            MPI_Bcast( (void*)optimized_parameters.get_data(), parameter_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


        }
        else if ( initial_guess == CLOSE_TO_ZERO ) {
            for(int idx = 0; idx < parameter_num-solution_guess_num; idx++) {
                optimized_parameters[idx] = distrib_real(gen)/100;
            }

#ifdef __MPI__        
            MPI_Bcast( (void*)optimized_parameters.get_data(), parameter_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

        }
        else {
            std::string err("bad value for initial guess");
        }

        if ( solution_guess_num > 0) {
            memcpy(optimized_parameters.get_data() + parameter_num-solution_guess_num, solution_guess, solution_guess_num*sizeof(double));
        }

        // starting number of gate block applied prior to the optimalized gate blocks
        int pre_gate_parameter_num = 0;

        // defining temporary variables for iteration cycles
        int block_idx_end;
        int block_idx_start = gates.size();
        gates.clear();
        int block_parameter_num;
        Gate* fixed_gate_post = new Gate( qbit_num );
        std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> gates_mtxs_post;

        // the identity matrix used in the calculations
        Matrix Identity =  create_identity( matrix_size );


        Matrix_real solution_guess_tmp;


        // maximal number of outer iterations overriden by config
        long long max_outer_iterations_loc;
        if ( config.count("max_outer_iterations") > 0 ) {
            config["max_outer_iterations"].get_property( max_outer_iterations_loc );
         
        }
        else {
            max_outer_iterations_loc =max_outer_iterations;
        }


        //measure the time for the decomposition
        tbb::tick_count start_time = tbb::tick_count::now();

        ////////////////////////////////////////
        // Start the iterations
        long long iter_idx;
        for ( iter_idx=0; iter_idx<max_outer_iterations_loc; iter_idx++) {

            //determine the range of blocks to be optimalized togedther
            block_idx_end = block_idx_start - optimization_block;
            if (block_idx_end < 0) {
                block_idx_end = 0;
            }

            // determine the number of free parameters to be optimized
            block_parameter_num = 0;
            for ( int block_idx=block_idx_start-1; block_idx>=block_idx_end; block_idx--) {
                block_parameter_num = block_parameter_num + gates_loc[block_idx]->get_parameter_num();
            }
       

            // ***** get applied the fixed gates applied before the optimized gates *****
            if (block_idx_start < (int)gates_loc.size() ) {
                std::vector<Gate*>::iterator fixed_gates_pre_it = gates.begin() + 1;
                //Matrix_real optimized_parameters_mtx(optimized_parameters, 1, parameter_num );
                Umtx = get_transformed_matrix(optimized_parameters_mtx, fixed_gates_pre_it, gates.size()-1, Umtx);
            }
            else {
                Umtx = Umtx_loc.copy();
            }

            // clear the gate list used in the previous iterations
            gates.clear();

            if (optimized_parameters_mtx.size() > 0 ) {
//                qgd_free( optimized_parameters );
//                optimized_parameters = NULL;
                  optimized_parameters_mtx = Matrix_real(0,0);
            }


            // ***** get the fixed gates applied after the optimized gates *****
            // create a list of post gates matrices
            if (block_idx_start == (int)gates_loc.size() ) {
                // matrix of the fixed gates aplied after the gates to be varied
                double* fixed_parameters_post = optimized_parameters.get_data();
                std::vector<Gate*>::iterator fixed_gates_post_it = gates_loc.begin();

                gates_mtxs_post = get_gate_products(fixed_parameters_post, fixed_gates_post_it, block_idx_end);
            }

            // Create a general gate describing the cumulative effect of gates following the optimized gates
            if (block_idx_end > 0) {
                fixed_gate_post->set_matrix( gates_mtxs_post[block_idx_end-1] );
                gates.push_back( fixed_gate_post ); 
            }
            else {
                // release gate products
                gates_mtxs_post.clear();
                fixed_gate_post->set_matrix( Identity );
            }

            // create a list of gates for the optimization process
            for ( int idx=block_idx_end; idx<block_idx_start; idx++ ) {
                gates.push_back( gates_loc[idx] );
            }


            // constructing solution guess for the optimization
            parameter_num = block_parameter_num;
            if ( solution_guess_tmp.size() == 0 ) {
                solution_guess_tmp = Matrix_real(1, parameter_num);
            }
            else if ( parameter_num != (int)solution_guess_tmp.size() ) {
                solution_guess_tmp = Matrix_real(1, parameter_num);
            }
            memcpy( solution_guess_tmp.get_data(), optimized_parameters.get_data()+parameter_num_loc - pre_gate_parameter_num - block_parameter_num, parameter_num*sizeof(double) );

            // solve the optimization problem of the block
            solve_layer_optimization_problem( parameter_num, solution_guess_tmp );

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
            memcpy( optimized_parameters.get_data()+parameter_num_loc - pre_gate_parameter_num-block_parameter_num, optimized_parameters_mtx.get_data(), parameter_num*sizeof(double) );

            if (block_idx_end == 0) {
                block_idx_start = gates_loc.size();
                pre_gate_parameter_num = 0;
            }
            else {
                block_idx_start = block_idx_start - optimization_block;
                pre_gate_parameter_num = pre_gate_parameter_num + block_parameter_num;
            }


            // optimization result is displayed in each 500th iteration
            if (iter_idx % 500 == 0) {                
                tbb::tick_count current_time = tbb::tick_count::now();
                std::stringstream sstream;
		sstream << "The minimum with " << layer_num << " layers after " << iter_idx << " outer iterations is " << current_minimum << " calculated in " << (current_time - start_time).seconds() << " seconds" << std::endl;
		print(sstream, 2);            
                start_time = tbb::tick_count::now();
            }


            // calculate the variance of the last 10 minimums
            double minvec_std = 0.0;
            for ( int kdx=0; kdx<min_vec_num; kdx++ ) {
                double tmp = minimum_vec[kdx] - minvec_mean;
                minvec_std += tmp*tmp;
            }
            minvec_std = sqrt(minvec_std/(min_vec_num-1));

            // conditions to break the iteration cycles
            if (std::abs(minvec_std/minimum_vec[min_vec_num-1]) < convergence_threshold ) {              
		std::stringstream sstream;
	        sstream << "The iterations converged to minimum " << current_minimum << " after " << iter_idx << " outer iterations with " << layer_num << " layers" << std::endl;
		print(sstream, 1);             
                break;
            }
            else if (check_optimization_solution()) {
      		std::stringstream sstream;
		sstream << "The minimum with " << layer_num << " layers after " << iter_idx << " outer iterations is " << current_minimum << std::endl;
		print(sstream, 1);  		               
                break;
            }


        }


        if (iter_idx == max_outer_iterations_loc && max_outer_iterations_loc>1) {            
		std::stringstream sstream;
		sstream << "Reached maximal number of outer iterations" << std::endl << std::endl;
		print(sstream, 1);
        }

        // restoring the parameters to originals
        optimization_block = optimization_block_loc;

        // store the result of the optimization
        gates.clear();
        gates = gates_loc;

        parameter_num = parameter_num_loc;


        optimized_parameters_mtx = Matrix_real( 1, parameter_num );
        memcpy( optimized_parameters_mtx.get_data(), optimized_parameters.get_data(), parameter_num*sizeof(double) );



        delete(fixed_gate_post);

        // restore the original unitary
        Umtx = Umtx_loc; // copy?


}




/**
@brief Abstarct function to be used to solve a single sub-layer optimization problem. The optimalized parameters are stored in attribute optimized_parameters.
@param 'num_of_parameters' The number of free parameters to be optimized
@param solution_guess Array containing the free parameters to be optimized.
*/
void Decomposition_Base::solve_layer_optimization_problem( int num_of_parameters, Matrix_real solution_guess) {
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

        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance") > 0 ) {
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }       

        return (std::abs(current_minimum - global_target_minimum) < optimization_tolerance_loc);

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
Matrix_real Decomposition_Base::get_optimized_parameters() {    
    
    return optimized_parameters_mtx.copy();

}

/**
@brief Call to get the optimized parameters.
@param ret Preallocated array to store the optimized parameters.
*/
void Decomposition_Base::get_optimized_parameters( double* ret ) {
    memcpy(ret, optimized_parameters_mtx.get_data(), parameter_num*sizeof(double));
    return;
}


/**
@brief Call to set the optimized parameters for initial optimization.
@param ret Preallocated array to store the optimized parameters.
*/
void Decomposition_Base::set_optimized_parameters( double* parameters, int num_of_parameters ) {

    if ( num_of_parameters == 0 ) {
        optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
        return;
    }

    if ( parameter_num != num_of_parameters ) {
        std::string err("Decomposition_Base::set_optimized_parameters: The number of parameters does not match with the free parameters of the circuit.");
        throw err;
    }

    optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
    memcpy( optimized_parameters_mtx.get_data(), parameters, num_of_parameters*sizeof(double) );

    return;
}

/**
@brief Calculate the transformed matrix resulting by an array of gates on the matrix Umtx
@param parameters An array containing the parameters of the U3 gates.
@param gates_it An iterator pointing to the first gate to be applied on the initial matrix.
@param num_of_gates The number of gates to be applied on the initial matrix
@return Returns with the transformed matrix.
*/
Matrix
Decomposition_Base::get_transformed_matrix( Matrix_real &parameters, std::vector<Gate*>::iterator gates_it, int num_of_gates ) {

    return get_transformed_matrix( parameters, gates_it, num_of_gates, Umtx );
}


/**
@brief Calculate the transformed matrix resulting by an array of gates on a given initial matrix.
@param parameters An array containing the parameters of the U3 gates.
@param gates_it An iterator pointing to the first gate to be applied on the initial matrix.
@param num_of_gates The number of gates to be applied on the initial matrix
@param initial_matrix The initial matrix wich is transformed by the given gates. (by deafult it is set to the attribute Umtx)
@return Returns with the transformed matrix.
*/
Matrix
Decomposition_Base::get_transformed_matrix( Matrix_real &parameters, std::vector<Gate*>::iterator gates_it, int num_of_gates, Matrix& initial_matrix ) {

    // The matrix to be returned
    Matrix ret_matrix = initial_matrix.copy();
    if (num_of_gates==0) {
        return ret_matrix;
    }


    // determine the number of parameters
    int parameters_num_total = 0;
    for (int idx=0; idx<num_of_gates; idx++) {

        // The current gate
        Gate* gate = *(gates_it++);
        parameters_num_total = parameters_num_total + gate->get_parameter_num();

    }


    // apply the gate operations on the inital matrix
    for (int idx=num_of_gates-0; idx>0; idx--) {

        // The current gate
        Gate* gate = *(--gates_it);
        parameters_num_total = parameters_num_total - gate->get_parameter_num();
        Matrix_real parameters_mtx( parameters.get_data()+parameters_num_total, 1, gate->get_parameter_num() );

        if (gate->get_type() == CNOT_OPERATION ) {
            CNOT* cnot_gate = static_cast<CNOT*>( gate );
            cnot_gate->apply_to(ret_matrix);
        }
        else if (gate->get_type() == CZ_OPERATION ) {
            CZ* cz_gate = static_cast<CZ*>( gate );
            cz_gate->apply_to(ret_matrix);
        }
        else if (gate->get_type() == CH_OPERATION ) {
            CH* ch_gate = static_cast<CH*>( gate );
            ch_gate->apply_to(ret_matrix);
        }
        else if (gate->get_type() == SYC_OPERATION ) {
            SYC* syc_gate = static_cast<SYC*>( gate );
            syc_gate->apply_to(ret_matrix);
        }
        else if (gate->get_type() == GENERAL_OPERATION ) {
            gate->apply_to(ret_matrix);
        }
        else if (gate->get_type() == U3_OPERATION ) {
            U3* u3_gate = static_cast<U3*>( gate );
            u3_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == RX_OPERATION ) {
            RX* rx_gate = static_cast<RX*>( gate );
            rx_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == RY_OPERATION ) {
            RY* ry_gate = static_cast<RY*>( gate );
            ry_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == CRY_OPERATION ) {
            CRY* cry_gate = static_cast<CRY*>( gate );
            cry_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == RZ_OPERATION ) {
            RZ* rz_gate = static_cast<RZ*>( gate );
            rz_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == RZ_P_OPERATION ) {
            RZ_P* rz_gate = static_cast<RZ_P*>( gate );
            rz_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == X_OPERATION ) {
            X* x_gate = static_cast<X*>( gate );
            x_gate->apply_to( ret_matrix );            
        }
        else if (gate->get_type() == Y_OPERATION ) {
            Y* y_gate = static_cast<Y*>( gate );
            y_gate->apply_to( ret_matrix );            
        }
        else if (gate->get_type() == Z_OPERATION ) {
            Z* z_gate = static_cast<Z*>( gate );
            z_gate->apply_to( ret_matrix );            
        }
        else if (gate->get_type() == SX_OPERATION ) {
            SX* sx_gate = static_cast<SX*>( gate );
            sx_gate->apply_to( ret_matrix );            
        }
        else if (gate->get_type() == UN_OPERATION ) {
            UN* un_gate = static_cast<UN*>( gate );
            un_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == ON_OPERATION ) {
            ON* on_gate = static_cast<ON*>( gate );
            on_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == COMPOSITE_OPERATION ) {
            Composite* com_gate = static_cast<Composite*>( gate );
            com_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == BLOCK_OPERATION ) {
            Gates_block* block_gate = static_cast<Gates_block*>( gate );
            block_gate->apply_to(parameters_mtx, ret_matrix);            
        }
        else if (gate->get_type() == ADAPTIVE_OPERATION ) {
            Adaptive* ad_gate = static_cast<Adaptive*>( gate );
            ad_gate->apply_to( parameters_mtx, ret_matrix);            
        }
        else {
            std::string err("Decomposition_Base::get_transformed_matrix: unimplemented gate");
            throw err;
        }


    }

    return ret_matrix;





}

/**
@brief Calculate the decomposed matrix resulted by the effect of the optimized gates on the unitary Umtx
@return Returns with the decomposed matrix.
*/
Matrix Decomposition_Base::get_decomposed_matrix() {
        
        return get_transformed_matrix( optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
}



/**
@brief Calculate the list of gate gate matrices such that the i>0-th element in the result list is the product of the gates of all 0<=n<i gates from the input list and the 0th element in the result list is the identity.
@param parameters An array containing the parameters of the gates.
@param gates_it An iterator pointing to the first gate.
@param num_of_gates The number of gates involved in the calculations
@return Returns with a vector of the product matrices.
*/
std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> 
Decomposition_Base::get_gate_products(double* parameters, std::vector<Gate*>::iterator gates_it, int num_of_gates) {


    // construct the vector of matrix representation of the gates
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> gate_mtxs(num_of_gates);

    // creating identity gate if no gates were involved in the calculations
    if (num_of_gates==0) {
        gate_mtxs.push_back( create_identity(matrix_size) );
        return gate_mtxs;
    }


    double* parameters_loc = parameters;
    Matrix mtx = create_identity(matrix_size);

    for (int idx=0; idx<num_of_gates; idx++) {
       

        // get the matrix representation of the gate
        Gate* gate = *gates_it;        
        Matrix_real parameters_loc_mtx( parameters_loc, 1, gate->get_parameter_num() );

        if (gate->get_type() == CNOT_OPERATION ) {
            CNOT* cnot_gate = static_cast<CNOT*>(gate);
            cnot_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == CZ_OPERATION ) {
            CZ* cz_gate = static_cast<CZ*>(gate);
            cz_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == CH_OPERATION ) {
            CH* ch_gate = static_cast<CH*>(gate);
            ch_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == SYC_OPERATION ) {
            SYC* syc_gate = static_cast<SYC*>(gate);
            syc_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == GENERAL_OPERATION ) {
            gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == U3_OPERATION ) {
            U3* u3_gate = static_cast<U3*>(gate);
            u3_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == RX_OPERATION ) {
            RX* rx_gate = static_cast<RX*>(gate);
            rx_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == RY_OPERATION ) {
            RY* ry_gate = static_cast<RY*>(gate);
            ry_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == CRY_OPERATION ) {
            CRY* cry_gate = static_cast<CRY*>(gate);
            cry_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == RZ_OPERATION ) {
            RZ* rz_gate = static_cast<RZ*>(gate);
            rz_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == RZ_P_OPERATION ) {
            RZ_P* rz_gate = static_cast<RZ_P*>(gate);
            rz_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == X_OPERATION ) {
            X* x_gate = static_cast<X*>(gate);
            x_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == Y_OPERATION ) {
            Y* y_gate = static_cast<Y*>(gate);
            y_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == Z_OPERATION ) {
            Z* z_gate = static_cast<Z*>(gate);
            z_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == SX_OPERATION ) {
            SX* sx_gate = static_cast<SX*>(gate);
            sx_gate->apply_from_right(mtx);
        }
        else if (gate->get_type() == UN_OPERATION ) {
            UN* un_gate = static_cast<UN*>(gate);
            un_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == ON_OPERATION ) {
            ON* on_gate = static_cast<ON*>(gate);
            on_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == COMPOSITE_OPERATION ) {
            Composite* com_gate = static_cast<Composite*>(gate);
            com_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == BLOCK_OPERATION ) {
            Gates_block* block_gate = static_cast<Gates_block*>(gate);
            block_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else if (gate->get_type() == ADAPTIVE_OPERATION ) {
            Adaptive* ad_gate = static_cast<Adaptive*>(gate);
            ad_gate->apply_from_right(parameters_loc_mtx, mtx);
        }
        else {
            std::string err("Decomposition_Base::get_gate_products: unimplemented gate");
            throw err;
        }

        parameters_loc = parameters_loc + gate->get_parameter_num();

        gate_mtxs[idx] = mtx.copy();     
        gates_it++;   

    }

    return gate_mtxs;

}

/**
@brief Apply an gates on the input matrix
@param gate_mtx The matrix of the gate.
@param input_matrix The input matrix to be transformed.
@return Returns with the transformed matrix
*/
Matrix
Decomposition_Base::apply_gate( Matrix& gate_mtx, Matrix& input_matrix ) {

    // Getting the transformed state upon the transformation given by gate
    return dot( gate_mtx, input_matrix );

}

/**
@brief Set the maximal number of layers used in the subdecomposition of the n-th qubit.
@param n The number of qubits for which the maximal number of layers should be used in the subdecomposition.
@param max_layer_num_in The maximal number of the gate layers used in the subdecomposition.
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
@param max_layer_num_in An <int,int> map containing the maximal number of the gate layers used in the subdecomposition.
@return Returns with 0 if succeded.
*/
int Decomposition_Base::set_max_layer_num( std::map<int, int> max_layer_num_in ) {


    for ( std::map<int,int>::iterator it = max_layer_num_in.begin(); it!=max_layer_num_in.end(); it++) {
        set_max_layer_num( it->first, it->second );
    }

    return 0;

}


/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Decomposition_Base::reorder_qubits( std::vector<int>  qbit_list) {

    Gates_block::reorder_qubits( qbit_list );

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
    max_layer_num_def[3] = 14;
    max_layer_num_def[4] = 60;
    max_layer_num_def[5] = 240;
    max_layer_num_def[6] = 1350;
    max_layer_num_def[7] = 7000;//6180;

}



/**
@brief Call to prepare the optimized gates to export. The gates are stored in the attribute gates
*/
void Decomposition_Base::prepare_gates_to_export() {

    std::vector<Gate*> gates_tmp = prepare_gates_to_export( gates, optimized_parameters_mtx.get_data() );

    // release the gates and replace them with the ones prepared to export
    gates.clear();
    gates = gates_tmp;

}



/**
@brief Call to prepare the optimized gates to export
@param ops A list of gates
@param parameters The parameters of the gates
@return Returns with a list of gate gates.
*/
std::vector<Gate*> Decomposition_Base::prepare_gates_to_export( std::vector<Gate*> ops, double* parameters ) {


    std::vector<Gate*> ops_ret;
    int parameter_idx = 0;


    for(std::vector<Gate*>::iterator it = ops.begin(); it != ops.end(); it++) {

        Gate* gate = *it;

        if (gate->get_type() == CNOT_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == CZ_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == CH_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == SYC_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == X_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == Y_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == Z_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == SX_OPERATION) {
            ops_ret.push_back( gate );
        }
        else if (gate->get_type() == U3_OPERATION) {

            // definig the U3 parameters
            double vartheta;
            double varphi;
            double varlambda;

            // get the inverse parameters of the U3 rotation

            U3* u3_gate = static_cast<U3*>(gate);

            if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_theta_parameter()) {
                vartheta = std::fmod( 2*parameters[parameter_idx], 4*M_PI);
                varphi = 0;
                varlambda =0;
                parameter_idx = parameter_idx + 1;

            }
            else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_phi_parameter()) {
                vartheta = 0;
                varphi = std::fmod( parameters[ parameter_idx ], 2*M_PI);
                varlambda =0;
                parameter_idx = parameter_idx + 1;
            }
            else if ((u3_gate->get_parameter_num() == 1) && u3_gate->is_lambda_parameter()) {
                vartheta = 0;
                varphi =  0;
                varlambda = std::fmod( parameters[ parameter_idx ], 2*M_PI);
                parameter_idx = parameter_idx + 1;
            }
            else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_phi_parameter() ) {
                vartheta = std::fmod( 2*parameters[ parameter_idx ], 4*M_PI);
                varphi = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                varlambda = 0;
                parameter_idx = parameter_idx + 2;
            }
            else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_theta_parameter() && u3_gate->is_lambda_parameter() ) {
                vartheta = std::fmod( 2*parameters[ parameter_idx ], 4*M_PI);
                varphi = 0;
                varlambda = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                parameter_idx = parameter_idx + 2;
            }
            else if ((u3_gate->get_parameter_num() == 2) && u3_gate->is_phi_parameter() && u3_gate->is_lambda_parameter() ) {
                vartheta = 0;
                varphi = std::fmod( parameters[ parameter_idx], 2*M_PI);
                varlambda = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                parameter_idx = parameter_idx + 2;
            }
            else if ((u3_gate->get_parameter_num() == 3)) {
                vartheta = std::fmod( 2*parameters[ parameter_idx ], 4*M_PI);
                varphi = std::fmod( parameters[ parameter_idx+1 ], 2*M_PI);
                varlambda = std::fmod( parameters[ parameter_idx+2 ], 2*M_PI);
                parameter_idx = parameter_idx + 3;
            }
            else {
                std::string err("bad value for initial guess");
                throw err;        
            }

            u3_gate->set_optimized_parameters( vartheta, varphi, varlambda );
            ops_ret.push_back( static_cast<Gate*>(u3_gate) );


        }
        else if (gate->get_type() == RX_OPERATION) {

            // definig the parameter of the rotational angle
            double vartheta;

            // get the inverse parameters of the RX rotation

            RX* rx_gate = static_cast<RX*>(gate);

            vartheta = std::fmod( 2*parameters[parameter_idx], 4*M_PI);
            parameter_idx = parameter_idx + 1;


            rx_gate->set_optimized_parameters( vartheta );
            ops_ret.push_back( static_cast<Gate*>(rx_gate) );


        }
        else if (gate->get_type() == RY_OPERATION) {

            // definig the parameter of the rotational angle
            double vartheta;

            // get the inverse parameters of the RY rotation

            RY* ry_gate = static_cast<RY*>(gate);

            vartheta = std::fmod( 2*parameters[parameter_idx], 4*M_PI);
            parameter_idx = parameter_idx + 1;


            ry_gate->set_optimized_parameters( vartheta );
            ops_ret.push_back( static_cast<Gate*>(ry_gate) );


        }
        else if (gate->get_type() == CRY_OPERATION) {

            // definig the parameter of the rotational angle
            double Theta;

            // get the inverse parameters of the RZ rotation

            CRY* cry_gate = static_cast<CRY*>(gate);

            Theta = std::fmod( 2.0*parameters[parameter_idx], 4*M_PI);
            parameter_idx = parameter_idx + 1;


            cry_gate->set_optimized_parameters( Theta );
            ops_ret.push_back( static_cast<Gate*>(cry_gate) );


        }
        else if (gate->get_type() == RZ_OPERATION) {

            // definig the parameter of the rotational angle
            double varphi;

            // get the inverse parameters of the RZ rotation

            RZ* rz_gate = static_cast<RZ*>(gate);

            varphi = std::fmod( 2*parameters[parameter_idx], 4*M_PI);
            parameter_idx = parameter_idx + 1;


            rz_gate->set_optimized_parameters( varphi );
            ops_ret.push_back( static_cast<Gate*>(rz_gate) );


        }
        else if (gate->get_type() == RZ_P_OPERATION) {

            // definig the parameter of the rotational angle
            double varphi;

            // get the inverse parameters of the RZ rotation

            RZ_P* rz_gate = static_cast<RZ_P*>(gate);

            varphi = std::fmod( parameters[parameter_idx], 2*M_PI);
            parameter_idx = parameter_idx + 1;


            rz_gate->set_optimized_parameters( varphi );
            ops_ret.push_back( static_cast<Gate*>(rz_gate) );


        }
        else if (gate->get_type() == UN_OPERATION) {

             // get the inverse parameters of the RZ rotation
            UN* un_gate = static_cast<UN*>(gate);

            Matrix_real optimized_parameters( parameters+parameter_idx, 1, (int)un_gate->get_parameter_num());
            parameter_idx = parameter_idx + un_gate->get_parameter_num();


            un_gate->set_optimized_parameters( optimized_parameters );
            ops_ret.push_back( static_cast<Gate*>(un_gate) );


        }
        else if (gate->get_type() == ON_OPERATION) {

            // get the inverse parameters of the RZ rotation
            ON* on_gate = static_cast<ON*>(gate);

            Matrix_real optimized_parameters( parameters+parameter_idx, 1, (int)on_gate->get_parameter_num());
            parameter_idx = parameter_idx + on_gate->get_parameter_num();


            on_gate->set_optimized_parameters( optimized_parameters );
            ops_ret.push_back( static_cast<Gate*>(on_gate) );


        }
        else if (gate->get_type() == COMPOSITE_OPERATION) {

            // get the inverse parameters of the RZ rotation
            Composite* com_gate = static_cast<Composite*>(gate);

            Matrix_real optimized_parameters( parameters+parameter_idx, 1, (int)com_gate->get_parameter_num());
            parameter_idx = parameter_idx + com_gate->get_parameter_num();


            com_gate->set_optimized_parameters( optimized_parameters );
            ops_ret.push_back( static_cast<Gate*>(com_gate) );


        }
        else if (gate->get_type() == ADAPTIVE_OPERATION) {

            // definig the parameter of the rotational angle
            double Theta;

            // get the inverse parameters of the RZ rotation

            Adaptive* ad_gate = static_cast<Adaptive*>(gate);

            Theta = std::fmod( 2*parameters[parameter_idx], 4*M_PI);
            parameter_idx = parameter_idx + 1;


            ad_gate->set_optimized_parameters( Theta );
            ops_ret.push_back( static_cast<Gate*>(ad_gate) );


        }
        else if (gate->get_type() == BLOCK_OPERATION) {
            Gates_block* block_gate = static_cast<Gates_block*>(gate);
            double* parameters_layer = parameters + parameter_idx;

            std::vector<Gate*> ops_loc = prepare_gates_to_export(block_gate, parameters_layer);
            parameter_idx = parameter_idx + block_gate->get_parameter_num();

            ops_ret.insert( ops_ret.end(), ops_loc.begin(), ops_loc.end() );
        }
        else {
            std::string err("Decomposition_Base::prepare_gates_to_export: unimplemented gate");
            throw err;
        }

    }


    return ops_ret;


}


/**
@brief Call to prepare the gates of an gate block to export
@param block_op A pointer to a block of gates
@param parameters The parameters of the gates
@return Returns with a list of gate gates.
*/
std::vector<Gate*> Decomposition_Base::prepare_gates_to_export( Gates_block* block_op, double* parameters ) {

    std::vector<Gate*> ops_tmp = block_op->get_gates();
    std::vector<Gate*> ops_ret = prepare_gates_to_export( ops_tmp, parameters );

    return ops_ret;

}





/**
@brief Call to prepare the optimized gates to export
@param n Integer labeling the n-th oepration  (n>=0).
@return Returns with a pointer to the n-th Gate, or with MULL if the n-th gate cant be retrived.
*/
Gate* Decomposition_Base::get_gate( int n ) {

    // get the n-th gate if exists
    if ( (unsigned int) n >= gates.size() ) {
        return NULL;
    }

    return gates[n];

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
@brief Call to set the threshold of convergence in the optimization processes.
@param convergence_threshold_in The value of the threshold. 
*/
void Decomposition_Base::set_convergence_threshold( double convergence_threshold_in ) {

    convergence_threshold = convergence_threshold_in;
    return;
}

/**
@brief Call to get the error of the decomposition
@return Returns with the error of the decomposition
*/
double Decomposition_Base::get_decomposition_error( ) {

    return decomposition_error;

}




/**
@brief Call to get the obtained minimum of the cost function
@return Returns with the minimum of the cost function
*/
double Decomposition_Base::get_current_minimum( ) {

    return current_minimum;

}

/**
@brief Call to get the current name of the project
@return Returns the name of the project
*/
std::string Decomposition_Base::get_project_name(){
	return project_name;
}

/**
@brief Call to set the name of the project
@param project_name_new pointer to the new project name
*/
void Decomposition_Base::set_project_name(std::string& project_name_new){
	project_name = project_name_new;
	return;
}


/**
@brief Call to calculate new global phase 
@param global_phase_factor The value of the phase
*/
void Decomposition_Base::calculate_new_global_phase_factor(QGD_Complex16 global_phase_factor_new){
	global_phase_factor = mult(global_phase_factor, global_phase_factor_new);
	return;
}

/**
@brief Call to get global phase 
@param global_phase_factor The value of the phase
*/
QGD_Complex16 Decomposition_Base::get_global_phase_factor( ){
	return global_phase_factor;
}

/**
@brief Call to set global phase 
@param global_phase_factor_new The value of the new phase
*/
void Decomposition_Base::set_global_phase(double new_global_phase){
	global_phase_factor.real = sqrt(2)*cos(new_global_phase);
	global_phase_factor.imag = sqrt(2)*sin(new_global_phase);
	return;
}

/**
@brief Call to apply global phase of U3 matrices to matrix
@param global_phase_factor The value of the phase
*/
void Decomposition_Base::apply_global_phase_factor(QGD_Complex16 global_phase_factor, Matrix& u3_gate){
	mult(global_phase_factor, u3_gate);
	return;
}

/**
@brief Call to apply the current global phase to the unitary matrix
@param global_phase_factor The value of the phase
*/
void Decomposition_Base::apply_global_phase_factor(){
	mult(global_phase_factor, Umtx);
	set_global_phase(0);
	return;
}


/**
@brief Call to export the unitary (with possible phase shift) into a binary file
@param filename The path to the file where the unitary is expored
*/
void Decomposition_Base::export_unitary(std::string& filename){
	FILE* pFile;
	if (project_name != ""){filename = project_name + "_" + filename;}

	const char* c_filename = filename.c_str();
	pFile = fopen(c_filename, "wb");
    	if (pFile==NULL) {
            fputs ("File error",stderr); 
            std::string error("Cannot open file.");
            throw error;
        }


    	fwrite(&Umtx.rows, sizeof(int), 1, pFile);
   	fwrite(&Umtx.cols, sizeof(int), 1, pFile);          
   	fwrite(Umtx.get_data(), sizeof(QGD_Complex16), Umtx.size(), pFile);
    	fclose(pFile);
	return;
}



/**
@brief Call to import the unitary (with possible phase shift) into a binary file
@param filename The path to the file from which the unitary is imported.
*/
Matrix Decomposition_Base::import_unitary_from_binary(std::string& filename){
	FILE* pFile;

	if (project_name != ""){filename = project_name + "_"  + filename;}

	const char* c_filename = filename.c_str();
	int cols;
	int rows;
	pFile = fopen(c_filename, "rb");
    	if (pFile==NULL) {
            fputs ("File error",stderr); 
            exit (1);
        }

	size_t fread_status;
        fread_status = fread(&rows, sizeof(int), 1, pFile);
	fread_status = fread(&cols, sizeof(int), 1, pFile);

        //TODO error handling for fread_status

	Matrix Umtx_ = Matrix(rows, cols);

	fread_status = fread(Umtx_.get_data(), sizeof(QGD_Complex16), rows*cols, pFile);
    	fclose(pFile);
	return Umtx_;
}



/**
@brief Set the number of qubits spanning the matrix of the gates stored in the block of gates.
@param qbit_num_in The number of qubits spanning the matrices.
*/
void Decomposition_Base::set_qbit_num( int qbit_num_in ) {

    // check the size of the unitary
    int matrix_size_loc = 1 << qbit_num_in;
    if ( matrix_size_loc != matrix_size ) {
        std::string err("Decomposition_Base::set_qbit_num: The new number of qubits is not in line with the input unitary to be decomposed");
        throw err;
    }

    // setting the number of qubits
    Gates_block::set_qbit_num(qbit_num_in);

    
}
