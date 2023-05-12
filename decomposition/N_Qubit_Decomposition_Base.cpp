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
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition_Base.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Adam.h"

#include "RL_experience.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif


static double adam_time = 0;
static double bfgs_time = 0;
static double pure_DFE_time = 0;

static double DFE_time = 0.0;
static double CPU_time = 0.0;


extern "C" int LAPACKE_dgesv( 	int  matrix_layout, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb); 	


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Base::N_Qubit_Decomposition_Base() {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = false;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;

    // The global minimum of the optimization problem
    global_target_minimum = 0;

    // logical variable indicating whether adaptive learning reate is used in the ADAM algorithm
    adaptive_eta = true;

    // parameter to contron the radius of parameter randomization around the curren tminimum
    radius = 1.0;
    randomization_rate = 0.3;

    // The chosen variant of the cost function
    cost_fnc = FROBENIUS_NORM;
    
    number_of_iters = 0;
 


    prev_cost_fnv_val = 1.0;
    //
    correction1_scale = 1/1.7;
    correction2_scale = 1/2.0;  

    iteration_threshold_of_randomization = 2500000;

    // number of utilized accelerators
    accelerator_num = 0;

    // set the trace offset
    trace_offset = 0;

    // unique id indentifying the instance of the class
    std::uniform_int_distribution<> distrib_int(0, INT_MAX);  
    int id = distrib_int(gen);


}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_Base::N_Qubit_Decomposition_Base( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, std::map<std::string, Config_Element>& config, guess_type initial_guess_in= CLOSE_TO_ZERO, int accelerator_num_in ) : Decomposition_Base(Umtx_in, qbit_num_in, config, initial_guess_in) {

    // logical value. Set true if finding the minimum number of gate layers is required (default), or false when the maximal number of two-qubit gates is used (ideal for general unitaries).
    optimize_layer_num  = optimize_layer_num_in;

    // A string describing the type of the class
    type = N_QUBIT_DECOMPOSITION_CLASS;

    // The global minimum of the optimization problem
    global_target_minimum = 0;
    
    number_of_iters = 0;

    // number of iteratrion loops in the optimization
    iteration_loops[2] = 3;

    // filling in numbers that were not given in the input
    for ( std::map<int,int>::iterator it = max_layer_num_def.begin(); it!=max_layer_num_def.end(); it++) {
        if ( max_layer_num.count( it->first ) == 0 ) {
            max_layer_num.insert( std::pair<int, int>(it->first,  it->second) );
        }
    }

    // logical variable indicating whether adaptive learning reate is used in the ADAM algorithm
    adaptive_eta = true;

    // parameter to contron the radius of parameter randomization around the curren tminimum
    radius = 1.0;
    randomization_rate = 0.3;

    // The chosen variant of the cost function
    cost_fnc = FROBENIUS_NORM;
    
    prev_cost_fnv_val = 1.0;
    //
    correction1_scale = 1/1.7;
    correction2_scale = 1/2.0; 

    iteration_threshold_of_randomization = 2500000;

    // set the trace offset
    trace_offset = 0;

    // unique id indentifying the instance of the class
    std::uniform_int_distribution<> distrib_int(0, INT_MAX);  
    id = distrib_int(gen);


#ifdef __DFE__

    // number of utilized accelerators
    accelerator_num = accelerator_num_in;
#else
    accelerator_num = 0;
#endif

}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Base::~N_Qubit_Decomposition_Base() {


#ifdef __DFE__
    if ( qbit_num >= 2 && get_accelerator_num() > 0 ) {
        unload_dfe_lib();//releive_DFE();
    }
#endif      


}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
N_Qubit_Decomposition_Base::add_finalyzing_layer() {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    // adding U3 gate to the block
    bool Theta = true;
    bool Phi = false;
    bool Lambda = true;

    for (int qbit=0; qbit<qbit_num; qbit++) {
        block->add_u3(qbit, Theta, Phi, Lambda);
    }

    // adding the opeartion block to the gates
    add_gate( block );

}




/**
@brief Calculate the error of the decomposition according to the spectral norm of \f$ U-U_{approx} \f$, where \f$ U_{approx} \f$ is the unitary produced by the decomposing quantum cirquit.
@param decomposed_matrix The decomposed matrix, i.e. the result of the decomposing gate structure applied on the initial unitary.
@return Returns with the calculated spectral norm.
*/
void
N_Qubit_Decomposition_Base::calc_decomposition_error(Matrix& decomposed_matrix ) {

	// (U-U_{approx}) (U-U_{approx})^\dagger = 2*I - U*U_{approx}^\dagger - U_{approx}*U^\dagger
	// U*U_{approx}^\dagger = decomposed_matrix_copy
	
 	/*Matrix A(matrix_size, matrix_size);
	QGD_Complex16* A_data = A.get_data();
	QGD_Complex16* decomposed_data = decomposed_matrix.get_data();
	QGD_Complex16 phase;
	phase.real = decomposed_matrix[0].real/(std::sqrt(decomposed_matrix[0].real*decomposed_matrix[0].real + decomposed_matrix[0].imag*decomposed_matrix[0].imag));
	phase.imag = -decomposed_matrix[0].imag/(std::sqrt(decomposed_matrix[0].real*decomposed_matrix[0].real + decomposed_matrix[0].imag*decomposed_matrix[0].imag));

	for (int idx=0; idx<matrix_size; idx++ ) {
		for (int jdx=0; jdx<matrix_size; jdx++ ) {
			
			if (idx==jdx) {
				QGD_Complex16 mtx_val = mult(phase, decomposed_data[idx*matrix_size+jdx]);
				A_data[idx*matrix_size+jdx].real = 2.0 - 2*mtx_val.real;
				A_data[idx*matrix_size+jdx].imag = 0;
			}
			else {
				QGD_Complex16 mtx_val_ij = mult(phase, decomposed_data[idx*matrix_size+jdx]);
				QGD_Complex16 mtx_val_ji = mult(phase, decomposed_data[jdx*matrix_size+idx]);
				A_data[idx*matrix_size+jdx].real = - mtx_val_ij.real - mtx_val_ji.real;
				A_data[idx*matrix_size+jdx].imag = - mtx_val_ij.imag + mtx_val_ji.imag;
			}

		}
	}


	Matrix alpha(matrix_size, 1);
	Matrix beta(matrix_size, 1);
	Matrix B = create_identity(matrix_size);

	// solve the generalized eigenvalue problem of I- 1/2
	LAPACKE_zggev( CblasRowMajor, 'N', 'N',
                          matrix_size, A.get_data(), matrix_size, B.get_data(),
                          matrix_size, alpha.get_data(),
                          beta.get_data(), NULL, matrix_size, NULL,
                          matrix_size );

	// determine the largest eigenvalue
	double eigval_max = 0;
	for (int idx=0; idx<matrix_size; idx++) {
		double eigval_abs = std::sqrt((alpha[idx].real*alpha[idx].real + alpha[idx].imag*alpha[idx].imag) / (beta[idx].real*beta[idx].real + beta[idx].imag*beta[idx].imag));
		if ( eigval_max < eigval_abs ) eigval_max = eigval_abs;		
	}
    
	// the norm is the square root of the largest einegvalue.*/
    if ( cost_fnc == FROBENIUS_NORM ) {
        decomposition_error =  get_cost_function(decomposed_matrix);
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        Matrix_real&& ret = get_cost_function_with_correction(decomposed_matrix, qbit_num);
        decomposition_error = ret[0] - std::sqrt(prev_cost_fnv_val)*ret[1]*correction1_scale;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        Matrix_real&& ret = get_cost_function_with_correction2(decomposed_matrix, qbit_num);
        decomposition_error = ret[0] - std::sqrt(prev_cost_fnv_val)*(ret[1]*correction1_scale + ret[2]*correction2_scale);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST){
       decomposition_error = get_hilbert_schmidt_test(decomposed_matrix);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
        Matrix&& ret = get_trace_with_correction(decomposed_matrix, qbit_num);
        double d = 1.0/decomposed_matrix.cols;
        decomposition_error = 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag));
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
        Matrix&& ret = get_trace_with_correction2(decomposed_matrix, qbit_num);
        double d = 1.0/decomposed_matrix.cols;
        decomposition_error = 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*(correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag)+correction2_scale*(ret[2].real*ret[2].real+ret[2].imag*ret[2].imag)));
    }
    else if ( cost_fnc == SUM_OF_SQUARES) {
        decomposition_error = get_cost_function_sum_of_squares(decomposed_matrix);
    }
    else {
        std::string err("N_Qubit_Decomposition_Base::optimization_problem: Cost function variant not implmented.");
        throw err;
    }

}



/**
@brief final optimization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
*/
void  N_Qubit_Decomposition_Base::final_optimization() {

	//The stringstream input to store the output messages.
	std::stringstream sstream;
	sstream << "***************************************************************" << std::endl;
	sstream << "Final fine tuning of the parameters in the " << qbit_num << "-qubit decomposition" << std::endl;
	sstream << "***************************************************************" << std::endl;
	print(sstream, 1);	    	

         


        //# setting the global minimum
        global_target_minimum = 0;

        if ( optimized_parameters_mtx.size() == 0 ) {
            solve_optimization_problem(NULL, 0);
        }
        else {
            current_minimum = optimization_problem(optimized_parameters_mtx.get_data());
            if ( check_optimization_solution() ) return;

            solve_optimization_problem(optimized_parameters_mtx.get_data(), parameter_num);
        }
}


/**
@brief Call to solve layer by layer the optimization problem via calling one of the implemented algorithms. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem( int num_of_parameters, gsl_vector *solution_guess_gsl) {

    switch ( alg ) {
        case ADAM:
            solve_layer_optimization_problem_ADAM( num_of_parameters, solution_guess_gsl);
            return;
        case ADAM_BATCHED:
            solve_layer_optimization_problem_ADAM_BATCHED( num_of_parameters, solution_guess_gsl);
            return;
        case AGENTS:
            solve_layer_optimization_problem_AGENTS( num_of_parameters, solution_guess_gsl);
            return;
        case COSINE:
            solve_layer_optimization_problem_COSINE( num_of_parameters, solution_guess_gsl);
            return;
        case AGENTS_COMBINED:
            solve_layer_optimization_problem_AGENTS_COMBINED( num_of_parameters, solution_guess_gsl);
            return;
        case BFGS:
            solve_layer_optimization_problem_BFGS( num_of_parameters, solution_guess_gsl);
            return;
        case BFGS2:
            solve_layer_optimization_problem_BFGS2( num_of_parameters, solution_guess_gsl);
            return;
        default:
            std::string error("N_Qubit_Decomposition_Base::solve_layer_optimization_problem: unimplemented optimization algorithm");
            throw error;
    }


}



/**
@brief Call to solve layer by layer the optimization problem via the COSINE algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_COSINE( int num_of_parameters, gsl_vector *solution_guess_gsl) {

#ifdef __DFE__
        if ( qbit_num >= 5 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif



        if (gates.size() == 0 ) {
            return;
        }


        double M_PI_half = M_PI/2;
        double M_PI_double = M_PI*2;


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
            std::uniform_real_distribution<> distrib_real(0, M_PI_double); 
            for ( int idx=0; idx<num_of_parameters; idx++) {
                solution_guess_gsl->data[idx] = distrib_real(gen);
            }
        }



//memset( solution_guess_gsl->data, 0.0, solution_guess_gsl->size*sizeof(double) );
        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }




        std::stringstream sstream;
        double optimization_time = 0.0;
        tbb::tick_count optimization_start = tbb::tick_count::now();


        // the result of the most successful agent:
        current_minimum = optimization_problem( optimized_parameters_mtx );

        

        // the array storing the optimized parameters
        Matrix_real solution_guess_tmp_mtx = Matrix_real( num_of_parameters, 1 );
        memcpy(solution_guess_tmp_mtx.get_data(), optimized_parameters_mtx.get_data(), num_of_parameters*sizeof(double) );

        int agent_num;
        if ( config.count("agent_num") > 0 ) { 
             long long value;                   
             config["agent_num"].get_property( value );  
             agent_num = (int) value;
        }
        else {
            agent_num = 64;
        }


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations") > 0 ) {
             config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance") > 0 ) {
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }



        // vector stroing the lates values of current minimums to identify convergence
        Matrix_real f0_vec(1, 100); 
        memset( f0_vec.get_data(), 0.0, f0_vec.size()*sizeof(double) );
        double f0_mean = 0.0;
        int f0_idx = 0;   


        Matrix_real param_update_mtx( num_of_parameters, 1 );


#ifdef __DFE__

        std::vector<Matrix_real> parameters_mtx_vec(num_of_parameters);
        parameters_mtx_vec.reserve(num_of_parameters); 



        // parameters for line search
        int line_points = 100;
        Matrix_real line_values;

        std::vector<Matrix_real> parameters_line_search_mtx_vec(line_points);
        parameters_line_search_mtx_vec.reserve(line_points); 
        
        Matrix_real f0_shifted_pi2_agents( agent_num, 1 );
        Matrix_real f0_shifted_pi_agents( agent_num, 1 );          
             


        for (unsigned long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++) {

            for( int idx=0; idx<num_of_parameters; idx++ ) {
                parameters_mtx_vec[idx] = solution_guess_tmp_mtx.copy();
            }




            for(int idx=0; idx<num_of_parameters; idx++) { 
                Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ]; 
                solution_guess_mtx_idx[idx] += M_PI_half;                
            }                 
      
            f0_shifted_pi2_agents = optimization_problem_batched( parameters_mtx_vec );  


            for(int idx=0; idx<num_of_parameters; idx++) { 
                Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                solution_guess_mtx_idx[idx] += M_PI_half;
            }   
             
            f0_shifted_pi_agents = optimization_problem_batched( parameters_mtx_vec );


          
                      
            for( int idx=0; idx<num_of_parameters; idx++ ) {
     
                double f0_shifted_pi         = f0_shifted_pi_agents[idx];
                double f0_shifted_pi2        = f0_shifted_pi2_agents[idx];     
            

                double A_times_cos = (current_minimum-f0_shifted_pi)/2;
                double offset      = (current_minimum+f0_shifted_pi)/2;

                double A_times_sin = offset - f0_shifted_pi2;

                double phi0 = atan2( A_times_sin, A_times_cos);


                double parameter_shift = phi0 > 0 ? M_PI-phi0 : -phi0-M_PI;

                    
                param_update_mtx[ idx ] = parameter_shift;	
		
		
                //revert the changed parameters
                Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[idx];  
                solution_guess_mtx_idx[ idx ]       = solution_guess_tmp_mtx[ idx ];  

            }
                    
            // perform line search over the deriction determined previously  
            for( int line_idx=0; line_idx<line_points; line_idx++ ) {

                Matrix_real parameters_line_idx = solution_guess_tmp_mtx.copy();

                for( int idx=0; idx<num_of_parameters; idx++ ) {
                    parameters_line_idx[idx] = std::fmod( parameters_line_idx[idx] + param_update_mtx[ idx ]*line_idx/line_points, M_PI_double);                    
                }

                parameters_line_search_mtx_vec[line_idx] = parameters_line_idx;

            }
                     
            line_values = optimization_problem_batched( parameters_line_search_mtx_vec ); 
//line_values.print_matrix(); 
                   

#else


        for (unsigned long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++) {



            tbb::parallel_for( tbb::blocked_range<int>(0, num_of_parameters, 10 ), [&](const tbb::blocked_range<int>& r) {

                Matrix_real solution_guess_tmp_mtx_loc = solution_guess_tmp_mtx.copy();


                for( int param_idx=r.begin(); param_idx<r.end(); param_idx++ ) {

                    double parameter_value_save = solution_guess_tmp_mtx_loc[param_idx];

                    solution_guess_tmp_mtx_loc[param_idx] += M_PI_half;
                    double f0_shifted_pi2 = optimization_problem( solution_guess_tmp_mtx_loc );

                    solution_guess_tmp_mtx_loc[param_idx] += M_PI_half;
                    double f0_shifted_pi = optimization_problem( solution_guess_tmp_mtx_loc );

                    solution_guess_tmp_mtx_loc[param_idx] += M_PI_half;
                    double f0_shifted_3pi2 = optimization_problem( solution_guess_tmp_mtx_loc );


                    double A_times_cos = (current_minimum-f0_shifted_pi)/2;
                    double A_times_sin = (f0_shifted_3pi2 - f0_shifted_pi2)/2;

                    //double amplitude = np.sqrt( A_times_cos**2 + A_times_sin**2 )
                    //print( "Amplitude: ", amplitude )

                    double phi0 = atan2( A_times_sin, A_times_cos);
                    //print( "phase: ", phi0 )

                    //offset = (f0+f0_shifted_pi)/2
                    //print( "offset: ", offset )


                    double parameter_shift = phi0 > 0 ? M_PI-phi0 : -phi0-M_PI;
		
                    //print( "minimal_parameter: ", minimal_parameter )
		
                    param_update_mtx[ param_idx ] = parameter_shift;	
		
                    //revert the parameter vector
                    solution_guess_tmp_mtx_loc[param_idx] = parameter_value_save;

                }

            });


            // perform line search over the deriction determined previously
            int line_points = 100;
            int grain_size = 10;
            Matrix_real line_values( line_points, 1);
             

            tbb::parallel_for( tbb::blocked_range<int>(0, line_points, grain_size ), [&](const tbb::blocked_range<int>& r) {

                Matrix_real solution_guess_tmp_mtx_loc = solution_guess_tmp_mtx.copy();

                for( int line_idx=r.begin(); line_idx<r.end(); line_idx++ ) {

                    // update parameters
                    for (int param_idx=0; param_idx<num_of_parameters; param_idx++) {
                        solution_guess_tmp_mtx_loc[param_idx] = std::fmod( solution_guess_tmp_mtx[param_idx] + param_update_mtx[ param_idx ]*line_idx/line_points, M_PI_double);
                    } 

                    line_values[line_idx] = optimization_problem( solution_guess_tmp_mtx_loc );


                }

            });


#endif


            // find the smallest value
            double f0_min = line_values[0];
            int idx_min = 0;
            for (int idx=1; idx<line_points; idx++) {
                if ( line_values[idx] < f0_min ) {
                    idx_min = idx;
                    f0_min = line_values[idx];
                }
            }

            current_minimum = f0_min;

            // update parameters
            for (int param_idx=0; param_idx<num_of_parameters; param_idx++) {
                solution_guess_tmp_mtx[param_idx] = std::fmod( solution_guess_tmp_mtx[param_idx] + param_update_mtx[ param_idx ]*idx_min/line_points, M_PI_double);
            } 




            // update the current cost function
            //current_minimum = optimization_problem( solution_guess_tmp_mtx );

            if ( iter_idx % 1000 == 0 ) {
                std::stringstream sstream;
                sstream << "COSINE: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum << std::endl;
                print(sstream, 0);   
                std::string filename("initial_circuit_iteration.binary");
                export_gate_list_to_binary(solution_guess_tmp_mtx, this, filename, verbose);


            }

            if (current_minimum < optimization_tolerance_loc ) {
                break;
            }
            
            
            // test local minimum convergence
            f0_mean = f0_mean + (current_minimum - f0_vec[ f0_idx ])/f0_vec.size();
            f0_vec[ f0_idx ] = current_minimum;
            f0_idx = (f0_idx + 1) % f0_vec.size();
    
            double var_f0 = 0.0;
            for (int idx=0; idx<f0_vec.size(); idx++) {
                var_f0 = var_f0 + (f0_vec[idx]-f0_mean)*(f0_vec[idx]-f0_mean);
            }
            var_f0 = std::sqrt(var_f0)/f0_vec.size();


     
            if ( std::abs( f0_mean - current_minimum) < 1e-7  && var_f0/f0_mean < 1e-7 ) {
                std::stringstream sstream;
                sstream << "COSINE: converged to minimum at iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum << std::endl;
                print(sstream, 0);   
                std::string filename("initial_circuit_iteration.binary");
                export_gate_list_to_binary(solution_guess_tmp_mtx, this, filename, verbose);
                

                break;
            }

        }
        
       

        memcpy( optimized_parameters_mtx.get_data(),  solution_guess_tmp_mtx.get_data(), num_of_parameters*sizeof(double) );
        

        sstream.str("");
        sstream << "obtained minimum: " << current_minimum << std::endl;


        tbb::tick_count optimization_end = tbb::tick_count::now();
        optimization_time  = optimization_time + (optimization_end-optimization_start).seconds();
        sstream << "COS time: " << adam_time << ", pure DFE time:  " << pure_DFE_time << " " << current_minimum << std::endl;
        
        print(sstream, 0); 

}


/**
@brief Call to solve layer by layer the optimization problem via the AGENT algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_AGENTS( int num_of_parameters, gsl_vector *solution_guess_gsl) {


#ifdef __DFE__
        if ( qbit_num >= 5 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif



        if (gates.size() == 0 ) {
            return;
        }


        double M_PI_half = M_PI/2;
        double M_PI_double = M_PI*2;


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
            std::uniform_real_distribution<> distrib_real(0, M_PI_double); 
            for ( int idx=0; idx<num_of_parameters; idx++) {
                solution_guess_gsl->data[idx] = distrib_real(gen);
            }
        }


#ifdef __MPI__        
        MPI_Bcast( (void*)solution_guess_gsl->data, num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif



//memset( solution_guess_gsl->data, 0.0, solution_guess_gsl->size*sizeof(double) );
        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }


        long long sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;
    

        tbb::tick_count optimization_start = tbb::tick_count::now();
        double optimization_time = 0.0;
pure_DFE_time = 0.0;




        
        current_minimum =   optimization_problem( optimized_parameters_mtx );  

        int max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_agent") > 0 ) {
             long long value;
             config["max_inner_iterations_agent"].get_property( value );
             max_inner_iterations_loc = (int) value;
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_agent") > 0 ) {
             double value;
             config["optimization_tolerance_agent"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }

        
        int agent_lifetime_loc;
        if ( config.count("agent_lifetime") > 0 ) {
             long long agent_lifetime_loc_tmp;
             config["agent_lifetime"].get_property( agent_lifetime_loc_tmp );  
             agent_lifetime_loc = (int)agent_lifetime_loc_tmp;
        }
        else {
            agent_lifetime_loc = 1000;
        }        



        std::stringstream sstream;
        sstream << "max_inner_iterations: " << max_inner_iterations_loc << std::endl;
        sstream << "agent_lifetime_loc: " << agent_lifetime_loc << std::endl;
        print(sstream, 2); 



        
        
        /// mutual exclusion to set the most successful agent
        tbb::spin_mutex agent_mutex;
        
        double agent_exploration_rate = 0.2;
        double agent_randomization_rate = 0.2;
        
        int agent_num;
        if ( config.count("agent_num") > 0 ) { 
             long long value;                   
             config["agent_num"].get_property( value );  
             agent_num = (int) value;
        }
        else {
            agent_num = 64;
        }
        
        sstream.str("");
        sstream << "AGENTS: number of agents " << agent_num << std::endl;
        print(sstream, 2);    
    
        
        bool terminate_optimization = false;
        
        // vector stroing the lates values of current minimums to identify convergence
        Matrix_real current_minimum_vec(1, 20); 
        memset( current_minimum_vec.get_data(), 0.0, current_minimum_vec.size()*sizeof(double) );
        double current_minimum_mean = 0.0;
        int current_minimum_idx = 0;   
        
        double var_current_minimum = DBL_MAX; 


        matrix_base<int> param_idx_agents( agent_num, 1 );

        // random generator of integers   
        std::uniform_int_distribution<> distrib_int(0, num_of_parameters-1);

        for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
            // initital paraneter index of the agents
            param_idx_agents[ agent_idx ] = distrib_int(gen);
        }

#ifdef __MPI__        
        MPI_Bcast( (void*)param_idx_agents.get_data(), agent_num, MPI_INT, 0, MPI_COMM_WORLD);
#endif


        int most_successfull_agent = 0;
 


tbb::tick_count t0_CPU = tbb::tick_count::now();

        // vector storing the parameter set usedby the individual agents.

        std::vector<Matrix_real> solution_guess_mtx_agents( agent_num );
        solution_guess_mtx_agents.reserve( agent_num );
        
        std::uniform_real_distribution<> distrib_real(0.0, M_PI_double);         
        
        for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
      
      
            // initialize random parameters for the agent            
            Matrix_real solution_guess_mtx_agent = Matrix_real( num_of_parameters, 1 );
            memset( solution_guess_mtx_agent.get_data(), 0.0, solution_guess_mtx_agent.size()*sizeof(double) );              

#ifdef __MPI__        
            if ( current_rank == 0 ) {
#endif

                if ( agent_idx == 0 ) {
                    memcpy( solution_guess_mtx_agent.get_data(), solution_guess_gsl->data, solution_guess_gsl->size*sizeof(double) );
                }
                else {
                    randomize_parameters( optimized_parameters_mtx, solution_guess_mtx_agent, current_minimum  ); 
                }


#ifdef __MPI__        
            } 
            
            MPI_Bcast( solution_guess_mtx_agent.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

            solution_guess_mtx_agents[ agent_idx ] = solution_guess_mtx_agent;

        }
        
        

        // array storing the current minimum of th eindividual agents
        Matrix_real current_minimum_agents;

        // intitial cost function for each of the agents
        current_minimum_agents = optimization_problem_batched( solution_guess_mtx_agents );


        // arrays to store some parameter values needed to be restored later
        Matrix_real parameter_value_save_agents( agent_num, 1 );    
       
        // arrays to store the cost functions at shifted parameters
        Matrix_real f0_shifted_pi2_agents( agent_num, 1 );
        Matrix_real f0_shifted_pi_agents( agent_num, 1 );                 
       


   
        // CPU time
        CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();       
       
        ///////////////////////////////////////////////////////////////////////////
        for (long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++) {
        
        
            // CPU time
            t0_CPU = tbb::tick_count::now();        


#ifdef __MPI__        

            memset( param_idx_agents.get_data(), 0, param_idx_agents.size()*sizeof(int) );
            memset( parameter_value_save_agents.get_data(), 0.0, parameter_value_save_agents.size()*sizeof(double) );            

            if ( current_rank == 0 ) {
#endif
        
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
            
                    // agent local parameter set
                    Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                
                    // determine parameter indices to be altered
                    int param_idx       = distrib_int(gen);
                    param_idx_agents[agent_idx] = param_idx;
                
                    // save the parameters to  be restored later
                    parameter_value_save_agents[agent_idx] = solution_guess_mtx_agent[param_idx];                
                
                                   
                }
       
#ifdef __MPI__  
            }
                  
            MPI_Bcast( (void*)param_idx_agents.get_data(), agent_num, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast( (void*)parameter_value_save_agents.get_data(), agent_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);            
#endif        
                      
                      
            // calsulate the cist functions at shifted parameter values
            for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ]; 
                solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;                
            }   
            
            // CPU time              
            CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();  
            
            // calculate batched cost function                 
            f0_shifted_pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );              
            
            // CPU time
            t0_CPU = tbb::tick_count::now();                                         
                        

            for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];             
                solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;
            }  
            
            // CPU time             
            CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();        
            
            // calculate batched cost function                         
            f0_shifted_pi_agents = optimization_problem_batched( solution_guess_mtx_agents );             
                
                                                     
            // CPU time                                      
            t0_CPU = tbb::tick_count::now();                                  
            
            
            // determine the parameters of the cosine function and determine the parameter shift at the minimum
            for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {

                double current_minimum_agent = current_minimum_agents[agent_idx];         
                double f0_shifted_pi         = f0_shifted_pi_agents[agent_idx];
                double f0_shifted_pi2        = f0_shifted_pi2_agents[agent_idx];                                                  
            

                double A_times_cos = (current_minimum_agent-f0_shifted_pi)/2;
                double offset      = (current_minimum_agent+f0_shifted_pi)/2;

                double A_times_sin = offset - f0_shifted_pi2;

                double phi0 = atan2( A_times_sin, A_times_cos);


                double parameter_shift = phi0 > 0 ? M_PI-phi0 : -phi0-M_PI;
		
		
                //update  the parameter vector
                Matrix_real& solution_guess_mtx_agent                    = solution_guess_mtx_agents[ agent_idx ];                             
                solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift; 
                
            }
               
            // CPU time                                                     
            CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();
            
            
            // determine the current minimum  at the shifted parameters        
            current_minimum_agents = optimization_problem_batched( solution_guess_mtx_agents ); 
  
            // CPU time                                        
            t0_CPU = tbb::tick_count::now();        


            // generate random numbers to manage the behavior of the agents
            Matrix_real random_numbers(   agent_num, 2 );
            memset( random_numbers.get_data(), 0.0, 2*agent_num*sizeof(double) );
            
#ifdef __MPI__        
            if ( current_rank == 0 ) {
#endif

                std::uniform_real_distribution<> distrib_to_choose(0.0, 1.0);

                for ( int agent_idx=0; agent_idx<2*agent_num; agent_idx++ ) {           
                    random_numbers[agent_idx] = distrib_to_choose( gen );
                }

#ifdef __MPI__    
            }    
            MPI_Bcast( random_numbers.get_data(), 2*agent_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif   



            // build up probability distribution to use to chose between the agents
            Matrix_real agent_probs(  current_minimum_agents.size(), 1 );

            // create probability distribution in each 1000-th iteration
            if ( iter_idx % agent_lifetime_loc == 0 ) {
                double prob_sum = 0.0;
                double current_minimum_agents_min = DBL_MAX;
                for( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {
                    if ( current_minimum_agents_min > current_minimum_agents[agent_idx] ) {
                        current_minimum_agents_min = current_minimum_agents[agent_idx];
                    }
                }


                for( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {
                    double prob_loc = exp( (current_minimum_agents_min - current_minimum_agents[agent_idx])*40.0/current_minimum_agents_min );
                    agent_probs[agent_idx] = prob_loc;
                    prob_sum = prob_sum + prob_loc;
                }

                for( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {
                    agent_probs[agent_idx] = agent_probs[agent_idx]/prob_sum;
                }


            }

            
            // govern the behavior of the agents
            for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {
                double& current_minimum_agent = current_minimum_agents[ agent_idx ];
                   
           
                
                if (current_minimum_agents[agent_idx] < optimization_tolerance_loc ) {
                    terminate_optimization = true;                    
                }  
                
               
                
                Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];                             

                // look for the best agent periodicaly
                if ( iter_idx % agent_lifetime_loc == 0 )
                {
                             
                    if ( current_minimum_agent <= current_minimum ) {

                        most_successfull_agent = agent_idx;
                    
                        // export the parameters of the curremt, most successful agent
                        memcpy(optimized_parameters_mtx.get_data(), solution_guess_mtx_agent.get_data(), num_of_parameters*sizeof(double) );

                        std::string filename("initial_circuit_iteration.binary");
                        if (project_name != "") { 
                            filename=project_name+ "_"  +filename;
                        }
                        export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);

                        
                        current_minimum = current_minimum_agent;      
                        
                                   
                        
                    }
                    else {
                        // less successful agent migh choose to keep their current state, or choose the state of more successful agents
                        
#ifdef __MPI__        
                        if ( current_rank == 0 ) {
#endif
                                                
                            double random_num = random_numbers[ agent_idx*random_numbers.stride ]; 
                                         
                            if ( random_num < agent_exploration_rate && agent_idx != most_successfull_agent) {
                                // choose the state of the most succesfull agent
                            
                                std::stringstream sstream;
                                sstream << "agent " << agent_idx << ": adopts the state of the most succesful agent. " << most_successfull_agent << std::endl;
                                print(sstream, 5);  

                            
                                random_num = random_numbers[ agent_idx*random_numbers.stride + 1 ];
                            
                                if ( random_num < agent_randomization_rate ) {
                                    randomize_parameters( optimized_parameters_mtx, solution_guess_mtx_agent, radius  );                              
                                }     

                       
                            }
                            else {
                                // keep the current state  of the agent                    
                            }

#ifdef __MPI__        
                        }
                        
                        MPI_Bcast( (void*)solution_guess_mtx_agent.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif     
                                                                                  
                    }
                 

                    // test global convergence 
                    if ( agent_idx == 0 ) {
                        current_minimum_mean = current_minimum_mean + (current_minimum - current_minimum_vec[ current_minimum_idx ])/current_minimum_vec.size();
                        current_minimum_vec[ current_minimum_idx ] = current_minimum;
                        current_minimum_idx = (current_minimum_idx + 1) % current_minimum_vec.size();
    
                        var_current_minimum = 0.0;
                        for (int idx=0; idx<current_minimum_vec.size(); idx++) {
                            var_current_minimum = var_current_minimum + (current_minimum_vec[idx]-current_minimum_mean)*(current_minimum_vec[idx]-current_minimum_mean);
                        }
                        var_current_minimum = std::sqrt(var_current_minimum)/current_minimum_vec.size();
                                  
                            
                        if ( std::abs( current_minimum_mean - current_minimum) < 1e-7  && var_current_minimum < 1e-7 ) {
                            std::stringstream sstream;
                            sstream << "AGENTS, iterations converged to "<< current_minimum << std::endl;
                            print(sstream, 0); 
                            terminate_optimization = true;
                        }                    
                   }   
                    

                    
                }   


                if ( iter_idx % 2000 == 0 && agent_idx == 0) {
                    std::stringstream sstream;
                    sstream << "AGENTS, agent " << agent_idx << ": processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%";
                    sstream << ", current minimum of agent 0: " << current_minimum_agents[ 0 ] << " global current minimum: " << current_minimum  << " CPU time: " << CPU_time;
                    sstream << " DFE_time: " << DFE_time << " pure DFE time: " << pure_DFE_time << std::endl;
                    print(sstream, 0); 
                }


           
                
#ifdef __MPI__    
                MPI_Barrier(MPI_COMM_WORLD);
#endif                                       
                
            }  // for agent_idx                        
CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();       
                                  
            
            // terminate the agent if the whole optimization problem was solved
            if ( terminate_optimization ) {                   
                break;                    
            }      
        
        }



        tbb::tick_count optimization_end = tbb::tick_count::now();
        optimization_time  = optimization_time + (optimization_end-optimization_start).seconds();
        sstream << "AGENTS time: " << adam_time << ", pure DFE time:  " << pure_DFE_time << " " << current_minimum << std::endl;

}



/**
@brief Call to solve layer by layer the optimization problem via the AGENT COMBINED algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_AGENTS_COMBINED( int num_of_parameters, gsl_vector *solution_guess_gsl)  {



    optimized_parameters_mtx = Matrix_real(solution_guess_gsl->data, solution_guess_gsl->size, 1);

    for( int loop_idx=0; loop_idx<1; loop_idx++ ) {

        gsl_vector *solution_guess_gsl_AGENTS = gsl_vector_alloc(num_of_parameters);
        memcpy( solution_guess_gsl_AGENTS->data, optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );

        solve_layer_optimization_problem_AGENTS( num_of_parameters, solution_guess_gsl_AGENTS );


        gsl_vector *solution_guess_gsl_COSINE = gsl_vector_alloc(num_of_parameters);
        memcpy( solution_guess_gsl_COSINE->data, optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );

        solve_layer_optimization_problem_BFGS( num_of_parameters, solution_guess_gsl_COSINE );

    }
        

}

/**
@brief Call to solve layer by layer the optimization problem via batched ADAM algorithm. (optimal for larger problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_ADAM_BATCHED( int num_of_parameters, gsl_vector *solution_guess_gsl) {


#ifdef __DFE__
        if ( qbit_num >= 2 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif



        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }
//memset( solution_guess_gsl->data, 0.0, solution_guess_gsl->size*sizeof(double) );

        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }


        int random_shift_count = 0;
        int sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;
    

        tbb::tick_count adam_start = tbb::tick_count::now();
        adam_time = 0.0;
pure_DFE_time = 0.0;
        Adam optimizer;
        optimizer.initialize_moment_and_variance( num_of_parameters );



        // the array storing the optimized parameters
        gsl_vector* grad_gsl = gsl_vector_alloc(num_of_parameters);
        gsl_vector* solution_guess_tmp = gsl_vector_alloc(num_of_parameters);
        memcpy(solution_guess_tmp->data, solution_guess_gsl->data, num_of_parameters*sizeof(double) );

        Matrix_real solution_guess_tmp_mtx = Matrix_real( solution_guess_tmp->data, num_of_parameters, 1 );
        Matrix_real grad_mtx = Matrix_real( grad_gsl->data, num_of_parameters, 1 );
        //solution_guess_tmp_mtx.print_matrix();


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_agent") > 0 ) {
             config["max_inner_iterations_agent"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }

        double f0 = DBL_MAX;
        std::stringstream sstream;
        sstream << "iter_max: " << max_inner_iterations_loc << ", randomization threshold: " << iteration_threshold_of_randomization << ", randomization radius: " << radius << std::endl;
        print(sstream, 2); 

        int ADAM_status = 0;



        Matrix Umtx_orig = Umtx;
        int batch_size_min = Umtx_orig.cols*5/6;
        std::uniform_int_distribution<> distrib_trace_offset(0, Umtx_orig.cols-batch_size_min);


        int batch_num = 100;
        for (int batch_idx=0; batch_idx<batch_num; batch_idx++ ) {

            trace_offset = distrib_trace_offset(gen);

            std::uniform_int_distribution<> distrib_col_num(batch_size_min, Umtx_orig.cols-trace_offset);
            int col_num = distrib_col_num(gen);

            // create a slice from the original Umtx
            Matrix Umtx_slice(Umtx_orig.rows, col_num);
            for (int row_idx=0; row_idx<Umtx_orig.rows; row_idx++) {
                memcpy( Umtx_slice.get_data() + row_idx*Umtx_slice.stride, Umtx_orig.get_data() + row_idx*Umtx_orig.stride + trace_offset, col_num*sizeof(QGD_Complex16) );
            }

            Umtx = Umtx_slice;

            for ( long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++ ) {

            


                optimization_problem_combined( solution_guess_tmp, (void*)(this), &f0, grad_gsl );

                prev_cost_fnv_val = f0;
  
                if (sub_iter_idx == 1 ) {
                    current_minimum_hold = f0;   
               
                    if ( adaptive_eta )  { 
                        optimizer.eta = optimizer.eta > 1e-3 ? optimizer.eta : 1e-3; 
                        //std::cout << "reset learning rate to " << optimizer.eta << std::endl;
                    }                 

                }


                if (current_minimum_hold*0.95 > f0 || (current_minimum_hold*0.97 > f0 && f0 < 1e-3) ||  (current_minimum_hold*0.99 > f0 && f0 < 1e-4) ) {
                    sub_iter_idx = 0;
                    current_minimum_hold = f0;        
                }
    
    
                if (current_minimum > f0 ) {
                    current_minimum = f0;
                    memcpy( optimized_parameters_mtx.get_data(),  solution_guess_tmp->data, num_of_parameters*sizeof(double) );
                    //double new_eta = 1e-3 * f0 * f0;
                
                    if ( adaptive_eta )  {
                        double new_eta = 1e-3 * f0;
                        optimizer.eta = new_eta > 1e-6 ? new_eta : 1e-6;
                        optimizer.eta = new_eta < 1e-1 ? new_eta : 1e-1;
                    }
                
                }
    

                if ( iter_idx % 5000 == 0 ) {

                    Matrix matrix_new = get_transformed_matrix( optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );

                    std::stringstream sstream;
                    sstream << "ADAM: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100;
                    sstream << "\%, current minimum:" << current_minimum << ", pure cost function:" << get_cost_function(matrix_new, trace_offset) << std::endl;
                    print(sstream, 0);   
                    std::string filename("initial_circuit_iteration.binary");
                    export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);

                }

//std::cout << grad_norm  << std::endl;
                if (f0 < optimization_tolerance || random_shift_count > random_shift_count_max ) {
                    break;
                }



                // calculate the gradient norm
                double norm = 0.0;
                for ( int grad_idx=0; grad_idx<num_of_parameters; grad_idx++ ) {
                    norm += grad_gsl->data[grad_idx]*grad_gsl->data[grad_idx];
                }
                norm = std::sqrt(norm);
                    

                if ( sub_iter_idx> iteration_threshold_of_randomization || ADAM_status != 0 ) {

                    //random_shift_count++;
                    sub_iter_idx = 0;
                    random_shift_count++;
                    current_minimum_hold = current_minimum;   


                
                    std::stringstream sstream;
                    if ( ADAM_status == 0 ) {
                        sstream << "ADAM: initiate randomization at " << f0 << ", gradient norm " << norm << std::endl;
                    }
                    else {
                        sstream << "ADAM: leaving local minimum " << f0 << ", gradient norm " << norm << " eta: " << optimizer.eta << std::endl;
                    }
                    print(sstream, 0);   
                    
                    Matrix_real solution_guess_tmp_mtx( solution_guess_tmp->data, solution_guess_tmp->size, 1);
                    randomize_parameters(optimized_parameters_mtx, solution_guess_tmp_mtx, f0 );
        
                    optimizer.reset();
                    optimizer.initialize_moment_and_variance( num_of_parameters );   

                    ADAM_status = 0;   

                    //optimizer.eta = 1e-3;
        
                }

                else {
                    ADAM_status = optimizer.update(solution_guess_tmp_mtx, grad_mtx, f0);
                }

                sub_iter_idx++;

            }




        }

        sstream.str("");
        sstream << "obtained minimum: " << current_minimum << std::endl;


        gsl_vector_free(grad_gsl);
        gsl_vector_free(solution_guess_tmp);
        tbb::tick_count adam_end = tbb::tick_count::now();
        adam_time  = adam_time + (adam_end-adam_start).seconds();
        sstream << "adam time: " << adam_time << ", pure DFE time:  " << pure_DFE_time << " " << f0 << std::endl;
        
        print(sstream, 0); 
        

}

/**
@brief Call to solve layer by layer the optimization problem via ADAM algorithm. (optimal for larger problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_ADAM( int num_of_parameters, gsl_vector *solution_guess_gsl) {

#ifdef __DFE__
        if ( qbit_num >= 2 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif



        if (gates.size() == 0 ) {
            return;
        }



        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }
//memset( solution_guess_gsl->data, 0.0, solution_guess_gsl->size*sizeof(double) );

        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }

        int random_shift_count = 0;
        long long sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;
    

        tbb::tick_count adam_start = tbb::tick_count::now();
        adam_time = 0.0;
pure_DFE_time = 0.0;
        Adam optimizer;
        optimizer.initialize_moment_and_variance( num_of_parameters );



        // the array storing the optimized parameters
        gsl_vector* grad_gsl = gsl_vector_alloc(num_of_parameters);
        gsl_vector* solution_guess_tmp = gsl_vector_alloc(num_of_parameters);
        memcpy(solution_guess_tmp->data, solution_guess_gsl->data, num_of_parameters*sizeof(double) );

        Matrix_real solution_guess_tmp_mtx = Matrix_real( solution_guess_tmp->data, num_of_parameters, 1 );
        Matrix_real grad_mtx = Matrix_real( grad_gsl->data, num_of_parameters, 1 );
        //solution_guess_tmp_mtx.print_matrix();






        int ADAM_status = 0;

        int randomization_successful = 0;


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations") > 0 ) {
            config["max_inner_iterations"].get_property( max_inner_iterations_loc );         
        }
        else {
            max_inner_iterations_loc =max_inner_iterations;
        }

        long long iteration_threshold_of_randomization_loc;
        if ( config.count("randomization_threshold") > 0 ) {
            config["randomization_threshold"].get_property( iteration_threshold_of_randomization_loc );  
        }
        else {
            iteration_threshold_of_randomization_loc = iteration_threshold_of_randomization;
        }
        
        
        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance") > 0 ) {
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }        


        double f0 = DBL_MAX;
        std::stringstream sstream;
        sstream << "max_inner_iterations: " << max_inner_iterations_loc << ", randomization threshold: " << iteration_threshold_of_randomization_loc << ", randomization radius: " << radius << std::endl;
        print(sstream, 2); 
        

        for ( long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++ ) {

            number_of_iters++;


            optimization_problem_combined( solution_guess_tmp, (void*)(this), &f0, grad_gsl );

            prev_cost_fnv_val = f0;
  
            if (sub_iter_idx == 1 ) {
                current_minimum_hold = f0;   
               
                if ( adaptive_eta )  { 
                    optimizer.eta = optimizer.eta > 1e-3 ? optimizer.eta : 1e-3; 
                    //std::cout << "reset learning rate to " << optimizer.eta << std::endl;
                }                 

            }


            if (current_minimum_hold*0.95 > f0 || (current_minimum_hold*0.97 > f0 && f0 < 1e-3) ||  (current_minimum_hold*0.99 > f0 && f0 < 1e-4) ) {
                sub_iter_idx = 0;
                current_minimum_hold = f0;        
            }
    
    
            if (current_minimum > f0 ) {
                current_minimum = f0;
                memcpy( optimized_parameters_mtx.get_data(),  solution_guess_tmp->data, num_of_parameters*sizeof(double) );
                //double new_eta = 1e-3 * f0 * f0;
                
                if ( adaptive_eta )  {
                    double new_eta = 1e-3 * f0;
                    optimizer.eta = new_eta > 1e-6 ? new_eta : 1e-6;
                    optimizer.eta = new_eta < 1e-1 ? new_eta : 1e-1;
                }
                
                randomization_successful = 1;
            }
    

            if ( iter_idx % 5000 == 0 ) {

                Matrix_real solution_guess_tmp_mtx( solution_guess_tmp->data, solution_guess_tmp->size, 1 );
                Matrix matrix_new = get_transformed_matrix( solution_guess_tmp_mtx, gates.begin(), gates.size(), Umtx );

                std::stringstream sstream;
                sstream << "ADAM: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum << ", current cost function:" << get_cost_function(matrix_new, trace_offset) << ", sub_iter_idx:" << sub_iter_idx <<std::endl;
                print(sstream, 0);   
                std::string filename("initial_circuit_iteration.binary");
                export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);
            }

//std::cout << grad_norm  << std::endl;
            if (f0 < optimization_tolerance_loc || random_shift_count > random_shift_count_max ) {
                break;
            }



                // calculate the gradient norm
                double norm = 0.0;
                for ( int grad_idx=0; grad_idx<num_of_parameters; grad_idx++ ) {
                    norm += grad_gsl->data[grad_idx]*grad_gsl->data[grad_idx];
                }
                norm = std::sqrt(norm);
                
//grad_mtx.print_matrix();
/*
            if ( ADAM_status == 0 && norm > 0.01 && optimizer.eta < 1e-4) {

                std::uniform_real_distribution<> distrib_prob(0.0, 1.0);
                if ( distrib_prob(gen) < 0.05 ) {
                    optimizer.eta = optimizer.eta*10;
                    std::cout << "Increasing learning rate at " << f0 << " to " << optimizer.eta << std::endl;
                }

            }
*/
/*

            if ( ADAM_status == 1 && norm > 0.01 ) {
                optimizer.eta = optimizer.eta > 1e-5 ? optimizer.eta/10 : 1e-6;
                std::cout << "Decreasing learning rate at " << f0 << " to " << optimizer.eta << std::endl;
                ADAM_status = 0;
            }

  */       

            if ( sub_iter_idx> iteration_threshold_of_randomization_loc || ADAM_status != 0 ) {

                //random_shift_count++;
                sub_iter_idx = 0;
                random_shift_count++;
                current_minimum_hold = current_minimum;   


                
                std::stringstream sstream;
                if ( ADAM_status == 0 ) {
                    sstream << "ADAM: initiate randomization at " << f0 << ", gradient norm " << norm << std::endl;
                }
                else {
                    sstream << "ADAM: leaving local minimum " << f0 << ", gradient norm " << norm << " eta: " << optimizer.eta << std::endl;
                }
                print(sstream, 0);   
                    
                 Matrix_real solution_guess_gsl_mtx( solution_guess_gsl->data, solution_guess_gsl->size, 1 );
                randomize_parameters(optimized_parameters_mtx, solution_guess_tmp_mtx, f0 );
                randomization_successful = 0;
        
                optimizer.reset();
                optimizer.initialize_moment_and_variance( num_of_parameters );   

                ADAM_status = 0;   

                //optimizer.eta = 1e-3;
        
            }

            else {
                ADAM_status = optimizer.update(solution_guess_tmp_mtx, grad_mtx, f0);
            }

            sub_iter_idx++;

        }
        sstream.str("");
        sstream << "obtained minimum: " << current_minimum << std::endl;


        gsl_vector_free(grad_gsl);
        gsl_vector_free(solution_guess_tmp);
        tbb::tick_count adam_end = tbb::tick_count::now();
        adam_time  = adam_time + (adam_end-adam_start).seconds();
        sstream << "adam time: " << adam_time << ", pure DFE time:  " << pure_DFE_time << " " << f0 << std::endl;

        print(sstream, 0); 

}



/**
@brief Call to solve layer by layer the optimization problem via BBFG algorithm. (optimal for smaller problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_BFGS( int num_of_parameters, gsl_vector *solution_guess_gsl) {


#ifdef __DFE__
        if ( qbit_num >= 2 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif

        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }


        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }

        // maximal number of iteration loops
        int iteration_loops_max;
        try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }

        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        // maximal number of inner iterations overriden by config
        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations") > 0 ) {
            config["max_inner_iterations"].get_property( max_inner_iterations_loc );         
        }
        else {
            max_inner_iterations_loc =max_inner_iterations;
        }


        // do the optimization loops
        for (int idx=0; idx<iteration_loops_max; idx++) {
	    
            long long iter = 0;
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

            gsl_multimin_fdfminimizer_set(s, &my_func, solution_guess_gsl, 0.01, 0.1);

            do {
                iter++;
                number_of_iters++;
                gsl_set_error_handler_off();
                status = gsl_multimin_fdfminimizer_iterate (s);

                if (status) {
                  break;
                }

                status = gsl_multimin_test_gradient (s->gradient, gradient_threshold);

            } while (status == GSL_CONTINUE && iter < max_inner_iterations_loc);

            if (current_minimum > s->f) {
                current_minimum = s->f;
                memcpy( optimized_parameters_mtx.get_data(), s->x->data, num_of_parameters*sizeof(double) );
                gsl_multimin_fdfminimizer_free (s);

                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + distrib_real(gen)/100;
                }
            }
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = solution_guess_gsl->data[jdx] + distrib_real(gen);
                }
                gsl_multimin_fdfminimizer_free (s);
            }

#ifdef __MPI__        
            MPI_Bcast( (void*)solution_guess_gsl->data, num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif



        }

}



/**
@brief Call to solve layer by layer the optimization problem via BFGS algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void N_Qubit_Decomposition_Base::solve_layer_optimization_problem_BFGS2( int num_of_parameters, gsl_vector *solution_guess_gsl) {


#ifdef __DFE__
        if ( qbit_num >= 2 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif


        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess_gsl == NULL) {
            solution_guess_gsl = gsl_vector_alloc(num_of_parameters);
        }


        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess_gsl->data, num_of_parameters*sizeof(double) );
        }

        // maximal number of iteration loops
        int iteration_loops_max;
        try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }


        int random_shift_count = 0;
        long long sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;





tbb::tick_count bfgs_start = tbb::tick_count::now();
bfgs_time = 0.0;


        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        // random generator of integers   
        std::uniform_int_distribution<> distrib_int(0, 5000);  


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations") > 0 ) {
            config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc =max_inner_iterations;
        }


        long long iteration_threshold_of_randomization_loc;
        if ( config.count("randomization_threshold") > 0 ) {
            config["randomization_threshold"].get_property( iteration_threshold_of_randomization_loc );  
        }
        else {
            iteration_threshold_of_randomization_loc = iteration_threshold_of_randomization;
        }

        std::stringstream sstream;
        sstream << "max_inner_iterations: " << max_inner_iterations_loc << ", randomization threshold: " << iteration_threshold_of_randomization_loc << std::endl;
        print(sstream, 2); 

        // do the optimization loops
        for (long long idx=0; idx<iteration_loops_max; idx++) {

            long long iter_idx = 0;
            int status = GSL_CONTINUE;

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

            gsl_multimin_fdfminimizer_set(s, &my_func, solution_guess_gsl, 0.01, 0.1);

            do {
                gsl_set_error_handler_off();
                
                if ( sub_iter_idx > iteration_threshold_of_randomization_loc || status != GSL_CONTINUE ) {

                    
                    sub_iter_idx = 0;
                    random_shift_count++;
                    current_minimum_hold = current_minimum;  

                    // calculate the gradient norm
                    gsl_vector* grad_gsl = gsl_vector_alloc(num_of_parameters);
                    optimization_problem_grad( solution_guess_gsl, this, grad_gsl );
                    double norm = 0.0;
                    for ( int grad_idx=0; grad_idx<num_of_parameters; grad_idx++ ) {
                        norm += grad_gsl->data[grad_idx]*grad_gsl->data[grad_idx];
                    }
                    norm = std::sqrt(norm);  
                    gsl_vector_free( grad_gsl );


                    std::stringstream sstream;
                    sstream << "BFGS2: leaving local minimum " << s->f << ", gradient norm " << norm  << std::endl;                    
                    print(sstream, 0);   
                    
                    Matrix_real solution_guess_gsl_mtx( solution_guess_gsl->data, solution_guess_gsl->size, 1 );
                    randomize_parameters(optimized_parameters_mtx, solution_guess_gsl_mtx, s->f );    

#ifdef __MPI__        
                    MPI_Bcast( (void*)solution_guess_gsl->data, num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
                    
                    status = 0;    
                    
                    gsl_multimin_fdfminimizer_free (s);                     
                    s = gsl_multimin_fdfminimizer_alloc (T, num_of_parameters);  
                    gsl_multimin_fdfminimizer_set(s, &my_func, solution_guess_gsl, 0.01, 0.1);                             
        
                }
                else {
                    status = gsl_multimin_fdfminimizer_iterate (s);
                }
                                
/*
                if (status) {
                  break;
                }
*/
                status = gsl_multimin_test_gradient (s->gradient, gradient_threshold);
                
                
                if (sub_iter_idx == 1 ) {
                     current_minimum_hold = s->f;    
                }


                if (current_minimum_hold*0.95 > s->f || (current_minimum_hold*0.97 > s->f && s->f < 1e-3) ||  (current_minimum_hold*0.99 > s->f && s->f < 1e-4) ) {
                     sub_iter_idx = 0;
                     current_minimum_hold = s->f;        
                }
    
    
                if (current_minimum > s->f ) {
                     current_minimum = s->f;
                     memcpy( optimized_parameters_mtx.get_data(),  s->x->data, num_of_parameters*sizeof(double) );
                }
    

                if ( iter_idx % 5000 == 0 ) {
                     std::stringstream sstream;
                     sstream << "BFGS2: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum << std::endl;
                     print(sstream, 2);  

                     std::string filename("initial_circuit_iteration.binary");
                     export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);
                }


                if (s->f < optimization_tolerance || random_shift_count > random_shift_count_max ) {
                    break;
                }


                sub_iter_idx++;
                iter_idx++;
                number_of_iters++;

            } while (iter_idx < max_inner_iterations_loc && s->f > optimization_tolerance);

            if (current_minimum > s->f) {
                current_minimum = s->f;
                memcpy( optimized_parameters_mtx.get_data(), s->x->data, num_of_parameters*sizeof(double) );                

                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = optimized_parameters_mtx[jdx] + distrib_real(gen)*2*M_PI/100;
                }
            }
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess_gsl->data[jdx] = optimized_parameters_mtx[jdx] + distrib_real(gen)*2*M_PI;
                }
            }

#ifdef __MPI__        
            MPI_Bcast( (void*)solution_guess_gsl->data, num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            
            gsl_multimin_fdfminimizer_free (s);
            
            if (current_minimum < optimization_tolerance ) {
                break;
            }



        }

        tbb::tick_count bfgs_end = tbb::tick_count::now();
        bfgs_time  = bfgs_time + (bfgs_end-bfgs_start).seconds();
        std::cout << "bfgs2 time: " << bfgs_time << " " << current_minimum << std::endl;

}


/**
@brief Call to randomize the parameter.
@param input The parameters are randomized around the values stores in this array
@param output The randomized parameters are stored within this array
@param f0 weight in the randomiztaion (output = input + rand()*sqrt(f0) ).
*/
void N_Qubit_Decomposition_Base::randomize_parameters( Matrix_real& input, Matrix_real& output, const double& f0  ) {

    // random generator of real numbers   
    std::uniform_real_distribution<> distrib_prob(0.0, 1.0);
    std::uniform_real_distribution<> distrib_real(-2*M_PI, 2*M_PI);


    double radius_loc;
    if ( config.count("Randomized_Radius") > 0 ) {
        config["Randomized_Radius"].get_property( radius_loc );  
    }
    else {
        radius_loc = radius;
    }

    const int num_of_parameters = input.size();

    int changed_parameters = 0;
    for ( int jdx=0; jdx<num_of_parameters; jdx++) {
        if ( distrib_prob(gen) <= randomization_rate ) {
            output[jdx] = input[jdx] + distrib_real(gen)*std::sqrt(f0)*radius_loc;
            changed_parameters++;
        }
        else {
            output[jdx] = input[jdx];
        }
    }

#ifdef __MPI__  
        //MPI_Bcast( (void*)output.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


}


/**
// @brief The optimization problem of the final optimization
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
    
    return optimization_problem( parameters_mtx );


}


/**
// @brief The optimization problem of the final optimization
// @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
// @return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( Matrix_real& parameters ) {

    // get the transformed matrix with the gates in the list
    if ( parameters.size() != parameter_num ) {
        std::stringstream sstream;
	sstream << "N_Qubit_Decomposition_Base::optimization_problem: Number of free paramaters should be " << parameter_num << ", but got " << parameters.size() << std::endl;
        print(sstream, 0);	  
        exit(-1);
    }


    Matrix matrix_new = get_transformed_matrix( parameters, gates.begin(), gates.size(), Umtx );
//matrix_new.print_matrix();

    if ( cost_fnc == FROBENIUS_NORM ) {
        return get_cost_function(matrix_new, trace_offset);
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        Matrix_real&& ret = get_cost_function_with_correction(matrix_new, qbit_num, trace_offset);
        return ret[0] - std::sqrt(prev_cost_fnv_val)*ret[1]*correction1_scale;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        Matrix_real&& ret = get_cost_function_with_correction2(matrix_new, qbit_num, trace_offset);
        return ret[0] - std::sqrt(prev_cost_fnv_val)*(ret[1]*correction1_scale + ret[2]*correction2_scale);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST){
        return get_hilbert_schmidt_test(matrix_new);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
        Matrix&& ret = get_trace_with_correction(matrix_new, qbit_num);
        double d = 1.0/matrix_new.cols;
        return 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag));
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
        Matrix&& ret = get_trace_with_correction2(matrix_new, qbit_num);
        double d = 1.0/matrix_new.cols;
        return 1 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(prev_cost_fnv_val)*(correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag)+correction2_scale*(ret[2].real*ret[2].real+ret[2].imag*ret[2].imag)));
    }
    else if ( cost_fnc == SUM_OF_SQUARES) {
        return get_cost_function_sum_of_squares(matrix_new);
    }
    else {
        std::string err("N_Qubit_Decomposition_Base::optimization_problem: Cost function variant not implmented.");
        throw err;
    }

}


/**
@brief The optimization problem of the final optimization with batched input (implemented only for the Frobenius norm cost function)
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
Matrix_real 
N_Qubit_Decomposition_Base::optimization_problem_batched( std::vector<Matrix_real>& parameters_vec) {

tbb::tick_count t0_DFE = tbb::tick_count::now();        


        Matrix_real cost_fnc_mtx(parameters_vec.size(), 1);
        
#ifdef __DFE__
    if ( get_accelerator_num() > 0 ) {
        int gatesNum, gateSetNum, redundantGateSets;
        DFEgate_kernel_type* DFEgates = convert_to_batched_DFE_gates( parameters_vec, gatesNum, gateSetNum, redundantGateSets );                        
            
        Matrix_real trace_DFE_mtx(gateSetNum, 3);
        
tbb::tick_count t0_DFE_pure = tbb::tick_count::now();         
#ifdef __MPI__
        // the number of decomposing layers are divided between the MPI processes

        int mpi_gateSetNum = gateSetNum / world_size;
        int mpi_starting_gateSetIdx = gateSetNum/world_size * current_rank;

        Matrix_real mpi_trace_DFE_mtx(mpi_gateSetNum, 3);

        lock_lib();
        calcqgdKernelDFE( Umtx.rows, Umtx.cols, DFEgates+mpi_starting_gateSetIdx*gatesNum, gatesNum, mpi_gateSetNum, trace_offset, mpi_trace_DFE_mtx.get_data() );
        unlock_lib();

        int bytes = mpi_trace_DFE_mtx.size()*sizeof(double);
        MPI_Allgather(mpi_trace_DFE_mtx.get_data(), bytes, MPI_BYTE, trace_DFE_mtx.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);

#else
        lock_lib();
        calcqgdKernelDFE( Umtx.rows, Umtx.cols, DFEgates, gatesNum, gateSetNum, trace_offset, trace_DFE_mtx.get_data() );
        unlock_lib();    
                                                                      
#endif  
pure_DFE_time += (tbb::tick_count::now() - t0_DFE_pure).seconds();       

        // calculate the cost function
        for ( int idx=0; idx<parameters_vec.size(); idx++ ) {
            cost_fnc_mtx[idx] = 1-trace_DFE_mtx[idx*3]/Umtx.cols;
        }


        delete[] DFEgates;

    }
    else{

#endif

        tbb::parallel_for( 0, (int)parameters_vec.size(), 1, [&]( int idx) {
            cost_fnc_mtx[idx] = optimization_problem( parameters_vec[idx] );
        });
       

   


#ifdef __DFE__  
    }
#endif

DFE_time += (tbb::tick_count::now() - t0_DFE).seconds();       
    return cost_fnc_mtx;
        
}


/**
// @brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param ret_temp A matrix to store trace in for gradient for HS test 
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( const gsl_vector* parameters, void* void_instance, Matrix ret_temp) {

    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);
    std::vector<Gate*> gates_loc = instance->get_gates();

    // get the transformed matrix with the gates in the list
    Matrix Umtx_loc = instance->get_Umtx();
    Matrix_real parameters_mtx(parameters->data, 1, instance->get_parameter_num() );
    Matrix matrix_new = instance->get_transformed_matrix( parameters_mtx, gates_loc.begin(), gates_loc.size(), Umtx_loc );

  
    cost_function_type cost_fnc = instance->get_cost_function_variant();

    if ( cost_fnc == FROBENIUS_NORM ) {
        return get_cost_function(matrix_new, instance->get_trace_offset());
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        double correction1_scale    = instance->get_correction1_scale();
        Matrix_real&& ret = get_cost_function_with_correction(matrix_new, instance->get_qbit_num(), instance->get_trace_offset());
        return ret[0] - 0*std::sqrt(instance->get_previous_cost_function_value())*ret[1]*correction1_scale;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        double correction1_scale    = instance->get_correction1_scale();
        double correction2_scale    = instance->get_correction2_scale();            
        Matrix_real&& ret = get_cost_function_with_correction2(matrix_new, instance->get_qbit_num(), instance->get_trace_offset());
        return ret[0] - std::sqrt(instance->get_previous_cost_function_value())*(ret[1]*correction1_scale + ret[2]*correction2_scale);
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST){
    	QGD_Complex16 trace_temp = get_trace(matrix_new);
    	ret_temp[0].real = trace_temp.real;
    	ret_temp[0].imag = trace_temp.imag;
    	double d = 1.0/matrix_new.cols;
        return 1.0-d*d*trace_temp.real*trace_temp.real-d*d*trace_temp.imag*trace_temp.imag;
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
        double correction1_scale = instance->get_correction1_scale();
        Matrix&& ret = get_trace_with_correction(matrix_new, instance->get_qbit_num());
        double d = 1.0/matrix_new.cols;
        for (int idx=0; idx<3; idx++){ret_temp[idx].real=ret[idx].real;ret_temp[idx].imag=ret[idx].imag;}
        return 1.0 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(instance->get_previous_cost_function_value())*correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag));
    }
    else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
        double correction1_scale    = instance->get_correction1_scale();
        double correction2_scale    = instance->get_correction2_scale();            
        Matrix&& ret = get_trace_with_correction2(matrix_new, instance->get_qbit_num());
        double d = 1.0/matrix_new.cols;
        for (int idx=0; idx<4; idx++){ret_temp[idx].real=ret[idx].real;ret_temp[idx].imag=ret[idx].imag;}
        return 1.0 - d*d*(ret[0].real*ret[0].real+ret[0].imag*ret[0].imag+std::sqrt(instance->get_previous_cost_function_value())*(correction1_scale*(ret[1].real*ret[1].real+ret[1].imag*ret[1].imag)+correction2_scale*(ret[2].real*ret[2].real+ret[2].imag*ret[2].imag)));
    }
    else if ( cost_fnc == SUM_OF_SQUARES) {
        return get_cost_function_sum_of_squares(matrix_new);
    }
    else {
        std::string err("N_Qubit_Decomposition_Base::optimization_problem: Cost function variant not implmented.");
        throw err;
    }


}

/**
// @brief The optimization problem of the final optimization
@param parameters A GNU Scientific Library containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double N_Qubit_Decomposition_Base::optimization_problem( const gsl_vector* parameters, void* void_instance) {
    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);
    Matrix ret(1,3);
    double cost_func = instance->optimization_problem(parameters, void_instance, ret);
    return cost_func;
}

Matrix_real N_Qubit_Decomposition_Base::optimization_problem_batch( int batchsize, const gsl_vector* parameters, void* void_instance) {
    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);
    Matrix_real costs(batchsize,1);
    // the number of free parameters
    int parameter_num_loc = instance->get_parameter_num();
#ifdef __DFE__
    if ( instance->qbit_num >= 2 && instance->get_accelerator_num() > 0 ) {
        int trace_offset_loc = instance->get_trace_offset();
        // the variant of the cost function
        cost_function_type cost_fnc = instance->get_cost_function_variant();
    
        // value of the cost function from the previous iteration to weigth the correction to the trace
        double prev_cost_fnv_val = instance->get_previous_cost_function_value();
        double correction1_scale    = instance->get_correction1_scale();
        double correction2_scale    = instance->get_correction2_scale();    
        Matrix&& Umtx_loc = instance->get_Umtx();
        Matrix_real trace_DFE_mtx(batchsize, 3);
        int gatesNum;
        Matrix_real parameters_mtx(parameters->data, 1, parameters->size);
        DFEgate_kernel_type* DFEgates = instance->convert_to_DFE_gates( parameters_mtx, gatesNum);
        lock_lib();
        calcqgdKernelDFE( Umtx_loc.rows, Umtx_loc.cols, DFEgates, parameter_num_loc, batchsize, trace_offset_loc, trace_DFE_mtx.get_data() );
        unlock_lib();
        for (int idx=0; idx<batchsize; idx++) {
            if ( cost_fnc == FROBENIUS_NORM ) {
                costs[idx] = 1-trace_DFE_mtx[3*idx]/Umtx_loc.cols;
            } else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
                costs[idx] = 1 - (trace_DFE_mtx[3*idx] + 0*std::sqrt(prev_cost_fnv_val)*trace_DFE_mtx[3*idx+1]*correction1_scale)/Umtx_loc.cols;
            }
            else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
                costs[idx] = 1 - (trace_DFE_mtx[3*idx] + std::sqrt(prev_cost_fnv_val)*(trace_DFE_mtx[3*idx+1]*correction1_scale + trace_DFE_mtx[3*idx+2]*correction2_scale))/Umtx_loc.cols;
            }
            else {
                std::string err("N_Qubit_Decomposition_Base::optimization_problem_batch: Cost function variant not implmented.");
                throw err;
            }
        }
    } else {
#else
    tbb::parallel_for( tbb::blocked_range<int>(0,batchsize,2), [&](tbb::blocked_range<int> r) {
        Matrix ret(batchsize,3);
        for (int idx=r.begin(); idx<r.end(); ++idx) {
            gsl_vector_view view = gsl_vector_subvector(parameters, parameter_num_loc * idx, parameter_num_loc);
            costs[idx] = instance->optimization_problem(&view.vector, void_instance, ret);
        }
    });
#endif
#ifdef __DFE__
    }
#endif
    return costs;
}

/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
void N_Qubit_Decomposition_Base::optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad ) {

    // The function value at x0
    double f0;

    // calculate the approximate gradient
    optimization_problem_combined( parameters, void_instance, &f0, grad);

}


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
void N_Qubit_Decomposition_Base::optimization_problem_combined( const gsl_vector* parameters, void* void_instance, double* f0, gsl_vector* grad ) {

    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);

    // the number of free parameters
    int parameter_num_loc = instance->get_parameter_num();

    // the variant of the cost function
    cost_function_type cost_fnc = instance->get_cost_function_variant();

    // value of the cost function from the previous iteration to weigth the correction to the trace
    double prev_cost_fnv_val = instance->get_previous_cost_function_value();
    double correction1_scale    = instance->get_correction1_scale();
    double correction2_scale    = instance->get_correction2_scale();    

    int qbit_num = instance->get_qbit_num();
    int trace_offset_loc = instance->get_trace_offset();

#ifdef __DFE__

///////////////////////////////////////
//std::cout << "number of qubits: " << instance->qbit_num << std::endl;
//tbb::tick_count t0_DFE = tbb::tick_count::now();/////////////////////////////////    
if ( instance->qbit_num >= 5 && instance->get_accelerator_num() > 0 ) {
    Matrix_real parameters_mtx(parameters->data, 1, parameters->size);

    int gatesNum, redundantGateSets, gateSetNum;
    DFEgate_kernel_type* DFEgates = instance->convert_to_DFE_gates_with_derivates( parameters_mtx, gatesNum, gateSetNum, redundantGateSets );

    Matrix&& Umtx_loc = instance->get_Umtx();   
    Matrix_real trace_DFE_mtx(gateSetNum, 3);


#ifdef __MPI__
    // the number of decomposing layers are divided between the MPI processes

    int mpi_gateSetNum = gateSetNum / instance->world_size;
    int mpi_starting_gateSetIdx = gateSetNum/instance->world_size * instance->current_rank;

    Matrix_real mpi_trace_DFE_mtx(mpi_gateSetNum, 3);

    lock_lib();
    calcqgdKernelDFE( Umtx_loc.rows, Umtx_loc.cols, DFEgates+mpi_starting_gateSetIdx*gatesNum, gatesNum, mpi_gateSetNum, trace_offset_loc, mpi_trace_DFE_mtx.get_data() );
    unlock_lib();

    int bytes = mpi_trace_DFE_mtx.size()*sizeof(double);
    MPI_Allgather(mpi_trace_DFE_mtx.get_data(), bytes, MPI_BYTE, trace_DFE_mtx.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);

#else

    lock_lib();
    calcqgdKernelDFE( Umtx_loc.rows, Umtx_loc.cols, DFEgates, gatesNum, gateSetNum, trace_offset_loc, trace_DFE_mtx.get_data() );
    unlock_lib();

#endif  

    std::stringstream sstream;
    sstream << *f0 << " " << 1.0 - trace_DFE_mtx[0]/Umtx_loc.cols << " " << trace_DFE_mtx[1]/Umtx_loc.cols << " " << trace_DFE_mtx[2]/Umtx_loc.cols << std::endl;
    instance->print(sstream, 5);	
    
  
    if ( cost_fnc == FROBENIUS_NORM ) {
        *f0 = 1-trace_DFE_mtx[0]/Umtx_loc.cols;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
        *f0 = 1 - (trace_DFE_mtx[0] + 0*std::sqrt(prev_cost_fnv_val)*trace_DFE_mtx[1]*correction1_scale)/Umtx_loc.cols;
    }
    else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
        *f0 = 1 - (trace_DFE_mtx[0] + std::sqrt(prev_cost_fnv_val)*(trace_DFE_mtx[1]*correction1_scale + trace_DFE_mtx[2]*correction2_scale))/Umtx_loc.cols;
    }
    else {
        std::string err("N_Qubit_Decomposition_Base::optimization_problem_combined: Cost function variant not implmented.");
        throw err;
    }

    //double f0_DFE = *f0;

    //Matrix_real grad_components_DFE_mtx(1, parameter_num_loc);
    for (int idx=0; idx<parameter_num_loc; idx++) {

        if ( cost_fnc == FROBENIUS_NORM ) {
            gsl_vector_set(grad, idx, -trace_DFE_mtx[3*(idx+1)]/Umtx_loc.cols);
        }
        else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
            gsl_vector_set(grad, idx, -(trace_DFE_mtx[3*(idx+1)] + std::sqrt(prev_cost_fnv_val)*trace_DFE_mtx[3*(idx+1)+1]*correction1_scale)/Umtx_loc.cols);
        }
        else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
            gsl_vector_set(grad, idx, -(trace_DFE_mtx[3*(idx+1)] + std::sqrt(prev_cost_fnv_val)*(trace_DFE_mtx[3*(idx+1)+1]*correction1_scale + trace_DFE_mtx[3*(idx+1)+2]*correction2_scale))/Umtx_loc.cols );
        }
        else {
            std::string err("N_Qubit_Decomposition_Base::optimization_problem_combined: Cost function variant not implmented.");
            throw err;
        }

        //grad_components_DFE_mtx[idx] = gsl_vector_get( grad, idx );


    }

    delete[] DFEgates;

//tbb::tick_count t1_DFE = tbb::tick_count::now();/////////////////////////////////
//std::cout << "uploaded data to DFE: " << (int)(gatesNum*gateSetNum*sizeof(DFEgate_kernel_type)) << " bytes" << std::endl;
//std::cout << "time elapsed DFE: " << (t1_DFE-t0_DFE).seconds() << ", expected time: " << (((double)Umtx_loc.rows*(double)Umtx_loc.cols*gatesNum*gateSetNum/get_chained_gates_num()/4 + 4578*3*get_chained_gates_num()))/350000000 + 0.001<< std::endl;

///////////////////////////////////////
}
else {

#endif

#ifdef __DFE__
tbb::tick_count t0_CPU = tbb::tick_count::now();/////////////////////////////////
#endif

    Matrix_real cost_function_terms;

    // vector containing gradients of the transformed matrix
    std::vector<Matrix> Umtx_deriv;
    Matrix trace_tmp(1,3);

    tbb::parallel_invoke(
        [&]{
            *f0 = instance->optimization_problem(parameters, reinterpret_cast<void*>(instance), trace_tmp); 
        },
        [&]{
            Matrix&& Umtx_loc = instance->get_Umtx();   
            Matrix_real parameters_mtx(parameters->data, 1, parameters->size);
            Umtx_deriv = instance->apply_derivate_to( parameters_mtx, Umtx_loc );
        });




    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            double grad_comp;
            if ( cost_fnc == FROBENIUS_NORM ) {
                grad_comp = (get_cost_function(Umtx_deriv[idx], trace_offset_loc) - 1.0);
            }
            else if ( cost_fnc == FROBENIUS_NORM_CORRECTION1 ) {
                Matrix_real deriv_tmp = get_cost_function_with_correction( Umtx_deriv[idx], qbit_num, trace_offset_loc );
                grad_comp = (deriv_tmp[0] - std::sqrt(prev_cost_fnv_val)*deriv_tmp[1]*correction1_scale - 1.0);
            }
            else if ( cost_fnc == FROBENIUS_NORM_CORRECTION2 ) {
                Matrix_real deriv_tmp = get_cost_function_with_correction2( Umtx_deriv[idx], qbit_num, trace_offset_loc );
                grad_comp = (deriv_tmp[0] - std::sqrt(prev_cost_fnv_val)*(deriv_tmp[1]*correction1_scale + deriv_tmp[2]*correction2_scale) - 1.0);
            }
            else if (cost_fnc == HILBERT_SCHMIDT_TEST){
                double d = 1.0/Umtx_deriv[idx].cols;
                QGD_Complex16 deriv_tmp = get_trace(Umtx_deriv[idx]);
                grad_comp = -2.0*d*d*trace_tmp[0].real*deriv_tmp.real-2.0*d*d*trace_tmp[0].imag*deriv_tmp.imag;
            }
            else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION1 ){
                Matrix&&  deriv_tmp = get_trace_with_correction( Umtx_deriv[idx], qbit_num);
                double d = 1.0/Umtx_deriv[idx].cols;
                grad_comp = -2.0*d*d* (trace_tmp[0].real*deriv_tmp[0].real+trace_tmp[0].imag*deriv_tmp[0].imag+std::sqrt(prev_cost_fnv_val)*correction1_scale*(trace_tmp[1].real*deriv_tmp[1].real+trace_tmp[1].imag*deriv_tmp[1].imag));
            }
            else if ( cost_fnc == HILBERT_SCHMIDT_TEST_CORRECTION2 ){
                Matrix&& deriv_tmp = get_trace_with_correction2( Umtx_deriv[idx], qbit_num);
                double d = 1.0/Umtx_deriv[idx].cols;
                grad_comp = -2.0*d*d* (trace_tmp[0].real*deriv_tmp[0].real+trace_tmp[0].imag*deriv_tmp[0].imag+std::sqrt(prev_cost_fnv_val)*(correction1_scale*(trace_tmp[1].real*deriv_tmp[1].real+trace_tmp[1].imag*deriv_tmp[1].imag) + correction2_scale*(trace_tmp[2].real*deriv_tmp[2].real+trace_tmp[2].imag*deriv_tmp[2].imag)));
            }
            else if ( cost_fnc == SUM_OF_SQUARES) {
                grad_comp = get_cost_function_sum_of_squares(Umtx_deriv[idx]);
            }
            else {
                std::string err("N_Qubit_Decomposition_Base::optimization_problem_combined: Cost function variant not implmented.");
                throw err;
            }
            
//            grad->data[idx] = grad_comp;
            gsl_vector_set(grad, idx, grad_comp);



        }
    });

    std::stringstream sstream;
    sstream << *f0 << std::endl;
    instance->print(sstream, 5);	

#ifdef __DFE__
}

/*
tbb::tick_count t1_CPU = tbb::tick_count::now();/////////////////////////////////
std::cout << "time elapsed CPU: " << (t1_CPU-t0_CPU).seconds() << " number of parameters: " << parameter_num_loc << std::endl;
std::cout << "cost function CPU: " << *f0 << " and DFE: " << f0_DFE << std::endl;

for ( int idx=0; idx<parameter_num_loc; idx++ ) {

    double diff = std::sqrt((grad_components_DFE_mtx[idx]-gsl_vector_get(grad, idx))*(grad_components_DFE_mtx[idx]-gsl_vector_get(grad, idx)));
    if ( diff > 1e-5 ) {
        std::cout << "DFE and CPU cost functions differs at index " << idx << " " <<  grad_components_DFE_mtx[idx] << " and " <<  gsl_vector_get(grad, idx) << std::endl;
        
    }   

}



std::cout << "N_Qubit_Decomposition_Base::optimization_problem_combined" << std::endl;
std::string error("N_Qubit_Decomposition_Base::optimization_problem_combined");
        throw error;
*/
#endif

/*

    // adjust gradient components corresponding to adaptive gates
    for (int idx=3*qbit_num; idx<parameter_num_loc; idx=idx+7 ) {
        double grad_comp = gsl_vector_get(grad, idx);
        grad_comp = grad_comp * std::sin( parameters->data[idx])*0.5*M_PI;
        gsl_vector_set(grad, idx, grad_comp);
    }
*/

}

void N_Qubit_Decomposition_Base::optimization_problem_combined_unitary( const gsl_vector* parameters, void* void_instance, Matrix& Umtx, std::vector<Matrix>& Umtx_deriv ) {
    // vector containing gradients of the transformed matrix
    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);

    tbb::parallel_invoke(
        [&]{
            Matrix Umtx_loc = instance->get_Umtx();
            Matrix_real parameters_mtx(parameters->data, 1, parameters->size);
            Umtx = instance->get_transformed_matrix( parameters_mtx, instance->gates.begin(), instance->gates.size(), Umtx_loc );
        },
        [&]{
            Matrix Umtx_loc = instance->get_Umtx();
            Matrix_real parameters_mtx(parameters->data, 1, parameters->size);
            Umtx_deriv = instance->apply_derivate_to( parameters_mtx, Umtx_loc );
        });
}

/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters The parameters for which the cost fuction shoule be calculated
@param f0 The value of the cost function at x0.
@param grad An array storing the calculated gradient components
*/
void N_Qubit_Decomposition_Base::optimization_problem_combined( const Matrix_real& parameters, double* f0, Matrix_real& grad ) {

#ifdef __DFE__

    lock_lib();

    if ( get_accelerator_num() > 0 && get_initialize_id() != id ) {
        std::string err("The uploaded unitary to the DFE might not be identical to the unitary stored by this specific class instance. Please upload the unitary to DFE by the Upload_Umtx_to_DFE() method.");
        throw err;
    }

#endif

    // create GSL wrappers around the pointers
    gsl_block block_tmp;
    block_tmp.data = parameters.get_data();
    block_tmp.size = parameters.size(); 

    gsl_vector parameters_gsl;
    parameters_gsl.data = parameters.get_data();
    parameters_gsl.size = parameters.size();
    parameters_gsl.stride = 1;   
    parameters_gsl.block = &block_tmp; 
    parameters_gsl.owner = 0; 
    
    
    gsl_block block_tmp2;
    block_tmp.data = grad.get_data();
    block_tmp.size = grad.size();        

    gsl_vector grad_gsl;
    grad_gsl.data = grad.get_data();
    grad_gsl.size = grad.size();
    grad_gsl.stride = 1;   
    grad_gsl.block = &block_tmp2; 
    grad_gsl.owner = 0;    

    // call the method to calculate the cost function and the gradients
    optimization_problem_combined( &parameters_gsl, this, f0, &grad_gsl );

#ifdef __DFE__
    unlock_lib();
#endif

}

void N_Qubit_Decomposition_Base::optimization_problem_combined_unitary( const Matrix_real& parameters, Matrix& Umtx, std::vector<Matrix>& Umtx_deriv ) {

    // create GSL wrappers around the pointers
    gsl_block block_tmp;
    block_tmp.data = parameters.get_data();
    block_tmp.size = parameters.size(); 

    gsl_vector parameters_gsl;
    parameters_gsl.data = parameters.get_data();
    parameters_gsl.size = parameters.size();
    parameters_gsl.stride = 1;   
    parameters_gsl.block = &block_tmp; 
    parameters_gsl.owner = 0; 

    // call the method to calculate the cost function and the gradients
    optimization_problem_combined_unitary( &parameters_gsl, this, Umtx, Umtx_deriv );

}

Matrix_real N_Qubit_Decomposition_Base::optimization_problem_batch( Matrix_real parameters )
{
    // create GSL wrappers around the pointers
    gsl_block block_tmp;
    block_tmp.data = parameters.get_data();
    block_tmp.size = parameters.size(); 

    gsl_vector parameters_gsl;
    parameters_gsl.data = parameters.get_data();
    parameters_gsl.size = parameters.size();
    parameters_gsl.stride = 1; //assert parameters.cols == parameters.stride == get_num_parameters()...   
    parameters_gsl.block = &block_tmp; 
    parameters_gsl.owner = 0; 
    
    Matrix_real result = optimization_problem_batch(parameters.rows, &parameters_gsl, this);
    return result;
}


/**
@brief Call to get the variant of the cost function used in the calculations
*/
cost_function_type 
N_Qubit_Decomposition_Base::get_cost_function_variant() {

    return cost_fnc;

}


/**
@brief Call to set the variant of the cost function used in the calculations
@param variant The variant of the cost function from the enumaration cost_function_type
*/
void 
N_Qubit_Decomposition_Base::set_cost_function_variant( cost_function_type variant  ) {

    cost_fnc = variant;

    std::stringstream sstream;
    sstream << "N_Qubit_Decomposition_Base::set_cost_function_variant: Cost function variant set to " << cost_fnc << std::endl;
    print(sstream, 2);	


}



/**
@brief ?????????????
*/
void N_Qubit_Decomposition_Base::set_max_inner_iterations( int max_inner_iterations_in  ) {

    max_inner_iterations = max_inner_iterations_in;
    
}



/**
@brief ?????????????
*/
void N_Qubit_Decomposition_Base::set_random_shift_count_max( int random_shift_count_max_in  ) {

    random_shift_count_max = random_shift_count_max_in;

}


/**
@brief ?????????????
*/
void N_Qubit_Decomposition_Base::set_optimizer( optimization_aglorithms alg_in ) {

    alg = alg_in;

    switch ( alg ) {
        case ADAM:
            max_inner_iterations = 1e5; 
            random_shift_count_max = 100;
            gradient_threshold = 1e-8;
            max_outer_iterations = 1;
            return;

        case ADAM_BATCHED:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            gradient_threshold = 1e-8;
            max_outer_iterations = 1;
            return;

        case COSINE:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            gradient_threshold = 1e-8;
            max_outer_iterations = 1;
            return;

        case AGENTS:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            gradient_threshold = 1e-8;
            max_outer_iterations = 1;
            return;

        case AGENTS_COMBINED:
            max_inner_iterations = 2.5e3;
            random_shift_count_max = 3;
            gradient_threshold = 1e-8;
            max_outer_iterations = 1;
            return;

        case BFGS:
            max_inner_iterations = 100;
            gradient_threshold = 1e-1;
            random_shift_count_max = 1;  
            max_outer_iterations = 1e8; 
            return;

        case BFGS2:
            max_inner_iterations = 1e5;
            random_shift_count_max = 100;
            gradient_threshold = 1e-8;
            max_outer_iterations = 1;
            return;

        default:
            std::string error("N_Qubit_Decomposition_Base::solve_layer_optimization_problem: unimplemented optimization algorithm");
            throw error;
    }



}



/**
@brief ?????????????
*/
void 
N_Qubit_Decomposition_Base::set_adaptive_eta( bool adaptive_eta_in  ) {

    adaptive_eta = adaptive_eta_in;

}



/**
@brief ?????????????
*/
void 
N_Qubit_Decomposition_Base::set_randomized_radius( double radius_in  ) {

    radius = radius_in;

}


/**
@brief ???????????
*/
double 
N_Qubit_Decomposition_Base::get_previous_cost_function_value() {

    return prev_cost_fnv_val;

}



/**
@brief ???????????
*/
double 
N_Qubit_Decomposition_Base::get_correction1_scale() {

    return correction1_scale;

}


/**
@brief ??????????????
@param ?????????
*/
void 
N_Qubit_Decomposition_Base::get_correction1_scale( const double& scale ) {


    correction1_scale = scale;

}




/**
@brief ???????????
*/
double 
N_Qubit_Decomposition_Base::get_correction2_scale() {

    return correction2_scale;

}


/**
@brief ??????????????
@param ?????????
*/
void 
N_Qubit_Decomposition_Base::get_correction2_scale( const double& scale ) {


    correction2_scale = scale;

}




/**
@brief ???????????
*/
long 
N_Qubit_Decomposition_Base::get_iteration_threshold_of_randomization() {

    return iteration_threshold_of_randomization;

}


/**
@brief ??????????????
@param ?????????
*/
void 
N_Qubit_Decomposition_Base::set_iteration_threshold_of_randomization( const unsigned long long& threshold ) {


    iteration_threshold_of_randomization = threshold;

}
/**
@brief Get the number of iterations.
*/
int 
N_Qubit_Decomposition_Base::get_num_iters() {

    return number_of_iters;

}


/**
@brief Get the trace offset used in the evaluation of the cost function
*/
int 
N_Qubit_Decomposition_Base::get_trace_offset() {

    return trace_offset;

}


/**
@brief Set the trace offset used in the evaluation of the cost function
*/
void 
N_Qubit_Decomposition_Base::set_trace_offset(int trace_offset_in) {


    if ( (trace_offset_in + Umtx.cols) > Umtx.rows ) {
        std::string error("N_Qubit_Decomposition_Base::set_trace_offset: trace offset must be smaller or equal to the difference of the rows and columns in the input unitary.");
        throw error;

    }

    
    trace_offset = trace_offset_in;


    std::stringstream sstream;
    sstream << "N_Qubit_Decomposition_Base::set_trace_offset: trace offset set to " << trace_offset << std::endl;
    print(sstream, 2);	

}


#ifdef __DFE__

void 
N_Qubit_Decomposition_Base::upload_Umtx_to_DFE() {

    lock_lib();

    if ( get_initialize_id() != id ) {
        // initialize DFE library
        init_dfe_lib( accelerator_num, qbit_num, id );
    }

    uploadMatrix2DFE( Umtx );


    unlock_lib();

}


/**
@brief Get the number of accelerators to be reserved on DFEs on users demand. 
*/
int 
N_Qubit_Decomposition_Base::get_accelerator_num() {

    return accelerator_num;

}


#endif

