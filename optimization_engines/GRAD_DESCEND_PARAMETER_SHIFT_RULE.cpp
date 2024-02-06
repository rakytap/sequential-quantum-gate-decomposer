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
/*! \file GRAD_DESCEND_PARAMETER_SHIFT_RULE
    \brief Implementation of the GRAD_DESCEND_PARAMETER_SHIFT_RULE optimization srategy
*/


#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"


#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif


/**
@brief Call to solve layer by layer the optimization problem via the GRAD_DESCEND_PARAMETER_SHIFT_RULE algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_GRAD_DESCEND_PARAMETER_SHIFT_RULE( int num_of_parameters, Matrix_real& solution_guess) {


        if ( cost_fnc != FROBENIUS_NORM && cost_fnc != VQE ) {
            std::string err("Optimization_Interface::solve_layer_optimization_problem_COSINE: Only cost functions FROBENIUS_NORM and VQE are implemented for this strategy");
            throw err;
        }


#ifdef __DFE__
        if ( qbit_num >= 5 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif

        tbb::tick_count t0_CPU = tbb::tick_count::now();

        if (gates.size() == 0 ) {
            return;
        }



        double M_PI_half    = M_PI/2;
        double M_PI_double  = M_PI*2.0;


        if (solution_guess.size() == 0 ) {
            solution_guess = Matrix_real(num_of_parameters,1);
            std::uniform_real_distribution<> distrib_real(0, M_PI_double); 
            for ( int idx=0; idx<num_of_parameters; idx++) {
                solution_guess[idx] = distrib_real(gen);
            }

        }



        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );
        }




        std::stringstream sstream;
        double optimization_time = 0.0;
        tbb::tick_count optimization_start = tbb::tick_count::now();


        // the current result
        current_minimum = optimization_problem( optimized_parameters_mtx );
        number_of_iters = number_of_iters + 1; 
        

        // the array storing the optimized parameters
        Matrix_real solution_guess_tmp_mtx = Matrix_real( num_of_parameters, 1 );
        memcpy(solution_guess_tmp_mtx.get_data(), optimized_parameters_mtx.get_data(), num_of_parameters*sizeof(double) );

        int batch_size;
        if ( config.count("batch_size_grad_descend_shift_rule") > 0 ) { 
             long long value;                   
             config["batch_size_grad_descend_shift_rule"].get_property( value );  
             batch_size = (int) value;
        }
        else if ( config.count("batch_size") > 0 ) { 
             long long value;                   
             config["batch_size"].get_property( value );  
             batch_size = (int) value;
        }
        else {
            batch_size = 64;
        }


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_grad_descend_shift_rule") > 0 ) {
             config["max_inner_iterations_grad_descend_shift_rule"].get_property( max_inner_iterations_loc );  
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
             config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }
        
        
        long long export_circuit_2_binary_loc;
        if ( config.count("export_circuit_2_binary_grad_descend_shift_rule") > 0 ) {
             config["export_circuit_2_binary_grad_descend_shift_rule"].get_property( export_circuit_2_binary_loc );  
        }
        else if ( config.count("export_circuit_2_binary") > 0 ) {
             config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
        }
        else {
            export_circuit_2_binary_loc = 0;
        }        


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_grad_descend_shift_rule") > 0 ) {
             config["optimization_tolerance_grad_descend_shift_rule"].get_property( optimization_tolerance_loc );  
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
             double value;
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }


        // the learning rate
        double eta_loc;
        if ( config.count("eta_grad_descend_shift_rule") > 0 ) {
             config["eta_grad_descend_shift_rule"].get_property( eta_loc ); 
        }
        if ( config.count("eta") > 0 ) {
             long long tmp;
             config["eta"].get_property( eta_loc );  
        }
        else {
            eta_loc = 1e-3;
        }


        // vector stroing the lates values of current minimums to identify convergence
        Matrix_real f0_vec(1, 100); 
        memset( f0_vec.get_data(), 0.0, f0_vec.size()*sizeof(double) );
        double f0_mean = 0.0;
        int f0_idx = 0; 



        Matrix_real param_update_mtx( batch_size, 1 );
        matrix_base<int> param_idx_agents( batch_size, 1 );

        // random generator of integers   
        std::uniform_int_distribution<> distrib_int(0, num_of_parameters-1);

        std::vector<Matrix_real> parameters_mtx_vec(batch_size);
        parameters_mtx_vec.reserve(batch_size); 

                 
             

        bool three_point_line_search               =    cost_fnc == FROBENIUS_NORM;
        bool three_point_line_search_double_period =    cost_fnc == VQE;


        for (unsigned long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++) {

            for( int idx=0; idx<batch_size; idx++ ) {
                parameters_mtx_vec[idx] = solution_guess_tmp_mtx.copy();

                // The index array of the chosen parameters
                param_idx_agents[ idx ] = distrib_int(gen);
            }


#ifdef __MPI__             
            MPI_Bcast( (void*)param_idx_agents.get_data(), batch_size, MPI_INT, 0, MPI_COMM_WORLD);
#endif     


          
            for(int idx=0; idx<batch_size; idx++) { 
                Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ]; 
                solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_half;                
            }                 
      
            Matrix_real f0_shifted_pi2_agents = optimization_problem_batched( parameters_mtx_vec );  


            for(int idx=0; idx<batch_size; idx++) { 
                Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                solution_guess_mtx_idx[ param_idx_agents[idx] ] -= M_PI;
            }   
             
            Matrix_real f0_shifted_pi_agents = optimization_problem_batched( parameters_mtx_vec );

            for( int idx=0; idx<batch_size; idx++ ) {
     
                double f0_shifted_pi         = f0_shifted_pi_agents[idx];
                double f0_shifted_pi2        = f0_shifted_pi2_agents[idx];     
            



                double grad_component = 0.5*(f0_shifted_pi2 - f0_shifted_pi);

                                   
                param_update_mtx[ idx ] = grad_component * eta_loc;

                //revert the changed parameters
                Matrix_real& solution_guess_mtx_idx             = parameters_mtx_vec[idx];  
                solution_guess_mtx_idx[ param_idx_agents[idx] ] = solution_guess_tmp_mtx[ param_idx_agents[idx] ];  	

            }

            

            // parameters for line search
            int line_points = 128;  

            std::vector<Matrix_real> parameters_line_search_mtx_vec(line_points);
            parameters_line_search_mtx_vec.reserve(line_points);         
                    
            // perform line search over the deriction determined previously  
            for( int line_idx=0; line_idx<line_points; line_idx++ ) {

                Matrix_real parameters_line_idx = solution_guess_tmp_mtx.copy();

                for( int idx=0; idx<batch_size; idx++ ) {
                    parameters_line_idx[ param_idx_agents[idx] ] += param_update_mtx[ idx ]*(double)line_idx/line_points;                    
                }

                parameters_line_search_mtx_vec[ line_idx] = parameters_line_idx;

            }
           

            Matrix_real line_values = optimization_problem_batched( parameters_line_search_mtx_vec ); 
                   

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
            for (int param_idx=0; param_idx<batch_size; param_idx++) {
                solution_guess_tmp_mtx[ param_idx_agents[param_idx] ] += param_update_mtx[ param_idx ]*(double)idx_min/line_points;
            } 

#ifdef __MPI__   
            MPI_Bcast( solution_guess_tmp_mtx.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


            // update the current cost function
            //current_minimum = optimization_problem( solution_guess_tmp_mtx );

            if ( iter_idx % 50 == 0 ) {
                std::stringstream sstream;
                sstream << "GRAD_DESCEND_SHIFT_RULE: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum;
                sstream << " circuit simulation time: " << circuit_simulation_time  << std::endl;
                print(sstream, 0);   
                if ( export_circuit_2_binary_loc > 0 ) {
                    std::string filename("initial_circuit_iteration.binary");
                    if (project_name != "") { 
                        filename=project_name+ "_"  +filename;
                    }
                    export_gate_list_to_binary(solution_guess_tmp_mtx, this, filename, verbose);
                }

                memcpy( optimized_parameters_mtx.get_data(),  solution_guess_tmp_mtx.get_data(), num_of_parameters*sizeof(double) );

                export_current_cost_fnc(current_minimum);


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
                sstream << "GRAD_DESCEND_SHIFT_RULE: converged to minimum at iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum;
                sstream << " circuit simulation time: " << circuit_simulation_time  << std::endl;
                print(sstream, 0);   
                if ( export_circuit_2_binary_loc > 0 ) {
                    std::string filename("initial_circuit_iteration.binary");
                    if (project_name != "") { 
                        filename=project_name+ "_"  +filename;
                    }
                    export_gate_list_to_binary(solution_guess_tmp_mtx, this, filename, verbose);
                }
                

                break;
            }

        }
        
       

        memcpy( optimized_parameters_mtx.get_data(),  solution_guess_tmp_mtx.get_data(), num_of_parameters*sizeof(double) );
        
        // CPU time
        CPU_time += (tbb::tick_count::now() - t0_CPU).seconds(); 
        
        sstream.str("");
        sstream << "obtained minimum: " << current_minimum << std::endl;


        tbb::tick_count optimization_end = tbb::tick_count::now();
        optimization_time  = optimization_time + (optimization_end-optimization_start).seconds();
        sstream << "GRAD_DESCEND_SHIFT_RULE time: " << CPU_time << " seconds, obtained minimum: " << current_minimum << std::endl;
        
        print(sstream, 0); 

}


