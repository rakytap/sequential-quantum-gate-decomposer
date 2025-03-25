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
/*! \file COSINE.cpp
    \brief Implementation of the COSINE optimization srategy
*/


#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"

#include "RL_experience.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif




/**
@brief Call to solve layer by layer the optimization problem via the COSINE algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_COSINE( int num_of_parameters, Matrix_real& solution_guess) {


        if ( ((cost_fnc != FROBENIUS_NORM) && (cost_fnc != HILBERT_SCHMIDT_TEST)) && cost_fnc != VQE   ){
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


        double M_PI_quarter = M_PI/4;
        double M_PI_half    = M_PI/2;
        double M_PI_double  = M_PI*2;

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
        //export_current_cost_fnc(current_minimum);
        

        // the array storing the optimized parameters
        Matrix_real solution_guess_tmp_mtx = Matrix_real( num_of_parameters, 1 );
        memcpy(solution_guess_tmp_mtx.get_data(), optimized_parameters_mtx.get_data(), num_of_parameters*sizeof(double) );

        int batch_size;
        if ( config.count("batch_size_cosine") > 0 ) { 
             long long value;                   
             config["batch_size_cosine"].get_property( value );  
             batch_size = (int) value;
        }
        else if ( config.count("batch_size") > 0 ) { 
             long long value;                   
             config["batch_size"].get_property( value );  
             batch_size = (int) value;
        }
        else {
            batch_size = 128 <= num_of_parameters ? 128 : num_of_parameters;
            
        }
        
        if( batch_size > num_of_parameters ) {
            std::string err("Optimization_Interface::solve_layer_optimization_problem_COSINE: batch size should be lower or equal to the number of free parameters");
            throw err;
        }


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_cosine") > 0 ) {
             config["max_inner_iterations_cosine"].get_property( max_inner_iterations_loc );  
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
             config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }
        
        
        long long export_circuit_2_binary_loc;
        if ( config.count("export_circuit_2_binary_cosine") > 0 ) {
             config["export_circuit_2_binary_cosine"].get_property( export_circuit_2_binary_loc );  
        }
        else if ( config.count("export_circuit_2_binary") > 0 ) {
             config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
        }
        else {
            export_circuit_2_binary_loc = 0;
        }        


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_cosine") > 0 ) {
             config["optimization_tolerance_cosine"].get_property( optimization_tolerance_loc );  
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
             double value;
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }
        
        
        // The number if iterations after which the current results are displed/exported
        int output_periodicity;
        if ( config.count("output_periodicity_cosine") > 0 ) {
             long long value = 1;
             config["output_periodicity_cosine"].get_property( value ); 
             output_periodicity = (int) value;
        }
        if ( config.count("output_periodicity") > 0 ) {
             long long value = 1;
             config["output_periodicity"].get_property( value ); 
             output_periodicity = (int) value;
        }
        else {
            output_periodicity = 50;
        }
        
        int linesearch_points = 3;
        if ( config.count("linesearch_points_cosine") > 0 ) {
             long long value;                   
             config["linesearch_points_cosine"].get_property( value );
             linesearch_points = (int) value;
        }
        else if ( config.count("linesearch_points") > 0 ) {
             long long value;                   
             config["linesearch_points"].get_property( value );
             linesearch_points = (int) value;
        }        


        if ( output_periodicity == 0 ) {
            std::string err("Output periodicity should be greater than zero");
            throw err;
        }   

        // vector stroing the lates values of current minimums to identify convergence
        Matrix_real f0_vec(1, 100); 
        memset( f0_vec.get_data(), 0.0, f0_vec.size()*sizeof(double) );
        double f0_mean = 0.0;
        int f0_idx = 0; 



        Matrix_real param_update_mtx( batch_size, 1 );
        matrix_base<int> param_idx_agents( batch_size, 1 );


        std::vector<Matrix_real> parameters_mtx_vec(batch_size);
        parameters_mtx_vec.reserve(batch_size); 

                 
             

        bool three_point_line_search               =    cost_fnc == FROBENIUS_NORM;
        bool three_point_line_search_double_period =    cost_fnc == VQE ;
        bool five_point_line_search                =    (cost_fnc == HILBERT_SCHMIDT_TEST) || ( cost_fnc == VQE && linesearch_points == 5 ); 

        for (unsigned long long iter_idx=0; iter_idx<max_inner_iterations_loc; iter_idx++) {
        
            // build up a vector of indices providing a set from which we can draw random (but unique) choices
            std::vector<int> indices(num_of_parameters);
            indices.reserve(num_of_parameters);
            for( int idx=0; idx<num_of_parameters; idx++ ) {
                indices[idx] = idx;
            }        

            for( int idx=0; idx<batch_size; idx++ ) {
                parameters_mtx_vec[idx] = solution_guess_tmp_mtx.copy();

                // random generator of integers   
                std::uniform_int_distribution<> distrib_int(0, indices.size()-1);

                // The index array of the chosen parameters
                int chosen_idx = distrib_int(gen);
                param_idx_agents[ idx ] = indices[ chosen_idx ];
                indices.erase( indices.begin()+chosen_idx ); 
            }


#ifdef __MPI__             
            MPI_Bcast( (void*)param_idx_agents.get_data(), batch_size, MPI_INT, 0, MPI_COMM_WORLD);
#endif     


          
            if ( three_point_line_search ) {

                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ]; 
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_half;                
                }                 
      
                Matrix_real f0_shifted_pi2_agents = optimization_problem_batched( parameters_mtx_vec );  


                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_half;
                }   
             
                Matrix_real f0_shifted_pi_agents = optimization_problem_batched( parameters_mtx_vec );

                for( int idx=0; idx<batch_size; idx++ ) {
     
                    double f0_shifted_pi         = f0_shifted_pi_agents[idx];
                    double f0_shifted_pi2        = f0_shifted_pi2_agents[idx];     
            

                    double A_times_cos = (current_minimum-f0_shifted_pi)/2;
                    double offset      = (current_minimum+f0_shifted_pi)/2;

                    double A_times_sin = offset - f0_shifted_pi2;

                    double phi0 = atan2( A_times_sin, A_times_cos);


                    double parameter_shift = phi0 > 0 ? M_PI-phi0 : -phi0-M_PI;

                                   
                    param_update_mtx[ idx ] = parameter_shift;

                    //revert the changed parameters
                    Matrix_real& solution_guess_mtx_idx             = parameters_mtx_vec[idx];  
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] = solution_guess_tmp_mtx[ param_idx_agents[idx] ];  	

                }

            }
            else if ( three_point_line_search_double_period ) {

                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ]; 
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_quarter;                
                }                 
      
                Matrix_real f0_shifted_pi4_agents = optimization_problem_batched( parameters_mtx_vec );  


                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_quarter;
                }   
             
                Matrix_real f0_shifted_pi2_agents = optimization_problem_batched( parameters_mtx_vec );

                for( int idx=0; idx<batch_size; idx++ ) {
     
                    double f0_shifted_pi         = f0_shifted_pi2_agents[idx];
                    double f0_shifted_pi2        = f0_shifted_pi4_agents[idx];     
            

                    double A_times_cos = (current_minimum-f0_shifted_pi)/2;
                    double offset      = (current_minimum+f0_shifted_pi)/2;

                    double A_times_sin = offset - f0_shifted_pi2;

                    double phi0 = atan2( A_times_sin, A_times_cos);


                    double parameter_shift = phi0 > 0 ? M_PI_half-phi0/2 : -phi0/2-M_PI_half;

                                   
                    param_update_mtx[ idx ] = parameter_shift;

                    //revert the changed parameters
                    Matrix_real& solution_guess_mtx_idx             = parameters_mtx_vec[idx];  
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] = solution_guess_tmp_mtx[ param_idx_agents[idx] ];  	

                }



            }
            else if (five_point_line_search){
                
                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ]; 
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_quarter;                
                }                 
      
                Matrix_real f0_shifted_pi4_agents = optimization_problem_batched( parameters_mtx_vec );  


                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_quarter;
                }   
             
                Matrix_real f0_shifted_pi2_agents = optimization_problem_batched( parameters_mtx_vec );
                
                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_half;
                }   
             
                Matrix_real f0_shifted_pi_agents = optimization_problem_batched( parameters_mtx_vec );
                
                for(int idx=0; idx<batch_size; idx++) { 
                    Matrix_real& solution_guess_mtx_idx = parameters_mtx_vec[ idx ];             
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] += M_PI_half;
                }   
             
                Matrix_real f0_shifted_3pi2_agents = optimization_problem_batched( parameters_mtx_vec );
                                                                     
                
                // determine the parameters of the cosine function and determine the parameter shift at the minimum
                for( int idx=0; idx<batch_size; idx++ ) {
                
                    double parameter_value_original = solution_guess_tmp_mtx[ param_idx_agents[idx] ];

                    double f0_shifted_pi4        = f0_shifted_pi4_agents[idx];                    
                    double f0_shifted_pi2        = f0_shifted_pi2_agents[idx];                    
                    double f0_shifted_pi         = f0_shifted_pi_agents[idx];                   
                    double f0_shifted_3pi2       = f0_shifted_3pi2_agents[idx];   
/*                                                                              
                    f(p)          = kappa*sin(2p+xi) + gamma*sin(p+phi) + offset
                    f(p + pi/4)   = kappa*cos(2p+xi) + gamma*sin(p+pi/4+phi) + offset
                    f(p + pi/2)   = -kappa*sin(2p+xi) + gamma*cos(p+phi) + offset
                    f(p + pi)     = kappa*sin(2p+xi) - gamma*sin(p+phi) + offset
                    f(p + 3*pi/2) = -kappa*sin(2p+xi) - gamma*cos(p+phi) + offset
*/

                    double f1 = current_minimum -  f0_shifted_pi;
                    double f2 = f0_shifted_pi2 - f0_shifted_3pi2;

                    double gamma = sqrt( f1*f1 + f2*f2 )*0.5;
                    //print( "gamma: ", gamma )

                    double varphi = atan2( f1, f2) - parameter_value_original;
                    //print( "varphi: ", varphi )

                    double offset = 0.25*(current_minimum +  f0_shifted_pi + f0_shifted_pi2 + f0_shifted_3pi2);
                    double f3     = 0.5*(current_minimum +  f0_shifted_pi - 2*offset);
                    double f4     = f0_shifted_pi4 - offset - gamma*sin(parameter_value_original+M_PI_quarter+varphi);


                    double kappa = sqrt( f3*f3 + f4*f4);
                    //print( "kappa: ", kappa )

                    double xi = atan2( f3, f4) - 2*parameter_value_original;
                    //print( "xi: ", xi )


                    double f; 
                    double params[5];
                    params[0] = kappa;
                    params[1] = xi + 2*parameter_value_original;
                    params[2] = gamma;
                    params[3] = varphi + parameter_value_original;
                    params[4] = offset;
                    

                    Matrix_real parameter_shift(1,1);
                    if ( abs(gamma) > abs(kappa) ) {
                        parameter_shift[0] = 3*M_PI/2 - varphi - parameter_value_original;                        
                    }
                    else {
                        parameter_shift[0] = 3*M_PI/4 - xi/2 - parameter_value_original/2;
                        

                    }
                   
                    parameter_shift[0] = std::fmod( parameter_shift[0], M_PI_double);    
                                                 
                    BFGS_Powell cBFGS_Powell(HS_partial_optimization_problem_combined,(void*)&params);
                    f = cBFGS_Powell.Start_Optimization(parameter_shift, 10);
                    
		
                    param_update_mtx[ idx ] = std::fmod( parameter_shift[0], M_PI_double);

                    //revert the changed parameters
                    Matrix_real& solution_guess_mtx_idx             = parameters_mtx_vec[idx];  
                    solution_guess_mtx_idx[ param_idx_agents[idx] ] = solution_guess_tmp_mtx[ param_idx_agents[idx] ];  	
                    
                }    
            
            }
            else {
                std::string err("solve_layer_optimization_problem_COSINE: Not implemented method.");
                throw err;
            }
		
            
////////////////////////////////////////////////////////////////////////////////
            // the line search is converted onto a onediemnsional search between x0+a*param_update_mtx and x0+b*param_update_mtx
            double interval_coeff = 2.0/(sqrt(5.0) + 1); // = 1/tau in Fig 1 of  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1606817


            // lowest point on the search interval
            double point_a = 0.0;
            double point_b = 1.0;
            double epsilon = 1e-2;

            double interval_length = point_b - point_a;

            double x1 = point_a + interval_length*interval_coeff*interval_coeff;
            double x2; // = point_a + interval_length*interval_coeff;

            Matrix_real parameters_x1 = solution_guess_tmp_mtx.copy();
            Matrix_real parameters_x2;
            for( int idx=0; idx<batch_size; idx++ ) {
                parameters_x1[ param_idx_agents[idx] ] += param_update_mtx[ idx ]*x1;     
            }

            
            double val_x1 = optimization_problem( parameters_x1 ); 
            double val_x2;


		
            int iter_max = 50;
            int iter = 0;

            double current_best_point = point_a;
            double current_best_value = current_minimum;


            while( true ) {

                interval_length = point_b - point_a;

                if ( interval_length < epsilon) {
                    break;
                }

                if ( (x1-point_a) < (point_b-x1) ) {
                    // x1 is closer to "a" than to "b"  --> x2 should be closer to "b"
                    x2 = point_a + interval_length*interval_coeff;

                    parameters_x2 = solution_guess_tmp_mtx.copy();

                    for( int idx=0; idx<batch_size; idx++ ) {
                        parameters_x2[ param_idx_agents[idx] ] += param_update_mtx[ idx ]*x2;     
                    }

                    val_x2 = optimization_problem( parameters_x2 ); 
                }
                else {
                    // x1 should be always closer to "a"
                    x2      = x1; 
                    val_x2 = val_x1;

                    x1 = point_a + interval_length*interval_coeff*interval_coeff;

                    parameters_x1 = solution_guess_tmp_mtx.copy();

                    for( int idx=0; idx<batch_size; idx++ ) {
                        parameters_x1[ param_idx_agents[idx] ] += param_update_mtx[ idx ]*x1;     
                    }

                    val_x1 = optimization_problem( parameters_x1 ); 

                }                


         //std::cout << point_a << " " << x1 << " " << x2 << " " << point_b << std::endl;
         //std::cout << val_x1 << " " << val_x2 << " " << current_minimum << " " << std::endl;

                if ( val_x1 < val_x2 ) {
                    point_b = x2;
                    if ( current_best_value > val_x1 ) {
                        current_best_point = x1; 
                        current_best_value = val_x1;
                    }
                }
                else {
                    point_a = x1;
                    if ( current_best_value > val_x2 ) {
                        current_best_point = x2; 
                        current_best_value = val_x2;
                    }

                    x1 = x2;
                    val_x1 = val_x2;
                }
   

                iter = iter + 1;

                if ( iter > iter_max) {
                    std::cout << "line search not  converged: interval length: " << interval_length << " " << interval_coeff << std::endl;
                    break;
                }
            }
       
           // std::cout << "number of costfunction evaluations : " << iter+1 << " " << interval_coeff << std::endl;

            current_minimum = current_best_value;

            // update parameters
            for (int param_idx=0; param_idx<batch_size; param_idx++) {
                solution_guess_tmp_mtx[ param_idx_agents[param_idx] ] += param_update_mtx[ param_idx ]*current_best_point;
            } 

////////////////////////////////////////////////////////////////////////////////
/*
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
*/
#ifdef __MPI__   
            MPI_Bcast( solution_guess_tmp_mtx.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


            // update the current cost function
            //current_minimum = optimization_problem( solution_guess_tmp_mtx );

            if ( iter_idx % output_periodicity == 0 ) {
                std::stringstream sstream;
                sstream << "COSINE: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum;
                sstream << " circuit simulation time: " << circuit_simulation_time  << std::endl;
                print(sstream, 3);   
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


     
            /*if ( std::abs( f0_mean - current_minimum) < 1e-7  && var_f0/f0_mean < 1e-7 ) {
                std::stringstream sstream;
                sstream << "COSINE: converged to minimum at iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum;
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
            }*/

        }
        
       

        memcpy( optimized_parameters_mtx.get_data(),  solution_guess_tmp_mtx.get_data(), num_of_parameters*sizeof(double) );
        
        // CPU time
        CPU_time += (tbb::tick_count::now() - t0_CPU).seconds(); 
        
        sstream.str("");
        sstream << "obtained minimum: " << current_minimum << std::endl;


        tbb::tick_count optimization_end = tbb::tick_count::now();
        optimization_time  = optimization_time + (optimization_end-optimization_start).seconds();
        sstream << "COSINE time: " << CPU_time << " seconds, obtained minimum: " << current_minimum << std::endl;
        
        print(sstream, 0); 

}


