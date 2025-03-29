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
/*! \file AGENTS.cpp
    \brief Implementation of the AGENTS optimization srategy
*/

#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"

#include "RL_experience.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif



/**
@brief Call to solve layer by layer the optimization problem via the AGENT algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_AGENTS( int num_of_parameters, Matrix_real& solution_guess) {



        if ( ((cost_fnc != FROBENIUS_NORM) && (cost_fnc != HILBERT_SCHMIDT_TEST)) && cost_fnc != VQE  ) {
            std::string err("Optimization_Interface::solve_layer_optimization_problem_AGENTS: Only cost functions 0 and 3 are implemented");
            throw err;
        }

#ifdef __DFE__
        if ( qbit_num >= 5 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif


        if (gates.size() == 0 ) {
            return;
        }


        double M_PI_quarter = M_PI/4;
        double M_PI_half = M_PI/2;
        double M_PI_double = M_PI*2;

        if (solution_guess.size() == 0 ) {
            solution_guess = Matrix_real(num_of_parameters,1);
            std::uniform_real_distribution<> distrib_real(0, M_PI_double); 
            for ( int idx=0; idx<num_of_parameters; idx++) {
                solution_guess[idx] = distrib_real(gen);
            }

        }


#ifdef __MPI__        
        MPI_Bcast( (void*)solution_guess.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif





        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );
        }


        long long sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;
    

        tbb::tick_count optimization_start = tbb::tick_count::now();
        double optimization_time = 0.0;




        
        current_minimum =   optimization_problem( optimized_parameters_mtx ); 

        int max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_agent") > 0 ) {
             long long value;
             config["max_inner_iterations_agent"].get_property( value );
             max_inner_iterations_loc = (int) value;
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
             long long value;
             config["max_inner_iterations"].get_property( value );
             max_inner_iterations_loc = (int) value;
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_agent") > 0 ) {
             config["optimization_tolerance_agent"].get_property( optimization_tolerance_loc );
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }

        
        long long export_circuit_2_binary_loc;
        if ( config.count("export_circuit_2_binary_agent") > 0 ) {
             config["export_circuit_2_binary_agent"].get_property( export_circuit_2_binary_loc );  
        }
        else if ( config.count("export_circuit_2_binary") > 0 ) {
             config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
        }
        else {
            export_circuit_2_binary_loc = 0;
        }            

        
        int agent_lifetime_loc;
        if ( config.count("agent_lifetime_agent") > 0 ) {
             long long agent_lifetime_loc_tmp;
             config["agent_lifetime_agent"].get_property( agent_lifetime_loc_tmp );  
             agent_lifetime_loc = (int)agent_lifetime_loc_tmp;
        }
        else if ( config.count("agent_lifetime") > 0 ) {
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

        double agent_randomization_rate_loc = 0.2;
        if ( config.count("aagent_randomization_rate_agent") > 0 ) {
        
             config["agent_randomization_rate_agent"].get_property( agent_randomization_rate_loc );
        }
        else if ( config.count("agent_randomization_rate") > 0 ) {
             config["agent_randomization_rate"].get_property( agent_randomization_rate_loc );
        }
        
        
        
        int agent_num;
        if ( config.count("agent_num_agent") > 0 ) { 
             long long value;                   
             config["agent_num_agent"].get_property( value );  
             agent_num = (int) value;
        }
        else if ( config.count("agent_num") > 0 ) { 
             long long value;                   
             config["agent_num"].get_property( value );  
             agent_num = (int) value;
        }
        else {
            agent_num = 64;
        }


        double agent_exploration_rate = 0.2;
        if ( config.count("agent_exploration_rate_agent") > 0 ) {
        
             config["agent_exploration_rate_agent"].get_property( agent_exploration_rate );
        }
        else if ( config.count("agent_exploration_rate") > 0 ) {
             config["agent_exploration_rate"].get_property( agent_exploration_rate );
        }
        
        int convergence_length = 20;
        if ( config.count("convergence_length_agent") > 0 ) {
             long long value;                   
             config["convergence_length_agent"].get_property( value );
             convergence_length = (int) value;
        }
        else if ( config.count("convergence_length") > 0 ) {
             long long value;                   
             config["convergence_length"].get_property( value );
             convergence_length = (int) value;
        }

        int linesearch_points = 3;
        if ( config.count("linesearch_points_agent") > 0 ) {
             long long value;                   
             config["linesearch_points_agent"].get_property( value );
             linesearch_points = (int) value;
        }
        else if ( config.count("linesearch_points") > 0 ) {
             long long value;                   
             config["linesearch_points"].get_property( value );
             linesearch_points = (int) value;
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
            output_periodicity = 0;
        }        
        
        sstream.str("");
        sstream << "AGENTS: number of agents " << agent_num << std::endl;
        sstream << "AGENTS: exploration_rate " << agent_exploration_rate << std::endl;
        sstream << "AGENTS: lifetime " << agent_lifetime_loc << std::endl;
        print(sstream, 2);    
    
        
        bool terminate_optimization = false;
        
        // vector stroing the lates values of current minimums to identify convergence
        Matrix_real current_minimum_vec(1, convergence_length); 
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
                    memcpy( solution_guess_mtx_agent.get_data(), solution_guess.get_data(), solution_guess.size()*sizeof(double) );
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
        Matrix_real f0_shifted_pi2_agents;
        Matrix_real f0_shifted_pi_agents;                 
        Matrix_real f0_shifted_pi4_agents;    
        Matrix_real f0_shifted_3pi2_agents;           
       

        bool three_point_line_search               =    cost_fnc == FROBENIUS_NORM;
        bool three_point_line_search_double_period =    cost_fnc == VQE && linesearch_points == 3;
        bool five_point_line_search                =    cost_fnc == HILBERT_SCHMIDT_TEST || ( cost_fnc == VQE && linesearch_points == 5 );


   
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
                      

            if ( three_point_line_search ) {
                    
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
                    double amplitude = sqrt(A_times_sin*A_times_sin + A_times_cos*A_times_cos);
		//std::cout << amplitude << " " << offset << std::endl;


                    //update  the parameter vector
                    Matrix_real& solution_guess_mtx_agent                    = solution_guess_mtx_agents[ agent_idx ];  
                    solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift; 

                    current_minimum_agents[agent_idx] = offset - amplitude;
                    
                }
            }                   
            else if ( three_point_line_search_double_period ) {

                                      
                // calsulate the cist functions at shifted parameter values
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ]; 
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;                
                }   

                // CPU time              
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();  
            
                // calculate batched cost function                 
                f0_shifted_pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );              
            
                // CPU time
                t0_CPU = tbb::tick_count::now();                                         
                        

                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];             
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;
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


                    double parameter_shift = phi0 > 0 ? M_PI_half-phi0/2 : -phi0/2-M_PI_half;
                    double amplitude = sqrt(A_times_sin*A_times_sin + A_times_cos*A_times_cos);
		//std::cout << amplitude << " " << offset << std::endl;


                    //update  the parameter vector
                    Matrix_real& solution_guess_mtx_agent                    = solution_guess_mtx_agents[ agent_idx ];  
                    solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift; 

                    current_minimum_agents[agent_idx] = offset - amplitude;
                    
                }
/*
double max = -DBL_MAX;
double min = DBL_MAX;

double delta = 2*M_PI/1000;
for ( int idx =0; idx<1000; idx++ ) {

Matrix_real& solution_guess_mtx_agent                    = solution_guess_mtx_agents[ 0 ];  
solution_guess_mtx_agent[param_idx_agents[ 0 ]] += delta;
double rr = optimization_problem( solution_guess_mtx_agent );
//std::cout << rr << std::endl;

if ( rr < min ) {
    min = rr;
}


if ( rr > max ) {
    max = rr;
}

}

double amplitude = (max-min)/2;
double offset = (max+min)/2;
std::cout <<"kkk " << amplitude << " " << offset << " " << param_idx_agents[ 0 ] << " " << parameter_value_save_agents[ 0 ] <<  std::endl;


Matrix_real tmp = optimization_problem_batched( solution_guess_mtx_agents );  
tmp.print_matrix();
    
      current_minimum_agents.print_matrix();
exit(-1);        
*/
            }  
            else if ( five_point_line_search ){
           
                
                // calsulate the cist functions at shifted parameter values
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ]; 
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;                
                }   
            
                // CPU time              
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();  
            
                // calculate batched cost function                 
                f0_shifted_pi4_agents = optimization_problem_batched( solution_guess_mtx_agents );              
            
                // CPU time
                t0_CPU = tbb::tick_count::now();                                         
                        

                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];             
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;
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
                
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) { 
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];             
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;
                }  
            
                // CPU time             
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();        
            
                // calculate batched cost function                         
                f0_shifted_3pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );             
                
                                                     
                // CPU time                                      
                t0_CPU = tbb::tick_count::now();  
		                
                
                // determine the parameters of the cosine function and determine the parameter shift at the minimum
                for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {

                    double current_minimum_agent = current_minimum_agents[agent_idx];         
                    double f0_shifted_pi4        = f0_shifted_pi4_agents[agent_idx];                    
                    double f0_shifted_pi2        = f0_shifted_pi2_agents[agent_idx];                    
                    double f0_shifted_pi         = f0_shifted_pi_agents[agent_idx];                   
                    double f0_shifted_3pi2       = f0_shifted_3pi2_agents[agent_idx];   
/*                                                                              
                    f(p)          = kappa*sin(2p+xi) + gamma*sin(p+phi) + offset
                    f(p + pi/4)   = kappa*cos(2p+xi) + gamma*sin(p+pi/4+phi) + offset
                    f(p + pi/2)   = -kappa*sin(2p+xi) + gamma*cos(p+phi) + offset
                    f(p + pi)     = kappa*sin(2p+xi) - gamma*sin(p+phi) + offset
                    f(p + 3*pi/2) = -kappa*sin(2p+xi) - gamma*cos(p+phi) + offset
*/

                    double f1 = current_minimum_agent -  f0_shifted_pi;
                    double f2 = f0_shifted_pi2 - f0_shifted_3pi2;

                    double gamma = sqrt( f1*f1 + f2*f2 )*0.5;
                    //print( "gamma: ", gamma )

                    double varphi = atan2( f1, f2) - parameter_value_save_agents[ agent_idx ];
                    //print( "varphi: ", varphi )

                    double offset = 0.25*(current_minimum_agent +  f0_shifted_pi + f0_shifted_pi2 + f0_shifted_3pi2);
                    double f3     = 0.5*(current_minimum_agent +  f0_shifted_pi - 2*offset);
                    double f4     = f0_shifted_pi4 - offset - gamma*sin(parameter_value_save_agents[ agent_idx ]+M_PI_quarter+varphi);


                    double kappa = sqrt( f3*f3 + f4*f4);
                    //print( "kappa: ", kappa )

                    double xi = atan2( f3, f4) - 2*parameter_value_save_agents[ agent_idx ];
                    //print( "xi: ", xi )


                    double f; 
                    double params[5];
                    params[0] = kappa;
                    params[1] = xi + 2*parameter_value_save_agents[ agent_idx ];
                    params[2] = gamma;
                    params[3] = varphi + parameter_value_save_agents[ agent_idx ];
                    params[4] = offset;
                    

                    Matrix_real parameter_shift(1,1);
                    if ( abs(gamma) > abs(kappa) ) {
                        parameter_shift[0] = 3*M_PI/2 - varphi - parameter_value_save_agents[ agent_idx ];                        
                    }
                    else {
                        parameter_shift[0] = 3*M_PI/4 - xi/2 - parameter_value_save_agents[ agent_idx ]/2;
                        

                    }
                   
                    parameter_shift[0] = std::fmod( parameter_shift[0], M_PI_double);    
                                                 
                    BFGS_Powell cBFGS_Powell(HS_partial_optimization_problem_combined,(void*)&params);
                    f = cBFGS_Powell.Start_Optimization(parameter_shift, 10);
		
                    //update  the parameter vector
                    Matrix_real& solution_guess_mtx_agent                   = solution_guess_mtx_agents[ agent_idx ];                             
                    solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift[0]; 

                    current_minimum_agents[agent_idx] = f;
                    
                }                                                                                                                         
                
            }
    

                
            
               
            // CPU time                                                     
            CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();
            
  
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


/*
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
*/

            // ocassionaly recalculate teh current cost functions of the agents
            if ( iter_idx % agent_lifetime_loc == 0 )
            {
                // recalculate the current cost functions
                current_minimum_agents = optimization_problem_batched( solution_guess_mtx_agents ); 
            }


            
            // govern the behavior of the agents
            for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {
                double& current_minimum_agent = current_minimum_agents[ agent_idx ];
                   
           
                
                if (current_minimum_agent < optimization_tolerance_loc ) {
                    terminate_optimization = true;    
                    current_minimum = current_minimum_agent;      

                    most_successfull_agent = agent_idx;

                   Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];                     
                    
                    // export the parameters of the current, most successful agent
                    memcpy(optimized_parameters_mtx.get_data(), solution_guess_mtx_agent.get_data(), num_of_parameters*sizeof(double) );
                }  
                
               
                
                Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];                             

                // look for the best agent periodicaly
                if ( iter_idx % agent_lifetime_loc == 0 )
                {

                    
                            
                    if ( current_minimum_agent <= current_minimum ) {

                        most_successfull_agent = agent_idx;
                    
                        // export the parameters of the current, most successful agent
                        memcpy(optimized_parameters_mtx.get_data(), solution_guess_mtx_agent.get_data(), num_of_parameters*sizeof(double) );

                        if ( export_circuit_2_binary_loc > 0 ) {
                            std::string filename("initial_circuit_iteration.binary");
                            if (project_name != "") { 
                                filename=project_name+ "_"  +filename;
                            }
                            export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);
                        }

                        
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
                                
                                current_minimum_agents[ agent_idx ] = current_minimum_agents[ most_successfull_agent ];
                                memcpy( solution_guess_mtx_agent.get_data(), solution_guess_mtx_agents[ most_successfull_agent ].get_data(), solution_guess_mtx_agent.size()*sizeof(double) );

                            
                                random_num = random_numbers[ agent_idx*random_numbers.stride + 1 ];
                            
                                if ( random_num < agent_randomization_rate_loc ) {
                                    randomize_parameters( optimized_parameters_mtx, solution_guess_mtx_agent, current_minimum  );  
                                    current_minimum_agents[agent_idx] = optimization_problem( solution_guess_mtx_agent );   
                                }     

                       
                            }
                            else {
                                // keep the current state  of the agent                    
                            }

#ifdef __MPI__        
                        }
                        
                        MPI_Bcast( (void*)solution_guess_mtx_agent.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Bcast( (void*)current_minimum_agents.get_data(), agent_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif     
                                                                                  
                    }
                 

                    // test global convergence 
                    if ( agent_idx == 0 ) {
                 
                        if ( output_periodicity>0 && iter_idx % output_periodicity == 0 ) {
                            export_current_cost_fnc(current_minimum);
                        }

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

                if ( iter_idx % agent_lifetime_loc == 0 && agent_idx == 0) {
                    std::stringstream sstream;
                    sstream << "AGENTS, agent " << agent_idx << ": processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%";
                    sstream << ", current minimum of agent 0: " << current_minimum_agents[ 0 ] << " global current minimum: " << current_minimum  << " CPU time: " << CPU_time;
                    sstream << " circuit simulation time: " << circuit_simulation_time  << std::endl;
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
        sstream.str("");
        sstream << "AGENTS time: " << CPU_time << " " << current_minimum << std::endl;

        print(sstream, 0); 
}



/**
@brief Call to solve layer by layer the optimization problem via the AGENT COMBINED algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_AGENTS_COMBINED( int num_of_parameters, Matrix_real& solution_guess)  {



    optimized_parameters_mtx = Matrix_real(solution_guess.get_data(), solution_guess.size(), 1);

    for( int loop_idx=0; loop_idx<1; loop_idx++ ) {

        Matrix_real solution_guess_AGENTS(num_of_parameters ,1);
        memcpy( solution_guess_AGENTS.get_data(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );

        solve_layer_optimization_problem_AGENTS( num_of_parameters, solution_guess_AGENTS );


        Matrix_real solution_guess_COSINE(num_of_parameters, 1);
        memcpy( solution_guess_COSINE.get_data(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );

        solve_layer_optimization_problem_GRAD_DESCEND( num_of_parameters, solution_guess_COSINE );

    }
        

}


