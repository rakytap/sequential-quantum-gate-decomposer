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
/*! \file Optimization_Interface.cpp
    \brief Class implementing optimization engines
*/

#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Adam.h"

#include "RL_experience.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif




/**
@brief Call to solve layer by layer the optimization problem via batched ADAM algorithm. (optimal for larger problems) The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_ADAM_BATCHED( int num_of_parameters, Matrix_real& solution_guess) {


#ifdef __DFE__
        if ( qbit_num >= 5 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif



        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess.size() == 0 ) {
            solution_guess = Matrix_real(num_of_parameters,1);
        }

        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );
        }


        int random_shift_count = 0;
        int sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;
    

        tbb::tick_count adam_start = tbb::tick_count::now();
        CPU_time = 0.0;

        Adam optimizer;
        optimizer.initialize_moment_and_variance( num_of_parameters );



        // the array storing the optimized parameters
        Matrix_real solution_guess_tmp = Matrix_real( num_of_parameters, 1 );
        memcpy(solution_guess_tmp.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );

        Matrix_real grad_mtx = Matrix_real( num_of_parameters, 1 );
        //solution_guess_tmp_mtx.print_matrix();


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_adam_batched") > 0 ) {
             config["max_inner_iterations_adam_batched"].get_property( max_inner_iterations_loc );  
        }
        if ( config.count("max_inner_iterations") > 0 ) {
             config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }

        long long iteration_threshold_of_randomization_loc;
        if ( config.count("randomization_threshold_adam") > 0 ) {
            config["randomization_threshold_adam"].get_property( iteration_threshold_of_randomization_loc );  
        }
        else if ( config.count("randomization_threshold") > 0 ) {
            config["randomization_threshold"].get_property( iteration_threshold_of_randomization_loc );  
        }
        else {
            iteration_threshold_of_randomization_loc = 2500000;
        }

        
        long long export_circuit_2_binary_loc;
        if ( config.count("export_circuit_2_binary_adam") > 0 ) {
             config["export_circuit_2_binary_adam"].get_property( export_circuit_2_binary_loc );  
        }
        else if ( config.count("export_circuit_2_binary") > 0 ) {
             config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
        }
        else {
            export_circuit_2_binary_loc = 0;
        }            

        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_adam_batched") > 0 ) {
             double value;
             config["optimization_tolerance_adam_batched"].get_property( optimization_tolerance_loc );
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
             double value;
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }


        bool adaptive_eta_loc;
        if ( config.count("adaptive_eta_adam_batched") > 0 ) {
             long long tmp;
             config["adaptive_eta_adam_batched"].get_property( tmp );  
             adaptive_eta_loc = (bool)tmp;
        }
        if ( config.count("adaptive_eta") > 0 ) {
             long long tmp;
             config["adaptive_eta"].get_property( tmp );  
             adaptive_eta_loc = (bool)tmp;
        }
        else {
            adaptive_eta_loc = adaptive_eta;
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

        double f0 = DBL_MAX;
        std::stringstream sstream;
        sstream << "iter_max: " << max_inner_iterations_loc << ", randomization threshold: " << iteration_threshold_of_randomization_loc << std::endl;
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

            


                optimization_problem_combined( solution_guess_tmp, (void*)(this), &f0, grad_mtx );

                prev_cost_fnv_val = f0;
  
                if (sub_iter_idx == 1 ) {
                    current_minimum_hold = f0;   
               
                    if ( adaptive_eta_loc )  { 
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
                    memcpy( optimized_parameters_mtx.get_data(),  solution_guess.get_data(), num_of_parameters*sizeof(double) );
                    //double new_eta = 1e-3 * f0 * f0;
                
                    if ( adaptive_eta_loc )  {
                        double new_eta = 1e-3 * f0;
                        optimizer.eta = new_eta > 1e-6 ? new_eta : 1e-6;
                        optimizer.eta = new_eta < 1e-1 ? new_eta : 1e-1;
                    }
                
                }
    

                if ( iter_idx % 5000 == 0 ) {
                    
                    std::stringstream sstream;
                    sstream << "ADAM: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100;
                    sstream << "\%, current minimum:" << current_minimum << ", pure cost function:" << optimization_problem( optimized_parameters_mtx ) << std::endl;
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
                    norm += grad_mtx[grad_idx]*grad_mtx[grad_idx];
                }
                norm = std::sqrt(norm);
                    

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
                    

                    randomize_parameters(optimized_parameters_mtx, solution_guess_tmp, f0 );
        
                    optimizer.reset();
                    optimizer.initialize_moment_and_variance( num_of_parameters );   

                    ADAM_status = 0;   

                    //optimizer.eta = 1e-3;
        
                }

                else {
                    ADAM_status = optimizer.update(solution_guess_tmp, grad_mtx, f0);
                }

                sub_iter_idx++;

            }




        }

        sstream.str("");
        sstream << "obtained minimum: " << current_minimum << std::endl;


        tbb::tick_count adam_end = tbb::tick_count::now();
        CPU_time  = CPU_time + (adam_end-adam_start).seconds();
        sstream << "adam time: " << CPU_time  << " " << f0 << std::endl;
        
        print(sstream, 0); 
        

}

