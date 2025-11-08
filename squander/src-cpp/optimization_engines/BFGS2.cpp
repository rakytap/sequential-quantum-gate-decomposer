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
/*! \file BFGS2.cpp
    \brief Implementation of the BFGS2 optimization srategy
*/


#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "BFGS_Powell.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif



/**
@brief Call to solve layer by layer the optimization problem via BFGS algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_BFGS2( int num_of_parameters, Matrix_real solution_guess) {


#ifdef __DFE__
        if ( qbit_num >= 2 && get_accelerator_num() > 0 ) {
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

        // maximal number of iteration loops
        int iteration_loops_max;
        try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }


        double current_minimum_hold = current_minimum;





tbb::tick_count bfgs_start = tbb::tick_count::now();
CPU_time = 0.0;


        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 1.0);


        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_bfgs2") > 0 ) {
            config["max_inner_iterations_bfgs2"].get_property( max_inner_iterations_loc );  
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
            config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc =max_inner_iterations;
        }


        long long export_circuit_2_binary_loc;
        if ( config.count("export_circuit_2_binary_bfgs2") > 0 ) {
             config["export_circuit_2_binary_bfgs2"].get_property( export_circuit_2_binary_loc );  
        }
        else if ( config.count("export_circuit_2_binary") > 0 ) {
             config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
        }
        else {
            export_circuit_2_binary_loc = 0;
        }    
        


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_adam") > 0 ) {
             config["optimization_tolerance_adam"].get_property( optimization_tolerance_loc );  
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
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
            output_periodicity = 0;
        }        

    

        std::stringstream sstream;
        sstream << "max_inner_iterations: " << max_inner_iterations_loc  << std::endl;
        print(sstream, 2); 


        // --- Basin-hopping parameters (SciPy-like defaults) ---
        double bh_T = 1.0;                     // "temperature" for Metropolis acceptance
        double bh_stepsize = 0.5;
        long long bh_interval = 50;                // how often to adapt stepsize
        double bh_target_accept = 0.5;
        double bh_stepwise_factor = 0.9;
        // Allow overrides via config (all optional)
        if (config.count("bh_T") > 0)                             config["bh_T"].get_property(bh_T);
        if (config.count("bh_stepsize") > 0)                      config["bh_stepsize"].get_property(bh_stepsize);
        if (config.count("bh_interval") > 0) { long long v; config["bh_interval"].get_property(v); bh_interval = std::max<long long>(1, v); }
        if (config.count("bh_target_accept_rate") > 0)            config["bh_target_accept_rate"].get_property(bh_target_accept);
        if (config.count("bh_stepwise_factor") > 0)               config["bh_stepwise_factor"].get_property(bh_stepwise_factor);

        // Clamp a couple of parameters to SciPy’s expected ranges
        bh_target_accept = std::min(0.999, std::max(0.001, bh_target_accept));
        if (!(bh_stepwise_factor > 0.0 && bh_stepwise_factor < 1.0)) bh_stepwise_factor = 0.9;

        // ---------------- Basin-hopping driver ----------------
        long long accept_count_window = 0;
        long long window_len = 0;
        long long no_improve_count = 0;
        double stepsize_now = bh_stepsize;            // adaptive stepsize (SciPy-style)

        BFGS_Powell cBFGS_Powell(optimization_problem_combined, this);
        double f_trial = cBFGS_Powell.Start_Optimization(solution_guess, max_inner_iterations);
        if (f_trial < current_minimum) {
            current_minimum = f_trial;
            memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double));
        }        
        Matrix_real x_current = solution_guess.copy();   // current basin representative


        for (long long iter_idx=0; iter_idx<iteration_loops_max; iter_idx++) {

            for (int j = 0; j < num_of_parameters; ++j) {
                double delta = (distrib_real(gen) * 2.0 - 1.0) * stepsize_now * M_PI;
                solution_guess[j] = fmod(solution_guess[j] + delta, 2.0 * M_PI);
            }
        
            f_trial = cBFGS_Powell.Start_Optimization(solution_guess, max_inner_iterations);

            // --- Metropolis acceptance (always accept downhill; uphill with prob exp(-(f_new - f_old)/T))
            bool accept = false;
            if (f_trial <= current_minimum_hold) {
                accept = true;
            } else {
                double dE = f_trial - current_minimum_hold;
                double prob = std::exp(-dE / std::max(1e-300, bh_T));
                accept = (distrib_real(gen) < prob);
            }

            if (accept) {
                // move to new basin
                memcpy(x_current.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double));
                current_minimum_hold = f_trial;
                ++accept_count_window;
            } else {
                memcpy(solution_guess.get_data(), x_current.get_data(), num_of_parameters*sizeof(double));
            }

            // --- Track global best
            //bool improved_global = false;
            if (f_trial < current_minimum) {
                current_minimum = f_trial;  // keep public minimum in sync
                memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double));
                //improved_global = true;
                no_improve_count = 0;
            } else {
                ++no_improve_count;
            }

            if ( iter_idx % 5000 == 0 ) {
                    std::stringstream sstream;
                    sstream << "BFGS2: processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum << std::endl;
                    print(sstream, 2);  

                    if ( export_circuit_2_binary_loc>0) {
                        std::string filename("initial_circuit_iteration.binary");
                        if (project_name != "") { 
                            filename=project_name+ "_"  +filename;
                        }
                        export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);
                    }
            }

            if ( output_periodicity>0 && iter_idx % output_periodicity == 0 ) {
                export_current_cost_fnc(current_minimum);
            }

#ifdef __MPI__        
            MPI_Bcast( (void*)solution_guess.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            
            
            if (current_minimum < optimization_tolerance_loc ) {
                break;
            }
            if (no_improve_count >= random_shift_count_max) {
                break;  // SciPy's niter_success criterion
            }

            // --- Adaptive stepsize every 'interval' iterations (SciPy behavior)
            ++window_len;
            if (bh_interval > 0 && (window_len % bh_interval) == 0) {
                double accept_rate = (double)accept_count_window / (double)bh_interval;
                // If acceptance is high, enlarge steps; else shrink steps.
                if (accept_rate > bh_target_accept) {
                    stepsize_now /= bh_stepwise_factor;   // increase (since factor<1)
                } else {
                    stepsize_now *= bh_stepwise_factor;   // decrease
                }
                // reset window counters
                accept_count_window = 0;
                window_len = 0;
            }


        }

        tbb::tick_count bfgs_end = tbb::tick_count::now();
        CPU_time  = CPU_time + (bfgs_end-bfgs_start).seconds();
        //std::cout << "bfgs2 time: " << CPU_time << " " << current_minimum << std::endl;

}


