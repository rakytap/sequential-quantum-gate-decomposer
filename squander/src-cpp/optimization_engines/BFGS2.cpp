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


        int random_shift_count = 0;
        long long sub_iter_idx = 0;
        double current_minimum_hold = current_minimum;





tbb::tick_count bfgs_start = tbb::tick_count::now();
CPU_time = 0.0;


        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        // random generator of integers   
        std::uniform_int_distribution<> distrib_int(0, 5000);  


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

        // do the optimization loops
        double f = DBL_MAX;
        for (long long iter_idx=0; iter_idx<iteration_loops_max; iter_idx++) {

            
                BFGS_Powell cBFGS_Powell(optimization_problem_combined, this);
                double f = cBFGS_Powell.Start_Optimization(solution_guess, max_inner_iterations);           
                
                if (sub_iter_idx == 1 ) {
                     current_minimum_hold = f;    
                }


                if (current_minimum_hold*0.95 > f || (current_minimum_hold*0.97 > f && f < 1e-3) ||  (current_minimum_hold*0.99 > f && f < 1e-4) ) {
                     sub_iter_idx = 0;
                     current_minimum_hold = f;        
                }
    
    
                if (current_minimum > f ) {
                     current_minimum = f;
                     memcpy( optimized_parameters_mtx.get_data(),  solution_guess.get_data(), num_of_parameters*sizeof(double) );
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


                if (f < optimization_tolerance_loc || random_shift_count > random_shift_count_max ) {
                    break;
                }


                sub_iter_idx++;
                iter_idx++;
                

        

            if (current_minimum > f) {
                current_minimum = f;
                memcpy( optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );                

                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess[jdx] = optimized_parameters_mtx[jdx] + distrib_real(gen)*2*M_PI/100;
                }
            }
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess[jdx] = optimized_parameters_mtx[jdx] + distrib_real(gen)*2*M_PI;
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



        }

        tbb::tick_count bfgs_end = tbb::tick_count::now();
        CPU_time  = CPU_time + (bfgs_end-bfgs_start).seconds();
        std::cout << "bfgs2 time: " << CPU_time << " " << current_minimum << std::endl;

}


