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
/*! \file GRAD_DESCEND.cpp
    \brief Implementation of the gradient descend optimization srategy
*/


#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "grad_descend.h"


#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif




/**
@brief Call to solve layer by layer the optimization problem via the GRAD_DESCEND (line search in the direction determined by the gradient) algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess_gsl A GNU Scientific Library vector containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_GRAD_DESCEND( int num_of_parameters, Matrix_real& solution_guess) {


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
        /*int iteration_loops_max;
        try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }*/
        int iteration_loops_max;
        long long value;
        if ( config.count("max_iteration_loops_grad_descend") > 0 ) {
            config["max_iteration_loops_grad_descend"].get_property( value );
            iteration_loops_max = (int) value;
        }
        else if ( config.count("max_iteration_loops") > 0 ) {
            config["max_iteration_loops"].get_property( value );       
            iteration_loops_max = (int) value;  
        }
        else {
            try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
            }
            catch (...) {
                iteration_loops_max = 1;
            }
        }
        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        // maximal number of inner iterations overriden by config
        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_grad_descend") > 0 ) {
            config["max_inner_iterations_grad_descend"].get_property( max_inner_iterations_loc );         
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
            config["max_inner_iterations"].get_property( max_inner_iterations_loc );         
        }
        else {
            max_inner_iterations_loc =max_inner_iterations;
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


        // do the optimization loops
        for (long long idx=0; idx<iteration_loops_max; idx++) {
	    

            Grad_Descend cGrad_Descend(optimization_problem_combined, export_current_cost_fnc, this);
            double f = cGrad_Descend.Start_Optimization(solution_guess, max_inner_iterations_loc);

            if (current_minimum > f) {
                current_minimum = f;
                memcpy( optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess[jdx] = solution_guess[jdx] + distrib_real(gen)/100;
                }
            }
           
            else {
                for ( int jdx=0; jdx<num_of_parameters; jdx++) {
                    solution_guess[jdx] = solution_guess[jdx] + distrib_real(gen);
                }
            }


            if ( output_periodicity>0 && idx % output_periodicity == 0 ) {
                export_current_cost_fnc(current_minimum);
            }

#ifdef __MPI__        
            MPI_Bcast( (void*)solution_guess.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif



        }




}



