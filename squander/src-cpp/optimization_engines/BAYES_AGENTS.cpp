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
/*! \file BAYES_AGENTS.cpp
    \brief Implementation of the BAYES AGENTS optimization srategy
*/


#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Bayes_Opt.h"

#include "RL_experience.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif




void Optimization_Interface::solve_layer_optimization_problem_BAYES_AGENTS( int num_of_parameters, Matrix_real& solution_guess) {
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
        int iteration_loops_max = 1;
        /*try {
            iteration_loops_max = std::max(iteration_loops[qbit_num], 1);
        }
        catch (...) {
            iteration_loops_max = 1;
        }*/

        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        // maximal number of inner iterations overriden by config
        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_bayes_opt") > 0 ) {
            config["max_inner_iterations_bfgs"].get_property( max_inner_iterations_loc );         
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


        int fragment_size = 5;
        int leftover = num_of_parameters%fragment_size;
        int number_of_agents = num_of_parameters/fragment_size;
        Matrix_real cost_funcs(1,number_of_agents);
        Matrix_real parameters_agents(1,num_of_parameters);
        for (int iter=0;iter<max_inner_iterations_loc;iter++){
            tbb::parallel_for( tbb::blocked_range<int>(0,number_of_agents,2), [&](tbb::blocked_range<int> r) {
                for (int idx=r.begin(); idx<r.end(); ++idx) {
                int start = idx*fragment_size;
                Bayes_Opt_Beam cAgent(optimization_problem,this,start,solution_guess);
                Matrix_real solution_guess_agent(1,fragment_size);
                memcpy(solution_guess_agent.get_data(),solution_guess.get_data()+start,sizeof(double)*fragment_size);
                cost_funcs[idx] = cAgent.Start_Optimization(solution_guess_agent,25);
                memcpy(parameters_agents.get_data()+start,solution_guess_agent.get_data(),sizeof(double)*fragment_size);
                } 
            });
            double cost_func_leftover = 100000.;
            if (leftover!=0){
            Bayes_Opt_Beam cAgent(optimization_problem,this,fragment_size*number_of_agents,solution_guess);
            Matrix_real solution_guess_agent(1,leftover);
            memcpy(solution_guess_agent.get_data(),solution_guess.get_data()+fragment_size*number_of_agents,sizeof(double)*leftover);
            cost_func_leftover = cAgent.Start_Optimization(solution_guess_agent,25);
            memcpy(parameters_agents.get_data()+fragment_size*number_of_agents,solution_guess_agent.get_data(),sizeof(double)*leftover);
            }
            int best_idx=0;
            for (int idx=1;idx<number_of_agents;idx++){
                best_idx = (cost_funcs[best_idx]<cost_funcs[idx])? best_idx:idx;
            }
            
            if (cost_funcs[best_idx]<cost_func_leftover){
                if (current_minimum>cost_funcs[best_idx]){
                    current_minimum = cost_funcs[best_idx];
                    int start = best_idx*fragment_size;
                    memcpy(solution_guess.get_data() + start,parameters_agents.get_data()+start,sizeof(double)*fragment_size);
                }
            }
            else{
                if (current_minimum>cost_func_leftover && leftover!=0){
                current_minimum = cost_func_leftover;
                memcpy(solution_guess.get_data() + fragment_size*number_of_agents,parameters_agents.get_data()+fragment_size*number_of_agents,sizeof(double)*leftover);
                }
            }

            if ( output_periodicity>0 && iter % output_periodicity == 0 ) {
               export_current_cost_fnc(current_minimum);
            }
        }

}




