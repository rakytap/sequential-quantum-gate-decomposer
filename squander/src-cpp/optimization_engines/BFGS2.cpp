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

        bool use_basin_hopping = false;
        if ( config.count("use_basin_hopping") > 0 ) {
            config["use_basin_hopping"].get_property( use_basin_hopping );  
        }

        if (use_basin_hopping) {
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
        } else {
            enum strategy_enum { STRAT_BEST1BIN, STRAT_RAND1BIN };
            enum init_enum { INIT_RANDOM, INIT_LHS };

            // ---------------- DE hyperparameters (SciPy-ish) ----------------
            long long de_strategy = STRAT_BEST1BIN; // or STRAT_RAND1BIN
            long long de_popsize = 15; // SciPy default NP ~= 15*D (so popsize is NP/D). We map so NP = de_popsize*num_params.
            double de_mutation = 0.8;          // F
            double de_recombination = 0.7;     // CR
            double de_tol = 1e-6;
            long long de_init = INIT_RANDOM;    // or INIT_LHS
            long long de_polish = 1;           // run BFGS after DE?


            if (config.count("de_strategy") > 0)                 config["de_strategy"].get_property(de_strategy);
            if (config.count("de_popsize") > 0)                  config["de_popsize"].get_property(de_popsize);
            if (config.count("de_mutation") > 0)                 config["de_mutation"].get_property(de_mutation);
            if (config.count("de_recombination") > 0)            config["de_recombination"].get_property(de_recombination);
            if (config.count("de_tol") > 0)                      config["de_tol"].get_property(de_tol);
            if (config.count("de_init") > 0)                     config["de_init"].get_property(de_init);
            if (config.count("de_polish") > 0) { long long v;    config["de_polish"].get_property(v); de_polish = v ? 1 : 0; }

            // Sanity + derived
            if (de_mutation <= 0.0) de_mutation = 0.8;
            if (de_recombination < 0.0) de_recombination = 0.7; else if (de_recombination > 1.0) de_recombination = 1.0;
            if (de_popsize <= 0) de_popsize = 15;

            const int D = num_of_parameters;
            int NP = de_popsize * D;
            NP = std::max(5, NP);

            // ---------------- Population storage ----------------
            std::vector<Matrix_real> pop(NP, Matrix_real(D, 1));
            std::vector<double>      fit(NP, DBL_MAX);

            // Initialize population
            auto init_random = [&]() {
                for (int i = 0; i < NP; ++i) {
                    for (int j = 0; j < D; ++j) {
                        pop[i][j] = distrib_real(gen) * 2.0 * M_PI; // in [0, 2π)
                    }
                }
            };

            auto init_lhs = [&]() {
                // Simple stratified (Latin hypercube-like) per-dimension on [0, 2π)
                std::vector<int> idx(NP);
                for (int i = 0; i < NP; ++i) idx[i] = i;

                for (int j = 0; j < D; ++j) {
                    std::shuffle(idx.begin(), idx.end(), gen);
                    for (int i = 0; i < NP; ++i) {
                        double a = (double)idx[i] / (double)NP;
                        double b = (double)(idx[i] + 1) / (double)NP;
                        double r = a + (b - a) * distrib_real(gen);
                        pop[i][j] = fmod(r * (2.0 * M_PI), 2.0 * M_PI);
                    }
                }
            };

            if (de_init == INIT_LHS) init_lhs(); else init_random();

            // Evaluate initial population
        #ifdef __MPI__
            // If you want, you can distribute evaluations here; for simplicity we do sequential eval.
        #endif
            double best_f = DBL_MAX;
            int best_i = 0;
            for (int i = 0; i < NP; ++i) {
                fit[i] = this->optimization_problem(pop[i]);
                if (fit[i] < best_f) { best_f = fit[i]; best_i = i; }
            }
            if (best_f < current_minimum) {
                current_minimum = best_f;
                memcpy(optimized_parameters_mtx.get_data(), pop[best_i].get_data(), D * sizeof(double));
            }

            // ---------------- DE loop ----------------
            std::vector<int> idx_all(NP);
            for (int i = 0; i < NP; ++i) idx_all[i] = i;

            auto pick_3_distinct = [&](int exclude) {
                // pick r1, r2, r3 distinct and != exclude
                int r1, r2, r3;
                do { r1 = idx_all[std::uniform_int_distribution<int>(0, NP-1)(gen)]; } while (r1 == exclude);
                do { r2 = idx_all[std::uniform_int_distribution<int>(0, NP-1)(gen)]; } while (r2 == exclude || r2 == r1);
                do { r3 = idx_all[std::uniform_int_distribution<int>(0, NP-1)(gen)]; } while (r3 == exclude || r3 == r1 || r3 == r2);
                return std::tuple<int,int,int>(r1, r2, r3);
            };

            Matrix_real trial(D, 1), mutant(D, 1);
            long long gen_no_improve = 0;
            double last_best = best_f;

            for (long long generation = 0; generation < iteration_loops_max; ++generation) {

                for (int i = 0; i < NP; ++i) {
                    // ----- Mutation -----
                    if (de_strategy == STRAT_BEST1BIN) {
                        // mutant = best + F*(r1 - r2)
                        int r1, r2;
                        do { r1 = idx_all[std::uniform_int_distribution<int>(0, NP-1)(gen)]; } while (r1 == i || r1 == best_i);
                        do { r2 = idx_all[std::uniform_int_distribution<int>(0, NP-1)(gen)]; } while (r2 == i || r2 == r1 || r2 == best_i);
                        for (int j = 0; j < D; ++j) {
                            mutant[j] = fmod(pop[best_i][j] + de_mutation * (pop[r1][j] - pop[r2][j]), 2.0 * M_PI);
                        }
                    } else { // "rand1bin" default
                        // mutant = r1 + F*(r2 - r3)
                        int r1, r2, r3;
                        std::tie(r1, r2, r3) = pick_3_distinct(i);
                        for (int j = 0; j < D; ++j) {
                            mutant[j] = fmod(pop[r1][j] + de_mutation * (pop[r2][j] - pop[r3][j]), 2.0 * M_PI);
                        }
                    }

                    // ----- Binomial crossover -----
                    int jrand = std::uniform_int_distribution<int>(0, D - 1)(gen);
                    for (int j = 0; j < D; ++j) {
                        if (j == jrand || distrib_real(gen) < de_recombination) {
                            trial[j] = mutant[j];
                        } else {
                            trial[j] = pop[i][j];
                        }
                    }

                    // ----- Selection -----
                    double f_trial = this->optimization_problem(trial);
                    if (f_trial <= fit[i]) {
                        memcpy(pop[i].get_data(), trial.get_data(), D * sizeof(double));
                        fit[i] = f_trial;

                        if (f_trial < best_f) {
                            best_f = f_trial;
                            best_i = i;
                        }
                    }
                }

                // Update global best into project-wide state
                if (best_f < current_minimum) {
                    current_minimum = best_f;
                    memcpy(optimized_parameters_mtx.get_data(), pop[best_i].get_data(), D * sizeof(double));
                }

                // Progress / export
                if (generation % 50 == 0) {
                    std::stringstream sstream;
                    sstream << "DE: generation " << generation << "/" << iteration_loops_max
                            << ", best=" << current_minimum << std::endl;
                    print(sstream, 2);

                    if (export_circuit_2_binary_loc > 0) {
                        std::string filename("initial_circuit_iteration.binary");
                        if (project_name != "") filename = project_name + "_" + filename;
                        export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);
                    }
                }

                if (output_periodicity > 0 && generation % output_periodicity == 0) {
                    export_current_cost_fnc(current_minimum);
                }

        #ifdef __MPI__
                // Optional sync of best-so-far to workers
                MPI_Bcast((void*)optimized_parameters_mtx.get_data(), D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        #endif

                // Stopping: reached target
                if (current_minimum < optimization_tolerance_loc) break;

                // Stopping: no improvement by > de_tol in this generation
                if (std::abs(last_best - best_f) <= de_tol) {
                    ++gen_no_improve;
                } else {
                    gen_no_improve = 0;
                }
                last_best = best_f;

                if (gen_no_improve >= 1) {
                    // one stagnant generation with < de_tol delta (simple, effective)
                    break;
                }
            }

            // ---------------- Polish with BFGS (optional) ----------------
            if (de_polish) {
                BFGS_Powell cBFGS_Powell(optimization_problem_combined, this);
                auto params = optimized_parameters_mtx.copy();
                double f_pol = cBFGS_Powell.Start_Optimization(params, /*max_inner_iterations*/  std::max<long long>(200, D*50));
                if (f_pol < current_minimum) {
                    current_minimum = f_pol;
                    memcpy(optimized_parameters_mtx.get_data(), params.get_data(), D * sizeof(double));
                }
            }            
        }

        tbb::tick_count bfgs_end = tbb::tick_count::now();
        CPU_time  = CPU_time + (bfgs_end-bfgs_start).seconds();
        //std::cout << "bfgs2 time: " << CPU_time << " " << current_minimum << std::endl;

}


