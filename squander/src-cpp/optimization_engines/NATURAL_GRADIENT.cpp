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
#include "common.h"

#include <fstream>


#ifdef __DFE__
#include "common_DFE.h"
#endif
void Optimization_Interface::solve_layer_optimization_problem_NATURAL_GRADIENT( int num_of_parameters, Matrix_real& solution_guess) {


        double M_PI_quarter = M_PI/4;
        double M_PI_half    = M_PI/2;
        double M_PI_double  = M_PI*2.0;

        if ( cost_fnc != VQE ) {
            std::string err("Optimization_Interface::solve_layer_optimization_problem_NATURAL_GRADIENT: Only cost function VQE are implemented for this strategy");
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
        
        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_natural_gradient") > 0 ) {
             config["max_inner_iterations_natural_gradient"].get_property( max_inner_iterations_loc );  
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
             config["max_inner_iterations"].get_property( max_inner_iterations_loc );  
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_natural_gradient") > 0 ) {
             config["optimization_tolerance_natural_gradient"].get_property( optimization_tolerance_loc );  
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
        if ( config.count("eta_natural_gradient") > 0 ) {
             config["eta_natural_gradient"].get_property( eta_loc ); 
        }
        if ( config.count("eta") > 0 ) {
             long long tmp;
             config["eta"].get_property( eta_loc );  
        }
        else {
            eta_loc = 1e-3;
        }
        Matrix_real f0_vec(1, 100); 
        memset( f0_vec.get_data(), 0.0, f0_vec.size()*sizeof(double) );
        double f0_mean = 0.0;
        int f0_idx = 0; 
        double f0;
        std::vector<Matrix> State_deriv;
        Matrix_real F(num_of_parameters,num_of_parameters);
        Matrix State_loc;
        Matrix_real grad(1,num_of_parameters);
        for (int iter=0; iter<max_inner_iterations_loc; iter++){
            get_derivative_components(solution_guess_tmp_mtx,this,&f0,grad,State_deriv,State_loc);
            for (int idx=0; idx<num_of_parameters;idx++){
                for (int jdx=0; jdx<num_of_parameters; jdx++){
                    double comp1=0;
                    QGD_Complex16 comp2;
                    comp2.real=0;
                    comp2.imag=0;
                    QGD_Complex16 comp3;
                    comp3.real=0;
                    comp3.imag=0;
                    for (int mdx=0;mdx<State_loc.size();mdx++){
                        comp1 = comp1 + State_deriv[idx][mdx].real*State_deriv[jdx][mdx].real + State_deriv[idx][mdx].imag*State_deriv[jdx][mdx].imag;
                        comp2.real = comp2.real + State_deriv[idx][mdx].real*State_loc[mdx].real + State_deriv[idx][mdx].imag*State_loc[mdx].imag;
                        comp2.imag = comp2.imag + State_deriv[idx][mdx].real*State_loc[mdx].imag - State_deriv[idx][mdx].imag*State_loc[mdx].real;
                        comp3.real = comp3.real + State_deriv[jdx][mdx].real*State_loc[mdx].real + State_deriv[jdx][mdx].imag*State_loc[mdx].imag;
                        comp3.imag = comp3.imag + State_deriv[jdx][mdx].imag*State_loc[mdx].real - State_deriv[jdx][mdx].real*State_loc[mdx].imag;
                    }
                    F[idx*num_of_parameters+jdx] = comp1 - comp2.real*comp3.real + comp2.imag*comp3.imag;
                }
            }
            Matrix_real rhs_natgrad(1,num_of_parameters);
            conjugate_gradient(F,grad,rhs_natgrad,1e-3);
            for (int idx=0; idx<num_of_parameters; idx++){
                solution_guess_tmp_mtx[idx] = solution_guess_tmp_mtx[idx] - eta_loc*rhs_natgrad [idx];
            }
            number_of_iters = number_of_iters + 1; 
            if (current_minimum>f0){
                current_minimum = f0;
            }
            if (current_minimum<optimization_tolerance_loc){
                break;
            }
            std::stringstream sstream;
            sstream << "GRAD_DESCEND_SHIFT_RULE: processed iterations " << (double)iter/max_inner_iterations_loc*100 << "\%, current minimum:" << current_minimum<< "\n";
            print(sstream, 0);   
        }
        return;
}
