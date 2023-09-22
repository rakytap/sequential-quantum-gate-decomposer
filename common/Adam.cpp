/*

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
/*! \file Adam.cpp
    \brief A class for Adam optimization according to https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
*/

#include "Adam.h"
#include "tbb/tbb.h"

#include <cfloat>	

/** Nullary constructor of the class
@return An instance of the class
*/
Adam::Adam() {

    beta1 = 0.68;
    beta2 = 0.8;
    epsilon = 1e-4;
    eta = 0.001;



    reset();
	
#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif

}


/** Contructor of the class
@brief Constructor of the class.
@param ???????????????????????????????
@param ???????????????????????????????
@param ???????????????????????????????
@return An instance of the class
*/
Adam::Adam( double beta1_in, double beta2_in, double epsilon_in, double eta_in ) {

   
    beta1 = beta1_in;
    beta2 = beta2_in;
    epsilon = epsilon_in;
    eta = eta_in;


    reset();


#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif

}

/**
@brief Destructor of the class
*/

Adam::~Adam() {
}



/**
@brief ?????????????
*/
void Adam::reset() {


    mom = Matrix_real(0,0);
    var = Matrix_real(0,0);

    iter_t = 0;
    beta1_t = 1.0;
    beta2_t = 1.0;

    // vector stroing the lates values of cost function to test local minimum
    f0_vec = Matrix_real(1, 100); 
    memset( f0_vec.get_data(), 0.0, f0_vec.size()*sizeof(double) );
    f0_mean = 0.0;
    f0_idx = 0;
    
    
    // decreasing_test
    decreasing_vec = matrix_base<int>(1, 20); 
    memset( decreasing_vec.get_data(), -1, decreasing_vec.size()*sizeof(int) );
    decreasing_idx = 0;
    decreasing_test = -1.0;  
    
    // previous value of the cost function
    f0_prev = DBL_MAX;  


}


/**
@brief ?????????????
*/
void Adam::initialize_moment_and_variance(int parameter_num) {

    mom = Matrix_real(parameter_num,1);
    var = Matrix_real(parameter_num,1);  

    memset( mom.get_data(), 0.0, mom.size()*sizeof(double) );
    memset( var.get_data(), 0.0, var.size()*sizeof(double) );
}


/**
@brief Call to set the number of gate blocks to be optimized in one shot
@param optimization_block_in The number of gate blocks to be optimized in one shot
@return 0 optimizer is in decreasing stage, 1 converged to local minumum
*/
int Adam::update( Matrix_real& parameters, Matrix_real& grad, const double& f0 ) {

    int parameter_num = parameters.size();
    if ( parameter_num != grad.size() ) {
        std::string error("Adam::update: number of parameters shoulod be equal to the number of elements in gradient vector");
        throw error;
    }

    if ( mom.size() == 0 ) {
        initialize_moment_and_variance( parameter_num );
    }

    if ( parameter_num != mom.size() ) {
        std::string error("Adam::update: number of parameters shoulod be equal to the number of elements in momentum vector");
        throw error;
    }


    // test local minimum convergence
    f0_mean = f0_mean + (f0 - f0_vec[ f0_idx ])/f0_vec.size();
    f0_vec[ f0_idx ] = f0;
    f0_idx = (f0_idx + 1) % f0_vec.size();
    
    double var_f0 = 0.0;
    for (int idx=0; idx<f0_vec.size(); idx++) {
       var_f0 = var_f0 + (f0_vec[idx]-f0_mean)*(f0_vec[idx]-f0_mean);
    }
    var_f0 = std::sqrt(var_f0)/f0_vec.size();


    if ( f0 < f0_prev ) {
        if ( decreasing_vec[ decreasing_idx ] == 1 ) {
            // the decresing test did not changed
        }
        else {
            // element in decreasing vec changed from -1 to 1
            decreasing_test = decreasing_test + 2.0/decreasing_vec.size();
        }

        decreasing_vec[ decreasing_idx ] = 1;
    }
    else {
        if ( decreasing_vec[ decreasing_idx ] == 1 ) {
            // element in decreasing vec changed from 1 to -1
            decreasing_test = decreasing_test - 2.0/decreasing_vec.size();
        }
        else {
            // the decresing test did not changed
        }

        decreasing_vec[ decreasing_idx ] = -1;
    }


    decreasing_idx = (decreasing_idx + 1) % decreasing_vec.size();

    f0_prev = f0;




    // test barren plateau
    double grad_var = 0.0;
    for( int idx=0; idx<parameter_num; idx++ ) {
        grad_var += var[idx];
    }   
        
    int barren_plateau = 0;
    if ( grad_var < epsilon && decreasing_test > 0.7  ) {   
        // barren plateau
        barren_plateau = 1;
    }

     
    double* mom_data = mom.get_data();
    double* var_data = var.get_data();
    double* grad_data = grad.get_data();
    double* param_data = parameters.get_data();

tbb::task_arena ta(4);
ta.execute( [&](){
    tbb::parallel_for( 0, parameter_num, 1, [&](int idx) {
    //for (int idx=0; idx<parameter_num; idx++) {
        mom_data[idx] = beta1 * mom_data[idx] + (1-beta1) * grad_data[idx];
        var_data[idx] = beta2 * var_data[idx] + (1-beta2) * grad_data[idx] * grad_data[idx];

        // bias correction step
        beta1_t = beta1_t * beta1;
        double mom_bias_corr = mom_data[idx]/(1-beta1_t);

        beta2_t = beta2_t * beta2;
        double var_bias_corr = var_data[idx]/(1-beta2_t);

        // update parameters
        if ( barren_plateau ) {
            param_data[idx] = param_data[idx] - eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon/100);
        }
        else {
            param_data[idx] = param_data[idx] - eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon);
        }
        /*
        if ( std::abs(eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon)) > 1e-3 ) {
            std::cout << std::abs(eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon)) << std::endl;
        }
        */
        
    //}
    });

});


    iter_t++;



    int ADAM_status = 0;
    if ( std::abs( f0_mean - f0) < 1e-6 && decreasing_test <= 0.7 && var_f0/f0_mean < 1e-6 ) {
        // local minimum
        ADAM_status = 1;
    }
    else {
        ADAM_status = 0;
    }    

    return ADAM_status;



}


/**
@brief ?????????????
*/
double Adam::get_decreasing_test() {

    return decreasing_test;

}

