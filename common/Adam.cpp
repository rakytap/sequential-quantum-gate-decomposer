/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Adam.cpp
    \brief A class for Adam optimization according to https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
*/

#include "Adam.h"
#include "tbb/tbb.h"

	

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
*/
void Adam::update( Matrix_real& parameters, Matrix_real& grad ) {

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
        param_data[idx] = param_data[idx] - eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon);
        /*
        if ( std::abs(eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon)) > 1e-3 ) {
            std::cout << std::abs(eta * mom_bias_corr/(sqrt(var_bias_corr) + epsilon)) << std::endl;
        }
        */
        
    //}
    });

});

    iter_t++;
}

