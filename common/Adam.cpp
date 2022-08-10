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

	



/** Nullary constructor of the class
@return An instance of the class
*/
Adam::Adam() {

    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    eta = 0.01;



    mom = 0.0;
    var = 0.0;
    iter_t = 0;
	
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


    mom = 0.0;
    var = 0.0;
    iter_t = 0;


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
@brief Call to set the number of gate blocks to be optimized in one shot
@param optimization_block_in The number of gate blocks to be optimized in one shot
*/
void Adam::update( Matrix_real& parameters, Matrix_real& grad ) {


    //access the idx-th element of the parameters or grad:
    size_t idx=2;
    double par_value = parameters[idx];




}

