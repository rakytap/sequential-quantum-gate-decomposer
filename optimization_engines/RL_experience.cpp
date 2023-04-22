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
/*! \file RL_experience.cpp
    \brief A class for RL_experience 
*/

#include "RL_experience.h"
#include "tbb/tbb.h"

#include <cfloat>	

/** Nullary constructor of the class
@return An instance of the class
*/
RL_experience::RL_experience() {

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
RL_experience::RL_experience( double beta1_in, double beta2_in, double epsilon_in, double eta_in ) {

   
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

RL_experience::~RL_experience() {
}



/**
@brief ?????????????
*/
void RL_experience::reset() {



}


