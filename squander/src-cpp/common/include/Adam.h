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
/*! \file Adam.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/

#ifndef ADAM_H
#define ADAM_H

#include "matrix_real.h"
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>


/**
@brief A class for Adam optimization according to https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
*/
class Adam  {


public:
  
    // learning rate of the Adam algorithm
    double eta;  


protected:

    /// Store the number of OpenMP threads. (During the calculations OpenMP multithreading is turned off.)
    int num_threads;
    /// parameter beta1 of the Adam algorithm
    double beta1;
    /// parameter beta2 of the Adam algorithm
    double beta2;
    // epsilon regularization parameter of the Adam algorithm
    double epsilon;
  
    /// momentum parameter of the Adam algorithm
    Matrix_real mom;
    /// variance parameter of the Adam algorithm
    Matrix_real var;
    /// iteration index
    int64_t iter_t;

    /// beta1^t
    double beta1_t;
    /// beta2^t
    double beta2_t;   

    /// vector stroing the lates values of cost function values to test local minimum
    Matrix_real f0_vec; 
    /// Mean of the latest cost function values to test local minimum
    double f0_mean;
    /// current index in the f0_vec array
    int f0_idx;
    
    /// vector containing 1 if cost function decreased from previous value, and -1 if it increased
    matrix_base<int> decreasing_vec; 
    /// current index in the decreasing_vec array
    int decreasing_idx = 0;
    /// decreasing_test
    double decreasing_test;
    /// previous value of the cost function
    double f0_prev;


public:

/** Nullary constructor of the class
@return An instance of the class
*/
Adam();

/** Contructor of the class
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary to be decomposed.
@param initial_guess_in Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return An instance of the class
*/
Adam( double beta1_in, double beta2_in, double epsilon_in, double eta_in);

/**
@brief Destructor of the class
*/
virtual ~Adam();


/**
@brief ?????????????
*/
void reset();


/**
@brief ?????????????
*/
void initialize_moment_and_variance(int parameter_num);

/**
@brief Call to set the number of gate blocks to be optimized in one shot
@param optimization_block_in The number of gate blocks to be optimized in one shot
*/
int update( Matrix_real& parameters, Matrix_real& grad, const double& f0 );


/**
@brief ?????????????
*/
double get_decreasing_test();

};


#endif //Adam
