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
/*! \file RL_experience.h
    \brief Header file for a class ???
*/

#ifndef RLEXPERIENCE_H
#define RLEXPERIENCE_H

#include "matrix_real.h"
#include "matrix_base.h"
#include "Gates_block.h"
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <random>


/**
@brief A class for RL_experience optimization according to https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
*/
class RL_experience  {


public:
  
    // learning rate of the RL_experience algorithm
    double eta;  


//protected:

    /// number of involved parameters
    int parameter_num;

    // the number of iterations used during the training
    unsigned long long iteration_num;

    /// attribute stroing the gate structure
    Gates_block* gates;

    // array containing the probabilities of which parameter to choose in the next turn of the optimization. Element (i,j) means that after the i-th parameter the j-th parameter would be chosen with probability of A_ij.
    Matrix_real parameter_probs;


    /// array containing the counts of successive parameters used in the optimization. Element (i,j) means that after the i-th parameter the j-th parameter was chosen A_ij times.
    matrix_base<int> parameter_counts;

    /// total counts in one row of parameter_counts --- reset when probabilites are updated
    matrix_base<int> total_counts;

    /// total counts used to evasluate one row in parameter_probs
    matrix_base<unsigned long long> total_counts_probs;



    std::vector<int> history;


    ///
    double exploration_rate;



public:

/** Nullary constructor of the class
@return An instance of the class
*/
RL_experience();

/** Contructor of the class
@brief Constructor of the class.
@param gates_in
*/
RL_experience( Gates_block* gates_in, unsigned long long iteration_num_in );

/**
@brief Copy constructor.
@param experience An instance of class
@return Returns with the instance of the class.
*/
RL_experience(const RL_experience& experience );

/** 
@brief Call to draw the next index
@param curent_index The current index after which the next one is selected
@return Returns with the selected index
*/
int draw( const int& curent_index, std::mt19937& gen );


/**
@brief Destructor of the class
*/
virtual ~RL_experience();


/**
@brief ?????????????
*/
void reset();


/**
@brief Assignment operator.
@param experience An instance of class
@return Returns with the instance of the class.
*/
RL_experience& operator= (const RL_experience& experience );


/**
@brief Call to make a copy of the current instance.
@return Returns with the instance of the class.
*/
RL_experience copy();

/** 
@brief Call to update the trained probabilities and reset the counts
*/
void update_probs( );


void export_probabilities();


void import_probabilities();

};


#endif //RL_experience
