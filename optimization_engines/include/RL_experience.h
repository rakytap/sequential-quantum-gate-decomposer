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
/*! \file RL_experience.h
    \brief Header file for a class ???
*/

#ifndef RLEXPERIENCE_H
#define RLEXPERIENCE_H

#include "matrix_real.h"
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>


/**
@brief A class for RL_experience optimization according to https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
*/
class RL_experience  {


public:
  
    // learning rate of the RL_experience algorithm
    double eta;  


protected:



public:

/** Nullary constructor of the class
@return An instance of the class
*/
RL_experience();

/** Contructor of the class
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary to be decomposed.
@param initial_guess_in Type to guess the initial values for the optimization. Possible values: ZEROS=0, RANDOM=1, CLOSE_TO_ZERO=2
@return An instance of the class
*/
RL_experience( double beta1_in, double beta2_in, double epsilon_in, double eta_in);

/**
@brief Destructor of the class
*/
virtual ~RL_experience();


/**
@brief ?????????????
*/
void reset();



};


#endif //RL_experience
