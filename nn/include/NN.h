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
/*! \file Decomposition_Base.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/

#ifndef NN_H
#define NN_H

#include "matrix.h"
#include "matrix_real.h"


#include "Gates_block.h"
#include "CZ.h"
#include "CH.h"
#include "CNOT.h"
#include "U3.h"
#include "RX.h"
#include "X.h"
#include "SX.h"
#include "RY.h"
#include "CRY.h"
#include "RZ.h"
#include "SYC.h"
#include "UN.h"
#include "ON.h"
#include "Adaptive.h"
#include "Composite.h"
#include <map>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include "gsl/gsl_multimin.h"
#include "gsl/gsl_statistics.h"
#include <tbb/cache_aligned_allocator.h>




/**
@brief A class containing basic methods for the decomposition process.
*/
class NN {


public:
  
    /// number of gate blocks used in one shot of the optimization process
    int tt;

protected:




public:

/** 
@brief Nullary constructor of the class
@return An instance of the class
*/
NN();


/** 
@brief call retrieve the channels for the neural network associated with a single 2x2 kernel
@return return with an 1x4 array containing the chanels prepared for the neural network. (dimension 4 stands for theta_up, phi, theta_down , lambda)
*/
void get_nn_chanels_from_kernel( Matrix& Umtx, Matrix_real& chanels);

/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@return return with an array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension 4 stands for theta_up, phi, theta_down , lambda)
*/
Matrix_real get_nn_chanels( Matrix& Umtx);


};

#endif //NN
