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
/*! \file Decomposition_Base.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/

#ifndef NN_H
#define NN_H

#include "matrix.h"
#include "matrix_real.h"

#include <time.h>
#include <stdlib.h>
#include <random>

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
#include <tbb/cache_aligned_allocator.h>




/**
@brief A class containing basic methods for the decomposition process.
*/
class NN {


public:
  
    /// number of gate blocks used in one shot of the optimization process
    int tt;

protected:


    /// Will be used to obtain a seed for the random number engine
    std::random_device rd;  
    /// Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen; 
    /// connectivity between the wubits
    std::vector<matrix_base<int>> topology;
    /// Store the number of OpenMP threads. (During the calculations OpenMP multithreading is turned off.)
    int num_threads;




public:

/** 
@brief Nullary constructor of the class
@return An instance of the class
*/
NN();


/** Constructor of the class
@param The list of conenctions between the qubits
@return An instance of the class
*/
NN( std::vector<matrix_base<int>> topology_in );



/** 
@brief Call to construct random parameter, with limited number of non-trivial adaptive layers
@param num_of_parameters The number of parameters
*/
void create_randomized_parameters( int num_of_parameters, int qbit_num, int levels, Matrix_real& parameters, matrix_base<int8_t>& nontrivial_adaptive_layers );


/** 
@brief call retrieve the channels for the neural network associated with a single 2x2 kernel
@return return with an 1x4 array containing the chanels prepared for the neural network. (dimension 4 stands for theta_up, phi, theta_down , lambda)
*/
void get_nn_chanels_from_kernel( Matrix& kernel_up, Matrix& kernel_down, Matrix_real& chanels);

/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@param Umtx A unitary of dimension dim x dim, where dim is a power of 2.
@param target_qbit The target qubit for which the chanels are calculated
@param chanels output array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
*/
void get_nn_chanels( const Matrix& Umtx, const int& target_qbit, Matrix_real& chanels);

/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@param Umtx A unitary of dimension dim x dim, where dim is a power of 2.
@param chanels output array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
*/
void get_nn_chanels( int qbit_num, const Matrix& Umtx, Matrix_real& chanels);

/** 
@brief call retrieve the channels for the neural network associated with a single, randomly generated unitary
@param qbit_num The number of qubits
@param levels The number of adaptive levels to be randomly constructed
@param chanles output argument to return with an array containing the chanels prepared for the neural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
@param parameters output argument of the randomly created parameters
*/
void get_nn_chanels(int qbit_num, int levels, Matrix_real& chanels, matrix_base<int8_t>& nontrivial_adaptive_layers);


/** 
@brief call retrieve the channels for the neural network associated with a single, randomly generated unitary
@param qbit_num The number of qubits
@param levels The number of adaptive levels to be randomly constructed
@param samples_num The number of samples
@param chanles output argument to return with an array containing the chanels prepared for the neural network. The array has dimensions [ samples_num, dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
@param parameters output argument of the randomly created parameters
*/
void get_nn_chanels(int qbit_num, int levels, int samples_num, Matrix_real& chanels, matrix_base<int8_t>& nontrivial_adaptive_layers);

};

#endif //NN
