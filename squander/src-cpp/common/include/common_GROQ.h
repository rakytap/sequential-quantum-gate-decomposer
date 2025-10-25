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
/*! \file common_GROQ.h
    \brief Header file for Groq LPU support in state vector simulation
*/

#ifndef common_GROQ_H
#define common_GROQ_H


#include <omp.h>
#include "QGDTypes.h"
#include "dot.h"


#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <sstream>

#define DFE_LIB_SV "libsvDFE.so"




/**
@brief Call to get the identification number of the inititalization of the library
@return Returns with the identification number of the inititalization of the library
*/
int get_initialize_id();


/**
@brief Call to allocated Groq cards for calculations
@param reserved_device_num The number of Groq accelerator cards to be allocated for the calulations
@param initialize_id_in Identification number of the inititalization of the library
@return Returns with 1 on success
*/
int init_groq_sv_lib( const int reserved_device_num, int initialize_id_in );

/**
@brief Call to unload the programs from the reserved Groq cards
*/
void unload_groq_sv_lib();




/**
@brief Call to pefrom the state vector simulation on the Groq hardware
@param reserved_device_num The number of Groq accelerator cards to be allocated for the calulations
@param chosen_device_num The ordinal number of the Groq accelerator card on which the calculation should be performed (0<=chosen_device_num<reserved_device_num)
@param qbit_num The number of qubits
@param u3_qbit An array of the gate kernels
@param target_qbits The array of target qubits
@param control_qbits The array of control qubits
@param quantum_state The input state vector on which the transformation is applied. The transformed state is returned via this input.
@param id_in Identification number of the inititalized library
*/
void apply_to_groq_sv(int reserved_device_num, int chosen_device_num, int qbit_num, std::vector<Matrix>& u3_qbit, std::vector<int>& target_qbits, std::vector<int>& control_qbits, Matrix& quantum_state, int id_in);

#endif

