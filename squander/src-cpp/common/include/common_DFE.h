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
/*! \file common_DFE.h
    \brief Header file for DFE support
*/

#ifndef common_DFE_H
#define common_DFE_H


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


#define DFE_LIB_9QUBITS "libqgdDFE.so"
#define DFE_LIB_10QUBITS "libqgdDFE_10qubits.so"
#define DFE_LIB_SIM "libqgdDFE_SIM.so"

#define DFE_LIB_SV "libsvDFE.so"

extern "C"
{


/**
@brief Fixed point data related to a gate operation
@param Theta Value of Theta/2
@param Phi Value of Phi
@param Lambda Value of Lambda
@param target_qbit Qubit on which the gate is applied
@param control_qbit The control qubit. For single qubit operations control_qbit=-1
@param gate_type Gate type according to enumeration of gate_type defined in SQUANDER
@param metadata The most significat bit is set to 1 for derivated gate operation. Set the (8-i)-th bit to 1 if the i-th element of the 2x2 gate kernel should be zero in the derivated gate operation. (If the 0st and 3nd element in kernel matrix should be zero then metadat should be 5 + (1<<7), since 5 = 0101. The the leading 1<<7 bit indicates that a derivation is processed.)
*/
typedef struct {
	int32_t ThetaOver2;
	int32_t Phi;
	int32_t Lambda;
	int8_t target_qbit;
	int8_t control_qbit;
	int8_t gate_type;
	uint8_t metadata;
} DFEgate_kernel_type;


}



/**
@brief ????????????
@return ??????????
*/
size_t get_accelerator_avail_num();


/**
@brief ????????????
@return ??????????
*/
size_t get_accelerator_free_num();

/**
@brief ????????????
@return ??????????
*/
int get_initialize_id();

/**
@brief ????????????
@return ??????????
*/
int calcqgdKernelDFE(size_t rows, size_t cols, DFEgate_kernel_type* gates, int gatesNum, int gateSetNum, int traceOffset, double* trace);


/**
@brief ????????????
@return ??????????
*/
int get_chained_gates_num();




/**
@brief ????????????
@return ??????????
*/
void uploadMatrix2DFE( Matrix& input );



/**
@brief ????????????
@return ??????????
*/
void unload_dfe_lib();


/**
@brief ????????????
@return ??????????
*/
void lock_lib();


/**
@brief ????????????
@return ??????????
*/
void unlock_lib();




/**
@brief ????????????
@return ??????????
*/
int init_dfe_lib( const int accelerator_num, int qbit_num, int initialize_id_in);


int init_groq_sv_lib( const int accelerator_num );
void unload_groq_sv_lib();
unsigned int ctz(unsigned int v);
void apply_to_groq_sv(int device_num, std::vector<Matrix>& u3_qbit, Matrix& input, std::vector<int>& target_qbit, std::vector<int>& control_qbit);

#endif

