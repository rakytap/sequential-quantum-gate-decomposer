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
    \brief Header file for DFE support in unitary simulation
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
@brief Call to get the available number of accelerators
@return Retirns with the number of the available accelerators
*/
size_t get_accelerator_avail_num();


/**
@brief Call to get the number of free accelerators
@return Retirns with the number of the free accelerators
*/
size_t get_accelerator_free_num();

/**
@brief Call to get the identification number of the inititalization of the library
@return Returns with the identification number of the inititalization of the library
*/
int get_initialize_id();

/**
@brief Call to execute the calculation on the reserved DFE engines.
@param rows The number of rows in the input matrix
@param cols the number of columns in the input matrix
@param gates The metadata describing the gates to be applied on the input
@param gatesNum The number of the chained up gates.
@param gateSetNum Integer descibing how many individual gate chains are encoded in the gates input.
@param traceOffset In integer describing an offset in the trace calculation
@param trace The trace of the transformed unitaries are returned through this pointer
@return Return with 0 on success
*/
int calcqgdKernelDFE(size_t rows, size_t cols, DFEgate_kernel_type* gates, int gatesNum, int gateSetNum, int traceOffset, double* trace);


/**
@brief Call to retrieve the number of gates that should be chained up during the execution of the DFE library
@return Returns with the number of the chained gates.
*/
int get_chained_gates_num();




/**
@brief Call to upload the input matrix to the DFE engine
@param input The input matrix
*/
void uploadMatrix2DFE( Matrix& input );



/**
@brief Call to unload the DFE libarary and release the allocated devices
*/
void unload_dfe_lib();


/**
@brief Call to lock the access to the execution of the DFE library
*/
void lock_lib();


/**
@brief Call to unlock the access to the execution of the DFE library
*/
void unlock_lib();





/**
@brief Call to initialize the DFE library support and allocate the requested devices
@param accelerator_num The number of requested devices
@param qbit_num The number of the supported qubits
@param initialize_id_in Identification number of the inititalization of the library
@return Returns with the identification number of the inititalization of the library.
*/
int init_dfe_lib( const int accelerator_num, int qbit_num, int initialize_id_in);





#endif

