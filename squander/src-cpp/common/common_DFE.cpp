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
/*! \file common_DFE.cpp
    \brief Provides functions to link and manage data-flow accelerator libarries
*/


#include "common_DFE.h"
#include "matrix_base.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <tbb/queuing_rw_mutex.h>

namespace {

tbb::queuing_rw_mutex libmutex;

void unload_dfe_lib_unlocked();
int init_dfe_lib_unlocked(const int accelerator_num, int qbit_num, int initialize_id_in);
void uploadMatrix2DFE_unlocked(Matrix& input);

}


// pointer to the dynamically loaded DFE library
void* handle = NULL;

extern "C" {

size_t (*get_accelerator_avail_num_dll)() = NULL;
size_t (*get_accelerator_free_num_dll)() = NULL;
int (*calcqgdKernelDFE_dll)(size_t rows, size_t cols, DFEgate_kernel_type* gates, int gatesNum, int gateSetNum, int traceOffset, double* trace) = NULL;
int (*load2LMEM_dll)(QGD_Complex16* data, size_t rows, size_t cols) = NULL;
void (*releive_DFE_dll)() = NULL;
int (*initialize_DFE_dll)( int accelerator_num ) = NULL;
int (*get_chained_gates_num_dll)() = NULL;

}

// The ID of the class that has initialized the accelerator lib (used to not initialze again if not necessary)
int initialize_id = -1;


DFE_Lib_Read_Lock::DFE_Lib_Read_Lock() {
    lock.acquire(libmutex, false);
}


DFE_Lib_Read_Lock::~DFE_Lib_Read_Lock() {
    lock.release();
}



/**
@brief Call to upload the input matrix to the DFE engine
@param input The input matrix
*/
void uploadMatrix2DFE( Matrix& input ) {
    tbb::queuing_rw_mutex::scoped_lock lock(libmutex, true);
    uploadMatrix2DFE_unlocked(input);
}


namespace {

void uploadMatrix2DFE_unlocked( Matrix& input ) {

    std::cout << "size in bytes of uploading to DFE: " << input.size()*2*sizeof(float) << std::endl;    

    // load the data to LMEM
    if (load2LMEM_dll( input.get_data(), input.rows, input.cols )) initialize_id = -1;

}


void unload_dfe_lib_unlocked()
{
    if (handle) {
        releive_DFE_dll();
        dlclose(handle);
        handle = NULL;
    }
}


}


/**
@brief Call to unload the DFE libarary and release the allocated devices
*/
void unload_dfe_lib()
{
    tbb::queuing_rw_mutex::scoped_lock lock(libmutex, true);
    unload_dfe_lib_unlocked();
}







/**
@brief Call to initialize the DFE library support and allocate the requested devices
@param accelerator_num The number of requested devices
@param qbit_num The number of the supported qubits
@param initialize_id_in Identification number of the inititalization of the library
@return Returns with the identification number of the inititalization of the library.
*/
int init_dfe_lib( const int accelerator_num, int qbit_num, int initialize_id_in )  {
    tbb::queuing_rw_mutex::scoped_lock lock(libmutex, true);
    return init_dfe_lib_unlocked(accelerator_num, qbit_num, initialize_id_in);
}


namespace {

int init_dfe_lib_unlocked( const int accelerator_num, int qbit_num, int initialize_id_in )  {

    initialize_id = initialize_id_in;

    
    unload_dfe_lib_unlocked();


    std::string lib_name_DFE = qbit_num > 9 ? DFE_LIB_10QUBITS : DFE_LIB_9QUBITS;
    std::string lib_name     = getenv("SLIC_CONF") ? DFE_LIB_SIM : lib_name_DFE;

    // dynamic-loading the correct DFE permanent calculator (Simulator/DFE/single or dual) from shared libararies
    handle = dlopen(lib_name.c_str(), RTLD_NOW); //"MAXELEROSDIR"
    if (handle == NULL && qbit_num == 10 && !getenv("SLIC_CONF")) {
        handle = dlopen(DFE_LIB_9QUBITS, RTLD_NOW);
    }
    if (handle == NULL) {
        initialize_id = -1;
        std::string err("init_dfe_lib: failed to load library " + lib_name + " - " + std::string(dlerror()));
        throw err;
    } 
    else {

        get_accelerator_avail_num_dll = (size_t (*)())dlsym(handle, "get_accelerator_avail_num");
        get_accelerator_free_num_dll  = (size_t (*)())dlsym(handle, "get_accelerator_free_num");
        calcqgdKernelDFE_dll          = (int (*)(size_t, size_t, DFEgate_kernel_type*, int, int, int, double*))dlsym(handle, "calcqgdKernelDFE");
        load2LMEM_dll                 = (int (*)(QGD_Complex16*, size_t, size_t))dlsym(handle, "load2LMEM");
        releive_DFE_dll               = (void (*)())dlsym(handle, "releive_DFE");
        initialize_DFE_dll            = (int (*)(int))dlsym(handle, "initialize_DFE");
        get_chained_gates_num_dll     = (int (*)())dlsym(handle, "get_chained_gates_num");

        if (initialize_DFE_dll(accelerator_num)) initialize_id = -1;

    }
    return initialize_id;

}

}


void init_dfe_lib_and_upload(const int accelerator_num, int qbit_num, int initialize_id_in, Matrix& input) {
    tbb::queuing_rw_mutex::scoped_lock lock(libmutex, true);
    if (initialize_id != initialize_id_in) {
        init_dfe_lib_unlocked(accelerator_num, qbit_num, initialize_id_in);
    }
    uploadMatrix2DFE_unlocked(input);
}






/**
@brief Call to get the available number of accelerators
@return Retirns with the number of the available accelerators
*/
size_t get_accelerator_avail_num() {

    return get_accelerator_avail_num_dll();

}


/**
@brief Call to get the number of free accelerators
@return Retirns with the number of the free accelerators
*/
size_t get_accelerator_free_num() {

    return get_accelerator_free_num_dll();

}

/**
@brief Call to get the identification number of the inititalization of the library
@return Returns with the identification number of the inititalization of the library
*/
int get_initialize_id() {

    return initialize_id;

}

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
int calcqgdKernelDFE(size_t rows, size_t cols, DFEgate_kernel_type* gates, int gatesNum, int gateSetNum, int traceOffset, double* trace) {


    return calcqgdKernelDFE_dll(rows, cols, gates, gatesNum, gateSetNum, traceOffset, trace);

}



/**
@brief Call to retrieve the number of gates that should be chained up during the execution of the DFE library
@return Returns with the number of the chained gates.
*/
int get_chained_gates_num() {

    return get_chained_gates_num_dll();

}

