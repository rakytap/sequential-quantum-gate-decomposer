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
/*! \file common.cpp
    \brief Provides commonly used functions and wrappers to CBLAS functions.
*/

//
// @brief A base class responsible for constructing matrices of C-NOT, U3
// gates acting on the N-qubit space

#include "common_DFE.h"
#include "matrix_base.hpp"

#include <atomic>
#include <dlfcn.h>
#include <unistd.h>
#include <mutex>


// pointer to the dynamically loaded DFE library
void* handle = NULL;

/// reference counting of locking-unlocking the DFE accelerators
std::atomic_size_t read_count(0); //readers-writer problem semaphore

/// mutex to guard DFE lib locking and unlocking
std::recursive_mutex libmutex; //writing mutex
std::mutex libreadmutex; //reader mutex


extern "C" {
size_t (*get_accelerator_avail_num_dll)() = NULL;
size_t (*get_accelerator_free_num_dll)() = NULL;
int (*calcqgdKernelDFE_dll)(size_t rows, size_t cols, DFEgate_kernel_type* gates, int gatesNum, int gateSetNum, int traceOffset, double* trace) = NULL;
int (*load2LMEM_dll)(QGD_Complex16* data, size_t rows, size_t cols) = NULL;
void (*releive_DFE_dll)() = NULL;
int (*initialize_DFE_dll)( int accelerator_num ) = NULL;
int (*get_chained_gates_num_dll)() = NULL;

size_t (*get_accelerator_avail_num_sv_dll)() = NULL;
size_t (*get_accelerator_free_num_sv_dll)() = NULL;
int (*calcsvKernelGroq_dll)(int num_gates, float* gates, int* target_qubits, int* control_qubits, float* result_real, float* result_imag, int device_num) = NULL;
int (*load_sv_dll)(float* data, size_t num_qubits, size_t device_num) = NULL;
void (*releive_groq_sv_dll)() = NULL;
int (*initialize_groq_sv_dll)( int accelerator_num ) = NULL;
}

// The ID of the class initializing the DFE lib
int initialize_id = -1;



/**
@brief ????????????
@return ??????????
*/
void uploadMatrix2DFE( Matrix& input ) {

    std::cout << "size in bytes of uploading to DFE: " << input.size()*2*sizeof(float) << std::endl;    

    // load the data to LMEM
    if (load2LMEM_dll( input.get_data(), input.rows, input.cols )) initialize_id = -1;

}


/**
@brief ????????????
@return ??????????
*/
void unload_dfe_lib()
{
    const std::lock_guard<std::recursive_mutex> lock(libmutex);
    if (handle) {
        releive_DFE_dll();
        dlclose(handle);
        handle = NULL;
    }
}







/**
@brief ????????????
@return ??????????
*/
int init_dfe_lib( const int accelerator_num, int qbit_num, int initialize_id_in )  {

    const std::lock_guard<std::recursive_mutex> lock(libmutex);

    initialize_id = initialize_id_in;

    
    unload_dfe_lib();


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


/**
@brief ????????????
@return ??????????
*/
void lock_lib()
{
    const std::lock_guard<std::mutex> lock(libreadmutex);
    if (++read_count == 1) libmutex.lock();
}


/**
@brief ????????????
@return ??????????
*/
void unlock_lib()
{
    const std::lock_guard<std::mutex> lock(libreadmutex);
    if (--read_count == 0) libmutex.unlock();
}





/**
@brief ????????????
@return ??????????
*/
size_t get_accelerator_avail_num() {

    return get_accelerator_avail_num_dll();

}


/**
@brief ????????????
@return ??????????
*/
size_t get_accelerator_free_num() {

    return get_accelerator_free_num_dll();

}

/**
@brief ????????????
@return ??????????
*/
int get_initialize_id() {

    return initialize_id;

}

/**
@brief ????????????
@return ??????????
*/
int calcqgdKernelDFE(size_t rows, size_t cols, DFEgate_kernel_type* gates, int gatesNum, int gateSetNum, int traceOffset, double* trace) {


    return calcqgdKernelDFE_dll(rows, cols, gates, gatesNum, gateSetNum, traceOffset, trace);

}



/**
@brief ????????????
@return ??????????
*/
int get_chained_gates_num() {

    return get_chained_gates_num_dll();

}


// pointer to the dynamically loaded groq library
void* handle_sv = NULL;

void unload_groq_sv_lib()
{
    if (handle_sv) {
        releive_groq_sv_dll();
        dlclose(handle_sv);
        handle_sv = NULL;
    }
}

int init_groq_sv_lib( const int accelerator_num )  {  
    
    unload_groq_sv_lib();


    std::string lib_name     = DFE_LIB_SV;

    // dynamic-loading the Groq calculator from shared library
    handle_sv = dlopen(lib_name.c_str(), RTLD_NOW); //"MAXELEROSDIR"
    if (handle_sv == NULL) {
        std::string err("init_groq_lib: failed to load library " + lib_name + " - " + std::string(dlerror()));
        throw err;
    } 
    else {
        get_accelerator_avail_num_sv_dll = (size_t (*)())dlsym(handle_sv, "get_accelerator_avail_num_sv");
        get_accelerator_free_num_sv_dll  = (size_t (*)())dlsym(handle_sv, "get_accelerator_free_num_sv");
        calcsvKernelGroq_dll          = (int (*)(int, float*, int*, int*, float*, float*, int))dlsym(handle_sv, "calcsvKernelGroq");
        load_sv_dll                 = (int (*)(float*, size_t, size_t))dlsym(handle_sv, "prepare_state_vector");
        releive_groq_sv_dll               = (void (*)())dlsym(handle_sv, "releive_groq_sv");
        initialize_groq_sv_dll            = (int (*)(int))dlsym(handle_sv, "initialize_groq_sv");

        if (initialize_groq_sv_dll(accelerator_num)) return 0;

    }
    return 1;

}



void apply_to_groq_sv(int device_num, int qbit_num, std::vector<Matrix>& u3_qbit, Matrix& State, std::vector<int>& target_qbit, std::vector<int>& control_qbit) {

    //struct timespec starttime;
    //timespec_get(&starttime, TIME_UTC);

    size_t matrix_size = 1 << qbit_num;

    // the number of chips to be allocated for the calculations
    int alloc_dfes = 1;
    if (handle_sv == NULL && !init_groq_sv_lib(alloc_dfes)) {
        throw std::string("Could not load and initialize DFE library");
    }

    //struct timespec t;
    //timespec_get(&t, TIME_UTC);
    //printf("Total time on uploading the Groq program: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);


    if ( State.size() == 0 ) {
        if (load_sv_dll( NULL, qbit_num, device_num) ) {
            throw std::string("Error occured while reseting the state vector to Groq LPU");
        }

        State = Matrix( matrix_size, 1);

    }
    else {

	if ( State.size() != matrix_size ) {
            throw std::string("apply_to_groq_sv: the size of the input vector should be in match with the number of qubits");
        }

	if ( State.cols != 1 ) {
            throw std::string("apply_to_groq_sv: the input state should have a single column");
        }

        //timespec_get(&starttime, TIME_UTC);
        std::vector<float> inout;
        inout.reserve(State.size()*2);
        for (size_t idx = 0; idx < State.rows; idx++) {
            inout.push_back(State.data[idx].real);
            inout.push_back(State.data[idx].imag);
        }

        //timespec_get(&t, TIME_UTC);
        //printf("Total time on converting State data to float: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);


        //timespec_get(&starttime, TIME_UTC);
        if (load_sv_dll(inout.data(), qbit_num, device_num)) {
            throw std::string("Error occured while uploading state vector to Groq LPU");
        }

        //timespec_get(&t, TIME_UTC);
        //printf("Total time on uploading the state vector: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);

    }


    std::vector<float> gateMatrices;
    gateMatrices.reserve( 4*u3_qbit.size()*2 );
    for (const Matrix& m : u3_qbit) {
        for (size_t i = 0; i < m.rows; i++) {
            for (size_t j = 0; j < m.cols; j++) {

                gateMatrices.push_back( m.data[i*m.stride+j].real );
                gateMatrices.push_back( m.data[i*m.stride+j].imag );

            }
        }
    }

    //timespec_get(&starttime, TIME_UTC);
    // evaluate the gate kernels on the Groq chip
    matrix_base<float> transformed_sv_real( matrix_size, 1);
    matrix_base<float> transformed_sv_imag( matrix_size, 1);

    if (calcsvKernelGroq_dll(u3_qbit.size(), gateMatrices.data(), target_qbit.data(), control_qbit.data(), transformed_sv_real.get_data(), transformed_sv_imag.get_data(), device_num)) {
        throw std::string("Error running gate kernels on groq");
    }

    //timespec_get(&t, TIME_UTC);
    //printf("Total time on calcsvKernelGroq: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);


    //timespec_get(&starttime, TIME_UTC);  
    // transform the state vector to double representation  
    for (size_t idx = 0; idx < matrix_size; idx++) {
        State.data[idx].real = transformed_sv_real[idx];
        State.data[idx].imag = transformed_sv_imag[idx];        
    }
    //timespec_get(&t, TIME_UTC);
    //printf("Total time on transforming state vector to double: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);



}

