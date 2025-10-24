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
/*! \file common_GROQ.cpp
    \brief Provides functions to link and manage data-flow accelerator libraries for state vector simulation
*/


#include "common_GROQ.h"
#include "matrix_base.hpp"

#include <atomic>
#include <dlfcn.h>
#include <unistd.h>
#include <mutex>


// pointer to the dynamically loaded groq library (sv stands for state vector)
void* handle_sv = NULL;


/// mutex to guard DFE lib locking and unlocking
std::recursive_mutex libmutex; //writing mutex
std::mutex libreadmutex; //reader mutex


extern "C" {

size_t (*get_accelerator_avail_num_sv_dll)() = NULL;
size_t (*get_accelerator_free_num_sv_dll)() = NULL;
int (*calcsvKernelGroq_dll)(int num_gates, float* gates, int* target_qubits, int* control_qubits, float* result_real, float* result_imag, int device_num) = NULL;
int (*load_sv_dll)(float* data, size_t num_qubits, size_t device_num) = NULL;
void (*releive_groq_sv_dll)() = NULL;
int (*initialize_groq_sv_dll)( int accelerator_num ) = NULL;

}

// The ID of the class that has initialized the accelerator lib (used to not initialze again if not necessary)
int initialize_id = -1;



/**
@brief Call to get the identification number of the inititalization of the library
@return Returns with the identification number of the inititalization of the library
*/
int get_initialize_id() {

    return initialize_id;

}






/**
@brief Call to unload the programs from the reserved Groq cards
*/
void unload_groq_sv_lib()
{
    const std::lock_guard<std::recursive_mutex> lock(libmutex);
    
    if (handle_sv) {
        releive_groq_sv_dll();
        dlclose(handle_sv);
        handle_sv = NULL;
    }
}


/**
@brief Call to allocated Groq cards for calculations
@param reserved_device_num The number of Groq accelerator cards to be allocated for the calulations
@param initialize_id_in Identification number of the inititalization of the library
@return Returns with 1 on success
*/
int init_groq_sv_lib( const int reserved_device_num, int initialize_id_in )  {  

    const std::lock_guard<std::recursive_mutex> lock(libmutex);

    initialize_id = initialize_id_in;
        
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

        if (initialize_groq_sv_dll(reserved_device_num)) return 0;

    }
    return 1;

}


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
void apply_to_groq_sv(int reserved_device_num, int chosen_device_num, int qbit_num, std::vector<Matrix>& u3_qbit, std::vector<int>& target_qbits, std::vector<int>& control_qbits, Matrix& quantum_state, int id_in) {

    //struct timespec starttime;
    //timespec_get(&starttime, TIME_UTC);

    size_t matrix_size = 1 << qbit_num;

    // the number of chips to be allocated for the calculations
    if (handle_sv == NULL && !init_groq_sv_lib(reserved_device_num, id_in)) {
        throw std::string("Could not load and initialize Groq library");
    }

    //struct timespec t;
    //timespec_get(&t, TIME_UTC);
    //printf("Total time on uploading the Groq program: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);


{ 
    std::lock_guard<std::recursive_mutex> lock(libmutex);
    
    if ( quantum_state.size() == 0 ) {
        if (load_sv_dll( NULL, qbit_num, chosen_device_num) ) {
            throw std::string("Error occured while reseting the state vector to Groq LPU");
        }

        quantum_state = Matrix( matrix_size, 1);

    }
    else {

	    if ( quantum_state.size() != matrix_size ) {
            throw std::string("apply_to_groq_sv: the size of the input vector should be in match with the number of qubits");
        }

	    if ( quantum_state.cols != 1 ) {
            throw std::string("apply_to_groq_sv: the input state should have a single column");
        }

        //timespec_get(&starttime, TIME_UTC);
        std::vector<float> inout;
        inout.reserve(quantum_state.size()*2);
        for (size_t idx = 0; idx < quantum_state.rows; idx++) {
            inout.push_back(quantum_state.data[idx].real);
            inout.push_back(quantum_state.data[idx].imag);
        }

        //timespec_get(&t, TIME_UTC);
        //printf("Total time on converting State data to float: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);


        //timespec_get(&starttime, TIME_UTC);
        if (load_sv_dll(inout.data(), qbit_num, chosen_device_num)) {
            throw std::string("Error occured while uploading state vector to Groq LPU");
        }

        //timespec_get(&t, TIME_UTC);
        //printf("Total time on uploading the state vector: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);

    }
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

    if (calcsvKernelGroq_dll(u3_qbit.size(), gateMatrices.data(), target_qbits.data(), control_qbits.data(), transformed_sv_real.get_data(), transformed_sv_imag.get_data(), chosen_device_num)) {
        throw std::string("Error running gate kernels on groq");
    }

    //timespec_get(&t, TIME_UTC);
    //printf("Total time on calcsvKernelGroq: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);


    //timespec_get(&starttime, TIME_UTC);  
    // transform the state vector to double representation  
    for (size_t idx = 0; idx < matrix_size; idx++) {
        quantum_state.data[idx].real = transformed_sv_real[idx];
        quantum_state.data[idx].imag = transformed_sv_imag[idx];        
    }
    //timespec_get(&t, TIME_UTC);
    //printf("Total time on transforming state vector to double: %.9f\n", (t.tv_sec - starttime.tv_sec) + (t.tv_nsec - starttime.tv_nsec) / 1e9);



}

