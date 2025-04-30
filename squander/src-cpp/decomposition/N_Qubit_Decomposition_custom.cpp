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
/*! \file N_Qubit_Decomposition.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"



/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_custom::N_Qubit_Decomposition_custom() : Optimization_Interface() {

    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        set_optimizer( BFGS );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
    }
    else {
        set_optimizer( ADAM );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;
    }

}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_custom::N_Qubit_Decomposition_custom( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, std::map<std::string, Config_Element>& config, guess_type initial_guess_in= CLOSE_TO_ZERO, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, optimize_layer_num_in, config, initial_guess_in, accelerator_num) {


    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        set_optimizer( BFGS );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
    }
    else {
        set_optimizer( ADAM );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;
    }

}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_custom::~N_Qubit_Decomposition_custom() {

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
*/
void
N_Qubit_Decomposition_custom::start_decomposition() {


   //The stringstream input to store the output messages.
   std::stringstream sstream;
   sstream << "***************************************************************" << std::endl;
   sstream << "Starting to disentangle " << qbit_num << "-qubit matrix via custom gate structure" << std::endl;
   sstream << "***************************************************************" << std::endl << std::endl << std::endl;
   print(sstream, 1);	    	
   
    

    // temporarily turn off OpenMP parallelism
#if BLAS==0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    //measure the time for the decompositin
    tbb::tick_count start_time = tbb::tick_count::now();




    if (optimized_parameters_mtx.size() > 0 ) {
        sstream.str("");
        sstream << "cost function of the imported circuit: " << optimization_problem( optimized_parameters_mtx ) << std::endl;
        print(sstream, 1);
    }   


    // final tuning of the decomposition parameters
    final_optimization();

    // calculating the final error of the decomposition
    Matrix matrix_decomposed = Umtx.copy();
    apply_to(optimized_parameters_mtx, matrix_decomposed );
    calc_decomposition_error( matrix_decomposed );


    sstream.str("");
    sstream << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " layers with:" << std::endl;

      
        // get the number of gates used in the decomposition
    std::map<std::string, int>&& gate_nums = get_gate_nums();
    	
    for( auto it=gate_nums.begin(); it != gate_nums.end(); it++ ) {
        sstream << it->second << " " << it->first << " gates" << std::endl;
    } 

    	 
    sstream << std::endl;
    tbb::tick_count current_time = tbb::tick_count::now();

    sstream << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    print(sstream, 1);	    	
    	
       

#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif

}










