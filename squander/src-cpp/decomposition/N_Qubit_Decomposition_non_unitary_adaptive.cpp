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
/*! \file N_Qubit_Decomposition_non_unitary_adaptive.cpp
    \brief Class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#include "N_Qubit_Decomposition_non_unitary_adaptive.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Random_Orthogonal.h"
#include "Random_Unitary.h"
#include "n_aryGrayCodeCounter.h"

#include "X.h"

#include <time.h>
#include <stdlib.h>

#include <iostream>

#ifdef __DFE__
#include "common_DFE.h"
#endif

using namespace std;



/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive::N_Qubit_Decomposition_non_unitary_adaptive() : Optimization_Interface() {


    // set the level limit
    level_limit = 0;



    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        set_optimizer( BFGS );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
        max_inner_iterations = 10000;
    }
    else {
        set_optimizer( ADAM );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;
    }
    

    // Boolean variable to determine whether randomized adaptive layers are used or not
    randomized_adaptive_layers = false;


}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled_in Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive::N_Qubit_Decomposition_non_unitary_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {


    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        set_optimizer( BFGS );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
        
        max_inner_iterations = 10000;

    }
    else {
        set_optimizer( ADAM );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;

    }

    if( topology.size() == 0 ) {
        for( int qbit1=0; qbit1<qbit_num; qbit1++ ) {
            for( int qbit2=qbit1; qbit2<qbit_num; qbit2++ ) {
                matrix_base<int> edge(2,1);
                edge[0] = qbit1;
                edge[1] = qbit2;

                topology.push_back( edge );
            }
        }
    }


    // Boolean variable to determine whether randomized adaptive layers are used or not
    randomized_adaptive_layers = false;

current_minimum = 30;

}



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled_in Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive::N_Qubit_Decomposition_non_unitary_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {



    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // Maximal number of iteartions in the optimization process
    max_outer_iterations = 1;

    
    // setting the topology
    topology = topology_in;

    if( topology.size() == 0 ) {
        for( int qbit1=0; qbit1<qbit_num; qbit1++ ) {
            for( int qbit2=qbit1+1; qbit2<qbit_num; qbit2++ ) {
                matrix_base<int> edge(2,1);
                edge[0] = qbit1;
                edge[1] = qbit2;

                topology.push_back( edge );
            }
        }
    }




    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        alg = BFGS;

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
        max_inner_iterations = 10000;
    }
    else {
        alg = ADAM;

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;
    }

    // Boolean variable to determine whether randomized adaptive layers are used or not
    randomized_adaptive_layers = false;

}

/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_non_unitary_adaptive::~N_Qubit_Decomposition_non_unitary_adaptive() {

}





/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
*/
void
N_Qubit_Decomposition_non_unitary_adaptive::start_decomposition() {


    //The string stream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;

    print(sstream, 1);   


    // get the initial circuit including redundant 2-qbit blocks.
    get_initial_circuit();
    

    // finalyzing the gate structure by turning CRY gates inti CNOT gates and do optimization cycles to correct approximation in this transformation 
    // (CRY gates with small rotation angles are expressed with a single CNOT gate
    finalize_circuit();


}





/**
@brief ???????????????????
*/
void N_Qubit_Decomposition_non_unitary_adaptive::get_initial_circuit() {
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

    if (level_limit == 0 ) {
        std::stringstream sstream;
	sstream << "please increase level limit" << std::endl;
        print(sstream, 0);	
        return;
    }




    Gates_block* gate_structure_loc = NULL;
    if ( gates.size() > 0 ) {
        std::stringstream sstream;
        sstream << "Using imported gate structure for the decomposition." << std::endl;
        print(sstream, 1);	
        gate_structure_loc = optimize_imported_gate_structure(optimized_parameters_mtx);
    }
    else {
        std::stringstream sstream;
        sstream << "Construct initial gate structure for the decomposition." << std::endl;
        print(sstream, 1);
        gate_structure_loc = determine_initial_gate_structure(optimized_parameters_mtx);
    }





    long long export_circuit_2_binary_loc;
    if ( config.count("export_circuit_2_binary") > 0 ) {
        config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
    }
    else {
        export_circuit_2_binary_loc = 0;
    }     
        
        
    if ( export_circuit_2_binary_loc > 0 ) {
        std::string filename("circuit_squander.binary");
        if (project_name != "") {
            filename = project_name+ "_" +filename;
        }
        export_gate_list_to_binary(optimized_parameters_mtx, gate_structure_loc, filename, verbose);
        
        std::string unitaryname("unitary_squander.binary");
        if (project_name != "") {
            filename = project_name+ "_" +unitaryname;
        }
        export_unitary(unitaryname);
        
    }
    
    // store the created gate structure
    release_gates();
	combine( gate_structure_loc );
	delete( gate_structure_loc );
	
	
#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif
}




/**
@brief ???????????????????
*/
void N_Qubit_Decomposition_non_unitary_adaptive::finalize_circuit() {


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


	Gates_block* gate_structure_loc = NULL;
    if ( gates.size() > 0 ) {
        std::stringstream sstream;
        sstream << "Using imported gate structure for the compression." << std::endl;
        print(sstream, 1);	
	        
        gate_structure_loc =  static_cast<Gates_block*>(this)->clone();
    }
    else {
        std::stringstream sstream;
        sstream << "No circuit initialized." << std::endl;
        print(sstream, 1);
        return;
    }
    
    
    std::stringstream sstream;
    sstream.str("");
    sstream << "**************************************************************" << std::endl;
    sstream << "************ Final tuning of the Gate structure **************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    print(sstream, 1);	    	
    
	// maximal number of inner iterations overwritten by config
    if ( config.count("optimization_tolerance") > 0 ) {
        long long value;
        config["optimization_tolerance"].get_property( value );
        optimization_tolerance = (double)value; 
    }
    else {optimization_tolerance = 1e-4;}



    optimization_block = get_gate_num();


    sstream.str("");
    sstream << "cost function value before replacing trivial CZ_UN gates: " << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    print(sstream, 3);	

    Gates_block* gate_structure_tmp = replace_trivial_CZ_NU_gates( gate_structure_loc, optimized_parameters_mtx );
    Matrix_real optimized_parameters_save = optimized_parameters_mtx;

    release_gates();
    combine( gate_structure_tmp );

    sstream.str("");
    sstream << "cost function value before final optimization: " << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    print(sstream, 3);	

    release_gates();
    optimized_parameters_mtx = optimized_parameters_save;

    // solve the optimization problem
    N_Qubit_Decomposition_custom cDecomp_custom;


    std::map<std::string, Config_Element> config_copy;
    config_copy.insert(config.begin(), config.end());
    if ( config.count("max_inner_iterations_final") > 0 ) {
        long long val;
        config["max_inner_iterations_final"].get_property( val ); 
        Config_Element element;
        element.set_property( "max_inner_iterations", val ); 
        config_copy["max_inner_iterations"] = element;
    }


    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config_copy, initial_guess, accelerator_num);
    cDecomp_custom.set_custom_gate_structure( gate_structure_tmp );
    cDecomp_custom.set_optimized_parameters( optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size() );
    cDecomp_custom.set_optimization_blocks( gate_structure_loc->get_gate_num() );
    cDecomp_custom.set_max_iteration( max_outer_iterations );
    cDecomp_custom.set_verbose(verbose);
    cDecomp_custom.set_cost_function_variant( cost_fnc );
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance ); 
    cDecomp_custom.set_trace_offset( trace_offset ); 
    cDecomp_custom.set_optimizer( alg );  
    if (alg==ADAM || alg==BFGS2) { 
        int param_num_loc = gate_structure_loc->get_parameter_num();
        int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
        cDecomp_custom.set_random_shift_count_max( 10000 );         
    }
    else if ( alg==ADAM_BATCHED ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 2500;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
        cDecomp_custom.set_random_shift_count_max( 5 );   
    }
    else if ( alg==BFGS ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 10000;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
    }
    cDecomp_custom.start_decomposition();
    number_of_iters += cDecomp_custom.get_num_iters();

    current_minimum = cDecomp_custom.get_current_minimum();
    optimized_parameters_mtx = cDecomp_custom.get_optimized_parameters();


    combine( gate_structure_tmp );
    delete( gate_structure_tmp );
    delete( gate_structure_loc );

    sstream.str("");
    sstream << "cost function value after final optimization: " << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    print(sstream, 3);	
   


    long long export_circuit_2_binary_loc;
    if ( config.count("export_circuit_2_binary") > 0 ) {
        config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
    }
    else {
        export_circuit_2_binary_loc = 0;
    }       
    	
    	
    if ( export_circuit_2_binary_loc > 0 ) {
        std::string filename2("circuit_final.binary");

        if (project_name != "") {
            filename2=project_name+ "_"  +filename2;
        }

        export_gate_list_to_binary(optimized_parameters_mtx, this, filename2, verbose);  
    
    }

    decomposition_error = optimization_problem(optimized_parameters_mtx);
    
    // get the number of gates used in the decomposition
    gates_num gates_num = get_gate_nums();

    
    sstream.str("");
    sstream << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
      
        if ( gates_num.u3>0 ) sstream << gates_num.u3 << " U3 gates," << std::endl;
        if ( gates_num.rx>0 ) sstream << gates_num.rx << " RX gates," << std::endl;
        if ( gates_num.ry>0 ) sstream << gates_num.ry << " RY gates," << std::endl;
        if ( gates_num.rz>0 ) sstream << gates_num.rz << " RZ gates," << std::endl;
        if ( gates_num.cnot>0 ) sstream << gates_num.cnot << " CNOT gates," << std::endl;
        if ( gates_num.cz>0 ) sstream << gates_num.cz << " CZ gates," << std::endl;
        if ( gates_num.ch>0 ) sstream << gates_num.ch << " CH gates," << std::endl;
        if ( gates_num.x>0 ) sstream << gates_num.x << " X gates," << std::endl;
        if ( gates_num.sx>0 ) sstream << gates_num.sx << " SX gates," << std::endl; 
        if ( gates_num.syc>0 ) sstream << gates_num.syc << " Sycamore gates," << std::endl;   
        if ( gates_num.un>0 ) sstream << gates_num.un << " UN gates," << std::endl;
        if ( gates_num.cry>0 ) sstream << gates_num.cry << " CRY gates," << std::endl;  
        if ( gates_num.adap>0 ) sstream << gates_num.adap << " Adaptive gates," << std::endl;
        if ( gates_num.cz_nu>0 ) sstream << gates_num.cz_nu << " CZ_NU gates," << std::endl;
    
        sstream << std::endl;
    	print(sstream, 1);	    	
    	
#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif

}

/**
@brief ?????????????????????????
*/
void
N_Qubit_Decomposition_non_unitary_adaptive::add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit) {
	
        if ( control_qbit >= qbit_num || target_qbit>= qbit_num ) {
            std::string error( "N_Qubit_Decomposition_non_unitary_adaptive::add_two_qubit_block: Label of control/target qubit should be less than the number of qubits in the register.");	        
            throw error;         
        }
        
        if ( control_qbit == target_qbit ) {
            std::string error( "N_Qubit_Decomposition_non_unitary_adaptive::add_two_qubit_block: Target and control qubits should be different");	        
            throw error;         
        }        

        Gates_block* layer = new Gates_block( qbit_num );

        bool Theta = true;
        bool Phi = true;
        bool Lambda = false;
/*
layer->add_rz(target_qbit);
layer->add_ry(target_qbit);
layer->add_rz(target_qbit);     

layer->add_rz(control_qbit);
layer->add_ry(control_qbit);
layer->add_rz(control_qbit);     
*/
        layer->add_u3(target_qbit, Theta, Phi, Lambda);
        layer->add_u3(control_qbit, Theta, Phi, Lambda); 
        layer->add_cnot(target_qbit, control_qbit); // véletszerüen vagy cz_nu-t vagy cnot, vagy egy teljes layer

        gate_structure->add_gate(layer);

}




/**
@brief 
@param 
@param 
*/
N_Qubit_Decomposition_custom
N_Qubit_Decomposition_non_unitary_adaptive::perform_optimization(Gates_block* gate_structure_loc){


    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }     

         
    N_Qubit_Decomposition_custom cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    cDecomp_custom_random.set_custom_gate_structure( gate_structure_loc );
    cDecomp_custom_random.set_optimization_blocks( gate_structure_loc->get_gate_num() );
    cDecomp_custom_random.set_max_iteration( max_outer_iterations );
#ifndef __DFE__
    cDecomp_custom_random.set_verbose(verbose);
#else
    cDecomp_custom_random.set_verbose(0);
#endif
    cDecomp_custom_random.set_cost_function_variant( cost_fnc );
    cDecomp_custom_random.set_debugfile("");
    cDecomp_custom_random.set_optimization_tolerance( optimization_tolerance_loc );
    cDecomp_custom_random.set_trace_offset( trace_offset ); 
    cDecomp_custom_random.set_optimizer( alg );
    cDecomp_custom_random.set_project_name( project_name );
    if ( alg == ADAM || alg == BFGS2 ) {
         int param_num_loc = gate_structure_loc->get_parameter_num();
         int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
         cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
         cDecomp_custom_random.set_random_shift_count_max( 10000 ); 
    }
    else if ( alg==ADAM_BATCHED ) {
         cDecomp_custom_random.set_optimizer( alg );  
         int max_inner_iterations_loc = 2000;
         cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
         cDecomp_custom_random.set_random_shift_count_max( 5 );   
    }
    else if ( alg==BFGS ) {
         cDecomp_custom_random.set_optimizer( alg );  
         int max_inner_iterations_loc = 10000;
         cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
    }
                
            
    cDecomp_custom_random.start_decomposition();
    return cDecomp_custom_random;
}
                

/**
@brief 
@param 
@param 
*/
Gates_block* 
N_Qubit_Decomposition_non_unitary_adaptive::tree_search_over_gate_structures( int level_max ){

    tbb::spin_mutex tree_search_mutex;
    
   
    
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }     

    // construct the possible CNOT combinations within a single level
    // the number of possible CNOT connections netween the qubits (including topology constraints)
    int n_ary_limit_max = topology.size();
    
    matrix_base<int> possible_target_qbits(1, n_ary_limit_max);
    matrix_base<int> possible_control_qbits(1, n_ary_limit_max);    
    for( int element_idx = 0; element_idx<n_ary_limit_max; element_idx++ ) {

       matrix_base<int>& edge = topology[ element_idx ];
       possible_target_qbits[element_idx] = edge[0];
       possible_control_qbits[element_idx] = edge[1]; 
 
    }   
    
    
    Gates_block* gate_structure_best_solution = NULL;    
    bool found_optimal_solution = false;
    
    
     if (level_max == 0){


        Gates_block* gate_structure_loc = new Gates_block(qbit_num);

        add_finalyzing_layer( gate_structure_loc );
       
        
       
       
        tbb::tick_count start_time_loc = tbb::tick_count::now();


        N_Qubit_Decomposition_custom cDecomp_custom_random;

        std::stringstream sstream;
        sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
        print(sstream, 1);

	// solve the optimization problem in isolated optimization process
	cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
	cDecomp_custom_random.set_custom_gate_structure( gate_structure_loc );
	cDecomp_custom_random.set_optimization_blocks( gate_structure_loc->get_gate_num() );
	cDecomp_custom_random.set_max_iteration( max_outer_iterations );
	#ifndef __DFE__
	cDecomp_custom_random.set_verbose(verbose);
	#else
	cDecomp_custom_random.set_verbose(0);
	#endif
	cDecomp_custom_random.set_cost_function_variant( cost_fnc );
	cDecomp_custom_random.set_debugfile("");
	cDecomp_custom_random.set_optimization_tolerance( optimization_tolerance_loc );
	cDecomp_custom_random.set_trace_offset( trace_offset ); 
	cDecomp_custom_random.set_optimizer( alg );
	cDecomp_custom_random.set_project_name( project_name );
	if ( alg == ADAM || alg == BFGS2 ) {
	    int param_num_loc = gate_structure_loc->get_parameter_num();
	    int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
	    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
	    cDecomp_custom_random.set_random_shift_count_max( 10000 ); 
	}
	else if ( alg==ADAM_BATCHED ) {
	    cDecomp_custom_random.set_optimizer( alg );  
	    int max_inner_iterations_loc = 2000;
	    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
	    cDecomp_custom_random.set_random_shift_count_max( 5 );   
	}
	else if ( alg==BFGS ) {
	    cDecomp_custom_random.set_optimizer( alg );  
	    int max_inner_iterations_loc = 10000;
	    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
	}
		
	    
	cDecomp_custom_random.start_decomposition();


	 number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization           


	double current_minimum_tmp         = cDecomp_custom_random.get_current_minimum();
	sstream.str("");
	sstream << "Optimization with " << level_max << " levels converged to " << current_minimum_tmp;
	print(sstream, 1);
        
        
           
        
        if( current_minimum_tmp < current_minimum && !found_optimal_solution) {

            current_minimum = current_minimum_tmp;
            if ( gate_structure_best_solution != NULL ) {
                delete( gate_structure_best_solution );
            }
            
            gate_structure_best_solution = gate_structure_loc;
            gate_structure_loc = NULL;

            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
        }
        else 
        {
            delete( gate_structure_loc );
            gate_structure_loc = NULL;
        }

     
        if ( current_minimum < optimization_tolerance_loc && !found_optimal_solution)  { 
            found_optimal_solution = true;
        } 
    
        return gate_structure_best_solution;
   
    }
  
    // set the limits for the N-ary Gray counter
    

    matrix_base<int> n_ary_limits( 1, level_max ); //array containing the limits of the individual Gray code elements    
    memset( n_ary_limits.get_data(), n_ary_limit_max, n_ary_limits.size()*sizeof(int) );
    
    for( int idx=0; idx<n_ary_limits.size(); idx++) {
        n_ary_limits[idx] = n_ary_limit_max;
    }
        

    
    
    //n_aryGrayCodeCounter gcode_counter( n_ary_limits );



    int64_t iteration_max = pow( (int64_t)n_ary_limit_max, level_max );
    
    
    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads;
    concurrency = concurrency < iteration_max ? concurrency : iteration_max;  


    int parallel = get_parallel_configuration();
       
    int64_t work_batch = 1;
    if( parallel==0) {
        work_batch = concurrency;
    }


    tbb::parallel_for( tbb::blocked_range<int64_t>((int64_t)0, concurrency, work_batch), [&](tbb::blocked_range<int64_t> r) {
        for (int64_t job_idx=r.begin(); job_idx<r.end(); ++job_idx) { 
        
    //for( int64_t job_idx=0; job_idx<concurrency; job_idx++ ) {  
    
            // initial offset and upper boundary of the gray code counter
            int64_t work_batch = iteration_max/concurrency;
            int64_t initial_offset = job_idx*work_batch;
            int64_t offset_max = (job_idx+1)*work_batch-1;
        
            if ( job_idx == concurrency-1) {
                offset_max = iteration_max-1;
            } 

            //std::cout << initial_offset << " " << offset_max << " " << iteration_max << " " << work_batch << " " << concurrency << std::endl;

            n_aryGrayCodeCounter gcode_counter(n_ary_limits, initial_offset);  // see piquassoboost for deatils of the implementation
            gcode_counter.set_offset_max( offset_max );
      
        
            for (int64_t iter_idx=initial_offset; iter_idx<offset_max+1; iter_idx++ ) {       

    
                matrix_base<int> gcode = gcode_counter.get();
        
                // determine the target qubit indices and control qbit indices for the CNOT gates from the Gray code counter
                matrix_base<int> target_qbits(1, level_max);
                matrix_base<int> control_qbits(1, level_max);

        
                for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {
        
                    // gcode[idx] = target_qbit[idx] * n_ary_limit_max + control_qbit[idx], where control_qbit > target_qbit
            
                    int target_qbit = possible_target_qbits[ gcode[gcode_idx] ];            
                    int control_qbit = possible_control_qbits[ gcode[gcode_idx] ];
            
            
                    target_qbits[gcode_idx] = target_qbit;
                    control_qbits[gcode_idx] = control_qbit;  
            
            
                    //std::cout <<   target_qbit << " " << control_qbit << std::endl;        
                }

        
                //  ----------- contruct the gate structure to be optimized ----------- 
                Gates_block* gate_structure_loc = new Gates_block(qbit_num);  // cnot nélkül megyek mi lesz? 

                            
                for (int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++) {
            
                    // add new 2-qbit block to the circuit
                    add_two_qubit_block( gate_structure_loc, target_qbits[gcode_idx], control_qbits[gcode_idx]  );
                }
               
                // add finalyzing layer to the the gate structure
                add_finalyzing_layer( gate_structure_loc );
              
    

                // ----------- start the decomposition ----------- 
        
                //measure the time for the decompositin
                tbb::tick_count start_time_loc = tbb::tick_count::now();


                N_Qubit_Decomposition_custom cDecomp_custom_random;

                std::stringstream sstream;
                sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
                print(sstream, 1);
        
                // solve the optimization problem in isolated optimization process
                cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
                cDecomp_custom_random.set_custom_gate_structure( gate_structure_loc );
                cDecomp_custom_random.set_optimization_blocks( gate_structure_loc->get_gate_num() );
                cDecomp_custom_random.set_max_iteration( max_outer_iterations );
#ifndef __DFE__
                cDecomp_custom_random.set_verbose(verbose);
#else
                cDecomp_custom_random.set_verbose(0);
#endif
                cDecomp_custom_random.set_cost_function_variant( cost_fnc );
                cDecomp_custom_random.set_debugfile("");
                cDecomp_custom_random.set_optimization_tolerance( optimization_tolerance_loc );    
                cDecomp_custom_random.set_trace_offset( trace_offset ); 
                cDecomp_custom_random.set_optimizer( alg );    
                cDecomp_custom_random.set_project_name( project_name );
                if ( alg == ADAM || alg == BFGS2 ) {
                    int param_num_loc = gate_structure_loc->get_parameter_num();
                    int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_random.set_random_shift_count_max( 10000 ); 
                }
                else if ( alg==ADAM_BATCHED ) {
                    cDecomp_custom_random.set_optimizer( alg );  
                    int max_inner_iterations_loc = 2000;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_random.set_random_shift_count_max( 5 );   
                }
                else if ( alg==BFGS ) {
                    cDecomp_custom_random.set_optimizer( alg );  
                    int max_inner_iterations_loc = 10000;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                }
                
            
                cDecomp_custom_random.start_decomposition();
                

                
                number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization  
    
                double current_minimum_tmp         = cDecomp_custom_random.get_current_minimum();
                sstream.str("");
                sstream << "Optimization with " << level_max << " levels converged to " << current_minimum_tmp;
                print(sstream, 1);

                //std::cout << "Optimization with " << level_max << " levels converged to " << current_minimum_tmp << std::endl;
        
                {
                    tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};

                    if( current_minimum_tmp < current_minimum && !found_optimal_solution) {
                        current_minimum = current_minimum_tmp;

                        if ( gate_structure_best_solution != NULL ) {
                            delete( gate_structure_best_solution );
                        }
            
                        gate_structure_best_solution = gate_structure_loc;
                        gate_structure_loc = NULL;

                        optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                    }
                    else {
                        delete( gate_structure_loc );
                        gate_structure_loc = NULL;
                    }

     
                    if ( current_minimum < optimization_tolerance_loc && !found_optimal_solution)  {            
                        found_optimal_solution = true;
                    } 
    
                }

                if( found_optimal_solution ) {
 
                    if ( gate_structure_loc ) {
                        delete( gate_structure_loc );
                    } 

                    break;
                }
        

                // iterate the Gray code to the next element
                int changed_index, value_prev, value;
                if ( gcode_counter.next(changed_index, value_prev, value) ) {
                    // exit from the for loop if no further gcode is present
                    break;
                }   
        
        
            }

        }
    
    });


    return gate_structure_best_solution;


}




/**
@brief 
@param 
@param 
*/
Gates_block* 
N_Qubit_Decomposition_non_unitary_adaptive::CZ_nu_search( int level_max ){

    tbb::spin_mutex tree_search_mutex;
    
   
    
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }     

    
    
    Gates_block* gate_structure_best_solution = NULL;    
    bool found_optimal_solution = false;
    
    
     
    
        
    for( int job_idx=0; job_idx<level_max; job_idx++ ) {  
    

            //std::cout << initial_offset << " " << offset_max << " " << iteration_max << " " << work_batch << " " << concurrency << std::endl;

        
                //  ----------- contruct the gate structure to be optimized ----------- 
                Gates_block* gate_structure_loc = new Gates_block(qbit_num);  // cnot nélkül megyek mi lesz? 

                // add non unitary adaptive blocks
                for (int gcode_idx=0; gcode_idx<job_idx; gcode_idx++) {
                    add_adaptive_layers( gate_structure_loc );
                }
             /*                  
                for (int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++) {
            
                    // add new 2-qbit block to the circuit
                    add_two_qubit_block( gate_structure_loc, target_qbits[gcode_idx], control_qbits[gcode_idx]  );
                }
               */
                // add finalyzing layer to the the gate structure
                add_finalyzing_layer( gate_structure_loc );
              
    

                // ----------- start the decomposition ----------- 
        
                //measure the time for the decompositin
                tbb::tick_count start_time_loc = tbb::tick_count::now();


                N_Qubit_Decomposition_custom cDecomp_custom_random;

                std::stringstream sstream;
                sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
                print(sstream, 1);
        
                // solve the optimization problem in isolated optimization process
                cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
                cDecomp_custom_random.set_custom_gate_structure( gate_structure_loc );
                cDecomp_custom_random.set_optimization_blocks( gate_structure_loc->get_gate_num() );
                cDecomp_custom_random.set_max_iteration( max_outer_iterations );
#ifndef __DFE__
                cDecomp_custom_random.set_verbose(verbose);
#else
                cDecomp_custom_random.set_verbose(0);
#endif
                cDecomp_custom_random.set_cost_function_variant( cost_fnc );
                cDecomp_custom_random.set_debugfile("");
                cDecomp_custom_random.set_optimization_tolerance( optimization_tolerance_loc );    
                cDecomp_custom_random.set_trace_offset( trace_offset ); 
                cDecomp_custom_random.set_optimizer( alg );    
                cDecomp_custom_random.set_project_name( project_name );
                if ( alg == ADAM || alg == BFGS2 ) {
                    int param_num_loc = gate_structure_loc->get_parameter_num();
                    int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_random.set_random_shift_count_max( 10000 ); 
                }
                else if ( alg==ADAM_BATCHED ) {
                    cDecomp_custom_random.set_optimizer( alg );  
                    int max_inner_iterations_loc = 2000;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_random.set_random_shift_count_max( 5 );   
                }
                else if ( alg==BFGS ) {
                    cDecomp_custom_random.set_optimizer( alg );  
                    int max_inner_iterations_loc = 10000;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                }
                
            
                cDecomp_custom_random.start_decomposition();
                

                
                number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization  
    
                double current_minimum_tmp         = cDecomp_custom_random.get_current_minimum();
                sstream.str("");
                sstream << "Optimization with " << level_max << " levels converged to " << current_minimum_tmp;
                print(sstream, 1);

                //std::cout << "Optimization with " << level_max << " levels converged to " << current_minimum_tmp << std::endl;
        
                {
                    tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};

                    if( current_minimum_tmp < current_minimum && !found_optimal_solution) {
                        current_minimum = current_minimum_tmp;

                        if ( gate_structure_best_solution != NULL ) {
                            delete( gate_structure_best_solution );
                        }
            
                        gate_structure_best_solution = gate_structure_loc;
                        gate_structure_loc = NULL;

                        optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                    }
                    else {
                        delete( gate_structure_loc );
                        gate_structure_loc = NULL;
                    }

     
                    if ( current_minimum < optimization_tolerance_loc && !found_optimal_solution)  {            
                        found_optimal_solution = true;
                    } 
    
                }

                if( found_optimal_solution ) {
 
                    if ( gate_structure_loc ) {
                        delete( gate_structure_loc );
                    } 

                    break;
                }
        
 
        
        
            

        }
    
   

    return gate_structure_best_solution;


}



/**
@brief Call to optimize an imported gate structure
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_non_unitary_adaptive::optimize_imported_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

    Gates_block* gate_structure_loc = (static_cast<Gates_block*>(this))->clone();
        
    //measure the time for the decompositin
    tbb::tick_count start_time_loc = tbb::tick_count::now();

    std::stringstream sstream;
    sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
         
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }      

    // solve the optimization problem
    N_Qubit_Decomposition_custom cDecomp_custom;
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, initial_guess, accelerator_num);
    cDecomp_custom.set_custom_gate_structure( gate_structure_loc );
    cDecomp_custom.set_optimized_parameters( optimized_parameters_mtx_loc.get_data(), optimized_parameters_mtx_loc.size() );
    cDecomp_custom.set_optimization_blocks( gate_structure_loc->get_gate_num() );
    cDecomp_custom.set_max_iteration( max_outer_iterations );
    cDecomp_custom.set_verbose(verbose);
    cDecomp_custom.set_cost_function_variant( cost_fnc );
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance_loc ); 
    cDecomp_custom.set_trace_offset( trace_offset ); 
    cDecomp_custom.set_optimizer( alg );  
    cDecomp_custom.set_project_name( project_name );
    if (alg==ADAM || alg==BFGS2) { 
        int param_num_loc = gate_structure_loc->get_parameter_num();
        int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
        cDecomp_custom.set_random_shift_count_max( 10000 );          
    }
    else if ( alg==ADAM_BATCHED ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 2500;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
        cDecomp_custom.set_random_shift_count_max( 5 );  
    }
    else if ( alg==BFGS ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 10000;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );    
    }
    cDecomp_custom.start_decomposition();
    number_of_iters += cDecomp_custom.get_num_iters();
    //cDecomp_custom.list_gates(0);

    tbb::tick_count end_time_loc = tbb::tick_count::now();

    current_minimum = cDecomp_custom.get_current_minimum();
    optimized_parameters_mtx_loc = cDecomp_custom.get_optimized_parameters();



    if ( cDecomp_custom.get_current_minimum() < optimization_tolerance_loc ) {
        std::stringstream sstream;
	sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        print(sstream, 1);	
    }   
    else {
        std::stringstream sstream;
	sstream << "Optimization problem converged to " << cDecomp_custom.get_current_minimum() << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        print(sstream, 1);       
    }

    if (current_minimum > optimization_tolerance_loc) {
        std::stringstream sstream;
	sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl; 
        print(sstream, 1);             
        optimization_tolerance_loc = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    sstream.str("");
    sstream << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
    return gate_structure_loc;



}

/**
@brief Call determine the gate structure of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_non_unitary_adaptive::determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

    N_Qubit_Decomposition_custom cDecomp_custom_random;
    cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    double optimization_tolerance_loc;
    long long level_max; 
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  

    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }      
    
    if  (config.count("tree_level_max") > 0 ){
        std::cout << "lefut\n";
        config["tree_level_max"].get_property( level_max );
    } 
    else {
        level_max = 14;
    }
    

    Gates_block* gate_structure_loc = NULL;
    Gates_block* best_so_far = NULL;

    
    gate_structure_loc          = tree_search_over_gate_structures( 0 );
    best_so_far                 = gate_structure_loc;
    double best_minimum_so_far  = current_minimum; 

    //best_so_far = CZ_nu_search( level_max );

    for ( int level = 1; level <= level_max; level++ ) { 

        gate_structure_loc = tree_search_over_gate_structures( level );   

        if (current_minimum < best_minimum_so_far) { 

            best_minimum_so_far = current_minimum;

            delete( best_so_far ) ;

            best_so_far        = gate_structure_loc;  
            gate_structure_loc = NULL;
        }
        else{

            delete( gate_structure_loc );
            gate_structure_loc = NULL;

        }
        
        if (current_minimum < optimization_tolerance_loc ) {

            break;
        }

    }    
    
    
    if (current_minimum > optimization_tolerance_loc) {
       std::stringstream sstream;
       sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
       print(sstream, 1);              
       optimization_tolerance_loc = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    std::stringstream sstream;
    sstream << "Continue with the compression of gate structure consisting of " << best_so_far->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	

    return best_so_far;
       
}




/**
@brief Call to replace CZ_NU gates in the circuit that are close to either an identity or to a CNOT gate.
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_non_unitary_adaptive::replace_trivial_CZ_NU_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {

    Gates_block* gate_structure_ret = new Gates_block(qbit_num);

    int layer_num = gate_structure->get_gate_num();

    int parameter_idx = 0;
    for (int idx=0; idx<layer_num; idx++ ) {

        Gate* gate = gate_structure->get_gate(idx);

        if ( gate->get_type() != BLOCK_OPERATION ) {
           std::string err = "N_Qubit_Decomposition_non_unitary_adaptive::replace_trivial_CZ_NU_gates: Only block gates are accepted in this conversion.";
           std::stringstream sstream;
	   sstream << err << std::endl;
           print(sstream, 1);	
           throw(err);
        }

        Gates_block* block_op = static_cast<Gates_block*>(gate);
        //int param_num = gate->get_parameter_num();


        Gates_block* layer = block_op->clone();

        for ( int jdx=0; jdx<layer->get_gate_num(); jdx++ ) {

            Gate* gate_tmp = layer->get_gate(jdx);
            int param_num = gate_tmp->get_parameter_num();
                    
            double parameter = optimized_parameters[parameter_idx];

            if ( gate_tmp->get_type() == CZ_NU_OPERATION &&  std::cos(parameter) < -0.95 ) {

                // convert to CZ gate
                int target_qbit = gate_tmp->get_target_qbit();
                int control_qbit = gate_tmp->get_control_qbit();
                layer->release_gate( jdx );

                CZ*   cz_gate     = new CZ(qbit_num, target_qbit, control_qbit);

                layer->insert_gate( (Gate*)cz_gate, jdx);

                Matrix_real parameters_new(1, optimized_parameters.size()-1);
                memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                memcpy(parameters_new.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));

                parameter_idx += 0;
                optimized_parameters = parameters_new;

            }            
            else if ( gate_tmp->get_type() == CZ_NU_OPERATION &&  std::cos(parameter) > 0.95 ) {
                // release trivial gate  

                layer->release_gate( jdx );
                jdx--;
                Matrix_real parameters_new(1, optimized_parameters.size()-1);
                memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                memcpy(parameters_new.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));
                        
                parameter_idx += 0;
                optimized_parameters = parameters_new;
            }          
            else if ( gate_tmp->get_type() == CZ_NU_OPERATION ) {

                        // controlled Y rotation decomposed into 2 CNOT gates
                        std::cout << "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiopooooooooooooooooo " << sin(parameter) << std::endl;

                        parameter_idx += 1;

            }
            else {
                parameter_idx  += param_num;

            }

        }

        gate_structure_ret->add_gate((Gate*)layer);


    }

                           

    return gate_structure_ret;


}





/**
@brief Call to add adaptive layers to the gate structure stored by the class.
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::add_adaptive_layers() {

    add_adaptive_layers( this );

}

/**
@brief Call to add adaptive layers to the gate structure.
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::add_adaptive_layers( Gates_block* gate_structure ) {


    // create the new decomposing layer and add to the gate staructure
    Gates_block* layer = construct_adaptive_gate_layers();
    gate_structure->combine( layer );


}




/**
@brief Call to construct adaptive layers.
*/
Gates_block* 
N_Qubit_Decomposition_non_unitary_adaptive::construct_adaptive_gate_layers() {


    //The string stream input to store the output messages.
    std::stringstream sstream;

    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    std::vector<Gates_block* > layers;


    if ( topology.size() > 0 ) {
        for ( std::vector<matrix_base<int>>::iterator it=topology.begin(); it!=topology.end(); it++) {

            if ( it->size() != 2 ) {
                std::string err("The connectivity data should contains two qubits.");
                throw err;
            }

            int control_qbit_loc = (*it)[0];
            int target_qbit_loc = (*it)[1];

            if ( control_qbit_loc >= qbit_num || target_qbit_loc >= qbit_num ) {
                std::string err("Label of control/target qubit should be less than the number of qubits in the register.");
                throw err;          
            }

            Gates_block* layer = new Gates_block( qbit_num );

            bool Theta = true;
            bool Phi = true;
            bool Lambda = true;
            layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
            layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
            layer->add_cz_nu(target_qbit_loc, control_qbit_loc);

            layers.push_back(layer);


        }
    }
    else {  
    
        // sequ
        for (int target_qbit_loc = 0; target_qbit_loc<qbit_num; target_qbit_loc++) {
            for (int control_qbit_loc = target_qbit_loc+1; control_qbit_loc<qbit_num; control_qbit_loc++) {

                Gates_block* layer = new Gates_block( qbit_num );

                bool Theta = true;
                bool Phi = true;
                bool Lambda = true;
                layer->add_u3(target_qbit_loc, Theta, Phi, Lambda);
                layer->add_u3(control_qbit_loc, Theta, Phi, Lambda); 
                layer->add_cz_nu(target_qbit_loc, control_qbit_loc);

                layers.push_back(layer);
            }
        }

    }

/*
    for (int idx=0; idx<layers.size(); idx++) {
        Gates_block* layer = (Gates_block*)layers[idx];
        block->add_gate( layers[idx] );

    }
*/

    bool randomized_adaptive_layers_loc;
    if ( config.count("randomized_adaptive_layers") > 0 ) {
        config["randomized_adaptive_layers"].get_property( randomized_adaptive_layers_loc );  
    }
    else {
        randomized_adaptive_layers_loc = randomized_adaptive_layers;
    }


    // make difference between randomized adaptive layers and deterministic one
    if (randomized_adaptive_layers_loc) {

        std::uniform_int_distribution<> distrib_int(0, 5000);

        while (layers.size()>0) { 
            int idx = distrib_int(gen) % layers.size();

#ifdef __MPI__        
            MPI_Bcast( &idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
            block->add_gate( layers[idx] );
            layers.erase( layers.begin() + idx );
        }

    }
    else {
        while (layers.size()>0) { 
            block->add_gate( layers[0] );
            layers.erase( layers.begin() );
        }

    }


    return block;


}


/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure stored by the class.
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::add_finalyzing_layer() {

    add_finalyzing_layer( this );

}

/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::add_finalyzing_layer( Gates_block* gate_structure ) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );
/*
    block->add_un();
    block->add_ry(qbit_num-1);
*/
    for (int idx=0; idx<qbit_num; idx++) {
            bool Theta = true;
            bool Phi = true;
            bool Lambda = true;
block->add_rz(idx);
block->add_ry(idx);
block->add_rz(idx); 

             //block->add_u3(idx, Theta, Phi, Lambda);
//        block->add_ry(idx);
    }


    // adding the opeartion block to the gates
    if ( gate_structure == NULL ) {
        throw ("N_Qubit_Decomposition_non_unitary_adaptive::add_finalyzing_layer: gate_structure is null pointer");
    }
    else {
        gate_structure->add_gate( block );
    }


}




/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param filename
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::set_adaptive_gate_structure( std::string filename ) {

    if ( gates.size() > 0  ) {
        release_gates();
        optimized_parameters_mtx = Matrix_real(0,0);
    }

    Gates_block* gate_structure = import_gate_list_from_binary(optimized_parameters_mtx, filename, verbose);
    combine( gate_structure );
    delete gate_structure;

}

/**
@brief set unitary matrix from binary file
@param filename .binary file to import unitary from
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::set_unitary_from_file( std::string filename ) {

    Umtx = import_unitary_from_binary(filename);

#ifdef __DFE__
    if( qbit_num >= 5 ) {
        upload_Umtx_to_DFE();
    }
#endif

}
/**
@brief call to set Unitary from mtx
@param matrix to set over
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::set_unitary( Matrix& Umtx_new ) {

    Umtx = Umtx_new;

#ifdef __DFE__
    if( qbit_num >= 5 ) {
        upload_Umtx_to_DFE();
    }
#endif

}

/**
@brief Call to append custom layers to the gate structure that are intended to be used in the decomposition.
@param filename
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::add_adaptive_gate_structure( std::string filename ) {



    Matrix_real optimized_parameters_mtx_tmp;
    Gates_block* gate_structure_tmp = import_gate_list_from_binary(optimized_parameters_mtx_tmp, filename, verbose);

    if ( gates.size() > 0 ) {
        gate_structure_tmp->combine( static_cast<Gates_block*>(this) );

        release_gates();
        combine( gate_structure_tmp );
      

        Matrix_real optimized_parameters_mtx_tmp2( 1, optimized_parameters_mtx_tmp.size() + optimized_parameters_mtx.size() );
        memcpy( optimized_parameters_mtx_tmp2.get_data(), optimized_parameters_mtx_tmp.get_data(), optimized_parameters_mtx_tmp.size()*sizeof(double) );
        memcpy( optimized_parameters_mtx_tmp2.get_data()+optimized_parameters_mtx_tmp.size(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp2;
    }
    else {
        combine( gate_structure_tmp );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp;
    }

}



/**
@brief Call to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gfate structure is released.
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::apply_imported_gate_structure() {

    if ( gates.size() == 0 ) {
        return;
    }

    
    std::stringstream sstream;
    sstream << "The cost function before applying the imported gate structure is:" << optimization_problem( optimized_parameters_mtx )  << std::endl;   
    
    apply_to(  optimized_parameters_mtx, Umtx );
    release_gates();
    optimized_parameters_mtx = Matrix_real(0,0);
    

    sstream << "The cost function after applying the imported gate structure is:" << optimization_problem( optimized_parameters_mtx )  << std::endl;
    print(sstream, 3);	



}


/**
@brief Call to add an adaptive layer to the gate structure previously imported
@param filename
*/
void 
N_Qubit_Decomposition_non_unitary_adaptive::add_layer_to_imported_gate_structure() {


    std::stringstream sstream;
    sstream << "Add new layer to the adaptive gate structure." << std::endl;	        
    print(sstream, 2);

    Gates_block* layer = construct_adaptive_gate_layers();


    combine( layer );

    Matrix_real tmp( 1, optimized_parameters_mtx.size() + layer->get_parameter_num() );
    memset( tmp.get_data(), 0, tmp.size()*sizeof(double) );
    memcpy( tmp.get_data(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );

    optimized_parameters_mtx = tmp;    

}







