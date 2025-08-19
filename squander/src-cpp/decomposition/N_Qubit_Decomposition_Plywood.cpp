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
/*! \file N_Qubit_Decomposition_Tabu_Search.cpp
    \brief Class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#include "N_Qubit_Decomposition_Tree_Search.h"
#include "N_Qubit_Decomposition_Plywood.h"
#include "Random_Orthogonal.h"
#include "Random_Unitary.h"
#include "n_aryGrayCodeCounter.h"
#include <random>

#include "X.h"

#include <time.h>
#include <stdlib.h>

#include <iostream>

#ifdef __DFE__
#include "common_DFE.h"
#endif



/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Plywood::N_Qubit_Decomposition_Plywood() : N_Qubit_Decomposition_Tree_Search() {


}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param level_limit_in The maximal number of two-qubit gates in the decomposition
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Plywood::N_Qubit_Decomposition_Plywood( Matrix Umtx_in, int qbit_num_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : N_Qubit_Decomposition_Tree_Search( Umtx_in, qbit_num_in, config, accelerator_num) {
    name = "Plywood";
    jump_level=0;
    jump_edges = std::vector<matrix_base<int>>{};
    max_jump_iter = 0;
}



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param level_limit_in The maximal number of two-qubit gates in the decomposition
@param topology_in A list of <target_qubit, control_qubit> pairs describing the connectivity between qubits.
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Plywood::N_Qubit_Decomposition_Plywood( Matrix Umtx_in, int qbit_num_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config,  int accelerator_num) : N_Qubit_Decomposition_Tree_Search( Umtx_in, qbit_num_in, topology_in, config, accelerator_num ) {

    // A string labeling the gate operation
    name = "Plywood";
    jump_level=0;
    jump_edges = std::vector<matrix_base<int>>{};
    max_jump_iter = 0;
}

/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Plywood::~N_Qubit_Decomposition_Plywood() {

}





/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
*/
void
N_Qubit_Decomposition_Plywood::start_decomposition() {


    //The string stream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
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

    long long jump_level_loc;
    if ( config.count("jump_level") > 0 ) {
        config["jump_level"].get_property( jump_level_loc );  
    }
    else {
        jump_level_loc = 2;
    } 
    long long max_jump_iter_loc;
    if ( config.count("max_jump_iter") > 0 ) {
        config["max_jump_iter"].get_property( max_jump_iter_loc );  
    }
    else {
        max_jump_iter_loc = 10;
    }
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }      

    jump_level = (int)jump_level_loc;
    max_jump_iter = (int)max_jump_iter_loc;
    
    for (int level_idx=0; level_idx<jump_level; level_idx++){
        for( int element_idx = 0; element_idx<topology.size(); element_idx++ ) {
           matrix_base<int>& edge = topology[ element_idx ];
           jump_edges.push_back( edge );
        }    
    }

    int jump_idx = 0;
    Gates_block* gate_structure_loc = NULL;
    while ( current_minimum > optimization_tolerance_loc && jump_idx < max_jump_iter) {
        std::stringstream sstream;
        sstream << "Plywood starting jump iter " << jump_idx << "." << std::endl;
        print(sstream, 1);
        gate_structure_loc = determine_gate_structure(optimized_parameters_mtx);
        if ( current_minimum < optimization_tolerance_loc ) {
            break;
        }
        jump_edges = extract_edges_from_circuit(gate_structure_loc);
        
        for (int level_idx=0; level_idx<jump_level; level_idx++){
            for( int element_idx = 0; element_idx<topology.size(); element_idx++ ) {
                matrix_base<int>& edge = topology[ element_idx ];
                jump_edges.push_back( edge );
            }    
        }
        
        jump_idx++;
    }


    Gates_block* gate_structure_compressed =  parameter_based_compression(gate_structure_loc, optimized_parameters_mtx);
    // store the created gate structure
    release_gates();
	combine( gate_structure_compressed );
	delete( gate_structure_loc );
	delete( gate_structure_compressed );
	
    decomposition_error = current_minimum;
	
	
#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif
}

Gates_block* 
N_Qubit_Decomposition_Plywood::parameter_based_compression( Gates_block* gate_structure,  Matrix_real& optimized_parameters ) {

    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }      

    // solve the optimization problem
    N_Qubit_Decomposition_adaptive cDecomp_custom;
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_adaptive( Umtx.copy(), qbit_num, 10, 1, config, accelerator_num);
    cDecomp_custom.set_custom_gate_structure( gate_structure );
    cDecomp_custom.set_optimized_parameters( optimized_parameters.get_data(), optimized_parameters.size() );
    cDecomp_custom.set_optimization_blocks( gate_structure->get_gate_num() );
    cDecomp_custom.set_max_iteration( max_outer_iterations );
    cDecomp_custom.set_verbose(0);
    cDecomp_custom.set_cost_function_variant( cost_fnc );
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance_loc ); 
    cDecomp_custom.set_trace_offset( trace_offset ); 
    cDecomp_custom.set_optimizer( alg );  
    cDecomp_custom.set_project_name( project_name );
    if (alg==ADAM || alg==BFGS2) { 
        int param_num_loc = gate_structure->get_parameter_num();
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
    cDecomp_custom.compress_circuit_PBC();
    std::map<std::string, int>&& gate_nums = cDecomp_custom.get_gate_nums();
    optimized_parameters = cDecomp_custom.get_optimized_parameters();
    
    return (static_cast<Gates_block*>(&cDecomp_custom))->clone();
}


/**
@brief  Call to construnc a gate structure corresponding to the configuration of the two-qubit gates described by the Gray code  
@param gcode The N-ary Gray code describing the configuration of the two-qubit gates.
@return Returns with the generated circuit
*/
Gates_block* 
N_Qubit_Decomposition_Plywood::construct_gate_structure_from_Gray_code( const GrayCode& gcode ) {


    // determine the target qubit indices and control qbit indices for the CNOT gates from the Gray code counter
    matrix_base<int> target_qbits(1, gcode.size());
    matrix_base<int> control_qbits(1, gcode.size());

        
    for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {    
            
        int target_qbit = possible_target_qbits[ gcode[gcode_idx] ];            
        int control_qbit = possible_control_qbits[ gcode[gcode_idx] ];
            
            
        target_qbits[gcode_idx] = target_qbit;
        control_qbits[gcode_idx] = control_qbit;  
            
            
        //std::cout <<   target_qbit << " " << control_qbit << std::endl;        
    }

        
    //  ----------- contruct the gate structure to be optimized ----------- 
    Gates_block* gate_structure_loc = new Gates_block(qbit_num);
    
    int n_ary_limit_max = jump_edges.size();
    

    for( int element_idx = 0; element_idx<n_ary_limit_max; element_idx++ ) {

       matrix_base<int>& edge = jump_edges[ element_idx ];
       int target_qbit = edge[0];
       int control_qbit = edge[1]; 
       add_two_qubit_block( gate_structure_loc, target_qbit, control_qbit);
 
    }    
                            
    for (int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++) {      
            
        // add new 2-qbit block to the circuit
        add_two_qubit_block( gate_structure_loc, target_qbits[gcode_idx], control_qbits[gcode_idx]  );
    }
         
    // add finalyzing layer to the the gate structure
    add_finalyzing_layer( gate_structure_loc );
                
    return  gate_structure_loc;           

}

std::vector<matrix_base<int>> 
N_Qubit_Decomposition_Plywood::extract_edges_from_circuit( Gates_block* circuit ) {

    std::vector<matrix_base<int>> gate_edges{};

    Gates_block* flat_circuit = static_cast<Gates_block*>(circuit->get_flat_circuit());
    int gate_num = flat_circuit->get_gate_num();
    for (int gate_idx; gate_idx<gate_num; gate_idx++){
        Gate* gate_tmp = flat_circuit->get_gate(gate_idx);
        if (gate_tmp->get_control_qbit() != -1){
            matrix_base<int> new_edge(2,1);
            new_edge[0] = gate_tmp->get_target_qbit();
            new_edge[1] = gate_tmp->get_control_qbit();
            gate_edges.push_back(new_edge);
        }
    }
    
    return gate_edges;
}




