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
/*! \file N_Qubit_Decomposition_adaptive.cpp
    \brief Class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Random_Orthogonal.h"
#include "Random_Unitary.h"

#include "X.h"

#include <time.h>
#include <stdlib.h>


#ifdef __DFE__
#include "common_DFE.h"
#endif





/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive() : Optimization_Interface() {


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
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {


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
N_Qubit_Decomposition_adaptive::N_Qubit_Decomposition_adaptive( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {



    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // Maximal number of iteartions in the optimization process
    max_outer_iterations = 1;

    
    // setting the topology
    topology = topology_in;




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
N_Qubit_Decomposition_adaptive::~N_Qubit_Decomposition_adaptive() {

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
*/
void
N_Qubit_Decomposition_adaptive::start_decomposition() {


    //The stringstream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;

    print(sstream, 1);   



    // get the initial circuit including redundand 2-qbit blocks.
    get_initial_circuit();

    // comppress the gate structure
    compress_circuit();

   
    // finalyzing the gate structure by turning CRY gates inti CNOT gates and do optimization cycles to correct approximation in this transformation 
    // (CRY gates with small rotation angles are expressed with a single CNOT gate
    finalize_circuit();

}





/**
@brief Call to determine the initial gate structure (still conatining adaptive gates)
*/
void N_Qubit_Decomposition_adaptive::get_initial_circuit() {
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
@brief Call to compress the gate structure by removing adaptive layers.
*/
void N_Qubit_Decomposition_adaptive::compress_circuit() {

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

    std::stringstream sstream;
    sstream.str("");
    sstream << std::endl;
    sstream << std::endl;
    sstream << "**************************************************************" << std::endl;
    sstream << "***************** Compressing Gate structure *****************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    print(sstream, 1);	    	
    Gates_block* gate_structure_loc = NULL;
    if ( gates.size() > 0 ) {
        std::stringstream sstream;
        sstream << "Using imported gate structure for the compression." << std::endl;
        print(sstream, 1);	
		    
        gate_structure_loc =  static_cast<Gates_block*>(this)->clone();
    }
    else {
        std::stringstream sstream;
        sstream << "No circuit initalised." << std::endl;
        print(sstream, 1);
        return;
    }

    sstream.str("");       
    sstream << "Compressing gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
    sstream.str("");    
    
    
    int iter = 0;
    int uncompressed_iter_num = 0;
    
    long long export_circuit_2_binary_loc;
    if ( config.count("export_circuit_2_binary") > 0 ) {
        config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
    }
    else {
        export_circuit_2_binary_loc = 0;
    }      
    
    while ( iter<25 || uncompressed_iter_num <= 5 ) {
        std::stringstream sstream;
        sstream.str("");
        sstream << "iteration " << iter+1 << ": ";
        print(sstream, 1);	

       
        Gates_block* gate_structure_compressed = compress_gate_structure( gate_structure_loc );

        if ( gate_structure_compressed->get_gate_num() < gate_structure_loc->get_gate_num() ) {
            uncompressed_iter_num = 0;
        }
        else {
            uncompressed_iter_num++;
        }

        if ( gate_structure_compressed != gate_structure_loc ) {

            delete( gate_structure_loc );
            gate_structure_loc = gate_structure_compressed;
            gate_structure_compressed = NULL;
            


            if ( export_circuit_2_binary_loc > 0 ) {
                std::string filename("circuit_compression.binary");
                if (project_name != "") { 
                    filename=project_name+ "_"  +filename;
                }
                export_gate_list_to_binary(optimized_parameters_mtx, gate_structure_loc, filename, verbose); 
            
            
                std::string filename_unitary("unitary_compression.binary");
                if (project_name != "") { 
                    filename_unitary=project_name+ "_"  +filename_unitary;
                }  
                export_unitary(filename_unitary);
                
                
            }
        }

        iter++;

        if (uncompressed_iter_num>10) break;
            // store the decomposing gate structure
    }

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
@brief Call to replace adaptive gates with conventional gates and refine the gate structure.
*/
void N_Qubit_Decomposition_adaptive::finalize_circuit() {


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
        sstream << "No circuit initalised." << std::endl;
        print(sstream, 1);
        return;
    }
    
    
    std::stringstream sstream;
    sstream.str("");
    sstream << "**************************************************************" << std::endl;
    sstream << "************ Final tuning of the Gate structure **************" << std::endl;
    sstream << "**************************************************************" << std::endl;
    print(sstream, 1);	    	
    
	// maximal number of inner iterations overriden by config
    if ( config.count("optimization_tolerance") > 0 ) {
        long long value;
        config["optimization_tolerance"].get_property( value );
        optimization_tolerance = (double)value; 
    }
    else {optimization_tolerance = 1e-4;}



    optimization_block = get_gate_num();


    sstream.str("");
    sstream << "cost function value before replacing trivial CRY gates: " << optimization_problem(optimized_parameters_mtx.get_data()) << std::endl;
    print(sstream, 3);	
    	
    Gates_block* gate_structure_tmp = replace_trivial_CRY_gates( gate_structure_loc, optimized_parameters_mtx );
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
@brief Call to optimize an imported gate structure
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::optimize_imported_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

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
@brief Call determine the gate structrue of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

    // strages to store the optimized minimums in case of different cirquit depths
    std::vector<double> minimum_vec;
    std::vector<Gates_block*> gate_structure_vec;
    std::vector<Matrix_real> optimized_parameters_vec;
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }         
        
    


    int level = level_limit_min;
    while ( current_minimum > optimization_tolerance_loc && level <= level_limit) {

        // create gate structure to be optimized
        Gates_block* gate_structure_loc = new Gates_block(qbit_num);  
        
        optimized_parameters_mtx_loc = Matrix_real(0,0);
                   
        for (int idx=0; idx<level; idx++) {

            // create the new decomposing layer and add to the gate staructure
            add_adaptive_layers( gate_structure_loc );

        }
           
        // add finalyzing layer to the top of the gate structure
        add_finalyzing_layer( gate_structure_loc );
            

        //measure the time for the decompositin
        tbb::tick_count start_time_loc = tbb::tick_count::now();


        N_Qubit_Decomposition_custom cDecomp_custom_random, cDecomp_custom_close_to_zero;

        std::stringstream sstream;
        sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
        print(sstream, 1);

	
/*
#ifndef __DFE__
        // try the decomposition withrandom and with close to zero initial values
        tbb::parallel_invoke(
            [&]{            
#endif
*/
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
/*
#ifndef __DFE__
            },
            [&]{
                // solve the optimization problem in isolated optimization process
                cDecomp_custom_close_to_zero = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, CLOSE_TO_ZERO);
                cDecomp_custom_close_to_zero.set_custom_gate_structure( gate_structure_loc );
                cDecomp_custom_close_to_zero.set_optimization_blocks( gate_structure_loc->get_gate_num() );    
                cDecomp_custom_close_to_zero.set_max_iteration( max_outer_iterations );
                cDecomp_custom_close_to_zero.set_verbose(0);
                cDecomp_custom_close_to_zero.set_cost_function_variant( cost_fnc );
                cDecomp_custom_close_to_zero.set_debugfile("");
                cDecomp_custom_close_to_zero.set_optimization_tolerance( optimization_tolerance );  
                cDecomp_custom_close_to_zero.set_optimizer( alg );
                if ( alg == ADAM || alg == BFGS2 ) {
                    int param_num_loc = gate_structure_loc->get_parameter_num();
                    int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
                    cDecomp_custom_close_to_zero.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_close_to_zero.set_random_shift_count_max( 10000 ); 
                }
                cDecomp_custom.close_to_zero.set_trace_offset( trace_offset ); 
                cDecomp_custom_close_to_zero.start_decomposition(true);
                number_of_iters += cDecomp_custom_close_to_zero.get_num_iters();
               }
         );
#endif
*/
         tbb::tick_count end_time_loc = tbb::tick_count::now();
/*
#ifdef __DFE__
return NULL;
#endif
*/
         double current_minimum_random         = cDecomp_custom_random.get_current_minimum();
         double current_minimum_close_to_zero = cDecomp_custom_close_to_zero.get_current_minimum();
         double current_minimum_loc;


         // select between the results obtained for different initial value strategy
         if ( current_minimum_random < optimization_tolerance_loc && current_minimum_close_to_zero > optimization_tolerance_loc ) {
             current_minimum_loc = current_minimum_random;
             optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
             initial_guess = RANDOM;
         }
         else if ( current_minimum_random > optimization_tolerance_loc && current_minimum_close_to_zero < optimization_tolerance_loc ) {
             current_minimum_loc = current_minimum_close_to_zero;
             optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
             initial_guess = CLOSE_TO_ZERO;
         }
         else if ( current_minimum_random < optimization_tolerance_loc && current_minimum_close_to_zero < optimization_tolerance_loc ) {
             Matrix_real optimized_parameters_mtx_random = cDecomp_custom_random.get_optimized_parameters();
             Matrix_real optimized_parameters_mtx_close_to_zero = cDecomp_custom_close_to_zero.get_optimized_parameters();

             int panelty_random         = get_panelty(gate_structure_loc, optimized_parameters_mtx_random);
             int panelty_close_to_zero = get_panelty(gate_structure_loc, optimized_parameters_mtx_close_to_zero );

             if ( panelty_random < panelty_close_to_zero ) {
                 current_minimum_loc = current_minimum_random;
                 optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
                 initial_guess = RANDOM;
             }
             else {
                 current_minimum_loc = current_minimum_close_to_zero;
                 optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
                 initial_guess = CLOSE_TO_ZERO;
             }

        }
        else {
           if ( current_minimum_random < current_minimum_close_to_zero ) {
                current_minimum_loc = current_minimum_random;
                optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
                initial_guess = RANDOM;
           }
           else {
                current_minimum_loc = current_minimum_close_to_zero;
                optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
                initial_guess = CLOSE_TO_ZERO;
           }

        }

        minimum_vec.push_back(current_minimum_loc);
        gate_structure_vec.push_back(gate_structure_loc);
        optimized_parameters_vec.push_back(optimized_parameters_mtx_loc);



        if ( current_minimum_loc < optimization_tolerance_loc ) {
	    std::stringstream sstream;
            sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            print(sstream, 1);	       
            break;
        }   
        else {
            std::stringstream sstream;
            sstream << "Optimization problem converged to " << current_minimum_loc << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            print(sstream, 1);  
        }

        level++;
    }

//exit(-1);

    // find the best decomposition
    int idx_min = 0;
    double current_minimum = minimum_vec[0];
    for (int idx=1; idx<(int)minimum_vec.size(); idx++) {
        if( current_minimum > minimum_vec[idx] ) {
            idx_min = idx;
            current_minimum = minimum_vec[idx];
        }
    }
     

    Gates_block* gate_structure_loc = gate_structure_vec[idx_min];
    optimized_parameters_mtx_loc = optimized_parameters_vec[idx_min];

    // release unnecesarry data
    for (int idx=0; idx<(int)minimum_vec.size(); idx++) {
        if( idx == idx_min ) {
            continue;
        }
        delete( gate_structure_vec[idx] );
    }    
    minimum_vec.clear();
    gate_structure_vec.clear();
    optimized_parameters_vec.clear();
    


    if (current_minimum > optimization_tolerance_loc) {
       std::stringstream sstream;
       sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
       print(sstream, 1);              
       optimization_tolerance_loc = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }
    

    return gate_structure_loc;
       
}



/**
@brief Call to run compression iterations on the circuit. (Trying to remove a CRY block in each iteration)
@param gate_structure The gate structure to be optimized
*/
Gates_block*
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure ) {



    int layer_num_max;
    int layer_num_orig = gate_structure->get_gate_num()-1; // TODO: see line 1558 to explain the -1: the last finalyzing layer of U3 gates is not tested for removal
    if ( layer_num_orig < 50 ) layer_num_max = 10;
    else if ( layer_num_orig < 60 ) layer_num_max = 4;
    else layer_num_max = 2;

    // create a list of layers to be tested for removal.
    std::vector<int> layers_to_remove;
    layers_to_remove.reserve(layer_num_orig); 
    for (int idx=0; idx<layer_num_orig; idx++ ) { // TODO: see line 1558 to explain the -1
        layers_to_remove.push_back(idx);
    }   
    
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }         

    // random generator of integers   
    std::uniform_int_distribution<> distrib_int(0, 5000);  

    while ( (int)layers_to_remove.size() > layer_num_max ) {
        int remove_idx = distrib_int(gen) % layers_to_remove.size();
       
        layers_to_remove.erase( layers_to_remove.begin() + remove_idx );
    }
    
#ifdef __MPI__        
    MPI_Bcast( &layers_to_remove[0], layers_to_remove.size(), MPI_INT, 0, MPI_COMM_WORLD);
#endif    

    // make a copy of the original unitary. (By removing trivial gates global phase might be added to the unitary)
    Matrix&& Umtx_orig = Umtx.copy();

    int panelties_num = layer_num_max < layer_num_orig ? layer_num_max : layer_num_orig;

    if ( panelties_num == 0 ) {
        return gate_structure;
    }

    // preallocate panelties associated with the number of remaining two-qubit controlled gates
    std::vector<unsigned int> panelties(panelties_num, 1<<31);
    std::vector<Gates_block*> gate_structures_vec(panelties_num, NULL);
    std::vector<double> current_minimum_vec(panelties_num, DBL_MAX);
    std::vector<int> iteration_num_vec(panelties_num, 0);


    std::vector<Matrix_real> optimized_parameters_vec(panelties_num, Matrix_real(0,0));
    std::vector<Matrix> Umtx_vec(panelties_num, Matrix(0,0));
  


    for (int idx=0; idx<panelties_num; idx++) {

        Umtx = Umtx_orig.copy();

        double current_minimum_loc = DBL_MAX;//current_minimum;
        int iteration_num = 0;
        Matrix_real optimized_parameters_loc = optimized_parameters_mtx.copy();

        Gates_block* gate_structure_reduced = compress_gate_structure( gate_structure, layers_to_remove[idx], optimized_parameters_loc,  current_minimum_loc, iteration_num  );
        if ( optimized_parameters_loc.size() == 0 ) {
            optimized_parameters_loc = optimized_parameters_mtx.copy();
        }

        // remove further adaptive gates if possible
        Gates_block* gate_structure_tmp;
        if ( gate_structure_reduced->get_gate_num() ==  gate_structure->get_gate_num() ) {
            gate_structure_tmp = gate_structure_reduced->clone();
        }
        else {
            gate_structure_tmp = remove_trivial_gates( gate_structure_reduced, optimized_parameters_loc, current_minimum_loc ); //TODO: reverse gate order
        }
      
        panelties[idx]                = get_panelty(gate_structure_tmp, optimized_parameters_loc);
        gate_structures_vec[idx]      = gate_structure_tmp;
        current_minimum_vec[idx]      = current_minimum_loc;
        iteration_num_vec[idx]        = iteration_num; // the accumulated number of optimization iterations


        optimized_parameters_vec[idx] = optimized_parameters_loc;
        Umtx_vec[idx]                 = Umtx;
        

        delete(gate_structure_reduced);

#ifdef __DFE__
        if ( current_minimum_vec[idx] < optimization_tolerance_loc ) {
            break;
        }
#endif
    }



    // determine the reduction with the lowest penalty
    unsigned int panelty_min = panelties[0];
    unsigned int idx_min     = 0;

    for (size_t idx=0; idx<panelties.size(); idx++) {
        if ( panelty_min > panelties[idx] ) {
            panelty_min = panelties[idx];
            idx_min = idx;
        }

        else if ( panelty_min == panelties[idx] ) {

            if ( (distrib_int(gen) % 2) == 1 ) {
                idx_min = idx;

                panelty_min = panelties[idx];
            }

        }

    }

#ifdef __MPI__        
    MPI_Bcast( &idx_min, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif    


    // release gate structures other than the best one
    for (size_t idx=0; idx<panelties.size(); idx++) {
        if (idx==idx_min) {
            continue;
        }


        if ( gate_structures_vec[idx] == gate_structure) {
            continue;
        }

        if ( gate_structures_vec[idx]  ) {
            delete( gate_structures_vec[idx] );
            gate_structures_vec[idx] = NULL;
        }

    }


    gate_structure           = gate_structures_vec[idx_min];
    optimized_parameters_mtx = optimized_parameters_vec[idx_min];
    current_minimum          = current_minimum_vec[idx_min];
    number_of_iters         += iteration_num_vec[idx_min]; // the total number of the accumulated optimization iterations
    Umtx                     = Umtx_vec[idx_min];
    
    int layer_num = gate_structure->get_gate_num();

    if ( layer_num < layer_num_orig+1 ) {
       std::stringstream sstream;
       sstream << "gate structure reduced from " << layer_num_orig+1 << " to " << layer_num << " decomposing layers" << std::endl;
       print(sstream, 1);	
    }
    else {
       std::stringstream sstream;
       sstream << "gate structure kept at " << layer_num << " layers" << std::endl;
       print(sstream, 1);		            
    }


    return gate_structure;



}

/**
@brief Call to run compression iterations on the circuit. (Trying to remove a CRY block in each iteration)
@param gate_structure The gate structure to be optimized
@param layer_idx The layer to be removed from the circuit
@param optimized_parameters A matrix containing the initial parameters
@param current_minimum_loc (out) The current minimum that has been achieved.
@param iteration_num (out) The number of iterations that have been carried out during the optimization
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::compress_gate_structure( Gates_block* gate_structure, int layer_idx, Matrix_real& optimized_parameters, double& current_minimum_loc, int& iteration_num ) {

    // create reduced gate structure without layer indexed by layer_idx
    Gates_block* gate_structure_reduced = gate_structure->clone();
    gate_structure_reduced->release_gate( layer_idx );

    Matrix_real parameters_reduced;
    if ( optimized_parameters.size() > 0 ) {
        parameters_reduced = create_reduced_parameters( gate_structure, optimized_parameters, layer_idx );
    }
    else {
        parameters_reduced = Matrix_real(0, 0);
    }
    


    N_Qubit_Decomposition_custom cDecomp_custom;

    std::map<std::string, Config_Element> config_copy;
    config_copy.insert(config.begin(), config.end());
    if ( config.count("max_inner_iterations_compression") > 0 ) {
        long long val;
        config["max_inner_iterations_compression"].get_property( val ); 
        Config_Element element;
        element.set_property( "max_inner_iterations", val ); 
        config_copy["max_inner_iterations"] = element;
    }
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }       


       
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config_copy, initial_guess, accelerator_num);
    cDecomp_custom.set_custom_gate_structure( gate_structure_reduced );
    cDecomp_custom.set_optimized_parameters( parameters_reduced.get_data(), parameters_reduced.size() );
    cDecomp_custom.set_verbose(0);
    cDecomp_custom.set_cost_function_variant( cost_fnc );
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_max_iteration( max_outer_iterations );
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_blocks( gate_structure_reduced->get_gate_num() ) ;
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance_loc );
    cDecomp_custom.set_trace_offset( trace_offset ); 
    cDecomp_custom.set_optimizer( alg );
    if ( alg == ADAM || alg==BFGS2) {
        cDecomp_custom.set_max_inner_iterations( 1e5 );  
        cDecomp_custom.set_random_shift_count_max( 1 );        
    }
    else if ( alg==BFGS ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 100;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc ); 
    }
    cDecomp_custom.start_decomposition();
    iteration_num = cDecomp_custom.get_num_iters();
    double current_minimum_tmp = cDecomp_custom.get_current_minimum();

    if ( current_minimum_tmp < optimization_tolerance_loc ) {
        //cDecomp_custom.list_gates(0);
        optimized_parameters = cDecomp_custom.get_optimized_parameters();
        current_minimum_loc = current_minimum_tmp;
        return gate_structure_reduced;
    }


    return gate_structure->clone();

}


/**
@brief Call to get the panelty derived from the number of CRY and CNOT gates in the circuit
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the initial parameters
*/
unsigned int 
N_Qubit_Decomposition_adaptive::get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    int panelty = 0;

    // iterate over the elements of the parameter array
    int parameter_idx = 0;
    int layer_num = gate_structure->get_gate_num();
    //for ( int layer_idx=layer_num-1; layer_idx>=0; layer_idx--) {
    for ( int layer_idx=0; layer_idx<layer_num; layer_idx++) {
    
        Gates_block* layer = static_cast<Gates_block*>( gate_structure->get_gate( layer_idx ) );
    
        int gate_num = layer->get_gate_num();
        //for( int gate_idx=gate_num-1; gate_idx>=0; gate_idx-- ) {
        for( int gate_idx=0; gate_idx<gate_num; gate_idx++ ) {
        
            Gate* gate = layer->get_gate( gate_idx );

            double parameter = optimized_parameters[parameter_idx];
            parameter_idx = parameter_idx + gate->get_parameter_num(); 
 
            if ( gate->get_type() != ADAPTIVE_OPERATION ) {
               continue;
            }        
            
        
            if ( std::abs(std::sin(parameter)) < 0.999 && std::abs(std::cos(parameter)) < 1e-3 ) {
                // Condition of pure CNOT gate
                panelty += 1;
            }
            else if ( std::abs(std::sin(parameter)) < 1e-3 && std::abs(1-std::cos(parameter)) < 1e-3 ) {
                // Condition of pure Identity gate
                //panelty++;
            }
            else {
                // Condition of controlled rotation gate
                panelty += 2;
            }
        
        }

    }


    return panelty;


}


/**
@brief Call to replace CRY gates in the circuit that are close to either an identity or to a CNOT gate.
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the  parameters
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::replace_trivial_CRY_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {

    Gates_block* gate_structure_ret = new Gates_block(qbit_num);

    int layer_num = gate_structure->get_gate_num();

/*
    std::map<std::string, Config_Element> config_copy;
    config_copy.insert(config.begin(), config.end());
    N_Qubit_Decomposition_custom cDecomp_custom( Umtx.copy(), qbit_num, false, config_copy, initial_guess);
    cDecomp_custom.set_custom_gate_structure( gate_structure );
    std::cout << std::endl << "before removing trivial gate: " << cDecomp_custom.optimization_problem( optimized_parameters ) << std::endl;
*/
    int parameter_idx = 0;
    //for (int idx=layer_num-1; idx>=0; idx-- ) {
    for (int idx=0; idx<layer_num; idx++ ) {

        Gate* gate = gate_structure->get_gate(idx);

        if ( gate->get_type() != BLOCK_OPERATION ) {
           std::string err = "N_Qubit_Decomposition_adaptive::replace_trivial_adaptive_gates: Only block gates are accepted in this conversion.";
           throw( err );
        }

        Gates_block* block_op = static_cast<Gates_block*>(gate);
        //int param_num = gate->get_parameter_num();


        if (  true ) {//gate_structure->contains_adaptive_gate(idx) ) {

                Gates_block* layer = block_op->clone();

                //for ( int jdx=layer->get_gate_num()-1; jdx>=0; jdx-- ) {
                for ( int jdx=0; jdx<layer->get_gate_num(); jdx++ ) {

                    Gate* gate_tmp = layer->get_gate(jdx);
                    int param_num = gate_tmp->get_parameter_num();


                    double parameter = optimized_parameters[parameter_idx];
                    parameter = activation_function(parameter, 1);//limit_max);

//std::cout << param[0] << " " << (gate_tmp->get_type() == ADAPTIVE_OPERATION) << " "  << std::abs(std::sin(param[0])) << " "  << 1+std::cos(param[0]) << std::endl;


                    if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(parameter)) > 0.999 && std::abs(std::cos(parameter)) < 1e-3) {

                        // convert to CZ gate
                        int target_qbit = gate_tmp->get_target_qbit();
                        int control_qbit = gate_tmp->get_control_qbit();
                        layer->release_gate( jdx );

                        RX*   rx_gate_1   = new RX(qbit_num, target_qbit);
                        CZ*   cz_gate     = new CZ(qbit_num, target_qbit, control_qbit);
                        RX*   rx_gate_2   = new RX(qbit_num, target_qbit);
                        RZ* rz_gate     = new RZ(qbit_num, control_qbit);                        

                        Gates_block* czr_gate = new Gates_block(qbit_num);
                        czr_gate->add_gate(rx_gate_1);
                        czr_gate->add_gate(cz_gate);
                        czr_gate->add_gate(rx_gate_2);
                        czr_gate->add_gate(rz_gate);

                        layer->insert_gate( (Gate*)czr_gate, jdx);

                        Matrix_real parameters_new(1, optimized_parameters.size()+2);

                        
                        memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));

                        memcpy(parameters_new.get_data()+parameter_idx+3, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));

                        parameters_new[parameter_idx] = -M_PI/4; // rx_1 parameter
                        parameters_new[parameter_idx+1] = M_PI/4; // rx_2 parameter


                        if ( std::sin(parameter) < 0 ) {
//                            parameters_new[parameter_idx+2] = -M_PI/2; // rz parameter with original RZ_P gate, in this case no global phase occurs either
                            parameters_new[parameter_idx+2] = -M_PI/4; // rz parameter   
                            
                            QGD_Complex16 global_phase_factor_new;
                            global_phase_factor_new.real = std::cos( -M_PI/4 );
                            global_phase_factor_new.imag = std::sin( -M_PI/4 );
                            apply_global_phase_factor(global_phase_factor_new, Umtx);

                        }
                        else{
//                            parameters_new[parameter_idx+2] = M_PI/2; // rz parameter with original RZ_P gate, in this case no global phase occurs either
                            parameters_new[parameter_idx+2] = M_PI/4; // rz parameter   
                            
                            QGD_Complex16 global_phase_factor_new;
                            global_phase_factor_new.real = std::cos( M_PI/4 );
                            global_phase_factor_new.imag = std::sin( M_PI/4 );
                            apply_global_phase_factor(global_phase_factor_new, Umtx);

                        }


                        optimized_parameters = parameters_new;
                        parameter_idx += 3;



                    }

                    else if ( gate_tmp->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(parameter)) < 1e-3 && std::abs(1-std::cos(parameter)) < 1e-3  ) {
                        // release trivial gate  

                        layer->release_gate( jdx );
                        //jdx--;
                        Matrix_real parameters_new(1, optimized_parameters.size()-1);
                        memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                        memcpy(parameters_new.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));
                        optimized_parameters = parameters_new;


                    }

                    else if ( gate_tmp->get_type() == ADAPTIVE_OPERATION ) {
                    
                        // controlled Y rotation decomposed into 2 CNOT gates
                        int target_qbit = gate_tmp->get_target_qbit();
                        int control_qbit = gate_tmp->get_control_qbit();
                        layer->release_gate( jdx );

                        RY*   ry_gate_1   = new RY(qbit_num, target_qbit);
                        CNOT* cnot_gate_1 = new CNOT(qbit_num, target_qbit, control_qbit);
                        RY*   ry_gate_2   = new RY(qbit_num, target_qbit);
                        CNOT* cnot_gate_2 = new CNOT(qbit_num, target_qbit, control_qbit);

                        Gates_block* czr_gate = new Gates_block(qbit_num);
                        czr_gate->add_gate(ry_gate_1);
                        czr_gate->add_gate(cnot_gate_1);
                        czr_gate->add_gate(ry_gate_2);
                        czr_gate->add_gate(cnot_gate_2);

                        layer->insert_gate( (Gate*)czr_gate, jdx);

                        Matrix_real parameters_new(1, optimized_parameters.size()+1);
                        memcpy(parameters_new.get_data(), optimized_parameters.get_data(), parameter_idx*sizeof(double));
                        memcpy(parameters_new.get_data()+parameter_idx+2, optimized_parameters.get_data()+parameter_idx+1, (optimized_parameters.size()-parameter_idx-1)*sizeof(double));
                        optimized_parameters = parameters_new;

                        optimized_parameters[parameter_idx] = parameter/2; // ry_1 parameter
                        optimized_parameters[parameter_idx+1] = -parameter/2; // ry_2 parameter


                        parameter_idx += 2;

                    }

                    else {

                        parameter_idx  += param_num;

                    }



                }

                gate_structure_ret->add_gate((Gate*)layer);


        }

    }

/*
    N_Qubit_Decomposition_custom cDecomp_custom_( Umtx.copy(), qbit_num, false, config_copy, initial_guess);
    cDecomp_custom_.set_custom_gate_structure( gate_structure_ret );
    std::cout << std::endl << "after removing trivial gate: " << cDecomp_custom_.optimization_problem( optimized_parameters ) << std::endl;
    exit(2);          
*/
    return gate_structure_ret;


}

/**
@brief Call to remove those blocks from the circuit that contain a trivial CRY gate (i.e. CRY gate close to be an identity.) The U3 gates are merged with subsequent gates.
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the parameters
@param current_minimum_loc (out) The current minimum that has been achieved.
*/
Gates_block*
N_Qubit_Decomposition_adaptive::remove_trivial_gates( Gates_block* gate_structure, Matrix_real& optimized_parameters, double& current_minimum_loc ) {

    int layer_num = gate_structure->get_gate_num();
    int parameter_idx = 0;

    Matrix_real&& optimized_parameters_loc = optimized_parameters.copy();

    Gates_block* gate_structure_loc = gate_structure->clone();


    for (int idx=0; idx<layer_num; idx++ ) {

        Gates_block* layer = static_cast<Gates_block*>( gate_structure_loc->get_gate(idx) );

        int param_num = layer->get_parameter_num();

        if ( layer->get_gate_num() != 3 ) {
            // every adaptive layers contains a single adaptive gate and two U3 gates
            parameter_idx += param_num;
            continue;
        }

        Gate* gate_adaptive = layer->get_gate(2);
        double parameter = optimized_parameters_loc[parameter_idx+6]; // parameter for adaptive gate        
        parameter = activation_function(parameter, 1);//limit_max);
       
        if ( gate_adaptive->get_type() == ADAPTIVE_OPERATION &&  std::abs(std::sin(parameter)) < 1e-3 && std::abs(1-std::cos(parameter)) < 1e-3  ) {
            /*
            optimized_parameters_loc[parameter_idx+6] = 0.0;           
            std::map<std::string, Config_Element> config_copy;
            config_copy.insert(config.begin(), config.end());
            N_Qubit_Decomposition_custom cDecomp_custom( Umtx.copy(), qbit_num, false, config_copy, initial_guess);
            cDecomp_custom.set_custom_gate_structure( gate_structure_loc );
            std::cout << std::endl << "before removing trivial gate: " << cDecomp_custom.optimization_problem( optimized_parameters_loc ) << std::endl;
            */

            int parameter_idx_to_be_removed = parameter_idx;


            // find matching U3 gates into which the U3 gates in the current layer are merged
            std::vector<int>&& involved_qbits = layer->get_involved_qubits();
            for( size_t rdx=0; rdx<involved_qbits.size(); rdx++ ) {

                U3* U_gate_to_be_removed = static_cast<U3*>(layer->get_gate(rdx));
                int qbit_to_be_matched = U_gate_to_be_removed->get_target_qbit();

                int parameter_idx_loc = parameter_idx + layer->get_parameter_num(); 

                bool found_match = false;
                U3* matching_gate = NULL;

                // iterate over subsequent layers to find the maching gate
                for ( int kdx=idx+1; kdx<layer_num; kdx++ ) {      

                    Gates_block* layer_test = static_cast<Gates_block*>( gate_structure_loc->get_gate(kdx) );

                    // iterate over the gates in the tested layer
                    int gate_num = layer_test->get_gate_num();
                    for ( int hdx=0; hdx<gate_num; hdx++ ) {

                        Gate* gate_test = layer_test->get_gate(hdx);                   

                        if ( gate_test->get_type() == U3_OPERATION ) {
                            int target_qbit_loc = gate_test->get_target_qbit();

                            if ( qbit_to_be_matched == target_qbit_loc ) {
                                found_match = true;
                                matching_gate = static_cast<U3*>(gate_test);
                                break;                             
                            }


                        }
                        
                        parameter_idx_loc = parameter_idx_loc + gate_test->get_parameter_num();
                    }

                    if ( found_match ) break;
        

                }

                if ( found_match == false ) {
                    // TODO: append a matching U3 gate to the very end of the circuit
                    std::string err("N_Qubit_Decomposition_adaptive::remove_trivial_gates: No matching U3 gate was found. Need to append a U3 gate to the end, but this functionality is not developed yet."); 
                    throw err;       
                }

                Matrix_real param1( &optimized_parameters_loc[parameter_idx_to_be_removed], 1, U_gate_to_be_removed->get_parameter_num() );
                Matrix U3_matrix1 = U_gate_to_be_removed->calc_one_qubit_u3(param1[0], param1[1], param1[2] );

                Matrix_real param2( &optimized_parameters_loc[parameter_idx_loc], 1, matching_gate->get_parameter_num() );
                Matrix U3_matrix2 = matching_gate->calc_one_qubit_u3(param2[0], param2[1], param2[2] );

                Matrix U3_prod = dot(U3_matrix2, U3_matrix1);

                optimized_parameters_loc[parameter_idx_to_be_removed] = 0.0;
                optimized_parameters_loc[parameter_idx_to_be_removed+1] = 0.0;
                optimized_parameters_loc[parameter_idx_to_be_removed+2] = 0.0;
                parameter_idx_to_be_removed = parameter_idx_to_be_removed + U_gate_to_be_removed->get_parameter_num();

                // calculate the new theta/2, phi, lambda parameters from U3_prod, and replace them in param2
                //  global phase on Umtx
                double ctheta3_over2 = std::sqrt(U3_prod[0].real*U3_prod[0].real+U3_prod[0].imag*U3_prod[0].imag); // cos( theta/2 )
                double stheta3_over2 = std::sqrt(U3_prod[2].real*U3_prod[2].real+U3_prod[2].imag*U3_prod[2].imag); // sin( theta/2 )
                double theta3_over2 = std::atan2(stheta3_over2,ctheta3_over2); // theta/2

		double alpha = std::atan2(U3_prod[0].imag,U3_prod[0].real); // the global phase

		double lambda3;
		double phi3;

		if (std::abs(stheta3_over2)<4e-8){
		    lambda3 = (std::atan2(U3_prod[3].imag,U3_prod[3].real)-alpha)/2;
		    phi3 = lambda3;
		}
		else {
		    lambda3 = std::atan2(-1*U3_prod[1].imag,-1*U3_prod[1].real)-alpha;
		    phi3 = std::atan2(U3_prod[2].imag,U3_prod[2].real)-alpha;
		}

                // the product U3 matrix
		Matrix U3_new = matching_gate->calc_one_qubit_u3(theta3_over2,phi3,lambda3);
		QGD_Complex16 global_phase_factor_new;
		global_phase_factor_new.real = std::cos(alpha);
		global_phase_factor_new.imag = std::sin(alpha);
		apply_global_phase_factor(global_phase_factor_new, U3_new);
		
                // test for the product U3 matrix
		if (std::sqrt((U3_new[3].real-U3_prod[3].real)*(U3_new[3].real-U3_prod[3].real)) + std::sqrt((U3_new[3].imag-U3_prod[3].imag)*(U3_new[3].imag-U3_prod[3].imag)) < 1e-8 && (stheta3_over2*stheta3_over2+ctheta3_over2*ctheta3_over2) > 0.99) {

                    // setting the resulting parameters if test passed
                    
                    param2[0] = theta3_over2;
                    param2[1] = phi3;
                    param2[2] = lambda3;
                    apply_global_phase_factor(global_phase_factor_new, Umtx);

		}
		/*
                N_Qubit_Decomposition_custom cDecomp_custom__( Umtx.copy(), qbit_num, false, config_copy, initial_guess);
                cDecomp_custom__.set_custom_gate_structure( gate_structure_loc );
                std::cout << "right before removing a trivial gate: " << cDecomp_custom__.optimization_problem( optimized_parameters_loc ) << std::endl;
                */
            }


            std::stringstream sstream;
            sstream << "N_Qubit_Decomposition_adaptive::remove_trivial_gates: Removing trivial gateblock" << std::endl;
            print(sstream, 3);
	    
               
            // remove gate from the structure
            int iteration_num_loc = 0;
            Gates_block* gate_structure_tmp = compress_gate_structure( gate_structure_loc, idx, optimized_parameters_loc, current_minimum_loc, iteration_num_loc );
	    number_of_iters += iteration_num_loc;

	    /*
            N_Qubit_Decomposition_custom cDecomp_custom_( Umtx.copy(), qbit_num, false, config_copy, initial_guess);
            cDecomp_custom_.set_custom_gate_structure( gate_structure_tmp );
	    std::cout << "after removing a trivial gate: " << cDecomp_custom_.optimization_problem( optimized_parameters_loc ) << std::endl;	    
            */
            optimized_parameters = optimized_parameters_loc;
            delete( gate_structure_loc );
            gate_structure_loc = gate_structure_tmp;
            layer_num = gate_structure_loc->get_gate_num();   
            break;            


          
            
        }

        parameter_idx += param_num;
         

        
    }
    
//std::cout << "N_Qubit_Decomposition_adaptive::remove_trivial_gates :" << gate_structure->get_gate_num() << " reduced to " << gate_structure_loc->get_gate_num() << std::endl;
    return gate_structure_loc;




}


/**
@brief Call to remove those parameters from the array, which correspond to gates that are about to be removed from the circuit.
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the parameters
@param layer_idx The layer to be removed from the circuit
*/
Matrix_real 
N_Qubit_Decomposition_adaptive::create_reduced_parameters( Gates_block* gate_structure, Matrix_real& optimized_parameters, int layer_idx ) {


    // determine the index of the parameter that is about to delete
    int gates_num = gate_structure->get_gate_num();
    int parameter_idx = 0;
    for ( int idx=0; idx<layer_idx; idx++) {    
    //for ( int idx=gates_num-1; idx>layer_idx; idx--) {
        Gate* gate = gate_structure->get_gate( idx );
        parameter_idx += gate->get_parameter_num();
    }


    Gate* gate = gate_structure->get_gate( layer_idx );
    int param_num_removed = gate->get_parameter_num();

    Matrix_real reduced_parameters(1, optimized_parameters.size() - param_num_removed );
    memcpy( reduced_parameters.get_data(), optimized_parameters.get_data(), (parameter_idx)*sizeof(double));
    memcpy( reduced_parameters.get_data()+parameter_idx, optimized_parameters.get_data()+parameter_idx+param_num_removed, (optimized_parameters.size()-parameter_idx-param_num_removed)*sizeof(double));


    return reduced_parameters;
}







/**
@brief Call to add adaptive layers to the gate structure stored by the class.
*/
void 
N_Qubit_Decomposition_adaptive::add_adaptive_layers() {

    add_adaptive_layers( this );

}

/**
@brief Call to add adaptive layers to the gate structure.
*/
void 
N_Qubit_Decomposition_adaptive::add_adaptive_layers( Gates_block* gate_structure ) {


    // create the new decomposing layer and add to the gate staructure
    Gates_block* layer = construct_adaptive_gate_layers();
    gate_structure->combine( layer );


}




/**
@brief Call to construct adaptive layers.
*/
Gates_block* 
N_Qubit_Decomposition_adaptive::construct_adaptive_gate_layers() {


    //The stringstream input to store the output messages.
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
            layer->add_adaptive(target_qbit_loc, control_qbit_loc);

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
                layer->add_adaptive(target_qbit_loc, control_qbit_loc);

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
N_Qubit_Decomposition_adaptive::add_finalyzing_layer() {

    add_finalyzing_layer( this );

}

/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
void 
N_Qubit_Decomposition_adaptive::add_finalyzing_layer( Gates_block* gate_structure ) {


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
             block->add_u3(idx, Theta, Phi, Lambda);
//        block->add_ry(idx);
    }


    // adding the opeartion block to the gates
    if ( gate_structure == NULL ) {
        throw ("N_Qubit_Decomposition_adaptive::add_finalyzing_layer: gate_structure is null pointer");
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
N_Qubit_Decomposition_adaptive::set_adaptive_gate_structure( std::string filename ) {

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
N_Qubit_Decomposition_adaptive::set_unitary_from_file( std::string filename ) {

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
N_Qubit_Decomposition_adaptive::set_unitary( Matrix& Umtx_new ) {

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
N_Qubit_Decomposition_adaptive::add_adaptive_gate_structure( std::string filename ) { 



    Matrix_real optimized_parameters_mtx_tmp;
    Gates_block* gate_structure_tmp = import_gate_list_from_binary(optimized_parameters_mtx_tmp, filename, verbose);

    if ( gates.size() > 0 ) {
        gate_structure_tmp->combine( static_cast<Gates_block*>(this) );

        release_gates();
        combine( gate_structure_tmp );
      

        Matrix_real optimized_parameters_mtx_tmp2( 1, optimized_parameters_mtx_tmp.size() + optimized_parameters_mtx.size() );
        
        memcpy( optimized_parameters_mtx_tmp2.get_data(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );
        memcpy( optimized_parameters_mtx_tmp2.get_data()+optimized_parameters_mtx.size(), optimized_parameters_mtx_tmp.get_data(), optimized_parameters_mtx_tmp.size()*sizeof(double) );
        
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
N_Qubit_Decomposition_adaptive::apply_imported_gate_structure() {

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
N_Qubit_Decomposition_adaptive::add_layer_to_imported_gate_structure() {


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







