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
/*! \file N_Qubit_Decomposition_Tree_Search.cpp
    \brief Class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#include "N_Qubit_Decomposition_Tree_Search.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
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

using namespace std;



size_t VectorHash::operator() (const matrix_base<int>& gcode) const {

    int n_ary_limit_max = 3;
    size_t hash = 0;

    for( int idx=0; idx<gcode.size(); idx++ ) {
        hash = hash + pow( idx, n_ary_limit_max) * gcode[idx];
    }

    return hash;
}





/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search() : Optimization_Interface() {


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
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {


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
    
    // construct the possible CNOT combinations within a single level
    // the number of possible CNOT connections netween the qubits (including topology constraints)
    int n_ary_limit_max = topology.size();
    
    possible_target_qbits = matrix_base<int>(1, n_ary_limit_max);
    possible_control_qbits = matrix_base<int>(1, n_ary_limit_max);    
    for( int element_idx = 0; element_idx<n_ary_limit_max; element_idx++ ) {

       matrix_base<int>& edge = topology[ element_idx ];
       possible_target_qbits[element_idx] = edge[0];
       possible_control_qbits[element_idx] = edge[1]; 
 
    }   





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
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {



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
    
    
    // construct the possible CNOT combinations within a single level
    // the number of possible CNOT connections netween the qubits (including topology constraints)
    int n_ary_limit_max = topology.size();
    
    possible_target_qbits = matrix_base<int>(1, n_ary_limit_max);
    possible_control_qbits = matrix_base<int>(1, n_ary_limit_max);    
    for( int element_idx = 0; element_idx<n_ary_limit_max; element_idx++ ) {

       matrix_base<int>& edge = topology[ element_idx ];
       possible_target_qbits[element_idx] = edge[0];
       possible_control_qbits[element_idx] = edge[1]; 
 
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


}

/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Tree_Search::~N_Qubit_Decomposition_Tree_Search() {

}





/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
*/
void
N_Qubit_Decomposition_Tree_Search::start_decomposition() {


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

    if (level_limit == 0 ) {
        std::stringstream sstream;
        sstream << "please increase level limit" << std::endl;
        print(sstream, 0);	
        return;
    }




    Gates_block* gate_structure_loc = determine_initial_gate_structure(optimized_parameters_mtx);
    





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
@brief Call determine the gate structure of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_Tree_Search::determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

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
   
   
    GrayCode gcode_best_solution3 = tabu_search_over_gate_structures();
return construct_gate_structure_from_Gray_code( gcode_best_solution3 );
current_minimum = std::numeric_limits<double>::max();
    //exit(1);
    
    GrayCode gcode_best_solution;
    double minimum_best_solution  = current_minimum; 

    for ( int level = 0; level <= level_max; level++ ) { 

        GrayCode&& gcode = tree_search_over_gate_structures( level );   

        if (current_minimum < minimum_best_solution) { 

            minimum_best_solution = current_minimum;
            gcode_best_solution   = gcode;  
            
        }

        if (current_minimum < optimization_tolerance_loc ) {
            break;
        }
        


    }    
    
    
    if (current_minimum > optimization_tolerance_loc) {
       std::stringstream sstream;
       sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
       print(sstream, 1);              
    }
gcode_best_solution.print_matrix();
//exit(1);

    return construct_gate_structure_from_Gray_code( gcode_best_solution );
       
}





/**
@brief Call to perform tree search over possible gate structures with a given tree search depth.
@param level_num The number of decomposing levels (i.e. the maximal tree depth)
@return Returns with the best Gray-code corresponding to the best circuit. (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
GrayCode 
N_Qubit_Decomposition_Tree_Search::tree_search_over_gate_structures( int level_num ){

    tbb::spin_mutex tree_search_mutex;
    
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }     
    
    
    if (level_num == 0){

        // empty Gray code describing a circuit without two-qubit gates
        GrayCode gcode;
        Gates_block* gate_structure_loc = construct_gate_structure_from_Gray_code( gcode );
        
        std::stringstream sstream;
        sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
        print(sstream, 1);



        N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization( gate_structure_loc );

        number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization           


        double current_minimum_tmp         = cDecomp_custom_random.get_current_minimum();
        sstream.str("");
        sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp;
        print(sstream, 1);
        
        
        if( current_minimum_tmp < current_minimum ) {
            current_minimum = current_minimum_tmp;
            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
        }
     
        //std::cout << "iiiiiiiiiiiiiiiiii " << current_minimum_tmp << std::endl;
        delete( gate_structure_loc );
        return gcode;
   
    }
  
     
    GrayCode gcode_best_solution;
    bool found_optimal_solution = false;
    
    
    
    // set the limits for the N-ary Gray counter
    
    int n_ary_limit_max = topology.size();
    matrix_base<int> n_ary_limits( 1, level_num ); //array containing the limits of the individual Gray code elements    
    memset( n_ary_limits.get_data(), n_ary_limit_max, n_ary_limits.size()*sizeof(int) );
    
    for( int idx=0; idx<n_ary_limits.size(); idx++) {
        n_ary_limits[idx] = n_ary_limit_max;
    }


    int64_t iteration_max = pow( (int64_t)n_ary_limit_max, level_num );
    
    
    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads;
    concurrency = concurrency < iteration_max ? concurrency : iteration_max;  


    int parallel = get_parallel_configuration();
       
    int64_t work_batch = 1;
    if( parallel==0) {
        work_batch = concurrency;
    }

//std::cout << "levels " << level_num << std::endl;
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
            
                if( found_optimal_solution ) {
                    return;
                }
        

    
                GrayCode&& gcode = gcode_counter.get();               
                
        
                Gates_block* gate_structure_loc = construct_gate_structure_from_Gray_code( gcode );
             
    

                // ----------- start the decomposition ----------- 
        
                std::stringstream sstream;
                sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
                print(sstream, 1);
                
                N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization( gate_structure_loc );
                
                delete( gate_structure_loc );
                gate_structure_loc = NULL;
        
                
                number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization  
    
                double current_minimum_tmp         = cDecomp_custom_random.get_current_minimum();
                sstream.str("");
                sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp;
                print(sstream, 1);
                
               

                //std::cout << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp << std::endl;
        
                {
                    tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};

                    if( current_minimum_tmp < current_minimum && !found_optimal_solution) {
                    
                        current_minimum     = current_minimum_tmp;                        
                        gcode_best_solution = gcode;

                        optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                    }
                    
                     
     
                    if ( current_minimum < optimization_tolerance_loc && !found_optimal_solution)  {            
                        found_optimal_solution = true;
                    } 
    
                }

 
                /*
                for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {
                    std::cout << gcode[gcode_idx] << ", ";
                }
                std::cout << current_minimum_tmp  << std::endl;
                */

                // iterate the Gray code to the next element
                int changed_index, value_prev, value;
                if ( gcode_counter.next(changed_index, value_prev, value) ) {
                    // exit from the for loop if no further gcode is present
                    break;
                }   
        
        
            }

        }
    
    });


    return gcode_best_solution;


}



/** 
@brief Perform tabu serach over gate structures
@return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
GrayCode 
N_Qubit_Decomposition_Tree_Search::tabu_search_over_gate_structures() {


    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }     

    // set the limits for the N-ary Gray code
    /*
    int n_ary_limit_max = topology.size();
    matrix_base<int> n_ary_limits( 1, levels ); //array containing the limits of the individual Gray code elements    
    memset( n_ary_limits.get_data(), n_ary_limit_max, n_ary_limits.size()*sizeof(int) );
    
    for( int idx=0; idx<n_ary_limits.size(); idx++) {
        n_ary_limits[idx] = n_ary_limit_max;
    }

*/
    GrayCode gcode;
/*
    // initiate Gray code to structure containing no CNOT gates
    for( int idx=0; idx<gcode.size(); idx++ ) {
        gcode[idx] = -1;
    }
*/

    GrayCode gcode_best_solution = gcode;


    std::uniform_real_distribution<double> unif(0.0,1.0);
    std::default_random_engine re;
    
    double inverz_temperature = 1.0;
    std::vector<GrayCode> possible_gate_structures;

    while( true ) {
  




        Gates_block* gate_structure_loc = construct_gate_structure_from_Gray_code( gcode );
             

        // ----------- start the decomposition ----------- 
        
        std::stringstream sstream;
        sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
        print(sstream, 1);
                
        N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization( gate_structure_loc );
                
        delete( gate_structure_loc );
        gate_structure_loc = NULL;
        
                
        number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization  
    
        double current_minimum_tmp         = cDecomp_custom_random.get_current_minimum();
        sstream.str("");
        sstream << "Optimization with " << gcode.size() << " levels converged to " << current_minimum_tmp;
        print(sstream, 1);

/*
        std::cout << current_minimum << " " << current_minimum_tmp << std::endl;
        gcode.print_matrix();
*/


        tested_gate_structures.insert( gcode ); 
        
        
             
        

        if( current_minimum_tmp < current_minimum || possible_gate_structures.size() == 1) {
            // accept the current gate structure in tabu search
                   
            current_minimum     = current_minimum_tmp;                        
            gcode_best_solution = gcode;
            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
            
            possible_gate_structures.clear();            
            insert_into_best_solution( gcode, current_minimum_tmp ); 

        }
        else {
            // accept the current gate structure in tabu search with a given probability

            double random_double = unif(re);            
            double number_to_test = exp( -inverz_temperature*(current_minimum_tmp-current_minimum) );

            if( random_double < number_to_test ) {
                //std::cout << "accepting worse solution " << current_minimum << " " << current_minimum_tmp << std::endl;
                // accept the intermediate solution
                current_minimum     = current_minimum_tmp;                        
                gcode_best_solution = gcode;
                optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                
                possible_gate_structures.clear();            
                insert_into_best_solution( gcode, current_minimum_tmp ); 
            }

        }
          
        if ( current_minimum < optimization_tolerance_loc )  {  
//std::cout << "solution found" << std::endl;
//gcode_best_solution.print_matrix();          
            break;
        } 
        
        
        if( possible_gate_structures.size() == 0 ) {
            // determine possible gate structures that can be obtained with a single change (i.e. changing one two-qubit block)
            possible_gate_structures = determine_muted_structures( gcode_best_solution );
            
        }
        
       
        
        if( possible_gate_structures.size() == 0 ) {
        
            while( best_solutions.size() > 0 ) {
            
                auto pair = best_solutions[0];
                
                gcode_best_solution = std::get<0>(pair);
                current_minimum     = std::get<1>(pair);
                
                possible_gate_structures = determine_muted_structures( gcode_best_solution );
                
                if( possible_gate_structures.size() > 0 ) {
                    break;
                }
            
            }              
        
        }
        
        if ( possible_gate_structures.size() == 0 ) {
            break;
        }
        
/*
        std::cout << "uuuuuuuuuuuuuuuuuuuuuuuu size:" << possible_gate_structures.size() << std::endl;
        for(  int idx=0; idx<possible_gate_structures.size(); idx++ ) {

            GrayCode& gcode = possible_gate_structures[ idx ];

            gcode.print_matrix();

        }

std::cout << "uuuuuuuuuuuuuuuuuuuuuuuu 2" << std::endl;
*/
//int levels_current = gcode.size();
        gcode = draw_gate_structure_from_list( possible_gate_structures );   
/*
if ( levels_current < gcode.size() ) {
std::cout << " increasing the gate structure" << std::endl;
}           
else if ( levels_current > gcode.size() ) {
std::cout << " decreasing the gate structure" << std::endl;
}
  */  

    }
    
    return gcode_best_solution;

}



/** 
@brief ????
@param ????
@return Returns with the ????
*/
void 
N_Qubit_Decomposition_Tree_Search::insert_into_best_solution( const GrayCode& gcode_, double minimum_ ) {


    for( auto it=best_solutions.begin(); it!=best_solutions.end(); it++ ) {
    
        double minimum = std::get<1>( *it );
        
        if( minimum > minimum_) {
            best_solutions.insert( it, std::make_pair(gcode_, minimum_) );
            
            if( best_solutions.size() > 5 ) {
                best_solutions.erase( best_solutions.end() - 1 );
            }
        }
    
    }



}


/** 
@brief ????
@param ????
@return Returns with the ????
*/
std::vector<GrayCode> 
N_Qubit_Decomposition_Tree_Search::determine_muted_structures( const GrayCode& gcode ) {


    std::vector<GrayCode> possible_structures_list;
    int n_ary_limit_max = topology.size();
/*
    std::cout << "ooooooooooooo " << n_ary_limit_max << std::endl;
    for( int idx=0; idx<topology.size(); idx++ ) {
        topology[idx].print_matrix();
    }
*/

    // modify current two-qubit blocks
    for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {

        for( int gcode_element=0; gcode_element<n_ary_limit_max; gcode_element++ ) {

            GrayCode gcode_modified = gcode.copy();
            gcode_modified[gcode_idx] = gcode_element;

            // add the modified Gray code if not present in the list of visited gate structures
            if( tested_gate_structures.count( gcode_modified ) == 0 ) {
                possible_structures_list.push_back( gcode_modified );
            }

        }
        
 
    }
    
    // generate structures with a less two-qubit blocks by one
    for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {
    
        GrayCode&& gcode_modified = gcode.remove_Digit( gcode_idx );
        
        // add the modified Gray code if not present in the list of visited gate structures
        if( tested_gate_structures.count( gcode_modified ) == 0 ) {
            possible_structures_list.push_back( gcode_modified );
        }
    
    }

    
    // generates structure with an extra two-qubit block
    GrayCode&& gcode_extended = gcode.add_Digit( n_ary_limit_max );
    
    for( int gcode_element=0; gcode_element<n_ary_limit_max; gcode_element++ ) {
    
        GrayCode gcode_modified = gcode_extended.copy();
        gcode_modified[ gcode_extended.size()-1 ] = gcode_element;
        
        // add the modified Gray code if not present in the list of visited gate structures
        if( tested_gate_structures.count( gcode_modified ) == 0 ) {
            possible_structures_list.push_back( gcode_modified );
        }
    
    }

    return possible_structures_list;

}



/** 
@brief ????
@param ????
@return Returns with the ????
*/
GrayCode
N_Qubit_Decomposition_Tree_Search::draw_gate_structure_from_list( const std::vector<GrayCode>& gcodes ) {

    if ( gcodes.size() == 0 ) {
	std::string err("N_Qubit_Decomposition_Tree_Search::draw_gate_structure_from_list: The list of gates structure is empty." );
        throw( err );
    }

    GrayCode gcode = gcodes[0];

    int levels = gcode.size();

    // the probability distribution is weighted by the number of two-qubit gates in the gate structure
    // the probability weights should be smaller if containing more two-qubit gates
    matrix_base<int> weights( gcodes.size(), 1 );
    
    int fact = 4;

    for( int gcode_idx=0; gcode_idx<gcodes.size(); gcode_idx++ ) {

        gcode = gcodes[ gcode_idx ];
        weights[ gcode_idx ] = fact*(levels);

        for( int gcode_element_idx=0; gcode_element_idx<gcode.size(); gcode_element_idx++ ) {
            if( gcode[gcode_element_idx] > -1 ) {
                weights[ gcode_idx ] = weights[ gcode_idx ] - fact;
            }
        }
    
    }

/*
    std::cout << "weights" << std::endl;
    weights.print_matrix();
*/
    // calculate the sum of weights to normalize for the probability distribution
    int weight_sum = 0;
    for( int idx=0; idx<weights.size(); idx++ ) {
        weight_sum = weight_sum + weights[idx];
    }


    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,weight_sum); // distribution in range [0, weight_sum]

    int random_num = dist(rng);
    int weight_sum_partial = 0;
    int chosen_idx = 0;
    for( int idx=0; idx<weights.size(); idx++ ) {

        weight_sum_partial = weight_sum_partial + weights[idx];

        if( random_num < weight_sum_partial ) {
            chosen_idx = idx;
            break;
        }

    }

   

    GrayCode chosen_gcode = gcodes[ chosen_idx ];
    gcodes.erase( gcodes.begin() + chosen_idx );

    return chosen_gcode;

}



/** 
@brief ????
@param ????
@return Returns with the ????
*/
GrayCode
N_Qubit_Decomposition_Tree_Search::mutate_gate_structure( const GrayCode& gcode ) {

    GrayCode mutated_gcode = gcode.copy();

    return mutated_gcode;

}




/**
@brief Call to perform the optimization on the given gate structure
@param gate_structure_loc The gate structure to be optimized
*/
N_Qubit_Decomposition_custom
N_Qubit_Decomposition_Tree_Search::perform_optimization(Gates_block* gate_structure_loc){


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
@brief  Call to construnc a gate structure corresponding to the configuration of the two-qubit gates described by the Gray code  
@param gcode The N-ary Gray code describing the configuration of the two-qubit gates.
@return Returns with the generated circuit
*/
Gates_block* 
N_Qubit_Decomposition_Tree_Search::construct_gate_structure_from_Gray_code( const GrayCode& gcode ) {


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

                            
    for (int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++) {      
            
        // add new 2-qbit block to the circuit
        add_two_qubit_block( gate_structure_loc, target_qbits[gcode_idx], control_qbits[gcode_idx]  );
    }
         
    // add finalyzing layer to the the gate structure
    add_finalyzing_layer( gate_structure_loc );
                
    return  gate_structure_loc;           

}




/**
@brief Call to add two-qubit building block (two single qubit rotation blocks and one two-qubit gate) to the circuit
@param gate_structure Appending the two-qubit building block to this circuit
@param target_qbit The target qubit of the two-qubit gate
@param control_qbit The control qubit of the two-qubit gate
*/
void
N_Qubit_Decomposition_Tree_Search::add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit) {
	
        if ( control_qbit >= qbit_num || target_qbit>= qbit_num ) {
            std::string error( "N_Qubit_Decomposition_Tree_Search::add_two_qubit_block: Label of control/target qubit should be less than the number of qubits in the register.");	        
            throw error;         
        }
        
        if ( control_qbit == target_qbit ) {
            std::string error( "N_Qubit_Decomposition_Tree_Search::add_two_qubit_block: Target and control qubits should be different");	        
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
        layer->add_cnot(target_qbit, control_qbit); 
        gate_structure->add_gate(layer);

}






/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
void 
N_Qubit_Decomposition_Tree_Search::add_finalyzing_layer( Gates_block* gate_structure ) {


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
        throw ("N_Qubit_Decomposition_Tree_Search::add_finalyzing_layer: gate_structure is null pointer");
    }
    else {
        gate_structure->add_gate( block );
    }


}



/**
@brief call to set Unitary from mtx
@param matrix to set over
*/
void 
N_Qubit_Decomposition_Tree_Search::set_unitary( Matrix& Umtx_new ) {

    Umtx = Umtx_new;

#ifdef __DFE__
    if( qbit_num >= 5 ) {
        upload_Umtx_to_DFE();
    }
#endif

}






