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

#include "N_Qubit_Decomposition_Tabu_Search.h"
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



/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Tabu_Search::N_Qubit_Decomposition_Tabu_Search() : N_Qubit_Decomposition_Tree_Search() {


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
N_Qubit_Decomposition_Tabu_Search::N_Qubit_Decomposition_Tabu_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : N_Qubit_Decomposition_Tree_Search( Umtx_in, qbit_num_in, level_limit_in, config, accelerator_num) {


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
N_Qubit_Decomposition_Tabu_Search::N_Qubit_Decomposition_Tabu_Search( Matrix Umtx_in, int qbit_num_in, int level_limit_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : N_Qubit_Decomposition_Tree_Search( Umtx_in, qbit_num_in, level_limit_in, topology_in, config, accelerator_num ) {

    // A string labeling the gate operation
    name = "Tabu_Search";

}

/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Tabu_Search::~N_Qubit_Decomposition_Tabu_Search() {

}





/**
@brief Call determine the gate structure of the decomposing circuit. 
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_Tabu_Search::determine_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {


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

    level_limit = (int)level_max;
   
    GrayCode gcode_best_solution = tabu_search_over_gate_structures();


    if (current_minimum > optimization_tolerance_loc) {
       std::stringstream sstream;
       sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
       print(sstream, 1);              
    }
    
    return construct_gate_structure_from_Gray_code( gcode_best_solution );

}




/** 
@brief Perform tabu serach over gate structures
@return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
GrayCode 
N_Qubit_Decomposition_Tabu_Search::tabu_search_over_gate_structures() {


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
        
        
             
        

        if( current_minimum_tmp < current_minimum) {
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
            possible_gate_structures = determine_mutated_structures( gcode_best_solution );
            
        }
        
       
        
        if( possible_gate_structures.size() == 0 ) {

            while( best_solutions.size() > 0 ) {
            
                auto pair = best_solutions[0];
                best_solutions.erase( best_solutions.begin() );
                
                gcode_best_solution = std::get<0>(pair);
                current_minimum     = std::get<1>(pair);
                
                possible_gate_structures = determine_mutated_structures( gcode_best_solution );
          
                if( possible_gate_structures.size() > 0 || best_solutions.size() == 0 ) {
                    break;
                }
            
            }              
        
        }
        
        if ( possible_gate_structures.size() == 0 ) {
            std::cout << "tttttttttttttttttttttttttttttt " << best_solutions.size() <<  std::endl;
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
@brief Call to store a given solution among the best ones.
@param gcode_ The Gray code encoding the gate structure
@param minimum_ The achieved cost function minimum with the given gate structure
*/
void 
N_Qubit_Decomposition_Tabu_Search::insert_into_best_solution( const GrayCode& gcode_, double minimum_ ) {

//std::cout << "N_Qubit_Decomposition_Tabu_Search::insert_into_best_solution a " << best_solutions.size()  << " " << minimum_ << std::endl;

    if ( best_solutions.size() == 0 ) {
        best_solutions.insert( best_solutions.begin(), std::make_pair(gcode_, minimum_) );
    }

    for( auto it=best_solutions.begin(); it!=best_solutions.end(); it++ ) {
    
        double minimum = std::get<1>( *it );
        
        if( minimum > minimum_) {
            best_solutions.insert( it, std::make_pair(gcode_, minimum_) );            
            break;
        }
    
    }

    if( best_solutions.size() > 40 ) {
        best_solutions.erase( best_solutions.end() - 1 );
    }

//std::cout << "N_Qubit_Decomposition_Tabu_Search::insert_into_best_solution b " << best_solutions.size()  <<  std::endl;

}


/** 
@brief Call to generate a list of mutated gate structures. In each mutation a sigle two-qubit gate block is changed, added or removed.
@param gcode The Gray code encoding the gate structure around which we mutate the structure.
@return Returns with the list of modified gray code encoding the gate structures
*/
std::vector<GrayCode> 
N_Qubit_Decomposition_Tabu_Search::determine_mutated_structures( const GrayCode& gcode ) {


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


    if ( gcode.size() == level_limit ) {
        // dont add further two-qubit layer
        return possible_structures_list;
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
@brief Call to sample a gate structure from a list of gate structures to test in the optimization process 
@param gcodes The list of possible Gray codes encoding the gate structures.
@return Returns with the sampled Gray code. The chosen Gray code is removed from the input list.
*/
GrayCode
N_Qubit_Decomposition_Tabu_Search::draw_gate_structure_from_list( std::vector<GrayCode>& gcodes ) {

    if ( gcodes.size() == 0 ) {
	std::string err("N_Qubit_Decomposition_Tabu_Search::draw_gate_structure_from_list: The list of gates structure is empty." );
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






