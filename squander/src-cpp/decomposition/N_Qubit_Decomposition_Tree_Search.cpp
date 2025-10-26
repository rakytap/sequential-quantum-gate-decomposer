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
#include <queue>
#include <complex>
#include <cmath>
#include <algorithm>

#include "X.h"

#include <time.h>
#include <stdlib.h>

#include <iostream>

#ifdef __DFE__
#include "common_DFE.h"
#endif

using Discovery = std::vector<std::pair<std::vector<int>, GrayCode>>;
using LevelResult = std::tuple<std::set<std::vector<int>>, std::map<std::vector<int>, GrayCode>, std::vector<std::pair<std::vector<int>, GrayCode>>>;

// Initialize at depth 0 (identity only)
static inline LevelResult enumerate_unordered_cnot_BFS_level_init(int n) {
    std::vector<int> I(n, 0);
    for (int i = 0; i < n; ++i) I[i] = 1 << i;
    std::set<std::vector<int>> visited;
    visited.emplace(I);
    std::map<std::vector<int>, GrayCode> seq_pairs_of;
    seq_pairs_of.emplace(I, GrayCode{});
    // emit the root
    Discovery out_res;
    out_res.emplace_back(I, GrayCode{});
    return LevelResult{visited, seq_pairs_of, out_res};
}

// One expansion “level”: pop all items from L.q, try all unordered pairs in topology
// (both directions internally), record first-time discoveries and emit them immediately
// (BFS ⇒ minimal depth).
static inline LevelResult enumerate_unordered_cnot_BFS_level_step(
        LevelInfo& L,
        const std::vector<matrix_base<int>>& topology)
{
    auto& [visited, seq_pairs_of, q] = L;
    std::map<std::vector<int>, GrayCode> new_seq_pairs_of;
    Discovery out_res;
    while (!q.empty()) {
        auto A = q.back();
        q.pop_back();

        const auto& last_pairs = seq_pairs_of.at(A);
        for (int p = 0; p < (int)topology.size(); ++p) {
            // try both directions
            // ensure p is unordered i<j; assume caller provides that
            std::pair<int, int> m1 = {topology[p][0], topology[p][1]};
            std::pair<int, int> m2 = {topology[p][1], topology[p][0]};

            for (auto mv : {m1, m2}) {
                std::vector<int> B = A;
                if (mv.first != mv.second) {
                    B[mv.second] ^= B[mv.first];
                }

                if (visited.find(B) != visited.end()) {
                    continue; // discovered already (at minimal or earlier depth)
                }
                visited.emplace(B);

                // build sequences
                auto seqp = last_pairs.add_Digit(topology.size());
                seqp[seqp.size() - 1] = p;

                new_seq_pairs_of.emplace(B, std::move(seqp));

                // emit discovery: (depth+1, B, seq_pairs_of[B], seq_dir_of[B])
                const auto& ref_pairs = new_seq_pairs_of.at(B);
                out_res.emplace_back(B, ref_pairs);
            }
        }
    }
    return LevelResult{visited, new_seq_pairs_of, out_res};
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
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search( Matrix Umtx_in, int qbit_num_in, std::map<std::string, Config_Element>& config, int accelerator_num )
    : N_Qubit_Decomposition_Tree_Search(Umtx_in, qbit_num_in, {}, config, accelerator_num) {    
}



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param topology_in A list of <target_qubit, control_qubit> pairs describing the connectivity between qubits.
@param config std::map conatining custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search( Matrix Umtx_in, int qbit_num_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {



    // set the level limit
    level_limit = 0;

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

    Gates_block* gate_structure_loc = determine_gate_structure(optimized_parameters_mtx);
    


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

    decomposition_error = current_minimum;
	
	
#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif
}



/**
@brief Call determine the gate structure of the decomposing circuit. 
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
N_Qubit_Decomposition_Tree_Search::determine_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {


    double optimization_tolerance_loc;
    long long level_max; 
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  

    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }      
    
    if  (config.count("tree_level_max") > 0 ){
        config["tree_level_max"].get_property( level_max );
    } 
    else {
        level_max = 14;
    }
   
   
     level_limit = (int)level_max;
     
    if (level_limit == 0 ) {
        std::string error( "please increase level limit");	        
        throw error;      
    }

    GrayCode best_solution;
    double minimum_best_solution  = current_minimum; 
    LevelInfo li;
    auto all_cuts = unique_cuts(qbit_num);
    std::map<std::pair<int, int>, std::vector<int>> pair_affects;
    for (const auto& pair : topology) {
        std::vector<int> cuts;
        for (int i = 0; i < all_cuts.size(); ++i) {
            const auto& A = all_cuts[i];
            if ((std::find(A.begin(), A.end(), pair[0]) != A.end()) ^ (std::find(A.begin(), A.end(), pair[1]) != A.end())) {
                cuts.push_back(i);
            }
        }
        pair_affects[std::pair<int, int>(pair[0], pair[1])] = std::move(cuts);
    }
    CutInfo ci = std::make_tuple(all_cuts, pair_affects, std::map<GrayCode, std::vector<std::pair<int, double>>>());

    for ( int level = 0; level <= level_limit; level++ ) { 

        auto [best_at_level, nextli, nextprefixes] = tree_search_over_gate_structures( level, li, ci );   
        li.swap(nextli);
        std::get<2>(ci) = std::move(nextprefixes);

        if (current_minimum < minimum_best_solution) { 

            minimum_best_solution = current_minimum;
            best_solution   = best_at_level;
            
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

    return construct_gate_structure_from_Gray_code( best_solution );
       
}





/**
@brief Call to perform tree search over possible gate structures with a given tree search depth.
@param level_num The number of decomposing levels (i.e. the maximal tree depth)
@return Returns with the best Gray-code corresponding to the best circuit. (The associated gate structure can be costructed by function construct_gate_structure_from_Gray_code)
*/
std::tuple<GrayCode, LevelInfo, std::map<GrayCode, std::vector<std::pair<int, double>>>>
N_Qubit_Decomposition_Tree_Search::tree_search_over_gate_structures( int level_num, LevelInfo& li, CutInfo& ci ){

    tbb::spin_mutex tree_search_mutex;
    
    auto [all_cuts, pair_affects, prefixes] = ci;

    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }     
    
    GrayCode best_solution;
    volatile bool found_optimal_solution = false;


    const auto& [visited, seq_pairs_of, out_res] = level_num == 0 ? enumerate_unordered_cnot_BFS_level_init(qbit_num) : enumerate_unordered_cnot_BFS_level_step(li, topology);

    std::set<GrayCode> pairs_reduced;
    for ( const auto& item : out_res ) {
        pairs_reduced.insert( item.second );
    }
    std::vector<GrayCode> all_pairs = std::vector<GrayCode>(pairs_reduced.begin(), pairs_reduced.end());
    std::vector<std::pair<GrayCode, std::vector<std::pair<int, double>>>> all_osr_results;
    int64_t iteration_max = all_pairs.size();
    all_osr_results.reserve(iteration_max);


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

            for (int64_t iter_idx=initial_offset; iter_idx<offset_max+1; iter_idx++ ) {
                const auto& solution = all_pairs[iter_idx];
                if( found_optimal_solution ) {
                    return;
                }
        

    

                // ----------------------------------------------------------------                                
                std::vector<std::vector<std::pair<int, double>>> osr_results;
                osr_results.reserve(2);
                for (int revpass = 0; revpass < 2; revpass++) {
                    Gates_block* gate_structure_loc;
                    if (revpass == 0) gate_structure_loc = construct_gate_structure_from_Gray_code( solution );
                    else {
                        GrayCode reversed_solution = solution.copy();
                        std::reverse(reversed_solution.data, reversed_solution.data+reversed_solution.size());
                        if (reversed_solution == solution) continue; // already optimized in the forward pass
                        gate_structure_loc = construct_gate_structure_from_Gray_code( reversed_solution );
                    }
                
        

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
                    sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp << std::endl;
                    print(sstream, 1);

                    auto U = Umtx.copy();
                    auto params = cDecomp_custom_random.get_optimized_parameters();
                    cDecomp_custom_random.apply_to(params, U);
                    std::vector<std::pair<int, double>> osr_result;
                    osr_result.reserve(all_cuts.size());
                    for (const auto& cut : all_cuts) {
                        osr_result.emplace_back(operator_schmidt_rank(U.data, qbit_num, cut));
                    }
                    osr_results.emplace_back(std::move(osr_result));


                    //std::cout << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp << std::endl;
            
                    {
                        tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};

                        if( current_minimum_tmp < current_minimum && !found_optimal_solution) {
                        
                            current_minimum     = current_minimum_tmp;                        
                            best_solution = solution;

                            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                        }
                        
                        
        
                        if ( current_minimum < optimization_tolerance_loc && !found_optimal_solution)  {            
                            found_optimal_solution = true;
                        } 
                    }
                } // end of revpass loop

                auto lastprefix = solution.size() != 0 ? prefixes.at(solution.remove_Digit(solution.size()-1)) : std::vector<std::pair<int, double>>();
                std::vector<int> check_cuts;
                if (solution.size() != 0) check_cuts = pair_affects.at(std::pair<int, int>(possible_target_qbits[solution[solution.size()-1]], possible_control_qbits[solution[solution.size()-1]]));
                else {
                    check_cuts.resize(all_cuts.size());
                    std::iota(check_cuts.begin(), check_cuts.end(), 0);
                }
                auto best_osr = *std::min_element(osr_results.begin(), osr_results.end(), [&check_cuts](const std::vector<std::pair<int, double>>& a, const std::vector<std::pair<int, double>>& b) {
                    int max_ar = 0, sum_ar = 0, max_br = 0, sum_br = 0;
                    double sum_as0 = 0, sum_bs0 = 0;
                    for (int i : check_cuts) {
                        max_ar = std::max(max_ar, a[i].first);
                        sum_ar += a[i].first;
                        max_br = std::max(max_br, b[i].first);
                        sum_br += b[i].first;
                        sum_as0 += a[i].second;
                        sum_bs0 += b[i].second;
                    }
                    if (max_ar != max_br) return max_ar < max_br;
                    if (sum_ar != sum_br) return sum_ar < sum_br;
                    return sum_as0 < sum_bs0;
                });
                if (solution.size() == 0 || !(std::all_of(check_cuts.begin(), check_cuts.end(), [&lastprefix, &best_osr](int i) {
                    return lastprefix[i].first < best_osr[i].first;
                }) || (std::all_of(check_cuts.begin(), check_cuts.end(), [&lastprefix, &best_osr](int i) {
                    return lastprefix[i].first == best_osr[i].first;
                }) && std::any_of(check_cuts.begin(), check_cuts.end(), [&lastprefix, &best_osr](int i) {
                    return lastprefix[i].second < best_osr[i].second;
                }))))
                {
                    tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};
                    all_osr_results.emplace_back(std::move(solution.copy()), std::move(best_osr));
                }

                /*
                for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {
                    std::cout << gcode[gcode_idx] << ", ";
                }
                std::cout << current_minimum_tmp  << std::endl;
                */

        
        
            }

        }
    
    });

    std::sort(all_osr_results.begin(), all_osr_results.end(), [](const std::pair<GrayCode, std::vector<std::pair<int, double>>>& a, const std::pair<GrayCode, std::vector<std::pair<int, double>>>& b) {
        int max_ar = 0, sum_ar = 0;
        double sum_as0 = 0;
        for (const auto& [rnk, s0] : a.second) {
            max_ar = std::max(max_ar, rnk);
            sum_ar += rnk;
            sum_as0 += s0;
        }
        int max_br = 0, sum_br = 0;
        double sum_bs0 = 0;
        for (const auto& [rnk, s0] : b.second) {
            max_br = std::max(max_br, rnk);
            sum_br += rnk;
            sum_bs0 += s0;
        }
        if (max_ar != max_br) return max_ar < max_br;
        if (sum_ar != sum_br) return sum_ar < sum_br;
        return sum_as0 < sum_bs0;
    });
    long long beam_width = all_osr_results.size();
    if ( config.count("beam") > 0 ) {
        config["beam"].get_property( beam_width );  
    }
    beam_width = std::min<long long>(beam_width, all_osr_results.size());
    std::map<GrayCode, std::vector<std::pair<int, double>>> nextprefixes;
    for (long i = 0; i < beam_width; i++) {
        const auto& item = all_osr_results[i];
        nextprefixes[item.first] = item.second;
    }
    std::vector<std::vector<int>> next_q;
    next_q.reserve(out_res.size());
    for ( auto it = out_res.crbegin(); it != out_res.crend(); ++it ) {
        if ( nextprefixes.find( it->second ) == nextprefixes.end() ) {
            continue;
        }        
        next_q.push_back( it->first );
    }
    return std::make_tuple(best_solution, std::make_tuple(visited, seq_pairs_of, next_q), nextprefixes);


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
        layer->add_rz(target_qbit);
        layer->add_ry(target_qbit);
        layer->add_rz(target_qbit);     

        layer->add_rz(control_qbit);
        layer->add_ry(control_qbit);
        layer->add_rz(control_qbit);

        //layer->add_u3(target_qbit);
        //layer->add_u3(control_qbit);
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
        //block->add_rz(idx);
        //block->add_ry(idx);
        //block->add_rz(idx); 
        block->add_u3(idx);
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






