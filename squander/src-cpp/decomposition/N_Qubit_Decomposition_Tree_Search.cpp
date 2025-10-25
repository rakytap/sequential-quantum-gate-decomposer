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

#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
#define lapack_int     int
#define lapack_complex_double   double _Complex
extern "C" lapack_complex_double lapack_make_complex_double(double re, double im);
extern "C" lapack_int LAPACKE_zgesvd( int matrix_order, char jobu, char jobvt,
                            lapack_int m, lapack_int n, lapack_complex_double* a,
                            lapack_int lda, double* s, lapack_complex_double* u,
                            lapack_int ldu, lapack_complex_double* vt,
                            lapack_int ldvt, double* superb );

// Helper: extract bits at positions 'pos' from integer x into a packed integer (LSB order)
static inline int extract_bits(int x, const std::vector<int>& pos) {
    int y = 0, k = 0;
    for (int p : pos) { y |= ((x >> p) & 1) << k; ++k; }
    return y;
}

// Index: row-major 2^n x 2^n
static inline size_t rm_idx(int row, int col, int N) {
    return (size_t)row * (size_t)N + (size_t)col;
}

//https://www.sciencedirect.com/science/article/pii/S0024379518303446
//https://arxiv.org/abs/2007.02490
//https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.062430
//https://arxiv.org/pdf/2111.03132
// Build the (dA*dA) x (dB*dB) OSR matrix M for cut A|B from U (2^n x 2^n), row-major.
// M_{ (a' * dA + a), (b' * dB + b) } = U_{ (a',b'), (a,b) }.
static void build_osr_matrix(const QGD_Complex16* U, int n,
                             const std::vector<int>& A, // qubits on A
                             std::vector<QGD_Complex16>& M, int& m_rows, int& m_cols)
{
    std::vector<int> A_sorted = A;
    std::sort(A_sorted.begin(), A_sorted.end());
    std::vector<int> B;
    B.reserve(n - (int)A_sorted.size());
    for (int q = 0; q < n; ++q)
        if (!std::binary_search(A_sorted.begin(), A_sorted.end(), q)) B.push_back(q);

    const int dA = 1 << (int)A_sorted.size();
    const int dB = 1 << (n - (int)A_sorted.size());
    const int N  = 1 << n;

    m_rows = dA * dA;
    m_cols = dB * dB;
    M.assign((size_t)m_rows * (size_t)m_cols, QGD_Complex16{0.0, 0.0});

    // Row-major indexing: U[in + out*N] is element (in, out)
    for (int in = 0; in < N; ++in) {
        const int a  = extract_bits(in, A_sorted) * dA;
        const int b  = extract_bits(in, B) * dB;
        for (int out = 0; out < N; ++out) {
            const int ap = extract_bits(out, A_sorted);
            const int bp = extract_bits(out, B);
            const int r = a + ap;   // row in M
            const int c = b + bp;   // col in M
            M[rm_idx(r, c, m_cols)] = U[(size_t)in + (size_t)out * (size_t)N];
        }
    }
}

// Numerical rank via LAPACKE_zgesdd (SVD)
static int numerical_rank_osr(const std::vector<QGD_Complex16>& M, int m_rows, int m_cols, double tol)
{
    // Copy M because LAPACK overwrites input
    std::vector<lapack_complex_double> A(M.size());
    for (size_t i=0;i<M.size();++i) A[i] = lapack_make_complex_double(M[i].real, M[i].imag);

    std::vector<double> S(std::min(m_rows, m_cols));
    std::vector<double> superb(std::max(1, std::min(m_rows, m_cols) - 1));  // REQUIRED for complex *gesvd
    // We don’t need U/V; job='N' for economy; gesvd is fine too.
    int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR,
                              'N','N',
                              m_rows, m_cols,
                              A.data(), m_cols,
                              S.data(),
                              nullptr, 1,
                              nullptr, 1,
                              superb.data());
    if (info != 0) return 0; // fall back safely

    //std::copy(S.begin(), S.end(), std::ostream_iterator<double>(std::cout, " ")); std::cout << std::endl;
    int rnk = 0;
    for (double s : S) if (s > tol) ++rnk;
    return rnk;
}

// Public: operator-Schmidt rank across cut A|B
int operator_schmidt_rank(const QGD_Complex16* U, int n,
                          const std::vector<int>& A_qubits,
                          double tol = 1e-10)
{
    std::vector<QGD_Complex16> M;
    int mr=0, mc=0;
    build_osr_matrix(U, n, A_qubits, M, mr, mc);
    return numerical_rank_osr(M, mr, mc, tol);
}

// base-2 logarithm, rounding down
static inline uint32_t lg_down(uint32_t v) {
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
                                            r |= (v >> 1);
    return r;
}

// base-2 logarithm, rounding up
static inline uint32_t lg_up(uint32_t x) {
    return lg_down(x - 1) + 1;
}

// Lower bound on remaining CNOTs: max_A ceil(log2(OSR_A(U)))
int osr_cnot_lower_bound(const QGD_Complex16* U, int n,
                         const std::vector<std::vector<int>>& cuts,
                         double tol = 1e-10)
{
    int h = 0;
    for (const auto& A : cuts) {
        int r = operator_schmidt_rank(U, n, A, tol);
        if (r > 1) {
            int hb = lg_up(r);
            if (hb > h) h = hb;
        }
    }
    return h;
}

//Strong default is 1|all rest cuts and one "balanced" cut e.g. {0, 1} for n=4, {0, 1} or {0, 2} for n=5
std::vector<std::vector<int>> default_cuts(int n) {
    std::vector<std::vector<int>> cuts;
    for (int q=0;q<n;++q) cuts.push_back({q});
    if (n>=4) cuts.push_back({0,1});
    return cuts;
}
// Return all non-empty, non-full subsets of {0,…,n-1}.
std::vector<std::vector<int>> all_cuts(int n) {
    std::vector<std::vector<int>> cuts;

    if (n <= 1) return cuts;

    const int N = 1 << n;
    for (int mask = 1; mask < N - 1; ++mask) {
        std::vector<int> cut;
        cut.reserve(n);
        for (int q = 0; q < n; ++q)
            if (mask & (1 << q))
                cut.push_back(q);
        cuts.push_back(std::move(cut));
    }
    return cuts;
}


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

    for ( int level = 0; level <= level_limit; level++ ) { 

        const auto& res = tree_search_over_gate_structures( level, li );   
        li = res.second;

        if (current_minimum < minimum_best_solution) { 

            minimum_best_solution = current_minimum;
            best_solution   = res.first;  
            
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
std::pair<GrayCode, LevelInfo>
N_Qubit_Decomposition_Tree_Search::tree_search_over_gate_structures( int level_num, LevelInfo li ){

    tbb::spin_mutex tree_search_mutex;
    
    
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
    int64_t iteration_max = all_pairs.size();

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
        
                Gates_block* gate_structure_loc = construct_gate_structure_from_Gray_code( solution );
            
    

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


 
                /*
                for( int gcode_idx=0; gcode_idx<gcode.size(); gcode_idx++ ) {
                    std::cout << gcode[gcode_idx] << ", ";
                }
                std::cout << current_minimum_tmp  << std::endl;
                */

        
        
            }

        }
    
    });

    std::vector<std::vector<int>> next_q;
    for ( const auto& item : out_res ) {
        next_q.push_back( item.first );
    }
    return std::pair<GrayCode, LevelInfo>(best_solution, std::make_tuple(visited, seq_pairs_of, next_q));


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
/*
layer->add_rz(target_qbit);
layer->add_ry(target_qbit);
layer->add_rz(target_qbit);     

layer->add_rz(control_qbit);
layer->add_ry(control_qbit);
layer->add_rz(control_qbit);     
*/

        layer->add_u3(target_qbit);
        layer->add_u3(control_qbit);
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






