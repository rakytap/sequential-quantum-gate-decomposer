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
#include "n_aryGrayCodeCounter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <unordered_map>

/**
@brief Structure containing the result of a BFS level enumeration.
This structure contains the visited states, sequence pairs, and the output results from enumerating a single BFS level.
*/
struct LevelResult {
    /// Set of visited states (represented as vectors of integers)
    std::set<std::vector<int>> visited;
    /// Map from state vectors to their corresponding Gray code sequences
    std::map<std::vector<int>, GrayCodeCNOT> seq_pairs_of;
    /// Vector of output results (discoveries) from the BFS level enumeration
    std::vector<std::pair<std::vector<int>, GrayCodeCNOT>> out_res;
};

using Discovery = std::vector<std::pair<std::vector<int>, GrayCodeCNOT>>;

/**
@brief Initialize the breadth-first search (BFS) enumeration at depth 0 (identity state only).
This function sets up the initial state for BFS enumeration of CNOT gate structures. At depth 0,
the state represents the identity operation (no CNOT gates applied), where each qubit is in its
own computational basis state. The function creates the initial state vector I, where each element
I[i] = 2^i represents the i-th qubit's basis state, marks it as visited, and initializes the
sequence pairs mapping with an empty Gray code.

@param n The number of qubits in the system
@return Returns a LevelResult structure containing:
        - visited: Set containing the initial identity state I
        - seq_pairs_of: Map from the identity state to an empty Gray code sequence
        - out_res: Vector containing a single discovery pair (I, empty Gray code)
@note This function is the starting point for BFS enumeration. The identity state I is represented
      as a vector where I[i] = 2^i, which corresponds to the i-th qubit being in state |1⟩ while
      all others are in state |0⟩.
*/
static inline LevelResult enumerate_unordered_cnot_BFS_level_init(int n) {

    std::vector<int> I(n, 0);
    for (int i = 0; i < n; ++i)
        I[i] = 1 << i;
    std::set<std::vector<int>> visited;
    visited.emplace(I);
    std::map<std::vector<int>, GrayCodeCNOT> seq_pairs_of;
    seq_pairs_of.emplace(I, GrayCodeCNOT{});
    // emit the root
    Discovery out_res;
    out_res.emplace_back(I, GrayCodeCNOT{});

    LevelResult result;
    result.visited = std::move(visited);
    result.seq_pairs_of = std::move(seq_pairs_of);
    result.out_res = std::move(out_res);
    return result;
}

// Return true iff 'seq' (list of CNOT pairs) equals the canonical
// Kahn topological order under the tie-breaker: lexicographic by pair,
// then by original index (to stabilize identical pairs).
static int canonical_prefix_ok(const GrayCodeCNOT& path, const std::vector<matrix_base<int>>& topology) {
    const int m = static_cast<int>(path.size());
    if (m <= 1)
        return -1;

    // 2) per-qubit serial constraints: edge u->v if ops u,v share a qubit and u < v
    std::vector<std::vector<int>> succ(m);
    std::vector<int> indeg(m, 0);
    std::unordered_map<int, int> last_on; // qubit -> last op index touching it
    last_on.reserve(m * 2);

    for (int k = 0; k < m; ++k) {
        const int a = topology[path.data[k]][0];
        const int b = topology[path.data[k]][1];
        for (int q : {a, b}) {
            std::unordered_map<int, int>::iterator it = last_on.find(q);
            if (it != last_on.end()) {
                int prev = it->second;
                succ[prev].push_back(k);
                ++indeg[k];
                it->second = k;
            } else {
                last_on.emplace(q, k);
            }
        }
    }

    // 3) deterministic Kahn with min-heap by (pair, index)
    struct Node {
        std::pair<int, int> p;
        int idx;
    };
    struct Cmp {
        bool operator()(const Node& a, const Node& b) const {
            if (a.p != b.p)
                return a.p > b.p; // lexicographically smaller first
            return a.idx > b.idx; // then by original index
        }
    };
    std::priority_queue<Node, std::vector<Node>, Cmp> pq;
    for (int k = 0; k < m; ++k)
        if (indeg[k] == 0)
            pq.push(Node{std::make_pair(topology[path.data[k]][0], topology[path.data[k]][1]), k});

    // 4) walk canonical order and require it matches the given prefix exactly
    for (int pos = 0; pos < m; ++pos) {
        if (pq.empty())
            return pos; // malformed (shouldn’t happen)
        Node u = pq.top();
        pq.pop();
        if (u.idx != pos)
            return pos; // deviation: not canonical

        for (int v : succ[u.idx]) {
            if (--indeg[v] == 0)
                pq.push(Node{std::make_pair(topology[path.data[v]][0], topology[path.data[v]][1]), v});
        }
    }
    return -1;
}

static int is_unique_structure(const GrayCodeCNOT& path, const std::vector<matrix_base<int>>& topology) {
    for (int idx = 0; idx < path.size() - 3; idx++) {
        if (path.data[idx] == path.data[idx + 1] && path.data[idx] == path.data[idx + 2] && path.data[idx] == path.data[idx + 3]) {
            return false; // avoid more than 3 repeated CNOTs
        }
    }
    return canonical_prefix_ok(path, topology) < 0; // not canonical prefix
}

/**
@brief Perform one expansion level of breadth-first search (BFS) enumeration over CNOT gate structures.
This function processes all states in the current BFS level queue, applies all possible CNOT operations
from the topology, and discovers new states at the next depth level. It maintains the BFS property that
states are discovered at their minimal depth, ensuring optimal exploration of the gate structure space.

The function operates in two modes:
- When use_gl=true (Gray-Lin mode): Applies CNOT operations directly to state vectors using XOR operations,
  tracking actual quantum states reached by the circuit.
- When use_gl=false: Builds Gray code sequences representing gate orderings, with additional constraints
  to avoid repeated CNOTs (max 3 consecutive) and ensure canonical ordering.

@param L LevelInfo reference containing the current BFS state:
         - visited: Set of states already discovered (modified to include new discoveries)
         - seq_pairs_of: Map from states to their Gray code sequences (used for lookups)
         - q: Queue of states to process at the current level (emptied during processing)
@param topology Vector of CNOT pairs (target, control) representing allowed qubit connections.
                Each element is a matrix_base<int> with two elements [target, control].
@param use_gl If true, uses Gray-Lin mode (applies CNOTs directly to states).
              If false, builds sequence-based representations with canonical ordering constraints.
@return Returns a LevelResult structure containing:
        - visited: Updated set of visited states (includes all newly discovered states)
        - seq_pairs_of: Map from newly discovered states to their extended Gray code sequences
        - out_res: Vector of discovery pairs (state, Gray code) for all newly found states
@note The function modifies the input LevelInfo structure L by updating visited states and clearing
      the queue. New states are discovered by applying CNOT operations: B[target] ^= B[control] in
      Gray-Lin mode, or by extending Gray code sequences in sequence mode.
*/
static inline LevelResult enumerate_unordered_cnot_BFS_level_step(LevelInfo& L,
                                                                  const std::vector<matrix_base<int>>& topology,
                                                                  bool use_gl = true) {
    std::set<std::vector<int>>& visited = L.visited;
    std::map<std::vector<int>, GrayCodeCNOT>& seq_pairs_of = L.seq_pairs_of;
    std::vector<std::vector<int>>& q = L.q;
    std::map<std::vector<int>, GrayCodeCNOT> new_seq_pairs_of;
    Discovery out_res;
    while (!q.empty()) {

        std::vector<int> A = q.back();
        q.pop_back();

        const GrayCodeCNOT& last_pairs = seq_pairs_of.at(A);
        for (int p = 0; p < (int)topology.size(); ++p) {
            // try both directions
            // ensure p is unordered i<j; assume caller provides that
            std::pair<int, int> m1 = {topology[p][0], topology[p][1]};
            std::pair<int, int> m2 = {topology[p][1], topology[p][0]};

            if (!use_gl) {
                if (last_pairs.size() >= 3 &&
                    std::all_of(last_pairs.data + last_pairs.size() - 3, last_pairs.data + last_pairs.size(),
                                [p](const int& x) { return x == p; }))
                    continue; // avoid more than 3 repeated CNOTs
                GrayCodeCNOT seqp = last_pairs.add_Digit(static_cast<int>(topology.size()));
                seqp[seqp.size() - 1] = p;
                if (canonical_prefix_ok(seqp, topology) >= 0)
                    continue; // not canonical prefix
            }

            std::vector<std::pair<int, int>> allmv =
                use_gl ? std::vector<std::pair<int, int>>{m1, m2} : std::vector<std::pair<int, int>>{m1};

            for (std::pair<int, int> mv : allmv) {
                std::vector<int> B;
                if (use_gl) {
                    B = A;
                    if (mv.first != mv.second) {
                        B[mv.second] ^= B[mv.first];
                    }

                    if (visited.find(B) != visited.end()) {
                        continue; // discovered already (at minimal or earlier depth)
                    }
                } else {
                    B = std::vector<int>(last_pairs.data, last_pairs.data + last_pairs.size());
                    B.push_back(p);
                }
                visited.emplace(B);

                // build sequences
                GrayCodeCNOT seqp = last_pairs.add_Digit(static_cast<int>(topology.size()));
                seqp[seqp.size() - 1] = p;

                new_seq_pairs_of.emplace(B, std::move(seqp));

                // emit discovery: (depth+1, B, seq_pairs_of[B], seq_dir_of[B])
                const GrayCodeCNOT& ref_pairs = new_seq_pairs_of.at(B);
                out_res.emplace_back(std::move(B), ref_pairs);
            }
        }
    }
    LevelResult result;
    result.visited = std::move(visited);
    result.seq_pairs_of = std::move(new_seq_pairs_of);
    result.out_res = std::move(out_res);
    return result;
}


template <class Callback>
void generate_insertions_recursive(
    const GrayCodeCNOT& curpath,
    const std::vector<matrix_base<int>>& topology,
    const std::vector<int>& topo_filt,
    int num_cnot,
    std::vector<int>& places,
    std::vector<int>& pairs,
    int depth,
    int min_place,
    Callback&& callback,
    bool & early_stop)
{
    const int nslots = curpath.size() + 1;

    if (depth == num_cnot) {
        matrix_base<int8_t> limits = matrix_base<int8_t>(1, curpath.size()+num_cnot);
        std::fill(limits.data, limits.data + limits.size(), topology.size());
        GrayCodeCNOT out(limits);

        int j = 0, k = 0;
        for (int slot = 0; slot < nslots; ++slot) {
            while (j < num_cnot && places[j] == slot) {
                if (k > 2 && out[k-1] == pairs[j] && out[k-2] == pairs[j] && out[k-3] == pairs[j]) {
                    return; // avoid more than 3 repeated CNOTs
                }
                out[k++] = pairs[j];
                ++j;
            }
            if (slot < curpath.size()) {
                if (k > 2 && out[k-1] == curpath[slot] && out[k-2] == curpath[slot] && out[k-3] == curpath[slot]) {
                    return; // avoid more than 3 repeated CNOTs
                }
                out[k++] = curpath[slot];
            }
        }
        early_stop |= callback(out);
        return;
    }
    uint32_t used_mask = 0u;
    for (int d = 0; d < depth; d++) {
        used_mask |= (1u<<topology[pairs[d]][0]) | (1u<<topology[pairs[d]][1]);
    }
    for (int place = min_place; place < nslots; ++place) {
        if (depth != 0 && places[depth-1]+1 < place) {
            continue; // avoid insertions more than one place away
        }
        places[depth] = place;
        for (int topo_idx : topo_filt) {
            uint32_t edge_mask = (1u<<topology[topo_idx][0]) | (1u<<topology[topo_idx][1]);
            if (depth != 0 && (used_mask & edge_mask) == 0) continue;
            pairs[depth] = topo_idx;
            generate_insertions_recursive(
                curpath, topology, topo_filt, num_cnot,
                places, pairs, depth + 1, place,
                callback, early_stop);
            if (early_stop) return;
        }
    }
}

template <class Callback>
void generate_insertions(
    const GrayCodeCNOT& curpath,
    const std::vector<matrix_base<int>>& topology,
    const std::vector<int>& topo_filt,
    int num_cnot,
    Callback&& callback)
{
    std::vector<int> places(num_cnot);
    std::vector<int> pairs(num_cnot);
    bool early_stop = false;
    generate_insertions_recursive(
        curpath, topology, topo_filt, num_cnot,
        places, pairs, 0, 0,
        std::forward<Callback>(callback), early_stop);
}

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search() : Optimization_Interface() {

    // set the level limit
    level_limit = 0;

    // BFGS is better for smaller problems, while ADAM for larger ones
    if (qbit_num <= 5) {
        set_optimizer(BFGS);

        // Maximal number of iterations in the optimization process
        max_outer_iterations = 4;
        max_inner_iterations = 10000;
    } else {
        set_optimizer(ADAM);

        // Maximal number of iterations in the optimization process
        max_outer_iterations = 1;
    }
}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param config std::map containing custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search(Matrix Umtx_in, int qbit_num_in,
                                                                     std::map<std::string, Config_Element>& config,
                                                                     int accelerator_num)
    : N_Qubit_Decomposition_Tree_Search(Umtx_in, qbit_num_in, {}, config, accelerator_num) {}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param topology_in A list of <target_qubit, control_qubit> pairs describing the connectivity between qubits.
@param config std::map containing custom config parameters
@param accelerator_num The number of DFE accelerators used in the calculations
@return An instance of the class
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search(Matrix Umtx_in, int qbit_num_in,
                                                                     std::vector<matrix_base<int>> topology_in,
                                                                     std::map<std::string, Config_Element>& config,
                                                                     int accelerator_num)
    : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {

    // set the level limit
    level_limit = 0;

    // Maximal number of iterations in the optimization process
    max_outer_iterations = 1;

    // setting the topology
    topology = topology_in;

    if (topology.size() == 0) {
        for (int qbit1 = 0; qbit1 < qbit_num; qbit1++) {
            for (int qbit2 = qbit1 + 1; qbit2 < qbit_num; qbit2++) {
                matrix_base<int> edge(2, 1);
                edge[0] = qbit1;
                edge[1] = qbit2;

                topology.push_back(edge);
            }
        }
    } else {
        for (size_t idx = 0; idx < topology.size(); idx++) {
            if (topology[idx].size() != 2) {
                std::string error("invalid topology: each element should be a pair of integers");
                throw error;
            }
            if (topology[idx][0] < 0 || topology[idx][0] >= qbit_num || topology[idx][1] < 0 || topology[idx][1] >= qbit_num) {
                std::string error("invalid topology: qubit indices should be between 0 and qbit_num-1");
                throw error;
            }
            if (topology[idx][0] == topology[idx][1]) {
                std::string error("invalid topology: target and control qubits should be different");
                throw error;
            }
            if (topology[idx][0] > topology[idx][1]) {
                std::swap(topology[idx][0], topology[idx][1]);
            }
        }
    }

    // construct the possible CNOT combinations within a single level
    // the number of possible CNOT connections netween the qubits (including topology constraints)
    int n_ary_limit_max = static_cast<int>(topology.size());

    possible_target_qbits = matrix_base<int>(1, n_ary_limit_max);
    possible_control_qbits = matrix_base<int>(1, n_ary_limit_max);
    for (int element_idx = 0; element_idx < n_ary_limit_max; element_idx++) {

        matrix_base<int>& edge = topology[element_idx];
        possible_target_qbits[element_idx] = edge[0];
        possible_control_qbits[element_idx] = edge[1];
    }

    // BFGS is better for smaller problems, while ADAM for larger ones
    if (qbit_num <= 5) {
        alg = BFGS;

        // Maximal number of iterations in the optimization process
        max_outer_iterations = 4;
        max_inner_iterations = 10000;
    } else {
        alg = ADAM;

        // Maximal number of iterations in the optimization process
        max_outer_iterations = 1;
    }
}

/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_Tree_Search::~N_Qubit_Decomposition_Tree_Search() {}

/**
@brief Start the disentangling process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0>
when the disentangling of the qubits is done. Set to False to omit this procedure
*/
void N_Qubit_Decomposition_Tree_Search::start_decomposition() {

    // The string stream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;

    print(sstream, 1);

// temporarily turn off OpenMP parallelism
#if BLAS == 0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS == 1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS == 2 // OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    Gates_block* gate_structure_loc = determine_gate_structure(optimized_parameters_mtx);

    long long export_circuit_2_binary_loc;
    if (config.count("export_circuit_2_binary") > 0) {
        config["export_circuit_2_binary"].get_property(export_circuit_2_binary_loc);
    } else {
        export_circuit_2_binary_loc = 0;
    }

    if (export_circuit_2_binary_loc > 0) {
        std::string filename("circuit_squander.binary");
        if (project_name != "") {
            filename = project_name + "_" + filename;
        }
        export_gate_list_to_binary(optimized_parameters_mtx, gate_structure_loc, filename, verbose);

        std::string unitaryname("unitary_squander.binary");
        if (project_name != "") {
            filename = project_name + "_" + unitaryname;
        }
        export_unitary(unitaryname);
    }

    // store the created gate structure
    release_gates();
    combine(gate_structure_loc);
    delete (gate_structure_loc);

    decomposition_error = current_minimum;

#if BLAS == 0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS == 1 // MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS == 2 // OpenBLAS
    openblas_set_num_threads(num_threads);
#endif
}

/**
@brief Call to determine the gate structure of the decomposing circuit.
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
@return Returns a pointer to the gate structure of the decomposing circuit
*/
Gates_block* N_Qubit_Decomposition_Tree_Search::determine_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

    double optimization_tolerance_loc;
    long long level_max = 14;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);

    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    if (config.count("tree_level_max") > 0) {
        config["tree_level_max"].get_property(level_max);
    }
    long long use_osr = 1;
    if (config.count("use_osr") > 0) {
        config["use_osr"].get_property(use_osr);
    }
    long long use_graph_search = 1;
    if (config.count("use_graph_search") > 0) {
        config["use_graph_search"].get_property(use_graph_search);
    }

    long long stop_first_solution = 1;
    if (config.count("stop_first_solution") > 0) {
        config["stop_first_solution"].get_property(stop_first_solution);
    }

    level_limit = std::min(std::max((int)level_max, 0), 14);

    if (level_limit < 0) {
        std::string error("please increase level limit");
        throw error;
    }

    GrayCodeCNOT best_solution;
    std::vector<GrayCodeCNOT> all_solutions;
    if (use_graph_search) {
        all_solutions.emplace_back(tree_search_over_gate_structures_best_first());
    } else {

        double minimum_best_solution = current_minimum;
        LevelInfo li;
        std::vector<std::vector<int>> all_cuts = unique_cuts(qbit_num);
        std::sort(all_cuts.begin(), all_cuts.end(), [](const std::vector<int>& a, const std::vector<int>& b){
            if (a.size() != b.size()) return a.size() < b.size();
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        });
        std::map<std::pair<int, int>, std::vector<int>> pair_affects;
        for (const matrix_base<int>& pair : topology) {
            std::vector<int> cuts;
            for (size_t i = 0; i < all_cuts.size(); ++i) {
                const std::vector<int>& A = all_cuts[i];
                if ((std::find(A.begin(), A.end(), pair[0]) != A.end()) ^
                    (std::find(A.begin(), A.end(), pair[1]) != A.end())) {
                    cuts.push_back(static_cast<int>(i));
                }
            }
            pair_affects[std::pair<int, int>(pair[0], pair[1])] = std::move(cuts);
        }
        CutInfo ci(std::move(all_cuts), MinCnotBoundSolver(qbit_num, all_cuts, topology));

        for (int level = 0; level <= level_limit; level++) {
            GrayCodeCNOT gcode;
            if (use_osr) {
                if (qbit_num <= 1) {
                    all_solutions.emplace_back();
                    break;
                } else {
                    TreeSearchResult result = tree_search_over_gate_structures_osr(level, li, ci);
                    all_solutions.insert(all_solutions.end(), result.solutions.begin(), result.solutions.end());
                    std::swap(li, result.level_info);
                    ci.prefixes = std::move(result.prefixes);
                }
                if (stop_first_solution && all_solutions.size() > 0) {
                    break;
                }
            } else {
                gcode = std::move(tree_search_over_gate_structures(level));
                if (current_minimum < minimum_best_solution) {

                    minimum_best_solution = current_minimum;
                    best_solution = gcode;
                }

                if (current_minimum < optimization_tolerance_loc) {
                    break;
                }
            }
        }
    }
    if (use_osr || use_graph_search) {
        N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(nullptr);
        std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);
        std::vector<double> optimized_parameters;
        current_minimum = std::numeric_limits<double>::max();
        if (all_solutions.size() == 0) { return new Gates_block(qbit_num); }
        for (const GrayCodeCNOT& solution : all_solutions) {
            std::unique_ptr<Gates_block> gate_structure_loc;
            gate_structure_loc.reset(construct_gate_structure_from_Gray_code(solution));
            cDecomp_custom_random.set_custom_gate_structure(gate_structure_loc.get());
            cDecomp_custom_random.set_optimization_blocks(gate_structure_loc->get_gate_num());

            // ----------- start the decomposition -----------
            double current_minimum_tmp;
            for (int iter = 0; iter < 5; iter++) {
                optimized_parameters.resize(cDecomp_custom_random.get_parameter_num());
                for (size_t idx = 0; idx < optimized_parameters.size(); idx++) {
                    optimized_parameters[idx] = distrib_real(gen);
                }
                cDecomp_custom_random.set_optimized_parameters(optimized_parameters.data(),
                                                               static_cast<int>(optimized_parameters.size()));
                cDecomp_custom_random.start_decomposition();
                current_minimum_tmp = cDecomp_custom_random.get_current_minimum();
                if (current_minimum_tmp < optimization_tolerance_loc) {
                    break;
                }
            }
            if (current_minimum_tmp < current_minimum) {
                current_minimum = current_minimum_tmp;
                optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters().copy();
                best_solution = solution;
            }
            if (current_minimum < optimization_tolerance_loc && stop_first_solution) {
                break;
            }
        }
    }

    if (current_minimum > optimization_tolerance_loc) {
        std::stringstream sstream;
        sstream << "Decomposition did not reach prescribed high numerical precision." << std::endl;
        print(sstream, 1);
    }

    return construct_gate_structure_from_Gray_code(best_solution);
}

SearchNode N_Qubit_Decomposition_Tree_Search::evaluate_path(
    N_Qubit_Decomposition_custom& cDecomp_custom_random, MinCnotBoundSolver& osr_bound_solver,
    std::vector<std::vector<int>>& all_cuts, double Fnorm, double osr_tol,
    std::uniform_real_distribution<>& distrib_real, std::mt19937& gen,
    const GrayCodeCNOT& path) {
    SearchNode ev_results(path);
    std::unique_ptr<Gates_block> gate_structure_loc(
        construct_gate_structure_from_Gray_code(path, false));
    cDecomp_custom_random.set_custom_gate_structure(gate_structure_loc.get());
    cDecomp_custom_random.set_optimization_blocks(gate_structure_loc->get_gate_num());
    std::vector<double> optimized_parameters(cDecomp_custom_random.get_parameter_num());
    for (size_t idx = 0; idx < optimized_parameters.size(); idx++) {
        optimized_parameters[idx] = distrib_real(gen);
    }
    cDecomp_custom_random.set_optimized_parameters(optimized_parameters.data(),
                                                    static_cast<int>(optimized_parameters.size()));
    for (const std::vector<int>& cut : all_cuts) {
        if (cut.size() != 1) continue;
        int max_rank = 2*(int)std::min(cut.size(), qbit_num-cut.size());
        //int max_rank = 2;
        std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>> rank_result;
        for (int rank = max_rank-1; rank >= 0; rank--) {
            cDecomp_custom_random.set_osr_params({cut}, rank, false);
            //cDecomp_custom_random.set_osr_params(all_cuts, rank, true);
            cDecomp_custom_random.start_decomposition();
            Matrix U = Umtx.copy();
            Matrix_real params = cDecomp_custom_random.get_optimized_parameters();
            cDecomp_custom_random.apply_to(params, U);
            std::vector<std::pair<int, double>> osr_result;
            osr_result.reserve(all_cuts.size());
            int newrank = rank;
            for (const std::vector<int>& eval_cut : all_cuts) {
                osr_result.emplace_back(operator_schmidt_rank(U, qbit_num, eval_cut, Fnorm, osr_tol));
                if (cut == eval_cut) newrank = osr_result.back().first;
                //newrank = std::max(newrank, osr_result.back().first);
            }
            double best_kappa = std::numeric_limits<double>::infinity();
            std::vector<int> best_edge_counts;
            int min_cnots = osr_bound_solver.solve_min_cnots(osr_result, best_kappa, best_edge_counts);
            if (newrank <= rank || rank == max_rank-1)
                rank_result = std::make_tuple(min_cnots, best_kappa, std::move(best_edge_counts), std::move(osr_result));
            if (newrank > rank) break;
            rank = std::min(rank, newrank);
        }
        ev_results.osr_results.emplace_back(std::move(rank_result));
        //if (ev_results.size() == (all_cuts.size()+1)/2) break;
    }
    return ev_results;
};

std::vector<uint32_t> build_pred_mask(const GrayCodeCNOT& ops,
    const std::vector<matrix_base<int>>& topology) {
    const int m = static_cast<int>(ops.size());
    std::vector<uint32_t> pred_mask(m, 0);

    std::unordered_map<int,int> last_on;
    last_on.reserve(m * 2);

    for (int k = 0; k < m; ++k) {
        int a = topology[ops[k]][0];
        int b = topology[ops[k]][1];

        for (int q : {a, b}) {
            std::unordered_map<int,int>::iterator it = last_on.find(q);
            if (it != last_on.end()) {
                int prev = it->second;
                pred_mask[k] |= (1u << prev);
                it->second = k;
            } else {
                last_on.emplace(q, k);
            }
        }
    }

    return pred_mask;
}

bool contains_topological_subsequence(
    const GrayCodeCNOT& smallpath, const GrayCodeCNOT& bigpath,
    const std::vector<matrix_base<int>>& topology)
{
    std::vector<uint32_t> pred_mask = build_pred_mask(smallpath, topology);
    const int m = static_cast<int>(smallpath.size());
    if (m == 0) return true;
    if (m > 31) {
        // this should never happen
        throw std::runtime_error("pattern too large for uint32_t mask");
    }

    const uint32_t FULL = (1u << m) - 1u;

    // reachable[S] = whether subset S of small nodes can be matched
    // after scanning some prefix of big
    std::vector<char> reachable(1u << m, 0), next_reachable(1u << m, 0);
    reachable[0] = 1;

    for (int i = 0; i < bigpath.size(); i++) {
        int b = bigpath[i];
        next_reachable = reachable; // skipping b is always allowed

        for (uint32_t S = 0; S <= FULL; ++S) {
            if (!reachable[S]) continue;

            // try matching b to any currently available node u
            for (int u = 0; u < m; ++u) {
                uint32_t bit = 1u << u;
                if (S & bit) continue; // already matched

                // all predecessors of u must already be in S
                if ((pred_mask[u] & ~S) != 0) continue;

                // labels must match
                if (smallpath[u] != b) continue;

                next_reachable[S | bit] = 1;
            }
        }

        reachable.swap(next_reachable);

        if (reachable[FULL]) return true;
    }

    return reachable[FULL];
}

struct ForbiddenSubseqSet {
    std::vector<GrayCodeCNOT> patterns;
    const std::vector<matrix_base<int>>& topology;

    ForbiddenSubseqSet(const std::vector<matrix_base<int>>& topology) : topology(topology) {}

    // Returns true if candidate should be pruned
    bool contains_forbidden_subsequence(const GrayCodeCNOT& candidate) const {
        for (const GrayCodeCNOT& pat : patterns) {
            if (contains_topological_subsequence(pat, candidate, topology)) {
                return true;
            }
        }
        return false;
    }

    // Insert a newly discovered forbidden path, keeping only minimal patterns
    void insert_forbidden(const GrayCodeCNOT& path) {
        // If already covered by a smaller forbidden pattern, skip
        for (const GrayCodeCNOT& pat : patterns) {
            if (contains_topological_subsequence(pat, path, topology)) {
                return;
            }
        }

        // Remove any existing patterns that are supersets of the new one
        patterns.erase(
            std::remove_if(
                patterns.begin(), patterns.end(),
                [&](const GrayCodeCNOT& pat) {
                    return contains_topological_subsequence(pat, path, topology);
                }),
            patterns.end()
        );

        patterns.push_back(path);
    }
};

GrayCodeCNOT N_Qubit_Decomposition_Tree_Search::tree_search_over_gate_structures_best_first() {
    std::vector<std::vector<int>> all_cuts = unique_cuts(qbit_num);
    std::sort(all_cuts.begin(), all_cuts.end(), [](const std::vector<int>& a, const std::vector<int>& b){
        if (a.size() != b.size()) return a.size() < b.size();
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    });
    // If topology entries are actual gates, the path stores topology indices.
    double Fnorm = std::sqrt(static_cast<double>(1 << qbit_num));
    double osr_tol = 1e-3;
    MinCnotBoundSolver osr_bound_solver(qbit_num, all_cuts, topology);
    //std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<SearchNode>> heap;
    std::unique_ptr<SearchNode> top_heap;
    std::set<GrayCodeCNOT> visited;
    //ForbiddenSubseqSet forbidden(topology);

    N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(nullptr);
    cDecomp_custom_random.set_cost_function_variant(OSR_ENTANGLEMENT);
    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);

    std::function<bool(const GrayCodeCNOT&)> add_to_heap = [&](const GrayCodeCNOT& path) -> bool {
        if (!is_unique_structure(path, topology))
            return false; // not unique structure

        bool inserted = visited.insert(path).second;

        if (!inserted) {
            return false;
        }
        // if (forbidden.contains_forbidden_subsequence(path)) {
        //     return false;
        // }
        // for (int i = 0; i < path.size(); i++) {
        //     if (visited.find(path.remove_Digit(i)) == visited.end()) {
        //         return false;
        //     }
        // }
        
        //std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        SearchNode sn = evaluate_path(cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm, osr_tol, distrib_real, gen, path);
        //printf("%.2fs\n", std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()*1e-9);
        // if (path.size()+sn.get_min_cnots() > level_limit) {
        //     forbidden.insert_forbidden(path);
        //     return false;
        // }

        if (top_heap == nullptr || !(*top_heap < sn)) {
            top_heap.reset(new SearchNode(std::move(sn)));
        }
        // heap.emplace(sn);
        return true;
    };

    GrayCodeCNOT startpath;
    if (qbit_num > 1)
        add_to_heap(startpath);

    std::vector<int> full_topo_filter(topology.size());
    std::iota(full_topo_filter.begin(), full_topo_filter.end(), 0);

    while (top_heap != nullptr) {
        std::unique_ptr<SearchNode> cur(top_heap.release());
        visited.clear(); // clear visited to save memory, relying on the fact that we won't revisit nodes anyway
        if (cur->get_min_cnots() == 0) {
            return cur->path;
        }
        const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& cur_best_osr_result = cur->get_best_osr_result();
        const std::vector<int>& best_edge_counts = std::get<2>(cur_best_osr_result);
        std::vector<int> topo_filter;
        bool exact_edges = false;
        int num_cnot;
        if (!exact_edges) {
            num_cnot = 1;
            topo_filter.resize(topology.size());
            std::iota(topo_filter.begin(), topo_filter.end(), 0);
            std::sort(topo_filter.begin(), topo_filter.end(), [&](int a, int b){
                return best_edge_counts[a] > best_edge_counts[b];
            });
        } else {
            num_cnot = std::get<0>(cur_best_osr_result);
            topo_filter.reserve(std::get<0>(cur_best_osr_result));
            //topo_filter.resize(std::count_if(best_edge_counts.begin(), best_edge_counts.end(), [](int c){ return c > 0; }));
            for (size_t i = 0; i < best_edge_counts.size(); i++) {
                for (int j = 0; j < best_edge_counts[i]; j++) {
                    topo_filter.push_back(static_cast<int>(i));
                }
            }
        }

        while (true) {
            // safety guard
            if (cur->path.size() + num_cnot > level_limit) {
                return startpath;
            }

            generate_insertions(cur->path, topology, topo_filter, num_cnot,
                [&](const GrayCodeCNOT& newpath) {
                    if (add_to_heap(newpath)) {
                        //return cur > heap.top();
                        return top_heap->get_min_cnots() == 0;
                    }
                    return false;
                });

            //const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& top_best_osr_result = top_heap->get_best_osr_result();
            if (*cur > *top_heap || num_cnot == std::get<0>(cur_best_osr_result)) {
            // if (std::get<0>(top_best_osr_result) < std::get<0>(cur_best_osr_result) ||
            //     std::get<0>(top_best_osr_result) == std::get<0>(cur_best_osr_result) &&
            //     std::get<1>(top_best_osr_result) + 1e-3 < std::get<1>(cur_best_osr_result)) {
                break;
            }
            

            ++num_cnot;

        }

        // Optional beam trimming:
        // if beam_width > 0 and heap.size() > beam_width, can rebuild a trimmed heap here.
    }
    //printf("failed\n");
    return startpath;
}

/**
@brief Perform tree search over possible gate structures using Gray code enumeration and Operator Schmidt Rank (OSR)
optimization.

This function performs a breadth-first search (BFS) over gate structures represented as Gray codes. It enumerates
CNOT gate combinations at a given level, optimizes each structure using OSR-based cost function, and filters
candidates based on their operator Schmidt rank across different qubit cuts. The search is performed in parallel
using Intel TBB for improved performance.

The function uses a beam search approach, keeping only the best candidates (based on beam width configuration)
for further exploration. It maintains state information about visited gate structures and their corresponding
Gray code sequences to avoid redundant computations.

@param level_num The number of decomposing levels (i.e. the depth in the search tree). Level 0 corresponds
                 to the identity (no CNOT gates).
@param li LevelInfo reference that is updated with visited states and sequence pairs discovered at this level.
          This is used to track the BFS state across multiple calls.
@param ci CutInfo reference containing cut information (all possible qubit cuts) and prefixes (OSR results
          from previous levels). The prefixes map is updated with new OSR results for promising candidates.
@return Returns a TreeSearchResult structure containing:
        - solutions: Vector of successful Gray-code solutions that achieved zero operator Schmidt rank
        - level_info: Updated LevelInfo with visited states and sequence pairs for the next level
        - prefixes: Map of GrayCodeCNOT to OSR result pairs for candidates that passed the filtering criteria
@note The function modifies the input parameters li and ci to maintain state across multiple calls.
      The associated gate structure can be constructed from a Gray code using the function
      construct_gate_structure_from_Gray_code.
*/
TreeSearchResult N_Qubit_Decomposition_Tree_Search::tree_search_over_gate_structures_osr(int level_num, LevelInfo& li,
                                                                                         CutInfo& ci) {

    tbb::spin_mutex tree_search_mutex;

    std::vector<std::vector<int>>& all_cuts = ci.all_cuts;
    MinCnotBoundSolver& osr_bound_solver = ci.osr_bound_solver;
    std::map<GrayCodeCNOT, SearchNode>& prefixes = ci.prefixes;    

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }
    long long stop_first_solution = 1;
    if (config.count("stop_first_solution") > 0) {
        config["stop_first_solution"].get_property(stop_first_solution);
    }
    GrayCodeCNOT best_solution;
    volatile bool found_optimal_solution = false;

    LevelResult level_result = level_num == 0 ? enumerate_unordered_cnot_BFS_level_init(qbit_num)
                                              : enumerate_unordered_cnot_BFS_level_step(li, topology, false);
    const std::set<std::vector<int>>& visited = level_result.visited;
    const std::map<std::vector<int>, GrayCodeCNOT>& seq_pairs_of = level_result.seq_pairs_of;
    const std::vector<std::pair<std::vector<int>, GrayCodeCNOT>>& out_res = level_result.out_res;

    std::set<GrayCodeCNOT> pairs_reduced;
    for (const std::pair<std::vector<int>, GrayCodeCNOT>& item : out_res) {
        pairs_reduced.insert(item.second);
    }
    std::vector<GrayCodeCNOT> all_pairs(pairs_reduced.begin(), pairs_reduced.end());
    std::set<SearchNode> all_osr_results;
    int64_t iteration_max = all_pairs.size();
    std::vector<GrayCodeCNOT> successful_solutions;
    double Fnorm = std::sqrt(static_cast<double>(1 << qbit_num));
    double osr_tol = 1e-3;

    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads;
    concurrency = concurrency < iteration_max ? concurrency : iteration_max;
    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);

    int parallel = get_parallel_configuration();

    int64_t work_batch = 1;
    if (parallel == 0) {
        work_batch = concurrency;
    }

    // std::cout << "levels " << level_num << std::endl;
    tbb::parallel_for(
        tbb::blocked_range<int64_t>((int64_t)0, concurrency, work_batch), [&](tbb::blocked_range<int64_t> r) {
            N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(nullptr);
            cDecomp_custom_random.set_cost_function_variant(OSR_ENTANGLEMENT);
            std::mt19937 ts_gen(std::random_device{}());

            for (int64_t job_idx = r.begin(); job_idx < r.end(); ++job_idx) {

                // for( int64_t job_idx=0; job_idx<concurrency; job_idx++ ) {

                // initial offset and upper boundary of the gray code counter
                int64_t work_batch = iteration_max / concurrency;
                int64_t initial_offset = job_idx * work_batch;
                int64_t offset_max = (job_idx + 1) * work_batch - 1;

                if (job_idx == concurrency - 1) {
                    offset_max = iteration_max - 1;
                }

                // std::cout << initial_offset << " " << offset_max << " " << iteration_max << " " << work_batch << " "
                // << concurrency << std::endl;

                for (int64_t iter_idx = initial_offset; iter_idx < offset_max + 1; iter_idx++) {
                    if (stop_first_solution && found_optimal_solution) {
                        break;
                    }
                    const GrayCodeCNOT& solution = all_pairs[iter_idx];

                    SearchNode sn = evaluate_path(cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm, osr_tol, distrib_real, ts_gen, solution);
                    number_of_iters +=
                        cDecomp_custom_random
                            .get_num_iters(); // retrieve the number of iterations spent on optimization

                    const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& osr_result = sn.get_best_osr_result();
                    bool isWorse = false;
                    for (int idx = 0; idx < solution.size(); idx++) {
                        const GrayCodeCNOT& prefix = solution.remove_Digit(idx);
                        std::map<GrayCodeCNOT, SearchNode>::const_iterator prefix_it = prefixes.find(prefix);
                        if (prefix_it == prefixes.end()) {
                            isWorse = true;
                            break;
                        }
                        //if (sn > *prefix_it)
                        const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& prefix_osr_result = prefix_it->second.get_best_osr_result();
                        if (std::get<0>(osr_result) > std::get<0>(prefix_osr_result) ||
                                   (std::get<0>(osr_result) == std::get<0>(prefix_osr_result) &&
                                    std::get<1>(osr_result) + 1e-3 < std::get<1>(prefix_osr_result))) {
                            isWorse = true;
                            break;
                        }
                    }
                    int cnot_lower_bound = std::get<0>(osr_result);
                    if (cnot_lower_bound <= level_limit - level_num && !isWorse) {
                        tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};
                        all_osr_results.emplace(std::move(sn));
                        if (cnot_lower_bound == 0) {
                            found_optimal_solution = true;
                            successful_solutions.push_back(solution.copy());
                        }
                    }

                    /*for( int gcode_idx=0; gcode_idx<solution.size(); gcode_idx++ ) {
                        std::cout << solution[gcode_idx] << ", ";
                    }
                    std::cout << current_minimum << std::endl;*/
                }
            }
        });

    long long beam_width = all_osr_results.size();
    if (config.count("beam") > 0) {
        config["beam"].get_property(beam_width);
        if (beam_width <= 0) beam_width = all_osr_results.size();
    }
    beam_width = std::min<long long>(beam_width, all_osr_results.size());
    std::map<GrayCodeCNOT, SearchNode> nextprefixes;
    for (std::set<SearchNode>::iterator item = all_osr_results.begin(); item != all_osr_results.end() && beam_width > 0; ++item, --beam_width) {
        nextprefixes.emplace(item->path, std::move(*item));
    }
    std::vector<std::vector<int>> next_q;
    next_q.reserve(out_res.size());
    for (std::vector<std::pair<std::vector<int>, GrayCodeCNOT>>::const_reverse_iterator it = out_res.crbegin();
         it != out_res.crend(); ++it) {
        if (nextprefixes.find(it->second) == nextprefixes.end()) {
            continue;
        }
        next_q.push_back(it->first);
    }
    TreeSearchResult result;
    result.solutions = std::move(successful_solutions);
    result.level_info.visited = std::move(visited);
    result.level_info.seq_pairs_of = std::move(seq_pairs_of);
    result.level_info.q = std::move(next_q);
    result.prefixes = std::move(nextprefixes);
    return result;
}

/**
@brief Call to perform tree search over possible gate structures with a given tree search depth.
@param level_num The number of decomposing levels (i.e. the maximal tree depth)
@return Returns the best Gray-code corresponding to the best circuit. The associated gate structure can be constructed
by function construct_gate_structure_from_Gray_code
*/
GrayCodeCNOT N_Qubit_Decomposition_Tree_Search::tree_search_over_gate_structures(int level_num) {

    tbb::spin_mutex tree_search_mutex;

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    if (level_num == 0) {

        // empty Gray code describing a circuit without two-qubit gates
        GrayCodeCNOT gcode;
        Gates_block* gate_structure_loc = construct_gate_structure_from_Gray_code(gcode);

        std::stringstream sstream;
        sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers."
                << std::endl;
        print(sstream, 1);

        N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(gate_structure_loc);

        number_of_iters +=
            cDecomp_custom_random.get_num_iters(); // retrieve the number of iterations spent on optimization

        double current_minimum_tmp = cDecomp_custom_random.get_current_minimum();
        sstream.str("");
        sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp;
        print(sstream, 1);

        if (current_minimum_tmp < current_minimum) {
            current_minimum = current_minimum_tmp;
            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
        }

        // std::cout << "iiiiiiiiiiiiiiiiii " << current_minimum_tmp << std::endl;
        delete (gate_structure_loc);
        return gcode;
    }

    GrayCodeCNOT gcode_best_solution;
    bool found_optimal_solution = false;

    // set the limits for the N-ary Gray counter

    int n_ary_limit_max = static_cast<int>(topology.size());
    matrix_base<int8_t> n_ary_limits_int8(1, level_num); // array containing the limits of the individual Gray code elements
    memset(n_ary_limits_int8.get_data(), n_ary_limit_max, n_ary_limits_int8.size() * sizeof(int8_t));
    matrix_base<int> n_ary_limits(1, level_num); // array containing the limits of the individual Gray code elements
    memset(n_ary_limits.get_data(), n_ary_limit_max, n_ary_limits.size() * sizeof(int));

    for (int idx = 0; idx < n_ary_limits.size(); idx++) {
        n_ary_limits[idx] = n_ary_limit_max;
        n_ary_limits_int8[idx] = n_ary_limit_max;
    }

    int64_t iteration_max =
        static_cast<int64_t>(pow(static_cast<double>(n_ary_limit_max), static_cast<double>(level_num)));

    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads;
    concurrency = concurrency < iteration_max ? concurrency : iteration_max;

    int parallel = get_parallel_configuration();

    int64_t work_batch = 1;
    if (parallel == 0) {
        work_batch = concurrency;
    }

    // std::cout << "levels " << level_num << std::endl;
    tbb::parallel_for(
        tbb::blocked_range<int64_t>((int64_t)0, concurrency, work_batch), [&](tbb::blocked_range<int64_t> r) {
            for (int64_t job_idx = r.begin(); job_idx < r.end(); ++job_idx) {

                // for( int64_t job_idx=0; job_idx<concurrency; job_idx++ ) {

                // initial offset and upper boundary of the gray code counter
                int64_t work_batch = iteration_max / concurrency;
                int64_t initial_offset = job_idx * work_batch;
                int64_t offset_max = (job_idx + 1) * work_batch - 1;

                if (job_idx == concurrency - 1) {
                    offset_max = iteration_max - 1;
                }

                // std::cout << initial_offset << " " << offset_max << " " << iteration_max << " " << work_batch << " "
                // << concurrency << std::endl;

                n_aryGrayCodeCounter gcode_counter(
                    n_ary_limits, initial_offset); // see piquassoboost for details of the implementation
                gcode_counter.set_offset_max(offset_max);
                GrayCodeCNOT gcode(n_ary_limits_int8);

                for (int64_t iter_idx = initial_offset; iter_idx < offset_max + 1; iter_idx++) {

                    if (found_optimal_solution) {
                        return;
                    }

                    GrayCode&& gcodeint = gcode_counter.get();
                    std::transform(gcodeint.data, gcodeint.data + gcodeint.size(), gcode.data,
                                   [](int val) { return static_cast<int8_t>(val); });

                    if (!is_unique_structure(gcode, topology)) continue;

                    Gates_block* gate_structure_loc = construct_gate_structure_from_Gray_code(gcode);

                    // ----------- start the decomposition -----------

                    std::stringstream sstream;
                    sstream << "Starting optimization with " << gate_structure_loc->get_gate_num()
                            << " decomposing layers." << std::endl;
                    print(sstream, 1);

                    N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(gate_structure_loc);

                    delete (gate_structure_loc);
                    gate_structure_loc = NULL;

                    number_of_iters += cDecomp_custom_random
                                           .get_num_iters(); // retrieve the number of iterations spent on optimization

                    double current_minimum_tmp = cDecomp_custom_random.get_current_minimum();
                    sstream.str("");
                    sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp;
                    print(sstream, 1);

                    // std::cout << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp
                    // << std::endl;

                    {
                        tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};

                        if (current_minimum_tmp < current_minimum && !found_optimal_solution) {

                            current_minimum = current_minimum_tmp;
                            gcode_best_solution = gcode;

                            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                        }

                        if (current_minimum < optimization_tolerance_loc && !found_optimal_solution) {
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
                    if (gcode_counter.next(changed_index, value_prev, value)) {
                        // exit from the for loop if no further gcode is present
                        break;
                    }
                }
            }
        });

    return gcode_best_solution;
}

/**
@brief Call to perform the optimization on the given gate structure
@param gate_structure_loc The gate structure to be optimized (can be nullptr)
@return Returns an instance of N_Qubit_Decomposition_custom with optimized parameters
*/
N_Qubit_Decomposition_custom N_Qubit_Decomposition_Tree_Search::perform_optimization(Gates_block* gate_structure_loc) {

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    N_Qubit_Decomposition_custom cDecomp_custom_random =
        N_Qubit_Decomposition_custom(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    if (gate_structure_loc != nullptr) {
        cDecomp_custom_random.set_custom_gate_structure(gate_structure_loc);
        cDecomp_custom_random.set_optimization_blocks(gate_structure_loc->get_gate_num());
    }
    cDecomp_custom_random.set_max_iteration(max_outer_iterations);
#ifndef __DFE__
    cDecomp_custom_random.set_verbose(verbose);
#else
    cDecomp_custom_random.set_verbose(0);
#endif
    cDecomp_custom_random.set_cost_function_variant(cost_fnc);
    cDecomp_custom_random.set_debugfile("");
    cDecomp_custom_random.set_optimization_tolerance(optimization_tolerance_loc);
    cDecomp_custom_random.set_trace_offset(trace_offset);
    cDecomp_custom_random.set_optimizer(alg);
    cDecomp_custom_random.set_project_name(project_name);
    if (alg == ADAM || alg == BFGS2) {
        int max_inner_iterations_loc = 10000;
        if (gate_structure_loc != nullptr) {
            int param_num_loc = gate_structure_loc->get_parameter_num();
            max_inner_iterations_loc = static_cast<int>((double)param_num_loc / 852 * 10000000.0);
        }
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
        cDecomp_custom_random.set_random_shift_count_max(5);
    } else if (alg == ADAM_BATCHED) {
        cDecomp_custom_random.set_optimizer(alg);
        int max_inner_iterations_loc = 2000;
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
        cDecomp_custom_random.set_random_shift_count_max(5);
    } else if (alg == BFGS) {
        cDecomp_custom_random.set_optimizer(alg);
        int max_inner_iterations_loc = 10000;
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
    }

    if (gate_structure_loc != nullptr)
        cDecomp_custom_random.start_decomposition();
    return cDecomp_custom_random;
}

/**
@brief Call to construct a gate structure corresponding to the configuration of the two-qubit gates described by the
Gray code
@param gcode The N-ary Gray code describing the configuration of the two-qubit gates
@param finalize If true, adds a finalizing layer of single-qubit rotations on all qubits
@return Returns a pointer to the generated circuit gate structure
*/
Gates_block* N_Qubit_Decomposition_Tree_Search::construct_gate_structure_from_Gray_code(const GrayCodeCNOT& gcode,
                                                                                        bool finalize) {

    // determine the target qubit indices and control qbit indices for the CNOT gates from the Gray code counter
    matrix_base<int> target_qbits(1, gcode.size());
    matrix_base<int> control_qbits(1, gcode.size());

    for (int gcode_idx = 0; gcode_idx < gcode.size(); gcode_idx++) {

        int target_qbit = possible_target_qbits[gcode[gcode_idx]];
        int control_qbit = possible_control_qbits[gcode[gcode_idx]];

        target_qbits[gcode_idx] = target_qbit;
        control_qbits[gcode_idx] = control_qbit;

        // std::cout <<   target_qbit << " " << control_qbit << std::endl;
    }

    //  ----------- contruct the gate structure to be optimized -----------
    Gates_block* gate_structure_loc = new Gates_block(qbit_num);

    for (int gcode_idx = 0; gcode_idx < gcode.size(); gcode_idx++) {

        // add new 2-qbit block to the circuit
        add_two_qubit_block(gate_structure_loc, target_qbits[gcode_idx], control_qbits[gcode_idx]);
    }

    // add finalizing layer to the gate structure
    if (finalize)
        add_finalyzing_layer(gate_structure_loc);

    return gate_structure_loc;
}

/**
@brief Call to add two-qubit building block (two single qubit rotation blocks and one two-qubit gate) to the circuit
@param gate_structure Appending the two-qubit building block to this circuit
@param target_qbit The target qubit of the two-qubit gate
@param control_qbit The control qubit of the two-qubit gate
*/
void N_Qubit_Decomposition_Tree_Search::add_two_qubit_block(Gates_block* gate_structure, int target_qbit,
                                                            int control_qbit) {

    if (control_qbit >= qbit_num || target_qbit >= qbit_num) {
        std::string error("N_Qubit_Decomposition_Tree_Search::add_two_qubit_block: Label of control/target qubit "
                          "should be less than the number of qubits in the register.");
        throw error;
    }

    if (control_qbit == target_qbit) {
        std::string error(
            "N_Qubit_Decomposition_Tree_Search::add_two_qubit_block: Target and control qubits should be different");
        throw error;
    }

    Gates_block* layer = new Gates_block(qbit_num);
    /*layer->add_rz(target_qbit);
    layer->add_ry(target_qbit);
    layer->add_rz(target_qbit);

    layer->add_rz(control_qbit);
    layer->add_ry(control_qbit);
    layer->add_rz(control_qbit);*/

    layer->add_u3(target_qbit);
    layer->add_u3(control_qbit);
    layer->add_cnot(target_qbit, control_qbit);
    gate_structure->add_gate(layer);
}

/**
@brief Call to add finalizing layer (single qubit rotations on all of the qubits) to the gate structure
@param gate_structure The gate structure to append the finalizing layer to
*/
void N_Qubit_Decomposition_Tree_Search::add_finalyzing_layer(Gates_block* gate_structure) {

    // creating block of gates
    Gates_block* block = new Gates_block(qbit_num);
    /*
        block->add_un();
        block->add_ry(qbit_num-1);
    */
    for (int idx = 0; idx < qbit_num; idx++) {
        // block->add_rz(idx);
        // block->add_ry(idx);
        // block->add_rz(idx);
        block->add_u3(idx);
        // block->add_u3(idx, Theta, Phi, Lambda);
        //        block->add_ry(idx);
    }

    // adding the operation block to the gates
    if (gate_structure == NULL) {
        throw("N_Qubit_Decomposition_Tree_Search::add_finalyzing_layer: gate_structure is null pointer");
    } else {
        gate_structure->add_gate(block);
    }
}

/**
@brief Call to set unitary matrix from a matrix
@param Umtx_new The unitary matrix to set
*/
void N_Qubit_Decomposition_Tree_Search::set_unitary(Matrix& Umtx_new) {

    Umtx = Umtx_new;
}
