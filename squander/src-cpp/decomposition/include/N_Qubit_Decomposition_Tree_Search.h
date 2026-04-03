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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file N_Qubit_Decomposition_Tree_Search.h
    \brief Header file for a class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#ifndef N_Qubit_Decomposition_Tree_Search_H
#define N_Qubit_Decomposition_Tree_Search_H
#include "GrayCode.h"
#include "N_Qubit_Decomposition_custom.h"

#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using GrayCodeCNOT = GrayCode_base<int8_t>;

/**
@brief Structure containing level information for breadth-first search over gate structures.
This structure tracks visited states, sequence pairs, and the queue of states to process during BFS enumeration.
*/
struct LevelInfo {
    /// Set of visited states (represented as vectors of integers)
    std::set<std::vector<int>> visited;
    /// Map from state vectors to their corresponding Gray code sequences
    std::map<std::vector<int>, GrayCodeCNOT> seq_pairs_of;
    /// Queue of states to be processed in the next BFS level
    std::vector<std::vector<int>> q;
};

// topology: vector of pairs/qubit-edges, e.g. {(0,1), (0,2), ...}
// cuts: each cut is a vector<int> listing one side of the bipartition
// cut_bounds: required cross-cut counts for each cut

class MinCnotBoundSolver {
public:
    MinCnotBoundSolver(int num_qubits,
                       const std::vector<std::vector<int>>& cuts,
                       const std::vector<matrix_base<int>>& topology)
        : num_qubits_(num_qubits), num_edges_(static_cast<int>(topology.size())), cuts_(cuts) {
        build_cut_to_edges(topology);
    }

    int solve_min_cnots(const std::vector<std::pair<int, double> >& cut_bounds, int max_total = -1) const {
        // return std::max_element(cut_bounds.begin(), cut_bounds.end(),
        //                         [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        //                             return a.first < b.first;
        //                         })->first;
        if (cut_bounds.size() != cuts_.size()) {
            throw std::invalid_argument("cut_bounds size must match cuts size");
        }

        std::vector<int> edge_counts(num_edges_, 0);

        for (int total = 0;; ++total) {
            if (max_total >= 0 && total > max_total) {
                return -1; // not found within bound
            }
            if (feasible_for_some_composition(total, edge_counts, cut_bounds, 0, 0)) {
                return total;
            }
        }
    }
    int solve_min_cnots(const std::vector<std::pair<int, double> >& cut_bounds, double& best_kappa, std::vector<int>& best_edge_counts, int max_total = -1) const {
        if (cut_bounds.size() != cuts_.size()) {
            throw std::invalid_argument("cut_bounds size must match cuts size");
        }
        best_edge_counts.clear();
        best_kappa = std::numeric_limits<double>::infinity();

        std::vector<int> edge_counts(num_edges_, 0);

        for (int total = 0;; ++total) {
            if (max_total >= 0 && total > max_total) {
                return -1; // not found within bound
            }
            if (best_feasible_for_some_composition(total, edge_counts, cut_bounds, 0, 0, best_kappa, best_edge_counts)) {
                return total;
            }
        }
    }
private:
    int num_qubits_;
    int num_edges_;
    std::vector<std::vector<int>> cuts_;
    std::vector<std::vector<int>> cut_to_edges_;

    void build_cut_to_edges(const std::vector<matrix_base<int>>& topology) {
        cut_to_edges_.clear();
        cut_to_edges_.reserve(cuts_.size());

        for (const std::vector<int>& cut : cuts_) {
            std::vector<char> in_cut(num_qubits_, 0);
            for (int q : cut) {
                in_cut[q] = 1;
            }

            std::vector<int> crossing_edges;
            crossing_edges.reserve(topology.size());

            for (int i = 0; i < static_cast<int>(topology.size()); ++i) {
                if (in_cut[topology[i][0]] != in_cut[topology[i][1]]) {
                    crossing_edges.push_back(i);
                }
            }
            cut_to_edges_.push_back(std::move(crossing_edges));
        }
    }

    bool composition_satisfies(const std::vector<int>& edge_counts,
                              const std::vector<std::pair<int, double> >& cut_bounds) const {
        for (int c = 0; c < static_cast<int>(cut_to_edges_.size()); ++c) {
            const int bound = cut_bounds[c].first;
            if (bound <= 0) continue;

            int sum = 0;
            for (int edge_idx : cut_to_edges_[c]) {
                sum += edge_counts[edge_idx];
                if (sum >= bound) break;
            }
            if (sum < bound) {
                return false;
            }
        }
        return true;
    }

    bool feasible_for_some_composition(int total,
                                       std::vector<int>& edge_counts,
                                       const std::vector<std::pair<int, double> >& cut_bounds,
                                       int pos,
                                       int used_sum) const {
        const int m = static_cast<int>(edge_counts.size());

        if (pos == m - 1) {
            edge_counts[pos] = total - used_sum;
            return composition_satisfies(edge_counts, cut_bounds);
        }

        const int remaining = total - used_sum;
        for (int x = 0; x <= remaining; ++x) {
            edge_counts[pos] = x;
            if (feasible_for_some_composition(total, edge_counts, cut_bounds, pos + 1, used_sum + x)) {
                return true;
            }
        }
        return false;
    }
    bool best_feasible_for_some_composition(
        int total,
        std::vector<int>& edge_counts,
        const std::vector<std::pair<int, double>>& cut_bounds,
        int pos,
        int used_sum,
        double& best_kappa,
        std::vector<int>& best_edge_counts) const
    {
        const bool use_surplus = true; // whether to use surplus (coverage - bound) or just coverage for kappa objective
        const int m = static_cast<int>(edge_counts.size());

        if (pos == m - 1) {
            edge_counts[pos] = total - used_sum;

            if (!composition_satisfies(edge_counts, cut_bounds)) {
                return false;
            }

            // Secondary objective:
            // minimize sum_c kappa_c * coverage_c
            double kappa_obj = 0.0;
            for (int c = 0; c < static_cast<int>(cut_to_edges_.size()); ++c) {
                int coverage = 0;
                for (int edge_idx : cut_to_edges_[c]) {
                    coverage += edge_counts[edge_idx];
                }
                if (use_surplus) {
                    const int surplus = coverage - cut_bounds[c].first;
                    kappa_obj += cut_bounds[c].second * static_cast<double>(surplus);
                } else {
                    kappa_obj += cut_bounds[c].second * static_cast<double>(coverage);
                }
            }

            if (kappa_obj < best_kappa) {
                best_kappa = kappa_obj;
                best_edge_counts = edge_counts;
            }
            return true;
        }

        bool found = false;
        const int remaining = total - used_sum;
        for (int x = 0; x <= remaining; ++x) {
            edge_counts[pos] = x;
            if (best_feasible_for_some_composition(
                    total, edge_counts, cut_bounds, pos + 1, used_sum + x,
                    best_kappa, best_edge_counts)) {
                found = true;
            }
        }
        return found;
    }    
};

struct SearchNode {
    std::vector<std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>> osr_results;
    GrayCodeCNOT path;
    SearchNode(GrayCodeCNOT path) : path(path) {}
    int get_min_cnots() const {
        return std::get<0>(*std::min_element(osr_results.begin(), osr_results.end(),
                                [](const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& a, const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& b) {
                                    return std::get<0>(a) < std::get<0>(b);
                                }));
    }
    const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& get_best_osr_result() const {
        return *std::min_element(osr_results.begin(), osr_results.end(),
                                [](const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& a, const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& b) {
                                    if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                                    if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
                                    double a_sum = std::accumulate(std::get<3>(a).begin(), std::get<3>(a).end(), 0.0, [&a](double c, const std::pair<int, double>& d){ return c + d.first * std::get<3>(a).size() + d.second; });
                                    double b_sum = std::accumulate(std::get<3>(b).begin(), std::get<3>(b).end(), 0.0, [&b](double c, const std::pair<int, double>& d){ return c + d.first * std::get<3>(b).size() + d.second; });
                                    return a_sum < b_sum;
                                });
    }
    bool operator<(const SearchNode& other) const { return other > *this; }
    bool operator>(const SearchNode& other) const {
        // int min_cnots = get_min_cnots();
        // int other_min_cnots = other.get_min_cnots();
        // int tot_cnot = path.size() + min_cnots;
        // int other_tot_cnot = other.path.size() + other_min_cnots;
        // if (tot_cnot != other_tot_cnot)
        //     return tot_cnot > other_tot_cnot;
        //if (min_cnots != other_min_cnots)
        //    return min_cnots > other_min_cnots;
        const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& best_osr = get_best_osr_result();
        const std::tuple<int, double, std::vector<int>, std::vector<std::pair<int, double>>>& other_best_osr = other.get_best_osr_result();
        if (std::get<0>(best_osr) != std::get<0>(other_best_osr))
            return std::get<0>(best_osr) > std::get<0>(other_best_osr);
        if (std::get<1>(best_osr) != std::get<1>(other_best_osr))
            return std::get<1>(best_osr) > std::get<1>(other_best_osr);
        return std::accumulate(std::get<3>(best_osr).begin(), std::get<3>(best_osr).end(), 0.0, [&best_osr](double c, const std::pair<int, double>& d){ return c + d.first * std::get<3>(best_osr).size() + d.second; }) >
               std::accumulate(std::get<3>(other_best_osr).begin(), std::get<3>(other_best_osr).end(), 0.0, [&other_best_osr](double c, const std::pair<int, double>& d){ return c + d.first * std::get<3>(other_best_osr).size() + d.second; });
        /*int tot = 0;
        for (size_t i = 0; i < osr_results.size(); i++) {
            if (osr_results[i].first < other.osr_results[i].first) { tot -= 1; continue; }
            if (osr_results[i].first > other.osr_results[i].first) { tot += 1; continue; }
            if (ParetoFrontier::dominates(osr_results[i].second, other.osr_results[i].second)) { tot -= 1; continue; }
            if (ParetoFrontier::dominates(other.osr_results[i].second, osr_results[i].second)) { tot += 1; continue; }
        }
        if (tot != 0) return tot > 0;
        return std::accumulate(osr_results.begin(), osr_results.end(), 0.0,
            [](double a, const std::pair<int, std::vector<std::pair<int, double>>>& b){
                return a + std::accumulate(b.second.begin(), b.second.end(), 0.0, [](double c, const std::pair<int, double>& d){ return c + d.first + d.second; });
            }) >
            std::accumulate(other.osr_results.begin(), other.osr_results.end(), 0.0,
            [](double a, const std::pair<int, std::vector<std::pair<int, double>>>& b){
                return a + std::accumulate(b.second.begin(), b.second.end(), 0.0, [](double c, const std::pair<int, double>& d){ return c + d.first + d.second; });
            });*/
    }
};

/**
@brief Structure containing cut information for operator Schmidt rank (OSR) analysis.
This structure tracks all possible qubit cuts, which cuts are affected by each CNOT pair,
and the OSR results (prefixes) for different Gray code sequences.
*/
struct CutInfo {
    /// Vector of all possible qubit cuts, where each cut is represented as a vector of qubit indices
    std::vector<std::vector<int>> all_cuts;
    /// Map from CNOT pair (target, control) to the indices of cuts that are affected by this pair
    MinCnotBoundSolver osr_bound_solver;
    /// Map from Gray code sequences to their OSR result pairs (rank, cost) for different cuts
    std::map<GrayCodeCNOT, SearchNode> prefixes;
    CutInfo(std::vector<std::vector<int>> all_cuts, MinCnotBoundSolver osr_bound_solver) : all_cuts(std::move(all_cuts)), osr_bound_solver(std::move(osr_bound_solver)) {}
};

/**
@brief Structure containing the result of tree search over gate structures with Gray code and level information.
*/
struct TreeSearchResult {
    /// Vector of successful Gray-code solutions
    std::vector<GrayCodeCNOT> solutions;
    /// Updated LevelInfo containing visited states and sequence pairs
    LevelInfo level_info;
    /// Map of GrayCodeCNOT to OSR (Operator Schmidt Rank) result pairs
    std::map<GrayCodeCNOT, SearchNode> prefixes;
};

/**
@brief A base class to determine the decomposition of an N-qubit unitary into a sequence of CNOT and U3 gates.
This class contains the non-template implementation of the decomposition class.
*/
class N_Qubit_Decomposition_Tree_Search : public Optimization_Interface {

  public:
  protected:
    /// The maximal number of adaptive layers used in the decomposition
    int level_limit;
    /// The minimal number of adaptive layers used in the decomposition
    int level_limit_min;
    /// A vector of index pairs encoding the connectivity between the qubits
    std::vector<matrix_base<int>> topology;

    /// List of possible target qubits according to the topology -- paired up with possible control qubits
    matrix_base<int> possible_target_qbits;
    /// List of possible control qubits according to the topology -- paired up with possible target qubits
    matrix_base<int> possible_control_qbits;

  public:
    /**
    @brief Nullary constructor of the class.
    @return An instance of the class
    */
    N_Qubit_Decomposition_Tree_Search();

    /**
    @brief Constructor of the class.
    @param Umtx_in The unitary matrix to be decomposed
    @param qbit_num_in The number of qubits spanning the unitary Umtx
    @param level_limit_in The maximal number of two-qubit gates in the decomposition
    @param config std::map conatining custom config parameters
    @param accelerator_num The number of DFE accelerators used in the calculations
    @return An instance of the class
    */
    N_Qubit_Decomposition_Tree_Search(Matrix Umtx_in, int qbit_num_in, std::map<std::string, Config_Element>& config,
                                      int accelerator_num = 0);

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
    N_Qubit_Decomposition_Tree_Search(Matrix Umtx_in, int qbit_num_in, std::vector<matrix_base<int>> topology_in,
                                      std::map<std::string, Config_Element>& config, int accelerator_num = 0);

    /**
    @brief Destructor of the class
    */
    virtual ~N_Qubit_Decomposition_Tree_Search();

    /**
    @brief Start the disentanglig process of the unitary
    @param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state
    |0> when the disentangling of the qubits is done. Set to False to omit this procedure
    @param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
    */
    virtual void start_decomposition();

    /**
    @brief Call determine the gate structrue of the decomposing circuit. (quantum circuit with CRY gates)
    @param optimized_parameters_mtx_loc A matrix containing the initial parameters
    */
    virtual Gates_block* determine_gate_structure(Matrix_real& optimized_parameters_mtx);

    /**
    @brief Call to add two-qubit building block (two single qubit rotation blocks and one two-qubit gate) to the circuit
    @param gate_structure Appending the two-qubit building block to this circuit
    @param target_qbit The target qubit of the two-qubit gate
    @param control_qbit The control qubit of the two-qubit gate
    */
    void add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit);

    /**
    @brief  Call to construct a gate structure corresponding to the configuration of the two-qubit gates described by
    the Gray code
    @param gcode The N-ary Gray code describing the configuration of the two-qubit gates.
    @return Returns with the generated circuit
    */
    Gates_block* construct_gate_structure_from_Gray_code(const GrayCodeCNOT& gcode, bool finalize = true);

    /**
    @brief Call to perform tree search over possible gate structures
    @param level_mum The number of decomposing levels (i.e. the maximal tree depth)
    @return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be
    costructed by function construct_gate_structure_from_Gray_code)
    */
    GrayCodeCNOT tree_search_over_gate_structures(int level_num);

    SearchNode evaluate_path(
        N_Qubit_Decomposition_custom& cDecomp_custom_random, MinCnotBoundSolver& osr_bound_solver,
        std::vector<std::vector<int>>& all_cuts, double Fnorm, double osr_tol,
        std::uniform_real_distribution<>& distrib_real, std::mt19937& gen,
        const GrayCodeCNOT& path);

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
    TreeSearchResult tree_search_over_gate_structures_osr(int level_num, LevelInfo& li, CutInfo& ci);
    GrayCodeCNOT tree_search_over_gate_structures_best_first();

    /**
    @brief Call to perform the optimization on the given gate structure
    @param gate_structure_loc The gate structure to be optimized
    */
    N_Qubit_Decomposition_custom perform_optimization(Gates_block* gate_structure_loc);

    // Bring base class add_finalyzing_layer into scope to avoid hiding
    using Optimization_Interface::add_finalyzing_layer;

    /**
    @brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
    */
    void add_finalyzing_layer(Gates_block* gate_structure);

    /**
    @brief Set unitary matrix
    @param matrix to set unitary to
    */
    void set_unitary(Matrix& Umtx_new);

    /**
    @brief Perform tabu serach over gate structures
    @return Returns with the best Gray-code corresponding to the best circuit (The associated gate structure can be
    costructed by function construct_gate_structure_from_Gray_code)
    */
    GrayCodeCNOT tabu_search_over_gate_structures();
};

#endif
