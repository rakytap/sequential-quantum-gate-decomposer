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
#include "OSR_Fan_Echo_Motifs.h"
#include "OSR_Pauli_Spine_Placement.h"
#include "OSR_Cut_Capacity.h"
#include "OSR_CNOT_Deflation.h"
#include "N_Qubit_Decomposition_custom.h"

#include "N_Qubit_Decomposition_Cost_Function.h"
#include "OSR_Motif_Library.h"
#include "n_aryGrayCodeCounter.h"

#include <algorithm>
#include <array>
#include <complex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <unordered_map>

namespace {
constexpr double DEFAULT_OSR_OPTIMIZATION_TOLERANCE = 1e-6;

// The Tree model stores theta/2, not Qiskit's full polar angle. Compose two
// adjacent local gates and recover a U3 representative modulo global phase.
std::array<double, 3> merge_tree_u3(const double* after, const double* before) {
    using Complex = std::complex<double>;
    auto matrix = [](const double* p) {
        const double c = std::cos(p[0]), s = std::sin(p[0]);
        return std::array<Complex, 4>{
            c, -s * std::polar(1.0, p[2]), s * std::polar(1.0, p[1]),
            c * std::polar(1.0, p[1] + p[2])
        };
    };
    const auto a = matrix(after), b = matrix(before);
    const Complex m00 = a[0] * b[0] + a[1] * b[2];
    const Complex m01 = a[0] * b[1] + a[1] * b[3];
    const Complex m10 = a[2] * b[0] + a[3] * b[2];
    const Complex m11 = a[2] * b[1] + a[3] * b[3];
    const double c = std::abs(m00), s = std::abs(m10);
    if (s < 1e-14) {
        return {0.0, 0.0, std::arg(m11) - std::arg(m00)};
    }
    const double phase = c >= 1e-14 ? std::arg(m00) : 0.0;
    return {std::atan2(s, c), std::arg(m10) - phase,
            std::arg(-m01) - phase};
}

struct TreeCNOTDeletion {
    bool certified = false;
    std::vector<double> parameters;
    double hs = std::numeric_limits<double>::max();
    double process_infidelity = std::numeric_limits<double>::max();
};

// Shared by certificate deflation and its direct component test/profiler.
// This is the existing chart transition, not a new synthesis strategy.
TreeCNOTDeletion refit_tree_cnot_deletion(
    Matrix& target_adjoint, int width, const std::vector<int>& targets,
    const std::vector<int>& controls, const std::vector<double>& parameters,
    int removed, int sign, std::map<std::string, Config_Element> fit_config,
    int hops, int evaluations, int seed, double tolerance) {
    const int length = targets.size();
    TreeCNOTDeletion result;
    if (length <= 0 || controls.size() != targets.size() ||
        parameters.size() != static_cast<size_t>(6 * length + 3 * width) ||
        removed < 0 || removed >= length || (sign != 1 && sign != -1)) {
        throw std::invalid_argument("Invalid native CNOT deletion chart");
    }
    std::vector<std::array<double, 3>> pending(width);
    std::vector<double> initial;
    initial.reserve(6 * (length - 1) + 3 * width);
    Gates_block structure(width);
    for (int idx = 0; idx < length; ++idx) {
        const int target = targets[idx], control = controls[idx];
        if (target < 0 || control < 0 || target >= width || control >= width || target == control) {
            throw std::invalid_argument("Invalid native CNOT deletion edge");
        }
        pending[target] = merge_tree_u3(parameters.data() + 6 * idx, pending[target].data());
        pending[control] = merge_tree_u3(parameters.data() + 6 * idx + 3, pending[control].data());
        if (idx == removed) {
            const std::array<double, 3> rx{sign * M_PI / 4, -M_PI / 2, M_PI / 2};
            const std::array<double, 3> rz{0.0, 0.0, sign * M_PI / 2};
            pending[target] = merge_tree_u3(rx.data(), pending[target].data());
            pending[control] = merge_tree_u3(rz.data(), pending[control].data());
            continue;
        }
        auto* block = new Gates_block(width);
        block->add_u3(target);
        block->add_u3(control);
        block->add_cnot(target, control);
        structure.add_gate(block);
        for (int wire : {target, control}) {
            initial.insert(initial.end(), pending[wire].begin(), pending[wire].end());
            pending[wire] = {0.0, 0.0, 0.0};
        }
    }
    auto* final = new Gates_block(width);
    for (int wire = 0; wire < width; ++wire) {
        pending[wire] = merge_tree_u3(parameters.data() + 6 * length + 3 * wire,
                                      pending[wire].data());
        initial.insert(initial.end(), pending[wire].begin(), pending[wire].end());
        final->add_u3(wire);
    }
    structure.add_gate(final);
    fit_config["parallel"].set_property("parallel", static_cast<long long>(0));
    fit_config["use_float"].set_property("use_float", false);
    fit_config["use_basin_hopping"].set_property("use_basin_hopping", true);
    fit_config["use_differential_evolution"].set_property("use_differential_evolution", false);
    fit_config["use_dual_annealing"].set_property("use_dual_annealing", false);
    fit_config["optimization_tolerance"].set_property("optimization_tolerance", tolerance);
    fit_config["optimization_tolerance_bfgs2"].set_property("optimization_tolerance_bfgs2", tolerance);
    fit_config["max_iteration_loops_bfgs2"].set_property("max_iteration_loops_bfgs2", static_cast<long long>(hops));
    N_Qubit_Decomposition_custom optimizer(target_adjoint.copy(), width, false, fit_config, RANDOM, 0);
    optimizer.set_verbose(0);
    optimizer.set_debugfile("");
    optimizer.set_optimizer(BFGS2);
    optimizer.set_random_seed(seed);
    optimizer.set_max_iteration(1);
    optimizer.set_max_inner_iterations(evaluations);
    optimizer.set_iteration_loops(width, hops);
    optimizer.set_optimization_tolerance(tolerance);
    optimizer.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
    optimizer.set_custom_gate_structure(&structure);
    optimizer.set_optimization_blocks(structure.get_gate_num());
    optimizer.set_optimized_parameters(initial.data(), static_cast<int>(initial.size()));
    optimizer.start_decomposition();
    Matrix_real fitted = optimizer.get_optimized_parameters();
    Matrix residual = target_adjoint.copy();
    optimizer.apply_to(fitted, residual);
    result.hs = get_hilbert_schmidt_test(residual);
    std::complex<double> trace(0.0, 0.0);
    for (int idx = 0; idx < residual.rows; ++idx) {
        const auto value = residual.data[idx * residual.stride + idx];
        trace += std::complex<double>(value.real, value.imag);
    }
    result.process_infidelity = std::max(0.0, 1.0 - std::norm(trace) /
        (double(residual.rows) * residual.rows));
    if (!std::isfinite(result.hs) || result.hs >= tolerance ||
        !std::isfinite(std::norm(trace)) || result.process_infidelity > tolerance) return result;
    for (int idx = 0; idx < fitted.size(); ++idx) {
        if (!std::isfinite(fitted[idx])) return result;
    }
    result.parameters.assign(fitted.get_data(), fitted.get_data() + fitted.size());
    result.certified = true;
    return result;
}

void trace_tree_cnot_deletion(std::ostream& output, int length, int removed,
                              int sign, const TreeCNOTDeletion& result) {
    output << "OSR CNOT deflation: from=" << length << ", to=" << length - 1
           << ", removed=" << removed << ", branch=" << sign << ", hs=" << result.hs
           << ", process_infidelity=" << result.process_infidelity
           << ", certified=1" << std::endl;
}

}

extern "C" int qgd_osr_refit_cnot_deletion(
    const QGD_Complex16* data, int width, int stride,
    const int* targets, const int* controls, int length,
    const double* parameters, int parameter_count,
    int removed, int sign, int hops, int evaluations, int seed,
    double tolerance, double* child_parameters, int child_parameter_count,
    double* hs, double* process_infidelity, int trace) {
    if (!data || !targets || !controls || !parameters || !child_parameters || !hs || !process_infidelity ||
        width < 2 || width > 4 || stride < (1 << width) || length <= 0 ||
        length > (std::numeric_limits<int>::max() - 3 * width) / 6 ||
        parameter_count != 6 * length + 3 * width ||
        child_parameter_count != 6 * (length - 1) + 3 * width ||
        removed < 0 || removed >= length || (sign != 1 && sign != -1) ||
        hops < 0 || evaluations < 1 || !std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) return -1;
    for (int idx = 0; idx < parameter_count; ++idx) if (!std::isfinite(parameters[idx])) return -1;
    try {
        Matrix target(1 << width, 1 << width);
        for (int row = 0; row < target.rows; ++row) {
            for (int col = 0; col < target.cols; ++col) {
                const auto value = data[row * stride + col];
                if (!std::isfinite(value.real) || !std::isfinite(value.imag)) return -1;
                target.data[row * target.stride + col] = value;
            }
        }
        auto result = refit_tree_cnot_deletion(target, width,
            std::vector<int>(targets, targets + length), std::vector<int>(controls, controls + length),
            std::vector<double>(parameters, parameters + parameter_count), removed, sign, {},
            hops, evaluations, seed, tolerance);
        *hs = result.hs;
        *process_infidelity = result.process_infidelity;
        if (!result.certified) return 0;
        std::copy(result.parameters.begin(), result.parameters.end(), child_parameters);
        if (trace) trace_tree_cnot_deletion(std::cout, length, removed, sign, result);
        return 1;
    } catch (...) {
        return -1;
    }
}

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
        std::fill(limits.data, limits.data + limits.size(), static_cast<int8_t>(topology.size()));
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
    // Insertions only need to be connected in the resulting circuit, not in
    // the inserted subword alone.  In particular, two disjoint inserted
    // edges can form a constructive motif through an inherited parent edge.
    // Seed the support with the parent path so such lifts are generated while
    // root expansions (where curpath is empty) retain the original connected
    // motif restriction.
    uint32_t used_mask = 0u;
    for (int parent_idx = 0; parent_idx < curpath.size(); ++parent_idx) {
        used_mask |= (1u << topology[curpath[parent_idx]][0]) |
                     (1u << topology[curpath[parent_idx]][1]);
    }
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
@brief Constructor of the class from a single precision unitary matrix.
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search(Matrix_float Umtx_in, int qbit_num_in,
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
@brief Constructor of the class from a single precision unitary matrix.
*/
N_Qubit_Decomposition_Tree_Search::N_Qubit_Decomposition_Tree_Search(Matrix_float Umtx_in, int qbit_num_in,
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

    bool export_circuit_2_binary_loc = false;
    if (config.count("export_circuit_2_binary") > 0) {
        config["export_circuit_2_binary"].get_property(export_circuit_2_binary_loc);
    } else {
        export_circuit_2_binary_loc = false;
    }

    if (export_circuit_2_binary_loc) {
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
    bool use_osr = true;
    if (config.count("use_osr") > 0) {
        config["use_osr"].get_property(use_osr);
    }
    bool use_graph_search = true;
    if (config.count("use_graph_search") > 0) {
        config["use_graph_search"].get_property(use_graph_search);
    }

    bool stop_first_solution = true;
    if (config.count("stop_first_solution") > 0) {
        config["stop_first_solution"].get_property(stop_first_solution);
    }

    // Fourteen is the default when the caller supplies no bound.  An explicit
    // tree_level_max (notably routing's fallback-CNOT-count minus one) must be
    // honored rather than silently truncated, or the exact router can miss a
    // valid strict improvement over its fallback.
    level_limit = std::max((int)level_max, 0);

    if (level_limit < 0) {
        std::string error("please increase level limit");
        throw error;
    }

    GrayCodeCNOT best_solution;
    std::vector<GrayCodeCNOT> all_solutions;
    if (use_graph_search) {
        all_solutions.emplace_back(tree_search_over_gate_structures_best_first());
        if (complete_logical_factor(level_limit, false)) {
            all_solutions.back() = best_first_solution_path;
        }
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

        // If OSR search did not find a fully disentangling solution, keep the
        // best prefix candidates discovered so far and evaluate them with
        // Hilbert-Schmidt optimization below.
        if (use_osr && all_solutions.empty() && !ci.prefixes.empty()) {
            all_solutions.reserve(ci.prefixes.size());
            for (std::map<GrayCodeCNOT, SearchNode>::const_iterator it = ci.prefixes.begin(); it != ci.prefixes.end(); ++it) {
                all_solutions.emplace_back(it->first.copy());
            }

            std::stringstream sstream;
            sstream << "OSR did not find a fully disentangled solution; evaluating best prefix candidates with Hilbert-Schmidt optimization." << std::endl;
            print(sstream, 1);
        }
    }
    double fixed_structure_acceptance_tolerance_loc =
        optimization_tolerance_loc;
    if (use_osr || use_graph_search) {
        if (config.count("synthesis_acceptance_tolerance") > 0) {
            double synthesis_acceptance_tolerance_loc;
            config["synthesis_acceptance_tolerance"].get_property(
                synthesis_acceptance_tolerance_loc
            );
            if (synthesis_acceptance_tolerance_loc > 0.0) {
                fixed_structure_acceptance_tolerance_loc = std::max(
                    fixed_structure_acceptance_tolerance_loc,
                    synthesis_acceptance_tolerance_loc
                );
            }
        }
        bool use_hilbert_schmidt_guidance = qbit_num >= 2;
        if (config.count("osr_use_hilbert_schmidt_guidance") > 0) {
            config["osr_use_hilbert_schmidt_guidance"].get_property(
                use_hilbert_schmidt_guidance
            );
        }
        N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(nullptr);
        if (use_hilbert_schmidt_guidance) {
            cDecomp_custom_random.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
            cDecomp_custom_random.set_iteration_loops(qbit_num, 0);
        } else if (config.count("max_iteration_loops") > 0) {
            long long iteration_loop_count;
            config["max_iteration_loops"].get_property(iteration_loop_count);
            cDecomp_custom_random.set_iteration_loops(
                qbit_num,
                static_cast<int>(std::max<long long>(iteration_loop_count, 1))
            );
        }
        std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);
        std::vector<double> optimized_parameters;
        current_minimum = std::numeric_limits<double>::max();
        if (all_solutions.size() == 0) {
            // Last-resort fallback: evaluate the current best-known structure.
            all_solutions.emplace_back(best_solution.copy());
        }
        int osr_zero_hs_attempts = 10;
        if (config.count("osr_zero_hs_attempts") > 0) {
            long long configured_attempts;
            config["osr_zero_hs_attempts"].get_property(
                configured_attempts
            );
            osr_zero_hs_attempts = static_cast<int>(
                std::max<long long>(configured_attempts, 1)
            );
        }
        for (const GrayCodeCNOT& solution : all_solutions) {
            uint64_t solution_seed = 1469598103934665603ULL;
            for (int idx = 0; idx < solution.size(); ++idx) {
                solution_seed ^= static_cast<uint64_t>(solution[idx] + 1);
                solution_seed *= 1099511628211ULL;
            }
            std::seed_seq solution_seed_sequence{
                static_cast<std::uint32_t>(solution_seed),
                static_cast<std::uint32_t>(solution_seed >> 32),
                static_cast<std::uint32_t>(qbit_num)
            };
            std::mt19937 solution_gen(solution_seed_sequence);
            std::unique_ptr<Gates_block> gate_structure_loc;
            gate_structure_loc.reset(construct_gate_structure_from_Gray_code(solution));
            cDecomp_custom_random.set_custom_gate_structure(gate_structure_loc.get());
            cDecomp_custom_random.set_optimization_blocks(gate_structure_loc->get_gate_num());

            // ----------- start the decomposition -----------
            double current_minimum_tmp =
                std::numeric_limits<double>::infinity();
            std::unique_ptr<Matrix_real> best_attempt_parameters;
            const int attempt_count =
                use_hilbert_schmidt_guidance && use_graph_search
                    ? (best_first_osr_solution_found
                           ? osr_zero_hs_attempts
                           : 1)
                    : 5;
            if (use_hilbert_schmidt_guidance && use_graph_search) {
                cDecomp_custom_random.set_iteration_loops(qbit_num, 0);
            }
            const bool has_hs_warm_start =
                use_hilbert_schmidt_guidance && use_graph_search &&
                solution == best_first_solution_path &&
                best_first_solution_parameters.size() ==
                    static_cast<size_t>(
                        cDecomp_custom_random.get_parameter_num()
                    );
            for (int iter = 0; iter < attempt_count; iter++) {
                optimized_parameters.resize(cDecomp_custom_random.get_parameter_num());
                if (has_hs_warm_start && iter == 0) {
                    std::copy(
                        best_first_solution_parameters.begin(),
                        best_first_solution_parameters.end(),
                        optimized_parameters.begin()
                    );
                } else {
                    for (size_t idx = 0;
                         idx < optimized_parameters.size(); ++idx) {
                        optimized_parameters[idx] = distrib_real(solution_gen);
                    }
                }
                cDecomp_custom_random.set_optimized_parameters(optimized_parameters.data(),
                                                               static_cast<int>(optimized_parameters.size()));
                cDecomp_custom_random.start_decomposition();
                Matrix_real fitted = cDecomp_custom_random.get_optimized_parameters();
                double attempt_minimum_tmp =
                    cDecomp_custom_random.get_current_minimum();
                if (use_hilbert_schmidt_guidance) {
                    // Certify the returned circuit, not a cached minimum in
                    // the inherited parent objective or another parameter fit.
                    Matrix certified_residual = Umtx.copy();
                    cDecomp_custom_random.apply_to(fitted, certified_residual);
                    attempt_minimum_tmp = get_hilbert_schmidt_test(certified_residual);
                }
                if (attempt_minimum_tmp < current_minimum_tmp) {
                    current_minimum_tmp = attempt_minimum_tmp;
                    best_attempt_parameters.reset(new Matrix_real(
                        fitted.copy()
                    ));
                }
                if (attempt_minimum_tmp < fixed_structure_acceptance_tolerance_loc) {
                    break;
                }
            }
            if (current_minimum_tmp < current_minimum) {
                current_minimum = current_minimum_tmp;
                optimized_parameters_mtx = best_attempt_parameters->copy();
                sync_optimized_parameters_float();
                best_solution = solution;
            }
            if (current_minimum < fixed_structure_acceptance_tolerance_loc && stop_first_solution) {
                break;
            }
        }
    }

    if (current_minimum > fixed_structure_acceptance_tolerance_loc) {
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
      const GrayCodeCNOT& path, const SearchNode* warm_start,
      bool run_optimization, int iteration_loop_override,
      int max_inner_iteration_override, int target_bound_override,
      const std::vector<double>* supplied_parameters,
      bool score_optimized_osr) {
        SearchNode ev_results(path, qbit_num >= 4);
    // On four or more qubits, cut-profile loss is only a necessary manifold
    // condition and can optimize a structure that fixed-structure synthesis
    // cannot realize. Fit candidate parameters against the target unitary,
    // then retain OSR as the exact discrete residual-bound calculation.
    bool use_hilbert_schmidt_guidance = qbit_num >= 2;
    if (config.count("osr_use_hilbert_schmidt_guidance") > 0) {
        config["osr_use_hilbert_schmidt_guidance"].get_property(
            use_hilbert_schmidt_guidance
        );
    }
    if (use_hilbert_schmidt_guidance && run_optimization) {
        const bool deferred_retry = iteration_loop_override < 0;
        long long guidance_inner_iterations = 250;
        if (config.count("osr_hs_guidance_inner_iterations") > 0) {
            config["osr_hs_guidance_inner_iterations"].get_property(
                guidance_inner_iterations
            );
        }
        // Every candidate gets one local HS solve. Only a candidate promoted
        // by the deferred frontier receives one basin hop before OSR is
        // measured again; broad per-candidate basin hopping remains disabled.
        iteration_loop_override = deferred_retry
            ? 1
            : std::max(iteration_loop_override, 0);
        if (max_inner_iteration_override <= 0) {
            max_inner_iteration_override = static_cast<int>(
                std::max<long long>(guidance_inner_iterations, 1)
            );
        }
    }
    std::unique_ptr<Gates_block> gate_structure_loc(
        construct_gate_structure_from_Gray_code(
            path, use_hilbert_schmidt_guidance
        ));
    cDecomp_custom_random.set_custom_gate_structure(gate_structure_loc.get());
    cDecomp_custom_random.set_optimization_blocks(gate_structure_loc->get_gate_num());
    std::vector<double> optimized_parameters(cDecomp_custom_random.get_parameter_num());
    bool inherited_parameters = false;
    std::vector<bool> inherited_blocks(path.size(), false);
    const size_t final_layer_parameters =
        use_hilbert_schmidt_guidance
            ? static_cast<size_t>(3 * qbit_num)
            : 0;
    if (supplied_parameters != nullptr) {
        if (supplied_parameters->size() != optimized_parameters.size()) {
            throw std::invalid_argument(
                "Supplied path parameters do not match the gate structure."
            );
        }
        optimized_parameters = *supplied_parameters;
        inherited_parameters = true;
    } else if (qbit_num >= 3 && warm_start != nullptr &&
        warm_start->optimized_parameters.size() ==
            static_cast<size_t>(6 * warm_start->path.size()) +
                final_layer_parameters &&
        optimized_parameters.size() ==
            static_cast<size_t>(6 * path.size()) +
                final_layer_parameters) {
        std::fill(
            optimized_parameters.begin(),
            optimized_parameters.end(), 0.0
        );
        int parent_idx = 0;
        for (int child_idx = 0;
             child_idx < path.size() && parent_idx < warm_start->path.size();
             ++child_idx) {
            if (path[child_idx] != warm_start->path[parent_idx])
                continue;
            std::copy_n(
                warm_start->optimized_parameters.begin() + 6 * parent_idx,
                6,
                optimized_parameters.begin() + 6 * child_idx
            );
            inherited_blocks[child_idx] = true;
            ++parent_idx;
        }
          inherited_parameters = parent_idx == warm_start->path.size();
          if (inherited_parameters) {
              if (final_layer_parameters > 0) {
                  std::copy_n(
                      warm_start->optimized_parameters.end() -
                          final_layer_parameters,
                      final_layer_parameters,
                      optimized_parameters.end() - final_layer_parameters
                  );
              }
              // Preserve the fitted parent and its final local layer. The
              // first projection should continue locally from that fit;
              // independent projection restarts still explore broad random
              // coordinates when local continuation is insufficient.
              const bool randomize_new_blocks =
                  (qbit_num >= 4 && run_optimization) ||
                  (qbit_num == 3 && iteration_loop_override > 0);
              double new_block_perturbation =
                  randomize_new_blocks ? M_PI : 0.0;
              if (qbit_num >= 4 &&
                  config.count("osr_warm_start_perturbation") > 0) {
                  config["osr_warm_start_perturbation"].get_property(
                      new_block_perturbation
                  );
                  if (!std::isfinite(new_block_perturbation) ||
                      new_block_perturbation < 0.0) {
                      throw std::invalid_argument(
                          "osr_warm_start_perturbation must be finite and nonnegative"
                      );
                  }
              }
              for (int child_idx = 0; child_idx < path.size(); ++child_idx) {
                  if (inherited_blocks[child_idx])
                      continue;
                  for (int param_idx = 0; param_idx < 6; ++param_idx) {
                      optimized_parameters[6 * child_idx + param_idx] =
                          (distrib_real(gen) / (2.0 * M_PI) - 0.5) *
                          2.0 * new_block_perturbation;
                  }
              }
          }
    }
    if (!inherited_parameters) {
        for (size_t idx = 0; idx < optimized_parameters.size(); idx++) {
            optimized_parameters[idx] = distrib_real(gen);
        }
    }
    cDecomp_custom_random.set_optimized_parameters(optimized_parameters.data(),
                                                    static_cast<int>(optimized_parameters.size()));
    Matrix U;
    Matrix_float U_float;
    Matrix_real_float params_float;
    // Tiny 3q/4q OSR problems receive no meaningful throughput benefit from
    // float32, while their singular-value rank boundaries and placement
    // ordering are precision-sensitive.  Keep the optional float path for
    // larger future partitions, but make the production 3q/4q graph signal
    // identical to the proven float64 path.
    const bool osr_uses_float = use_float && qbit_num >= 5;
    // Use one calibrated smooth objective for three- and four-qubit
    // partitions. The former 4q-only 0.01 default made the profile soft-min
    // effectively hard and silently ignored the configured generic value.
    double profile_temperature = 0.1;
    if (config.count("osr_profile_temperature") > 0)
        config["osr_profile_temperature"].get_property(profile_temperature);
    // Zero selects the hard maximum over cut losses. A positive temperature
    // makes worst-cut pressure differentiable during continuous optimization.
    double cut_smoothmax_temperature = 0.1;
    if (config.count("osr_cut_smoothmax_temperature") > 0)
        config["osr_cut_smoothmax_temperature"].get_property(
            cut_smoothmax_temperature
        );

    N_Qubit_Decomposition_custom joint_optimizer =
        perform_optimization(nullptr, true);
    // The path evaluator already owns the restart-specific generator. Keep
    // basin hopping in the same deterministic stream instead of allowing the
    // nested optimizer to reseed itself from std::random_device. This makes a
    // path/restart pair reproducible without selecting a circuit-specific
    // lucky seed.
    joint_optimizer.set_random_seed(gen());
    if (iteration_loop_override >= 0) {
        joint_optimizer.set_iteration_loops(
            qbit_num, iteration_loop_override
        );
    } else if (config.count("max_iteration_loops") > 0) {
        long long iteration_loop_count;
        config["max_iteration_loops"].get_property(iteration_loop_count);
        joint_optimizer.set_iteration_loops(
            qbit_num,
            static_cast<int>(std::max<long long>(iteration_loop_count, 1))
        );
    }
    if (max_inner_iteration_override >= 0) {
        joint_optimizer.set_max_inner_iterations(
            max_inner_iteration_override
        );
    }
    joint_optimizer.set_cost_function_variant(
        use_hilbert_schmidt_guidance
            ? HILBERT_SCHMIDT_TEST
            : OSR_ENTANGLEMENT
    );
    joint_optimizer.set_custom_gate_structure(gate_structure_loc.get());
    joint_optimizer.set_optimization_blocks(gate_structure_loc->get_gate_num());
    joint_optimizer.set_optimized_parameters(
        optimized_parameters.data(),
        static_cast<int>(optimized_parameters.size())
    );
    bool use_cnot_witnesses = false;
    if (config.count("osr_gate_spectral_witnesses") > 0)
        config["osr_gate_spectral_witnesses"].get_property(use_cnot_witnesses);
    double witness_tolerance = 1e-6;
    if (config.count("osr_optimization_tolerance") > 0)
        config["osr_optimization_tolerance"].get_property(witness_tolerance);
    witness_tolerance = std::min(witness_tolerance, osr_tol*osr_tol);
    auto evaluate_all_cuts = [&](Matrix_real& params) {
        if (osr_uses_float) {
            params.copy_to(params_float);
            Umtx_float.copy_to(U_float);
            joint_optimizer.apply_to(params_float, U_float);
        } else {
            Umtx.copy_to(U);
            joint_optimizer.apply_to(params, U);
        }
        std::vector<std::pair<int, double>> result;
        result.reserve(all_cuts.size());
        for (const std::vector<int>& cut : all_cuts) {
            if (use_cnot_witnesses && osr_uses_float)
                result.emplace_back(operator_schmidt_cnot_bound(
                    U_float, qbit_num, cut, Fnorm, witness_tolerance));
            else if (use_cnot_witnesses)
                result.emplace_back(operator_schmidt_cnot_bound(
                    U, qbit_num, cut, Fnorm, witness_tolerance));
            else if (osr_uses_float)
                result.emplace_back(operator_schmidt_rank(
                    U_float, qbit_num, cut, Fnorm, osr_tol
                ));
            else
                result.emplace_back(operator_schmidt_rank(
                    U, qbit_num, cut, Fnorm, osr_tol
                ));
        }
        return result;
    };

    bool profile_plateau_descent = false;
    if (config.count("osr_profile_plateau_descent") > 0) {
        config["osr_profile_plateau_descent"].get_property(
            profile_plateau_descent
        );
    }
    bool deferred_profile_plateau_descent = false;
    if (qbit_num >= 3 && qbit_num <= 4 &&
        config.count("osr_deferred_profile_plateau_descent") > 0) {
        config["osr_deferred_profile_plateau_descent"].get_property(
            deferred_profile_plateau_descent
        );
    }
    auto lower_budget_profile_loss = [&](int remaining_cnots) {
        // Every circuit with m-1 CNOTs induces one of these edge-cover
        // profiles. The discarded normalized Schmidt energy lower-bounds
        // its process infidelity on each cut. Minimize the worst-cut bound
        // over all profiles, rather than multiplying a tail above the current
        // rank by an edge-cover surplus that can be identically zero.
        const auto profiles = osr_bound_solver.enumerate_cut_coverages(
            std::max(remaining_cnots - 1, 0)
        );
        return osr_uses_float
            ? get_osr_entanglement_test(U_float, all_cuts, profiles, 0.0, 0.0, use_cnot_witnesses)
            : get_osr_entanglement_test(U, all_cuts, profiles, 0.0, 0.0, use_cnot_witnesses);
    };

    Matrix_real params = joint_optimizer.get_optimized_parameters();
    std::vector<std::pair<int, double>> initial_osr_result =
        evaluate_all_cuts(params);
    double initial_kappa = std::numeric_limits<double>::infinity();
    std::vector<int> initial_edge_counts;
    const int initial_min_cnots = osr_bound_solver.solve_min_cnots(
        initial_osr_result, initial_kappa, initial_edge_counts
    );
    bool use_parent_progress_target = qbit_num >= 4;
    if (config.count("osr_parent_progress_target") > 0) {
        config["osr_parent_progress_target"].get_property(
            use_parent_progress_target
        );
    }
      const int target_bound = target_bound_override >= 0
          ? target_bound_override
          : std::max(
                use_parent_progress_target && warm_start != nullptr
                    ? warm_start->get_min_cnots() - 1
                    : initial_min_cnots - 1,
                0
            );
    if (!run_optimization) {
        if (profile_plateau_descent || use_cnot_witnesses) {
            initial_kappa = lower_budget_profile_loss(initial_min_cnots);
        }
        // Identity-placement screens do not consume this auxiliary metric.
        // Rescoring fitted restart/certificate parameters does need it.
        if (deferred_profile_plateau_descent &&
            supplied_parameters != nullptr) {
            ev_results.lower_budget_profile_distance =
                lower_budget_profile_loss(initial_min_cnots);
        }
        // evaluate_all_cuts already reconstructed the exact residual above.
        // A fresh optimizer's cached decomposition error is still DBL_MAX:
        // reading it here silently erased the placement-specific objective.
        // Reuse that residual without another apply_to or optimization.
        if (use_hilbert_schmidt_guidance) {
            ev_results.screening_objective = osr_uses_float
                ? get_hilbert_schmidt_test(U_float)
                : get_hilbert_schmidt_test(U);
        } else {
            const auto screen_profiles =
                osr_bound_solver.enumerate_cut_coverages(target_bound);
            ev_results.screening_objective = osr_uses_float
                ? get_osr_entanglement_test(
                      U_float, all_cuts, screen_profiles, profile_temperature,
                      cut_smoothmax_temperature, use_cnot_witnesses)
                : get_osr_entanglement_test(
                      U, all_cuts, screen_profiles, profile_temperature,
                      cut_smoothmax_temperature, use_cnot_witnesses);
        }
        bool trace_screen = false;
        if (config.count("osr_trace_candidate_placements") > 0) {
            config["osr_trace_candidate_placements"].get_property(trace_screen);
        }
        if (trace_screen) {
            std::stringstream sstream;
            sstream << "OSR placement screen: cnots=" << path.size()
                    << ", residual=" << initial_min_cnots
                    << ", objective=" << ev_results.screening_objective
                    << std::endl;
            print(sstream, 2);
        }
        ev_results.osr_results.emplace_back(
            initial_min_cnots, initial_kappa,
            std::move(initial_edge_counts), std::move(initial_osr_result)
        );
        ev_results.optimized_parameters.resize(params.size());
        std::copy_n(
            params.get_data(), params.size(),
            ev_results.optimized_parameters.begin()
        );
        return ev_results;
    }
    int optimization_target_bound = target_bound;
    while (true) {
        std::vector<std::vector<int>> target_profiles =
            osr_bound_solver.enumerate_cut_coverages(
                optimization_target_bound
            );
          joint_optimizer.set_osr_params(
              all_cuts,
              target_profiles,
              profile_temperature,
              cut_smoothmax_temperature
          );
          joint_optimizer.start_decomposition();
          double smooth_objective =
              joint_optimizer.get_decomposition_error();
          double minimum_profile_loss =
              joint_optimizer.get_osr_min_profile_loss();
          params = joint_optimizer.get_optimized_parameters();
        ev_results.screening_objective = smooth_objective;
        if (!score_optimized_osr) {
            ev_results.optimized_parameters.resize(params.size());
            std::copy_n(
                params.get_data(), params.size(),
                ev_results.optimized_parameters.begin()
            );
            return ev_results;
        }
        std::vector<std::pair<int, double>> osr_result =
            evaluate_all_cuts(params);
        double best_kappa = std::numeric_limits<double>::infinity();
        std::vector<int> best_edge_counts;
        const int min_cnots = osr_bound_solver.solve_min_cnots(
            osr_result, best_kappa, best_edge_counts
        );
        if (profile_plateau_descent || use_cnot_witnesses) {
            best_kappa = lower_budget_profile_loss(min_cnots);
            minimum_profile_loss = best_kappa;
        }
        if (deferred_profile_plateau_descent) {
            ev_results.lower_budget_profile_distance =
                lower_budget_profile_loss(min_cnots);
        }
        {
            std::stringstream sstream;
            sstream << "OSR calibration: path_depth=" << path.size()
                    << ", target_bound=" << optimization_target_bound
                    << ", profiles=" << target_profiles.size()
                    << ", smooth=" << smooth_objective
                    << ", profile_min=" << minimum_profile_loss
                    << ", exact_min_cnots=" << min_cnots
                    << ", exact_kappa=" << best_kappa << std::endl;
            print(sstream, 2);
        }
        ev_results.osr_results.emplace_back(
            min_cnots, best_kappa, std::move(best_edge_counts),
            std::move(osr_result)
        );

        // A two-qubit problem has one cut and at most two residual CNOT
        // tiers.  Finishing the zero tier is effectively free and is needed
        // to absorb boundary SWAPs.  Wider problems retain the deliberately
        // fast one-tier greedy schedule.
        if (qbit_num != 2 || optimization_target_bound == 0)
            break;
        optimization_target_bound = 0;
    }
    ev_results.optimized_parameters.resize(params.size());
    std::copy_n(
        params.get_data(), params.size(), ev_results.optimized_parameters.begin()
    );
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
    std::vector<char> reachable(size_t(1) << m, 0), next_reachable(size_t(1) << m, 0);
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
    best_first_osr_solution_found = false;
    best_first_solution_path = GrayCodeCNOT();
    best_first_solution_parameters.clear();
    if (complete_logical_factor(level_limit, true)) {
        return best_first_solution_path;
    }
    std::vector<std::vector<int>> all_cuts = unique_cuts(qbit_num);
    std::sort(all_cuts.begin(), all_cuts.end(), [](const std::vector<int>& a, const std::vector<int>& b){
        if (a.size() != b.size()) return a.size() < b.size();
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    });
    // If topology entries are actual gates, the path stores topology indices.
    double Fnorm = std::sqrt(static_cast<double>(1 << qbit_num));
    double osr_optimization_tolerance_loc =
        DEFAULT_OSR_OPTIMIZATION_TOLERANCE;
    if (config.count("osr_optimization_tolerance") > 0) {
        config["osr_optimization_tolerance"].get_property(
            osr_optimization_tolerance_loc
        );
    }
    // When Hilbert-Schmidt fitting supplies the continuous objective, OSR is a
    // post-fit structural guide rather than an acceptance test.  The bounded
    // screening solve is deliberately approximate, so applying the tighter OSR
    // optimizer tolerance here can hide the correct edge before the final HS
    // solve gets a chance to refine it.  Keep the two meanings independently
    // configurable; final circuit acceptance is still guarded by the much
    // tighter synthesis_acceptance_tolerance.
    bool use_hilbert_schmidt_guidance = qbit_num >= 2;
    if (config.count("osr_use_hilbert_schmidt_guidance") > 0) {
        config["osr_use_hilbert_schmidt_guidance"].get_property(
            use_hilbert_schmidt_guidance
        );
    }
    double osr_guidance_tolerance_loc = 1e-4;
    if (config.count("osr_guidance_tolerance") > 0) {
        config["osr_guidance_tolerance"].get_property(
            osr_guidance_tolerance_loc
        );
    }
    const double osr_tol = std::sqrt(
        use_hilbert_schmidt_guidance
            ? osr_guidance_tolerance_loc
            : osr_optimization_tolerance_loc
    );
    MinCnotBoundSolver osr_bound_solver(qbit_num, all_cuts, topology);
    std::vector<double> target_edge_mutual_information(
        topology.size(), 0.0
    );
    if (qbit_num == 4) {
        std::vector<double> single_qubit_entropy(qbit_num, 0.0);
        for (int qbit = 0; qbit < qbit_num; ++qbit) {
            single_qubit_entropy[qbit] = operator_schmidt_entropy(
                Umtx, qbit_num, std::vector<int>{qbit}, Fnorm
            );
        }
        for (size_t edge = 0; edge < topology.size(); ++edge) {
            const int q0 = topology[edge][0];
            const int q1 = topology[edge][1];
            const double pair_entropy = operator_schmidt_entropy(
                Umtx, qbit_num, std::vector<int>{q0, q1}, Fnorm
            );
            target_edge_mutual_information[edge] = std::max(
                0.0,
                single_qubit_entropy[q0] +
                    single_qubit_entropy[q1] - pair_entropy
            );
        }
        bool trace_mutual_information = false;
        if (config.count("osr_trace_candidate_placements") > 0) {
            config["osr_trace_candidate_placements"].get_property(
                trace_mutual_information
            );
        }
        if (trace_mutual_information) {
            std::stringstream sstream;
            sstream << "OSR target edge mutual information:";
            for (size_t edge = 0; edge < topology.size(); ++edge) {
                sstream << " " << edge << "=("
                        << topology[edge][0] << ","
                        << topology[edge][1] << "):"
                        << target_edge_mutual_information[edge];
            }
            sstream << std::endl;
            print(sstream, 2);
        }
    }
    //std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<SearchNode>> heap;
    std::unique_ptr<SearchNode> top_heap;
    std::set<GrayCodeCNOT> visited;
    //ForbiddenSubseqSet forbidden(topology);

    N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(nullptr, true);
    cDecomp_custom_random.set_cost_function_variant(
        use_hilbert_schmidt_guidance
            ? HILBERT_SCHMIDT_TEST
            : OSR_ENTANGLEMENT
    );
    long long hilbert_schmidt_guidance_inner_iterations =
        qbit_num >= 4 ? 200 : 250;
    if (config.count("osr_hs_guidance_inner_iterations") > 0) {
        config["osr_hs_guidance_inner_iterations"].get_property(
            hilbert_schmidt_guidance_inner_iterations
        );
    }
    long long final_three_qubit_basin_hops = 1;
    if (config.count("osr_final_3q_basin_hops") > 0) {
        config["osr_final_3q_basin_hops"].get_property(
            final_three_qubit_basin_hops
        );
    }
    final_three_qubit_basin_hops = std::max<long long>(
        final_three_qubit_basin_hops, 0
    );
    bool rank_final_three_qubit_class = true;
    if (config.count("osr_rank_final_3q_class") > 0) {
        config["osr_rank_final_3q_class"].get_property(
            rank_final_three_qubit_class
        );
    }
    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);

    bool use_lazy_deferred_frontier = true;
    if (config.count("osr_lazy_deferred_frontier") > 0) {
        config["osr_lazy_deferred_frontier"].get_property(
            use_lazy_deferred_frontier
        );
    }
    // Four-qubit searches are long enough that consuming the shared RNG in a
    // different candidate order can turn the same path into a pathological
    // basin-hopping run. Seed each path independently there; routing supplies
    // deterministic target-specific seeds to three-qubit catalog synthesis.
    bool use_path_deterministic_rng = qbit_num >= 4;
    if (config.count("osr_path_deterministic_rng") > 0) {
        config["osr_path_deterministic_rng"].get_property(
            use_path_deterministic_rng
        );
    }
    long path_rng_seed = 0;
    if (config.count("random_seed") > 0) {
        config["random_seed"].get_property(path_rng_seed);
    }
      long long projection_restarts = qbit_num >= 4 ? 2 : 1;
      if (config.count("osr_hs_projection_restarts") > 0) {
          config["osr_hs_projection_restarts"].get_property(
              projection_restarts
          );
      }
      projection_restarts = std::max<long long>(projection_restarts, 1);
      bool eager_projection_restarts = true;
      if (config.count("osr_eager_projection_restarts") > 0) {
          config["osr_eager_projection_restarts"].get_property(
              eager_projection_restarts
          );
      }
      bool trace_candidate_placements = false;
      if (config.count("osr_trace_candidate_placements") > 0) {
          config["osr_trace_candidate_placements"].get_property(
          trace_candidate_placements
      );
      }
      // The old unset screening objective was constant across placements.
      // Keep its ordering until the corrected objective passes full-corpus
      // quality/completion replay; the measured node cost is always accurate.
      bool placement_objective_ordering = false;
      if (config.count("osr_placement_objective_ordering") > 0) {
          config["osr_placement_objective_ordering"].get_property(
              placement_objective_ordering
          );
      }
      bool rank_degenerate_cover_class = true;
      bool physical_cover_motifs = false;
      bool physical_cover_motifs_checked = false;
      if (config.count("osr_physical_cover_motifs") > 0) {
          config["osr_physical_cover_motifs"].get_property(physical_cover_motifs);
      }
      if (config.count("osr_rank_degenerate_cover_class") > 0) {
          config["osr_rank_degenerate_cover_class"].get_property(
              rank_degenerate_cover_class
          );
      }
      bool rank_constructive_pair_class = false;
      if (config.count("osr_rank_constructive_pair_class") > 0) {
          config["osr_rank_constructive_pair_class"].get_property(
              rank_constructive_pair_class
          );
      }
      bool screen_before_motif = false;
      if (config.count("osr_screen_before_motif") > 0) {
          config["osr_screen_before_motif"].get_property(
              screen_before_motif
          );
      }
      bool neutral_pair_projection = false;
      if (config.count("osr_neutral_pair_projection") > 0) {
          config["osr_neutral_pair_projection"].get_property(
              neutral_pair_projection
          );
      }

      bool certified_terminal_completion = false;
      if (config.count("osr_certified_terminal_completion") > 0) {
          config["osr_certified_terminal_completion"].get_property(
              certified_terminal_completion
          );
      }
      // The former partial-refinement experiment is an alias for the
      // certified continuation now; necessary spectra never replace a path.
      bool legacy_profile_refinement = false;
      if (config.count("osr_bidirectional_profile_refinement") > 0) {
          config["osr_bidirectional_profile_refinement"].get_property(
              legacy_profile_refinement
          );
      }
      certified_terminal_completion =
          (certified_terminal_completion || legacy_profile_refinement) &&
          use_hilbert_schmidt_guidance && qbit_num >= 3 && qbit_num <= 4;
      double terminal_tolerance = optimization_tolerance;
      if (config.count("synthesis_acceptance_tolerance") > 0) {
          config["synthesis_acceptance_tolerance"].get_property(
              terminal_tolerance
          );
      }
      bool geometric_completion_motifs = false;
      if (config.count("osr_geometric_completion_motifs") > 0) {
          config["osr_geometric_completion_motifs"].get_property(
              geometric_completion_motifs
          );
      }
      geometric_completion_motifs = geometric_completion_motifs &&
        use_hilbert_schmidt_guidance && qbit_num >= 3 && qbit_num <= 4;
      bool certified_cnot_deflation = false;
      if (config.count("osr_certified_cnot_deflation") > 0) {
          config["osr_certified_cnot_deflation"].get_property(
              certified_cnot_deflation
          );
      }
      certified_cnot_deflation = certified_cnot_deflation &&
          use_hilbert_schmidt_guidance && qbit_num >= 3 && qbit_num <= 4;
      if ((certified_terminal_completion || geometric_completion_motifs ||
           certified_cnot_deflation) &&
          (!std::isfinite(terminal_tolerance) ||
           terminal_tolerance <= 0.0 || terminal_tolerance >= 1.0)) {
          throw std::invalid_argument(
              "Terminal synthesis tolerance must be finite and in (0,1)"
          );
      }
      bool geometric_root_checked = false;
      int target_physical_floor = -1;
      int geometric_physical_floor = -1;
      std::vector<int> target_cut_requirements;
      std::unique_ptr<SearchNode> certified_terminal;
      if (certified_terminal_completion || geometric_completion_motifs) {
          std::vector<std::pair<int, double>> target_cuts;
          // Use a conservative structural error scale, never a stricter
          // rank classification than the requested synthesis accuracy.
          const double witness_tolerance = std::max(
              osr_guidance_tolerance_loc, 4.0 * terminal_tolerance
          );
          for (const auto& cut : all_cuts) {
              target_cuts.emplace_back(operator_schmidt_cnot_bound(
                  Umtx, qbit_num, cut, Fnorm, witness_tolerance
              ));
          }
          double surplus = 0.0;
          std::vector<int> counts;
          target_physical_floor = osr_bound_solver.solve_min_cnots(
              target_cuts, surplus, counts
          );
      }

    if (geometric_completion_motifs || certified_cnot_deflation) {
        // Approximate HS guidance and target-family eligibility have different
        // meanings. A weak interaction can be invisible at the coarse guide
        // scale but necessary at synthesis precision. Preserve the original
        // guide and terminal search; classify constructive charts separately.
        std::vector<int> capacity_minimums(qbit_num, 0);
        for (int root = 0; root < qbit_num; ++root) {
            const auto forward = squander_osr::cut_capacity(
                Umtx.data, qbit_num, Umtx.stride, root, terminal_tolerance, false);
            const auto inverse = squander_osr::cut_capacity(
                Umtx.data, qbit_num, Umtx.stride, root, terminal_tolerance, true);
            capacity_minimums[root] = std::max(forward.minimum_crossing_cnots,
                                              inverse.minimum_crossing_cnots);
        }
        std::vector<std::pair<int, double>> structural_cuts;
        for (const auto& cut : all_cuts) {
            structural_cuts.emplace_back(operator_schmidt_cnot_bound(
                Umtx, qbit_num, cut, Fnorm, 4.0 * terminal_tolerance
            ));
            int root = -1;
            if (cut.size() == 1) {
                root = cut.front();
            } else if (cut.size() + 1 == static_cast<size_t>(qbit_num)) {
                for (int wire = 0; wire < qbit_num; ++wire) {
                    if (std::find(cut.begin(), cut.end(), wire) == cut.end()) root = wire;
                }
            }
            if (root >= 0) {
                if (capacity_minimums[root] > structural_cuts.back().first) {
                    structural_cuts.back().first = capacity_minimums[root];
                    structural_cuts.back().second = 0.0;
                }
            }
            target_cut_requirements.push_back(structural_cuts.back().first);
        }
        if (geometric_completion_motifs) {
            double surplus = 0.0;
            std::vector<int> counts;
            geometric_physical_floor = osr_bound_solver.solve_min_cnots(
                structural_cuts, surplus, counts
            );
        }
    }

    if (certified_terminal_completion && qbit_num == 4 &&
        target_physical_floor > 0 && target_physical_floor <= level_limit) {
        // OSR one across a singleton cut means an exact tensor product.
        // Extract it by normalized block projection; no synthesis parameters
        // or circuit structure enter this algebraic factorization.
        using Complex = std::complex<double>;
        for (int spectator = 0; spectator < qbit_num; ++spectator) {
            std::vector<int> active;
            std::array<int, 4> compact{};
            compact.fill(-1);
            for (int qbit = 0; qbit < qbit_num; ++qbit) {
                if (qbit == spectator) continue;
                compact[qbit] = static_cast<int>(active.size());
                active.push_back(qbit);
            }
            if (operator_schmidt_cnot_bound(
                    Umtx, qbit_num, active, Fnorm,
                    std::max(osr_guidance_tolerance_loc,
                             4.0 * terminal_tolerance)).first != 0) {
                continue;
            }
            std::vector<matrix_base<int>> reduced_topology;
            std::vector<int> parent_digits;
            for (int edge = 0; edge < possible_target_qbits.size(); ++edge) {
                const int a = possible_target_qbits[edge];
                const int b = possible_control_qbits[edge];
                if (a == spectator || b == spectator) continue;
                matrix_base<int> reduced_edge(1, 2);
                reduced_edge[0] = compact[a];
                reduced_edge[1] = compact[b];
                reduced_topology.push_back(reduced_edge);
                parent_digits.push_back(edge);
            }
            std::array<bool, 3> connected{true, false, false};
            for (int pass = 0; pass < 3; ++pass) {
                for (const auto& edge : reduced_topology) {
                    if (connected[edge[0]] || connected[edge[1]]) {
                        connected[edge[0]] = connected[edge[1]] = true;
                    }
                }
            }
            // A spectator can still be a necessary routing bus. Do not
            // remove it when the active induced graph is disconnected.
            if (!std::all_of(connected.begin(), connected.end(),
                             [](bool value) { return value; })) {
                continue;
            }
            std::array<int, 8> indices{};
            for (int index = 0; index < 8; ++index) {
                for (int bit = 0; bit < 3; ++bit) {
                    indices[index] |= ((index >> bit) & 1) << active[bit];
                }
            }
            auto element = [&](int block, int row, int col) {
                const int full_row = indices[row] |
                    ((block / 2) << spectator);
                const int full_col = indices[col] |
                    ((block % 2) << spectator);
                const auto value = Umtx.data[
                    full_row * Umtx.stride + full_col
                ];
                return Complex(value.real, value.imag);
            };
            int reference = 0;
            double norm = 0.0;
            for (int block = 0; block < 4; ++block) {
                double candidate_norm = 0.0;
                for (int row = 0; row < 8; ++row) {
                    for (int col = 0; col < 8; ++col) {
                        candidate_norm += std::norm(element(block, row, col));
                    }
                }
                if (candidate_norm > norm) {
                    reference = block;
                    norm = candidate_norm;
                }
            }
            if (norm <= 0.0) continue;
            const double scale = std::sqrt(norm / 8.0);
            Matrix active_inverse(8, 8);
            for (int row = 0; row < 8; ++row) {
                for (int col = 0; col < 8; ++col) {
                    const Complex value = element(reference, row, col) / scale;
                    active_inverse.data[row * active_inverse.stride + col] =
                        QGD_Complex16{value.real(), value.imag()};
                }
            }
            std::array<Complex, 4> local_inverse{};
            for (int block = 0; block < 4; ++block) {
                for (int row = 0; row < 8; ++row) {
                    for (int col = 0; col < 8; ++col) {
                        local_inverse[block] +=
                            std::conj(element(reference, row, col) / scale) *
                            element(block, row, col) / 8.0;
                    }
                }
            }
            double reconstruction_error = 0.0;
            for (int block = 0; block < 4; ++block) {
                for (int row = 0; row < 8; ++row) {
                    for (int col = 0; col < 8; ++col) {
                        reconstruction_error += std::norm(
                            local_inverse[block] *
                                element(reference, row, col) / scale -
                                element(block, row, col));
                    }
                }
            }
            if (reconstruction_error / 16.0 >= terminal_tolerance) continue;
            auto reduced_config = config;
            reduced_config["use_float"].set_property("use_float", false);
            reduced_config["optimization_tolerance"].set_property(
                "optimization_tolerance", terminal_tolerance
            );
            reduced_config["osr_certified_terminal_completion"].set_property(
                "osr_certified_terminal_completion", true
            );
            N_Qubit_Decomposition_Tree_Search reduced(
                active_inverse, 3, reduced_topology, reduced_config, 0
            );
            reduced.set_verbose(0);
            reduced.set_optimizer(alg);
            reduced.set_random_seed(static_cast<int>(path_rng_seed));
            reduced.set_optimization_tolerance(terminal_tolerance);
            // The OSR floor is a lower bound, not a synthesis budget. An
            // exact spectator must not prevent the established 3q solver
            // from reaching a valid count above that floor. The spectator
            // adds no CNOTs, so inherit the parent's actual CNOT budget.
            reduced.level_limit = level_limit;
            const GrayCodeCNOT reduced_path =
                reduced.tree_search_over_gate_structures_best_first();
            if (!reduced.best_first_osr_solution_found ||
                reduced_path.size() > static_cast<size_t>(level_limit)) {
                continue;
            }
            const int depth = reduced_path.size();
            matrix_base<int8_t> limits(1, depth);
            std::fill_n(limits.data, limits.size(),
                        static_cast<int8_t>(topology.size()));
            GrayCodeCNOT embedded(limits);
            for (int idx = 0; idx < depth; ++idx) {
                embedded[idx] = static_cast<int8_t>(
                    parent_digits[reduced_path[idx]]
                );
            }
            const auto& reduced_parameters =
                reduced.best_first_solution_parameters;
            if (reduced_parameters.size() !=
                static_cast<size_t>(6 * depth + 9)) {
                throw std::logic_error("Product-cut parameter count mismatch");
            }
            std::vector<double> parameters(6 * depth + 12);
            std::copy_n(reduced_parameters.data(), 6 * depth,
                        parameters.data());
            for (int bit = 0; bit < 3; ++bit) {
                std::copy_n(reduced_parameters.data() + 6 * depth + 3 * bit, 3,
                            parameters.data() + 6 * depth + 3 * active[bit]);
            }
            // The spectator's forward factor is the adjoint of its extracted
            // inverse factor. Euler recovery ignores only a global phase.
            const Complex m00 = std::conj(local_inverse[0]);
            const Complex m01 = std::conj(local_inverse[2]);
            const Complex m10 = std::conj(local_inverse[1]);
            const Complex m11 = std::conj(local_inverse[3]);
            const double c = std::abs(m00), s = std::abs(m10);
            double* local = parameters.data() + 6 * depth + 3 * spectator;
            if (s < 1e-14) {
                local[0] = local[1] = 0.0;
                local[2] = std::arg(m11) - std::arg(m00);
            } else {
                const double phase = c >= 1e-14 ? std::arg(m00) : 0.0;
                local[0] = std::atan2(s, c);
                local[1] = std::arg(m10) - phase;
                local[2] = std::arg(-m01) - phase;
            }
            Matrix verification = Umtx.copy();
            Matrix_real view(parameters.data(), 1,
                             static_cast<int>(parameters.size()));
            std::unique_ptr<Gates_block> structure(
                construct_gate_structure_from_Gray_code(embedded, true)
            );
            structure->apply_to(view, verification, 0);
            const double hs = get_hilbert_schmidt_test(verification);
            if (trace_candidate_placements) {
                std::stringstream sstream;
                sstream << "OSR product-cut completion: spectator="
                        << spectator << ", cnots=" << depth
                        << ", floor=" << target_physical_floor
                        << ", hs=" << hs
                        << ", certified="
                        << (std::isfinite(hs) && hs < terminal_tolerance)
                        << std::endl;
                print(sstream, 2);
            }
            if (std::isfinite(hs) && hs < terminal_tolerance) {
                best_first_osr_solution_found = true;
                best_first_solution_path = embedded.copy();
                best_first_solution_parameters = std::move(parameters);
                return embedded;
            }
        }
    }

    auto complete_terminal_path = [&](
        const GrayCodeCNOT& path, SearchNode best, bool run_optimization
    ) -> SearchNode {
        const bool try_terminal = certified_terminal_completion &&
            path.size() + 1 == target_physical_floor &&
            path.size() < level_limit;
        if (!try_terminal || !run_optimization ||
            !use_hilbert_schmidt_guidance || path.size() == 0 ||
            best.get_min_cnots() == 0) {
            return best;
        }

        // HS proximity and residual CNOT complexity are different objectives.
        // Refine a fixed, nearest cut-cover family after the cheap HS fit.
        // These are necessary spectral conditions, not synthesis certificates.
        // Keep this experimental continuation off until full-corpus replay.
        const size_t parameter_count =
            static_cast<size_t>(6 * path.size() + 3 * qbit_num);
        if (best.optimized_parameters.size() != parameter_count) {
            throw std::logic_error(
                "OSR refinement requires the complete final local layer"
            );
        }
        std::vector<std::vector<int>> profiles;
        // Keep the exact edge-to-profile correspondence. A necessary
        // spectrum match is only a proposal for a constructive closer.
        for (const auto& edge : topology) {
            std::vector<int> profile;
            for (const auto& cut : all_cuts) {
                const bool a = std::find(cut.begin(), cut.end(), edge[0]) !=
                    cut.end();
                const bool b = std::find(cut.begin(), cut.end(), edge[1]) !=
                    cut.end();
                profile.push_back(a != b ? 1 : 0);
            }
            profiles.push_back(std::move(profile));
        }
        if (profiles.empty()) {
            return best;
        }
        const std::vector<double> hs_parameters = best.optimized_parameters;
        double refinement_tolerance = DEFAULT_OSR_OPTIMIZATION_TOLERANCE;
        if (config.count("osr_optimization_tolerance") > 0) {
            config["osr_optimization_tolerance"].get_property(
                refinement_tolerance
            );
        }
        if (!std::isfinite(refinement_tolerance) ||
            refinement_tolerance <= 0.0) {
            throw std::invalid_argument(
                "OSR refinement tolerance must be finite and positive"
            );
        }
        long long refinement_iterations = 1000;
        if (config.count("max_inner_iterations_bfgs2") > 0) {
            config["max_inner_iterations_bfgs2"].get_property(
                refinement_iterations
            );
        }
        refinement_iterations = std::max<long long>(
            refinement_iterations, 1
        );

        // Exact parameter map for C -> C-dagger. Each Tree block stores
        // U3(target), U3(control), CNOT, followed by the final local layer.
        // U3(theta, phi, lambda)-dagger = U3(-theta, -lambda, -phi).
        // Move the pending local rotations wire by wire when reversing CNOTs.
        auto inverse_parameters = [&](
            const GrayCodeCNOT& word, const std::vector<double>& parameters
        ) {
            std::vector<double> pending(3 * qbit_num);
            std::vector<double> inverse(parameters.size());
            auto dagger = [](const double* source, double* destination) {
                destination[0] = -source[0];
                destination[1] = -source[2];
                destination[2] = -source[1];
            };
            const size_t final_offset = static_cast<size_t>(6 * word.size());
            for (int qbit = 0; qbit < qbit_num; ++qbit) {
                dagger(parameters.data() + final_offset + 3 * qbit,
                       pending.data() + 3 * qbit);
            }
            for (int idx = 0; idx < word.size(); ++idx) {
                const int original_idx = word.size() - 1 - idx;
                const int first = possible_target_qbits[word[original_idx]];
                const int second = possible_control_qbits[word[original_idx]];
                std::copy_n(pending.data() + 3 * first, 3,
                            inverse.data() + 6 * idx);
                std::copy_n(pending.data() + 3 * second, 3,
                            inverse.data() + 6 * idx + 3);
                dagger(parameters.data() + 6 * original_idx,
                       pending.data() + 3 * first);
                dagger(parameters.data() + 6 * original_idx + 3,
                       pending.data() + 3 * second);
            }
            std::copy(pending.begin(), pending.end(),
                      inverse.begin() + final_offset);
            return inverse;
        };

        // Both frames start from the SAME HS fit. Optimizing C-dagger against
        // U-dagger gives (U-dagger*C)-dagger, whose cut singular spectra equal
        // those of the input-side residual. This reuses the native OSR gradient.
        for (int input_frame = 0; input_frame < 2; ++input_frame) {
            GrayCodeCNOT refinement_path = path.copy();
            Matrix refinement_target = Umtx.copy();
            std::vector<double> initial = hs_parameters;
            if (input_frame != 0) {
                for (int idx = 0; idx < path.size(); ++idx) {
                    refinement_path[idx] = path[path.size() - 1 - idx];
                }
                initial = inverse_parameters(path, hs_parameters);
                for (int row = 0; row < Umtx.rows; ++row) {
                    for (int col = 0; col < Umtx.cols; ++col) {
                        const QGD_Complex16 element =
                            Umtx.data[col * Umtx.stride + row];
                        refinement_target.data[
                            row * refinement_target.stride + col
                        ] = QGD_Complex16{element.real, -element.imag};
                    }
                }
            }
            auto refinement_config = config;
            refinement_config["use_float"].set_property("use_float", false);
            refinement_config["parallel"].set_property(
                "parallel", static_cast<long long>(0)
            );
            refinement_config["osr_gate_spectral_witnesses"].set_property(
                "osr_gate_spectral_witnesses", true
            );
            refinement_config["optimization_tolerance"].set_property(
                "optimization_tolerance", refinement_tolerance
            );
            refinement_config["max_inner_iterations_bfgs2"].set_property(
                "max_inner_iterations_bfgs2", refinement_iterations
            );
            N_Qubit_Decomposition_custom refinement(
                refinement_target.copy(), qbit_num, false,
                refinement_config, RANDOM, 0
            );
            std::unique_ptr<Gates_block> structure(
                construct_gate_structure_from_Gray_code(refinement_path, true)
            );
            refinement.set_custom_gate_structure(structure.get());
            refinement.set_optimization_blocks(structure->get_gate_num());
            refinement.set_optimized_parameters(
                initial.data(), static_cast<int>(initial.size())
            );
            refinement.set_verbose(0);
            refinement.set_debugfile("");
            refinement.set_random_seed(0);
            refinement.set_optimizer(BFGS2);
            refinement.set_max_iteration(1);
            refinement.set_iteration_loops(qbit_num, 0);
            refinement.set_max_inner_iterations(
                static_cast<int>(std::min<long long>(
                    refinement_iterations, std::numeric_limits<int>::max()
                ))
            );
            refinement.set_optimization_tolerance(refinement_tolerance);
            Matrix_real parameters = refinement.get_optimized_parameters();
            Matrix residual = refinement_target.copy();
            refinement.apply_to(parameters, residual);

            // Cache scalar family losses by (cut, crossing count), rather
            // than repeating every SVD for every edge-cover composition.
            std::vector<std::map<int, double>> cut_losses(all_cuts.size());
            size_t selected_profile = 0;
            double before_loss = std::numeric_limits<double>::infinity();
            for (size_t profile_idx = 0; profile_idx < profiles.size();
                 ++profile_idx) {
                double loss = 0.0;
                for (size_t cut_idx = 0; cut_idx < all_cuts.size(); ++cut_idx) {
                    const int crossings = profiles[profile_idx][cut_idx];
                    auto& cache = cut_losses[cut_idx];
                    auto found = cache.find(crossings);
                    if (found == cache.end()) {
                        std::vector<std::vector<int>> one_cut{
                            all_cuts[cut_idx]
                        };
                        const double cut_loss = get_osr_entanglement_test(
                            residual, one_cut,
                            std::vector<std::vector<int>>{{crossings}},
                            0.0, 0.0, true
                        );
                        found = cache.emplace(crossings, cut_loss).first;
                    }
                    loss = std::max(loss, found->second);
                }
                if (loss < before_loss) {
                    before_loss = loss;
                    selected_profile = profile_idx;
                }
            }
            refinement.set_cost_function_variant(OSR_ENTANGLEMENT);
            refinement.set_osr_params(
                all_cuts,
                std::vector<std::vector<int>>{profiles[selected_profile]},
                0.0, 0.1
            );
            refinement.start_decomposition();
            parameters = refinement.get_optimized_parameters();
            refinement_target.copy_to(residual);
            refinement.apply_to(parameters, residual);

            const double after_loss = get_osr_entanglement_test(
                residual, all_cuts,
                std::vector<std::vector<int>>{profiles[selected_profile]},
                0.0, 0.0, true
            );
            if (try_terminal && after_loss < refinement_tolerance) {
                const int edge = static_cast<int>(selected_profile);
                matrix_base<int8_t> closing_limits(1, 1);
                closing_limits[0] = static_cast<int8_t>(topology.size());
                GrayCodeCNOT closing_path(closing_limits);
                closing_path[0] = static_cast<int8_t>(edge);
                auto closing_config = refinement_config;
                closing_config["optimization_tolerance"].set_property(
                    "optimization_tolerance", refinement_tolerance
                );
                closing_config["optimization_tolerance_bfgs2"].set_property(
                    "optimization_tolerance_bfgs2", refinement_tolerance
                );
                closing_config["max_iteration_loops_bfgs2"].set_property(
                    "max_iteration_loops_bfgs2", static_cast<long long>(5)
                );
                closing_config["use_basin_hopping"].set_property(
                    "use_basin_hopping", true
                );
                closing_config["use_differential_evolution"].set_property(
                    "use_differential_evolution", false
                );
                closing_config["use_dual_annealing"].set_property(
                    "use_dual_annealing", false
                );
                N_Qubit_Decomposition_custom closing(
                    residual.copy(), qbit_num, false, closing_config, RANDOM, 0
                );
                std::unique_ptr<Gates_block> closing_structure(
                    construct_gate_structure_from_Gray_code(closing_path, true)
                );
                closing.set_custom_gate_structure(closing_structure.get());
                closing.set_optimization_blocks(closing_structure->get_gate_num());
                closing.set_verbose(0);
                closing.set_debugfile("");
                closing.set_optimizer(BFGS2);
                closing.set_random_seed(static_cast<int>(path_rng_seed));
                closing.set_max_iteration(1);
                closing.set_max_inner_iterations(static_cast<int>(
                    std::min<long long>(refinement_iterations,
                                        std::numeric_limits<int>::max())
                ));
                closing.set_iteration_loops(qbit_num, 5);
                closing.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
                closing.set_optimization_tolerance(refinement_tolerance);
                std::mt19937 closing_gen(static_cast<unsigned>(path_rng_seed));
                std::uniform_real_distribution<double> angle(0.0, 2 * M_PI);
                std::vector<double> initial_closing(
                    closing_structure->get_parameter_num()
                );
                for (double& value : initial_closing) value = angle(closing_gen);
                closing.set_optimized_parameters(
                    initial_closing.data(), static_cast<int>(initial_closing.size())
                );
                closing.start_decomposition();
                Matrix_real z = closing.get_optimized_parameters();

                const int depth = refinement_path.size();
                matrix_base<int8_t> child_limits(1, depth + 1);
                std::fill_n(child_limits.data, child_limits.size(),
                            static_cast<int8_t>(topology.size()));
                GrayCodeCNOT child_path(child_limits);
                for (int idx = 0; idx < depth; ++idx) {
                    child_path[idx] = refinement_path[idx];
                }
                child_path[depth] = static_cast<int8_t>(edge);
                std::vector<double> merged(6 * (depth + 1) + 3 * qbit_num);
                std::copy_n(parameters.get_data(), 6 * depth, merged.data());
                const int first = possible_target_qbits[edge];
                const int second = possible_control_qbits[edge];
                auto local = merge_tree_u3(
                    z.get_data(), parameters.get_data() + 6 * depth + 3 * first
                );
                std::copy(local.begin(), local.end(), merged.begin() + 6 * depth);
                local = merge_tree_u3(
                    z.get_data() + 3,
                    parameters.get_data() + 6 * depth + 3 * second
                );
                std::copy(local.begin(), local.end(),
                          merged.begin() + 6 * depth + 3);
                for (int qbit = 0; qbit < qbit_num; ++qbit) {
                    const double* final = z.get_data() + 6 + 3 * qbit;
                    if (qbit == first || qbit == second) {
                        std::copy_n(final, 3,
                            merged.begin() + 6 * (depth + 1) + 3 * qbit);
                    } else {
                        local = merge_tree_u3(final,
                            parameters.get_data() + 6 * depth + 3 * qbit);
                        std::copy(local.begin(), local.end(),
                            merged.begin() + 6 * (depth + 1) + 3 * qbit);
                    }
                }
                std::unique_ptr<Gates_block> child_structure(
                    construct_gate_structure_from_Gray_code(child_path, true)
                );
                auto polish_config = closing_config;
                polish_config["optimization_tolerance"].set_property(
                    "optimization_tolerance", terminal_tolerance
                );
                polish_config["max_iteration_loops_bfgs"].set_property(
                    "max_iteration_loops_bfgs", static_cast<long long>(5)
                );
                N_Qubit_Decomposition_custom polish(
                    refinement_target.copy(), qbit_num, false,
                    polish_config, RANDOM, 0
                );
                polish.set_custom_gate_structure(child_structure.get());
                polish.set_optimization_blocks(child_structure->get_gate_num());
                polish.set_optimized_parameters(
                    merged.data(), static_cast<int>(merged.size())
                );
                polish.set_verbose(0);
                polish.set_debugfile("");
                polish.set_optimizer(BFGS);
                polish.set_random_seed(static_cast<int>(path_rng_seed));
                polish.set_max_iteration(1);
                polish.set_max_inner_iterations(static_cast<int>(
                    std::min<long long>(refinement_iterations,
                                        std::numeric_limits<int>::max())
                ));
                polish.set_iteration_loops(qbit_num, 5);
                polish.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
                polish.set_optimization_tolerance(terminal_tolerance);
                // No angles are frozen: the composed fit is only the initial
                // point of a full joint fixed-structure optimization.
                polish.start_decomposition();
                Matrix_real polished = polish.get_optimized_parameters();
                std::vector<double> fitted(polished.get_data(),
                    polished.get_data() + polished.size());
                if (input_frame != 0) {
                    fitted = inverse_parameters(child_path, fitted);
                    for (int idx = 0; idx < (depth + 1) / 2; ++idx) {
                        std::swap(child_path[idx], child_path[depth - idx]);
                    }
                }
                Matrix verification = Umtx.copy();
                Matrix_real fitted_view(fitted.data(), 1,
                                        static_cast<int>(fitted.size()));
                std::unique_ptr<Gates_block> forward_structure(
                    construct_gate_structure_from_Gray_code(child_path, true)
                );
                forward_structure->apply_to(fitted_view, verification, 0);
                const double hs = get_hilbert_schmidt_test(verification);
                const bool certified = std::isfinite(hs) &&
                    hs < terminal_tolerance;
                if (trace_candidate_placements) {
                    std::stringstream sstream;
                    sstream << "OSR terminal completion: parent_depth=" << depth
                            << ", frame=" << (input_frame ? "input" : "output")
                            << ", edge=" << edge << ", cnots=" << depth + 1
                            << ", floor=" << target_physical_floor
                            << ", hs=" << hs << ", certified=" << certified
                            << std::endl;
                    print(sstream, 2);
                }
                if (certified) {
                    SearchNode completed(child_path, best.use_optimistic_score);
                    completed.optimized_parameters = std::move(fitted);
                    completed.screening_objective = hs;
                    completed.lower_budget_profile_distance = 0.0;
                    completed.osr_results.emplace_back(
                        0, 0.0, std::vector<int>(topology.size(), 0),
                        std::vector<std::pair<int, double>>(
                            all_cuts.size(), std::make_pair(0, 0.0))
                    );
                    // Only a complete certificate at the target lower bound
                    // can replace the current forward search, never a partial
                    // cut-profile improvement or an unverified incumbent.
                    certified_terminal.reset(new SearchNode(completed));
                    return completed;
                }
            }

        }
        return best;
    };

      auto evaluate_candidate_path = [&](
          const GrayCodeCNOT& path, const SearchNode* warm_start,
          bool run_optimization, int iteration_loop_override,
          int max_inner_iteration_override,
          int target_bound_override = -1,
          bool full_projection_restart_budget = false) -> SearchNode {
        const bool defer_postfit_osr =
            qbit_num >= 3 && run_optimization && eager_projection_restarts &&
            projection_restarts > 1;
        auto evaluate_once = [&] (
            long long restart_index,
            const SearchNode* restart_warm_start,
            bool score_osr,
            const std::vector<double>* initial_parameters
        ) -> SearchNode {
            if (use_path_deterministic_rng || !run_optimization) {
                // Deterministic replay must still retain independent restart
                // projections.  The previous 3q branch ignored
                // ``restart_index``, making every configured restart
                // identical when deterministic mode was enabled.
                const uint64_t restart_seed = qbit_num >= 4
                    ? static_cast<uint64_t>(restart_index)
                    : (static_cast<uint64_t>(path_rng_seed) << 32) ^
                          static_cast<uint64_t>(restart_index);
                uint64_t hash = 1469598103934665603ULL ^ restart_seed;
                for (int idx = 0; idx < path.size(); ++idx) {
                    hash ^= static_cast<uint64_t>(path[idx] + 1);
                    hash *= 1099511628211ULL;
                }
                auto path_gen = gen;
                path_gen.seed(static_cast<unsigned long>(
                    hash ^ (hash >> 32)
                ));
                return evaluate_path(
                    cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm,
                    osr_tol, distrib_real, path_gen, path, restart_warm_start,
                      run_optimization, iteration_loop_override,
                      max_inner_iteration_override, target_bound_override,
                      initial_parameters, score_osr
                );
            }
            return evaluate_path(
                cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm,
                osr_tol, distrib_real, gen, path, restart_warm_start,
                  run_optimization, iteration_loop_override,
                  max_inner_iteration_override, target_bound_override,
                  initial_parameters, score_osr
            );
        };

        SearchNode best = evaluate_once(
            0, warm_start, !defer_postfit_osr, nullptr
        );
        // Most paths are placement screens, not final synthesis candidates.
        // Spending the complete independent-projection portfolio on every
        // member of a finite one/two-CNOT class doubles its dominant cost.
        // Keep one deterministic projection for those screens and reserve
        // the configured restart portfolio for a promoted candidate or a
        // complete constructive motif. This is the same lazy policy already
        // used for basin-hopping iterations below.
        const bool first_projection_improves_parent =
            !defer_postfit_osr && warm_start != nullptr &&
            best.get_min_cnots() < warm_start->get_min_cnots();
        // Honor the requested projection portfolio on both supported
        // widths. The 3q default is still one solve; an explicit restart
        // request must not silently collapse to one because of its width.
        const long long restart_limit =
            qbit_num >= 3 && run_optimization &&
                    (eager_projection_restarts ||
                     full_projection_restart_budget ||
                     first_projection_improves_parent)
                ? projection_restarts
                : 1;
        for (long long restart_index = 1;
             restart_index < restart_limit; ++restart_index) {
            if (trace_candidate_placements) {
                std::stringstream sstream;
                sstream << "OSR projection restart: path_depth=" << path.size()
                        << ", restart_index=" << restart_index << std::endl;
                print(sstream, 2);
            }
            SearchNode retry = evaluate_once(
                restart_index, nullptr, !defer_postfit_osr, nullptr
            );
            // For identical circuit structures, select the closest HS
            // projection before measuring OSR. Thresholded rank is not a
            // valid discriminator between different local HS fits.
            if (retry.screening_objective < best.screening_objective) {
                best = std::move(retry);
            }
        }
        if (neutral_pair_projection && qbit_num >= 4 &&
            use_hilbert_schmidt_guidance && run_optimization &&
            warm_start != nullptr &&
            path.size() == warm_start->path.size() + 2) {
            const size_t final_layer_parameters =
                static_cast<size_t>(3 * qbit_num);
            const size_t parent_parameters =
                static_cast<size_t>(6 * warm_start->path.size()) +
                final_layer_parameters;
            if (warm_start->optimized_parameters.size() ==
                parent_parameters) {
                for (int insert_at = 0; insert_at + 1 < path.size();
                     ++insert_at) {
                    if (path[insert_at] != path[insert_at + 1]) {
                        continue;
                    }
                    bool matches_parent = true;
                    int parent_idx = 0;
                    for (int child_idx = 0; child_idx < path.size();
                         ++child_idx) {
                        if (child_idx == insert_at ||
                            child_idx == insert_at + 1) {
                            continue;
                        }
                        if (path[child_idx] !=
                            warm_start->path[parent_idx++]) {
                            matches_parent = false;
                            break;
                        }
                    }
                    if (!matches_parent) {
                        continue;
                    }
                    // The equal adjacent CNOTs cancel with identity U3s.
                    // Copying the other blocks embeds the fitted parent.
                    std::vector<double> neutral_parameters(
                        static_cast<size_t>(6 * path.size()) +
                            final_layer_parameters,
                        0.0
                    );
                    parent_idx = 0;
                    for (int child_idx = 0; child_idx < path.size();
                         ++child_idx) {
                        if (child_idx == insert_at ||
                            child_idx == insert_at + 1) {
                            continue;
                        }
                        std::copy_n(
                            warm_start->optimized_parameters.begin() +
                                6 * parent_idx++,
                            6,
                            neutral_parameters.begin() + 6 * child_idx
                        );
                    }
                    std::copy_n(
                        warm_start->optimized_parameters.end() -
                            final_layer_parameters,
                        final_layer_parameters,
                        neutral_parameters.end() - final_layer_parameters
                    );
                    SearchNode neutral = evaluate_once(
                        restart_limit, warm_start,
                        !defer_postfit_osr, &neutral_parameters
                    );
                    if (neutral.screening_objective <
                        best.screening_objective) {
                        best = std::move(neutral);
                    }
                    break;
                }
            }
        }
        if (defer_postfit_osr) {
            const double winning_screening_objective =
                best.screening_objective;
            SearchNode scored = evaluate_path(
                cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm,
                osr_tol, distrib_real, gen, path, nullptr, false,
                iteration_loop_override, max_inner_iteration_override,
                target_bound_override, &best.optimized_parameters, true
            );
            scored.screening_objective = winning_screening_objective;
            best = std::move(scored);
        }

        // The 3q constructive continuation is cheap enough for each fitted
        // candidate. On 4q it belongs to the committed forward state, not
        // every sibling that will immediately be discarded.
        if (qbit_num == 3) {
            best = complete_terminal_path(path, std::move(best),
                                          run_optimization);
        }
        return best;
    };

    // A zero cut-cover bound is necessary but not sufficient for synthesis:
    // the optimized circuit must also meet the same HS tolerance used by the
    // final fixed-structure pass. Otherwise keep extending this forward path.
    double synthesis_tolerance_loc = optimization_tolerance;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(
            synthesis_tolerance_loc
        );
    }
    if (config.count("synthesis_acceptance_tolerance") > 0) {
        double configured_tolerance;
        config["synthesis_acceptance_tolerance"].get_property(
            configured_tolerance
        );
        if (configured_tolerance > 0.0) {
            synthesis_tolerance_loc = std::max(
                synthesis_tolerance_loc, configured_tolerance
            );
        }
    }
    auto certify_zero_residual = [&](SearchNode& node) -> bool {
        if (node.get_min_cnots() != 0) {
            return false;
        }
        if (!use_hilbert_schmidt_guidance) {
            return true;
        }
        if (certified_terminal != nullptr &&
            node.path == certified_terminal->path &&
            node.optimized_parameters ==
                certified_terminal->optimized_parameters &&
            node.screening_objective < terminal_tolerance) {
            return true;
        }
        N_Qubit_Decomposition_custom verifier =
            perform_optimization(nullptr, false);
        // Zero-residual certification is a fixed-structure Hilbert-Schmidt
        // fit, not a graph-search guidance problem.  BFGS2's basin-hopping
        // budget consistently stops above the acceptance tolerance on
        // realizable 4q structures, while ordinary BFGS converges from the
        // same starts.  Keep BFGS2 for OSR candidate discovery and use the
        // smooth local solver for this final certificate only.
        verifier.set_optimizer(BFGS);
        // perform_optimization inherits the parent objective. This verifier
        // must use the same HS metric as the synthesis acceptance tolerance.
        verifier.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
        long long warm_loops = qbit_num >= 4 ? 5 : 0;
        if (config.count("osr_zero_hs_warm_loops") > 0) {
            config["osr_zero_hs_warm_loops"].get_property(warm_loops);
        }
        if (warm_loops < 0 ||
            warm_loops > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "osr_zero_hs_warm_loops must be a nonnegative int"
            );
        }
        std::unique_ptr<Gates_block> structure(
            construct_gate_structure_from_Gray_code(node.path)
        );
        verifier.set_custom_gate_structure(structure.get());
        verifier.set_optimization_blocks(structure->get_gate_num());

        int attempts = 10;
        if (config.count("osr_zero_hs_attempts") > 0) {
            long long configured_attempts;
            config["osr_zero_hs_attempts"].get_property(
                configured_attempts
            );
            attempts = static_cast<int>(
                std::max<long long>(configured_attempts, 1)
            );
        }
        uint64_t solution_seed = 1469598103934665603ULL;
        for (int idx = 0; idx < node.path.size(); ++idx) {
            solution_seed ^= static_cast<uint64_t>(node.path[idx] + 1);
            solution_seed *= 1099511628211ULL;
        }
        std::seed_seq seed_sequence{
            static_cast<std::uint32_t>(solution_seed),
            static_cast<std::uint32_t>(solution_seed >> 32),
            static_cast<std::uint32_t>(qbit_num)
        };
        std::mt19937 solution_gen(seed_sequence);
        std::uniform_real_distribution<> init_angle(0.0, 2 * M_PI);
        std::vector<double> parameters(verifier.get_parameter_num());
        if (parameters.size() != node.optimized_parameters.size()) {
            throw std::invalid_argument(
                "OSR and HS structures have different parameter counts"
            );
        }
        double best_hs = std::numeric_limits<double>::infinity();
        std::vector<double> best_parameters;
        for (int attempt = 0; attempt < attempts; ++attempt) {
            if (attempt == 0) {
                parameters = node.optimized_parameters;
            } else {
                for (double& angle : parameters) {
                    angle = init_angle(solution_gen);
                }
            }
            verifier.set_optimized_parameters(
                parameters.data(), static_cast<int>(parameters.size())
            );
            // Spend the reliable multi-loop HS continuation only on the
            // inherited fit. Independent random restarts remain single-pass.
            verifier.set_iteration_loops(
                qbit_num, attempt == 0 ? static_cast<int>(warm_loops) : 0
            );
            verifier.start_decomposition();
            Matrix_real fitted = verifier.get_optimized_parameters();
            Matrix certified_residual = Umtx.copy();
            verifier.apply_to(fitted, certified_residual);
            const double hs = get_hilbert_schmidt_test(certified_residual);
            if (hs < best_hs) {
                best_hs = hs;
                best_parameters.assign(
                    fitted.get_data(), fitted.get_data() + fitted.size()
                );
            }
            if (hs < synthesis_tolerance_loc) {
                break;
            }
        }
        if (best_hs < synthesis_tolerance_loc) {
            node.screening_objective = best_hs;
            node.optimized_parameters = std::move(best_parameters);
            return true;
        }
        // Screening rank and synthesis precision have different purposes.
        // Once the zero-rank screening claim fails its HS certificate,
        // retaining that coarse claim hides small but necessary entanglers
        // (for example weak controlled phases). Recompute the residual cuts
        // at certificate precision, using the best actual fit so that the
        // parameters, HS value and cut profile describe the same residual.
        const double certificate_osr_tol = std::min(
            osr_tol, std::sqrt(synthesis_tolerance_loc)
        );
        if (!best_parameters.empty() && certificate_osr_tol < osr_tol) {
            SearchNode rescored = evaluate_path(
                cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm,
                certificate_osr_tol, distrib_real, gen, node.path, nullptr,
                false, 0, 0, -1, &best_parameters, true
            );
            rescored.screening_objective = best_hs;
            node = std::move(rescored);
        }
        return false;
    };

    // Experimental forward-only continuation of a rank plateau. A complete
    // shallow insertion class is still evaluated before choosing its best
    // HS projection; integer OSR descents retain priority. This avoids
    // enumerating longer words from an unchanged root merely because a
    // full-rank residual has not crossed its next dyadic rank boundary yet.
    bool projection_plateau_descent = false;
    if (config.count("osr_projection_plateau_descent") > 0) {
        config["osr_projection_plateau_descent"].get_property(
            projection_plateau_descent
        );
    }
    bool profile_plateau_descent = false;
    if (config.count("osr_profile_plateau_descent") > 0) {
        config["osr_profile_plateau_descent"].get_property(
            profile_plateau_descent
        );
    }

    bool deferred_profile_plateau_descent = false;
    if (qbit_num >= 3 && qbit_num <= 4 &&
        config.count("osr_deferred_profile_plateau_descent") > 0) {
        config["osr_deferred_profile_plateau_descent"].get_property(
            deferred_profile_plateau_descent
        );
    }


    // Geometry-only proposals complement the optimistic OSR frontier. They
    // are not a complete library or a proof of global coverage. Repeated
    // overlapping interactions and paired blocks provide small constructive
    // charts without borrowing any target/source circuit decomposition.
    std::set<std::vector<int>> geometric_attempted_words;
    std::unique_ptr<SearchNode> geometric_certificate;
    std::vector<squander_osr::PauliSpinePlacement> geometric_placement_requirements;
    auto try_geometric_completion = [&](int cnot_bound)
        -> std::unique_ptr<SearchNode> {
        if (!geometric_completion_motifs ||
            cnot_bound < std::max(geometric_physical_floor, 1)) {
            return nullptr;
        }
        if (geometric_certificate != nullptr &&
            geometric_certificate->path.size() <= cnot_bound) {
            return std::unique_ptr<SearchNode>(
                new SearchNode(*geometric_certificate)
            );
        }
        const int edge_count = static_cast<int>(possible_target_qbits.size());
        std::vector<unsigned> supports(edge_count, 0);
        for (int edge = 0; edge < edge_count; ++edge) {
            supports[edge] = (unsigned(1) << possible_target_qbits[edge]) |
                            (unsigned(1) << possible_control_qbits[edge]);
        }
        const auto fan_echo_words = squander_osr::fan_echo_motifs(
            qbit_num, supports, cnot_bound
        );
        if (geometric_placement_requirements.empty()) {
            // Umtx stores the target adjoint. Read the desired unitary
            // through its stride, without copying or transposing a matrix.
            geometric_placement_requirements = squander_osr::pauli_spine_placements(
                qbit_num, terminal_tolerance, [&](int row, int col) {
                    const auto& value = Umtx.get_data()[col * Umtx.stride + row];
                    return std::complex<double>(value.real, -value.imag);
                }
            );
        }
        auto motif_config = config;
        motif_config["parallel"].set_property("parallel",
                                             static_cast<long long>(0));
        motif_config["use_float"].set_property("use_float", false);
        motif_config["use_basin_hopping"].set_property("use_basin_hopping", true);
        motif_config["use_differential_evolution"].set_property(
            "use_differential_evolution", false
        );
        motif_config["use_dual_annealing"].set_property(
            "use_dual_annealing", false
        );
        motif_config["optimization_tolerance"].set_property(
            "optimization_tolerance", terminal_tolerance
        );
        motif_config["optimization_tolerance_bfgs2"].set_property(
            "optimization_tolerance_bfgs2", terminal_tolerance
        );
        long long hops = 2;
        if (config.count("osr_geometric_completion_basin_hops") > 0) {
            config["osr_geometric_completion_basin_hops"].get_property(hops);
        }
        if (hops < 0 || hops > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "osr_geometric_completion_basin_hops must be a nonnegative int"
            );
        }
        motif_config["max_iteration_loops_bfgs2"].set_property(
            "max_iteration_loops_bfgs2", hops
        );
        // Preserve the established budget for existing charts. Newly added
        // complete fan-echo charts inherit the configured full local-fit
        // budget, independently of partition width.
        long long fan_echo_hops = hops;
        if (config.count("max_iteration_loops_bfgs2") > 0) {
            config["max_iteration_loops_bfgs2"].get_property(fan_echo_hops);
        } else if (config.count("max_iteration_loops") > 0) {
            config["max_iteration_loops"].get_property(fan_echo_hops);
        }
        fan_echo_hops = std::max(fan_echo_hops, hops);
        if (fan_echo_hops < 0 || fan_echo_hops > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "max_iteration_loops must be a nonnegative int"
            );
        }
        long long evaluations = 1000;
        if (config.count("max_inner_iterations_bfgs2") > 0) {
            config["max_inner_iterations_bfgs2"].get_property(evaluations);
        }
        evaluations = std::max<long long>(1, std::min<long long>(
            evaluations, std::numeric_limits<int>::max()
        ));
        for (int length = std::max(geometric_physical_floor, 1);
             length <= cnot_bound; ++length) {
            std::vector<std::vector<int>> words;
            std::set<std::vector<int>> distinct;
            std::set<std::vector<int>> full_budget_words;
            auto add_period = [&](const std::vector<int>& period) {
                std::vector<int> word(length);
                for (int idx = 0; idx < length; ++idx) {
                    word[idx] = period[idx % period.size()];
                }
                if (distinct.insert(word).second) {
                    words.push_back(std::move(word));
                }
            };
            if (qbit_num == 3) {
                for (int first = 0; first < edge_count; ++first) {
                    for (int second = 0; second < edge_count; ++second) {
                        if (first == second ||
                            (supports[first] | supports[second]) != unsigned(7)) {
                            continue;
                        }
                        add_period({first, second});
                        add_period({first, first, second, second});
                    }
                }
                if (edge_count == 3) {
                    std::vector<int> period{0, 1, 2};
                    do {
                        add_period(period);
                    } while (std::next_permutation(period.begin(), period.end()));
                }
            }
            for (const auto& word : fan_echo_words) {
                if (static_cast<int>(word.size()) == length &&
                    distinct.insert(word).second) {
                    full_budget_words.insert(word);
                    words.push_back(word);
                }
            }
            for (const auto& word : words) {
                if (!geometric_attempted_words.insert(word).second) continue;
                bool satisfies_cuts = true;
                for (size_t cut = 0; cut < all_cuts.size(); ++cut) {
                    int crossings = 0;
                    for (int digit : word) {
                        const auto& subset = all_cuts[cut];
                        const bool first = std::find(
                            subset.begin(), subset.end(),
                            possible_target_qbits[digit]
                        ) != subset.end();
                        const bool second = std::find(
                            subset.begin(), subset.end(),
                            possible_control_qbits[digit]
                        ) != subset.end();
                        crossings += first != second;
                    }
                    if (crossings < target_cut_requirements[cut]) {
                        satisfies_cuts = false;
                        break;
                    }
                }
                if (!satisfies_cuts) continue;
                if (!squander_osr::admits_pauli_spine_placement(
                        word, supports, geometric_placement_requirements)) continue;
                matrix_base<int8_t> limits(1, length);
                std::fill_n(limits.data, limits.size(),
                            static_cast<int8_t>(edge_count));
                GrayCodeCNOT path(limits);
                for (int idx = 0; idx < length; ++idx) {
                    path[idx] = static_cast<int8_t>(word[idx]);
                }
                std::unique_ptr<Gates_block> structure(
                    construct_gate_structure_from_Gray_code(path, true)
                );
                const bool fan_echo = full_budget_words.count(word) != 0;
                const long long word_hops = fan_echo ? fan_echo_hops : hops;
                auto word_config = motif_config;
                word_config["max_iteration_loops_bfgs2"].set_property(
                    "max_iteration_loops_bfgs2", word_hops
                );
                N_Qubit_Decomposition_custom optimizer(
                    Umtx.copy(), qbit_num, false, word_config, RANDOM, 0
                );
                optimizer.set_verbose(0);
                optimizer.set_debugfile("");
                optimizer.set_optimizer(BFGS2);
                optimizer.set_random_seed(static_cast<int>(path_rng_seed));
                optimizer.set_max_iteration(1);
                optimizer.set_max_inner_iterations(static_cast<int>(evaluations));
                optimizer.set_iteration_loops(qbit_num, static_cast<int>(word_hops));
                optimizer.set_optimization_tolerance(terminal_tolerance);
                optimizer.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
                optimizer.set_custom_gate_structure(structure.get());
                optimizer.set_optimization_blocks(structure->get_gate_num());
                std::mt19937 motif_gen(static_cast<unsigned>(path_rng_seed));
                std::uniform_real_distribution<double> angle(0.0, 2 * M_PI);
                std::vector<double> initial(structure->get_parameter_num());
                for (double& value : initial) value = angle(motif_gen);
                optimizer.set_optimized_parameters(
                    initial.data(), static_cast<int>(initial.size())
                );
                optimizer.start_decomposition();
                Matrix_real parameters = optimizer.get_optimized_parameters();
                Matrix residual = Umtx.copy();
                optimizer.apply_to(parameters, residual);
                const double hs = get_hilbert_schmidt_test(residual);
                if (trace_candidate_placements) {
                    std::stringstream sstream;
                    sstream << "OSR geometric chart: cnots=" << length
                            << ", floor=" << geometric_physical_floor
                            << ", hs=" << hs << ", hops=" << word_hops
                            << ", path=";
                    for (size_t idx = 0; idx < word.size(); ++idx) {
                        if (idx != 0) sstream << ',';
                        sstream << word[idx];
                    }
                    sstream << '\n';
                    print(sstream, 1);
                }
                if (!std::isfinite(hs) || hs >= terminal_tolerance) continue;
                std::unique_ptr<SearchNode> completed(new SearchNode(path, true));
                completed->optimized_parameters.assign(
                    parameters.get_data(), parameters.get_data() + parameters.size()
                );
                completed->screening_objective = hs;
                completed->lower_budget_profile_distance = 0.0;
                completed->osr_results.emplace_back(
                    0, 0.0, std::vector<int>(topology.size(), 0),
                    std::vector<std::pair<int, double>>(
                        all_cuts.size(), std::make_pair(0, 0.0)
                    )
                );
                if (trace_candidate_placements) {
                    std::stringstream sstream;
                    sstream << "OSR geometric completion: cnots=" << length
                            << ", floor=" << geometric_physical_floor
                            << ", bound=" << cnot_bound << ", hs=" << hs
                            << ", certified=1"
                            << ", motif=" << (fan_echo ? "fan_echo" : "legacy")
                            << ", hops=" << word_hops
                            << std::endl;
                    print(sstream, 2);
                }
                geometric_certificate.reset(new SearchNode(*completed));
                return completed;
            }
        }
        return nullptr;
    };

    auto deflate_certificate = [&](SearchNode node) -> SearchNode {
        if (!certified_cnot_deflation || node.path.size() == 0) return node;
        auto deflation_config = config;
        deflation_config["parallel"].set_property(
            "parallel", static_cast<long long>(0)
        );
        deflation_config["use_float"].set_property("use_float", false);
        deflation_config["use_basin_hopping"].set_property(
            "use_basin_hopping", true
        );
        deflation_config["use_differential_evolution"].set_property(
            "use_differential_evolution", false
        );
        deflation_config["use_dual_annealing"].set_property(
            "use_dual_annealing", false
        );
        deflation_config["optimization_tolerance"].set_property(
            "optimization_tolerance", terminal_tolerance
        );
        deflation_config["optimization_tolerance_bfgs2"].set_property(
            "optimization_tolerance_bfgs2", terminal_tolerance
        );
        long long hops = 2;
        if (config.count("osr_cnot_deflation_basin_hops") > 0) {
            config["osr_cnot_deflation_basin_hops"].get_property(hops);
        }
        if (hops < 0 || hops > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "osr_cnot_deflation_basin_hops must be a nonnegative int"
            );
        }
        deflation_config["max_iteration_loops_bfgs2"].set_property(
            "max_iteration_loops_bfgs2", hops
        );
        long long evaluations = 1000;
        if (config.count("max_inner_iterations_bfgs2") > 0) {
            config["max_inner_iterations_bfgs2"].get_property(evaluations);
        }
        evaluations = std::max<long long>(1, std::min<long long>(
            evaluations, std::numeric_limits<int>::max()
        ));
        auto crosses = [&](int digit, const std::vector<int>& subset) {
            const bool target = std::find(
                subset.begin(), subset.end(), possible_target_qbits[digit]
            ) != subset.end();
            const bool control = std::find(
                subset.begin(), subset.end(), possible_control_qbits[digit]
            ) != subset.end();
            return target != control;
        };
        while (node.path.size() > 0) {
            const int parent_length = node.path.size();
            const int child_length = parent_length - 1;
            if (node.optimized_parameters.size() !=
                static_cast<size_t>(6 * parent_length + 3 * qbit_num)) {
                throw std::logic_error(
                    "Certified Tree circuit has an inconsistent U3 layout"
                );
            }
            std::vector<int> word_targets(parent_length), word_controls(parent_length);
            for (int idx = 0; idx < parent_length; ++idx) {
                word_targets[idx] = possible_target_qbits[node.path[idx]];
                word_controls[idx] = possible_control_qbits[node.path[idx]];
            }
            std::vector<int> crossings(all_cuts.size(), 0);
            for (size_t cut = 0; cut < all_cuts.size(); ++cut) {
                for (int idx = 0; idx < parent_length; ++idx) {
                    crossings[cut] += crosses(node.path[idx], all_cuts[cut]);
                }
            }
            bool reduced = false;
            for (int removed = 0; removed < parent_length && !reduced;
                 ++removed) {
                bool satisfies_cuts = true;
                for (size_t cut = 0; cut < all_cuts.size(); ++cut) {
                    if (crossings[cut] -
                            crosses(node.path[removed], all_cuts[cut]) <
                        target_cut_requirements[cut]) {
                        satisfies_cuts = false;
                        break;
                    }
                }
                if (!satisfies_cuts) continue;
                // A removed CX replaced by identity is an HS stationary
                // seed: its normalized trace overlap is 1/2. Both products
                // Rx_target(+/-pi/2) Rz_control(+/-pi/2) attain the optimal
                // product-unitary overlap 1/sqrt(2), from CX's two equal
                // operator-Schmidt coefficients. Jointly refit all angles;
                // this exact chart transition is not a solution certificate.
                for (int sign : {1, -1}) {
                    GrayCodeCNOT child;
                    if (child_length > 0) {
                        matrix_base<int8_t> limits(1, child_length);
                        std::fill_n(limits.data, limits.size(),
                            static_cast<int8_t>(topology.size()));
                        child = GrayCodeCNOT(limits);
                    }
                    int child_idx = 0;
                    for (int idx = 0; idx < parent_length; ++idx) {
                        if (idx != removed) child[child_idx++] = node.path[idx];
                    }
                    auto fit = refit_tree_cnot_deletion(Umtx, qbit_num,
                        word_targets, word_controls, node.optimized_parameters, removed, sign,
                        deflation_config, static_cast<int>(hops), static_cast<int>(evaluations),
                        static_cast<int>(path_rng_seed), terminal_tolerance);
                    if (!fit.certified) continue;
                    SearchNode completed(child, true);
                    completed.optimized_parameters = std::move(fit.parameters);
                    completed.screening_objective = fit.hs;
                    completed.lower_budget_profile_distance = 0.0;
                    completed.osr_results.emplace_back(
                        0, 0.0, std::vector<int>(topology.size(), 0),
                        std::vector<std::pair<int, double>>(
                            all_cuts.size(), std::make_pair(0, 0.0))
                    );
                    if (trace_candidate_placements) {
                        std::stringstream sstream;
                        trace_tree_cnot_deletion(sstream, parent_length, removed, sign, fit);
                        print(sstream, 2);
                    }
                    node = std::move(completed);
                    reduced = true;
                    break;
                }
            }
            if (!reduced) break;
        }
        return node;
    };

    auto record_certificate = [&](SearchNode node) {
        // A proposal can replace an already certified OSR result only with
        // a strictly smaller, independently certified circuit. Failed fits
        // are cached by word; they never discard the original certificate.
        auto shorter = try_geometric_completion(node.path.size() - 1);
        if (shorter != nullptr) node = std::move(*shorter);
        node = deflate_certificate(std::move(node));
        best_first_osr_solution_found = true;
        best_first_solution_path = node.path.copy();
        best_first_solution_parameters = std::move(node.optimized_parameters);
        return best_first_solution_path.copy();
    };

    // A chart that saturates the existing physical lower bound is already
    // CNOT-optimal at the requested accuracy. Check this finite proposal
    // class before waiting for a first graph-search certificate. Failure
    // leaves the optimistic frontier and its CNOT budget unchanged.
    if (geometric_completion_motifs && geometric_physical_floor > 0 &&
        geometric_physical_floor <= level_limit) {
        auto minimal_chart = try_geometric_completion(geometric_physical_floor);
        if (minimal_chart != nullptr) {
            return record_certificate(std::move(*minimal_chart));
        }
    }

      std::function<bool(const GrayCodeCNOT&, const SearchNode*, bool)>
          add_to_heap = [&](const GrayCodeCNOT& path,
                            const SearchNode* warm_start,
                            bool full_restart_budget) -> bool {
        // The path length is the synthesized CNOT count.  Paths beyond the
        // strict-improvement bound cannot be valid answers and must not incur
        // an optimization.  This is an exact bound, not an OSR-residual prune.
        // A parent residual describes one parameter point, not every jointly
        // refitted child. Inserting a gate can change earlier local rotations,
        // so that residual is not a certified lower bound for this structure.
        if (path.size() > level_limit)
            return false;
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
          SearchNode sn = evaluate_candidate_path(
              path, warm_start, true,
              // A zero override skips the local solve altogether.  Every
              // padded insertion then has the same unoptimized HS=1 value,
              // erasing the placement signal and forcing exhaustive word
              // enumeration.  One loop is the intended inexpensive probe;
              // only a proven completion receives the full restart budget.
              full_restart_budget ? -1 : 1,
              full_restart_budget ? -1 : 100,
              -1,
              full_restart_budget
        );
        //printf("%.2fs\n", std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()*1e-9);

        if (top_heap == nullptr || !(*top_heap < sn)) {
            top_heap.reset(new SearchNode(std::move(sn)));
        }
        // heap.emplace(sn);
        return true;
    };

    GrayCodeCNOT startpath;
    if (qbit_num > 1)
        add_to_heap(startpath, nullptr, true);

    std::vector<int> full_topo_filter(topology.size());
    std::iota(full_topo_filter.begin(), full_topo_filter.end(), 0);

    if (use_lazy_deferred_frontier && top_heap != nullptr) {
        struct DeferredExpansion {
            std::shared_ptr<SearchNode> parent;
            std::vector<int> edge_order;
            std::vector<int> edge_rank;
            std::vector<GrayCodeCNOT> candidates;
            std::shared_ptr<SearchNode> plateau_candidate;
            std::shared_ptr<SearchNode> lookahead_candidate;
            std::shared_ptr<SearchNode> constructive_plateau_candidate;
            std::shared_ptr<SearchNode> constructive_descent_candidate;
            std::shared_ptr<SearchNode> fractional_plateau_candidate;
            std::vector<std::shared_ptr<SearchNode>> tied_descent_candidates;
            size_t next_candidate;
            int num_cnot;
            int max_insertion_cnots;
            bool plateau_expanded;
            bool full_restart_budget;
        };

        std::shared_ptr<SearchNode> root(top_heap.release());
        std::shared_ptr<SearchNode> best = root;
        std::vector<DeferredExpansion> frontier;

        auto is_full_weave_path = [&](const GrayCodeCNOT& path) {
            const size_t path_size = static_cast<size_t>(path.size());
            if (qbit_num < 4 || path_size < 8)
                return false;
            for (size_t idx = 0; idx + 7 < path_size; ++idx) {
                const int edge_a = path[idx];
                const int edge_b = path[idx + 1];
                const int edge_c = path[idx + 2];
                const int edge_d = path[idx + 6];
                if (path[idx + 3] != edge_b ||
                    path[idx + 4] != edge_a ||
                    path[idx + 5] != edge_c ||
                    path[idx + 7] != edge_d ||
                    edge_a == edge_b || edge_a == edge_c ||
                    edge_b == edge_c || edge_d == edge_a ||
                    edge_d == edge_b || edge_d == edge_c) {
                    continue;
                }
                int star_center = -1;
                for (int qbit = 0; qbit < qbit_num; ++qbit) {
                    const auto contains_qbit = [&](int edge) {
                        return topology[edge][0] == qbit ||
                               topology[edge][1] == qbit;
                    };
                    if (contains_qbit(edge_a) &&
                        contains_qbit(edge_b) &&
                        contains_qbit(edge_c)) {
                        star_center = qbit;
                        break;
                    }
                }
                if (star_center >= 0 &&
                    topology[edge_d][0] != star_center &&
                    topology[edge_d][1] != star_center) {
                    return true;
                }
            }
            return false;
        };

        auto paired_block_count = [&](const GrayCodeCNOT& path) {
            int count = 0;
            for (int idx = 0; idx + 3 < path.size(); ++idx) {
                const int edge_a = path[idx];
                const int edge_b = path[idx + 2];
                if (path[idx + 1] == edge_a &&
                    path[idx + 3] == edge_b && edge_a != edge_b) {
                    ++count;
                }
            }
            return count;
        };

        auto inherited_connectivity_score = [&](
            const GrayCodeCNOT& parent_path,
            const GrayCodeCNOT& child_path) {
            std::vector<bool> support(qbit_num, false);
            for (int idx = 0; idx < parent_path.size(); ++idx) {
                support[topology[parent_path[idx]][0]] = true;
                support[topology[parent_path[idx]][1]] = true;
            }
            int parent_idx = 0;
            int connected = 0;
            int growth = 0;
            for (int idx = 0; idx < child_path.size(); ++idx) {
                if (parent_idx < parent_path.size() &&
                    child_path[idx] == parent_path[parent_idx]) {
                    ++parent_idx;
                    continue;
                }
                const int q0 = topology[child_path[idx]][0];
                const int q1 = topology[child_path[idx]][1];
                if (!support[q0] && !support[q1]) {
                    continue;
                }
                ++connected;
                growth += static_cast<int>(!support[q0]) +
                          static_cast<int>(!support[q1]);
                support[q0] = true;
                support[q1] = true;
            }
            return std::make_tuple(-growth, -connected);
        };

        auto insertion_mutual_information_score = [&](
            const GrayCodeCNOT& parent_path,
            const GrayCodeCNOT& child_path) {
            int parent_idx = 0;
            double score = 0.0;
            for (int idx = 0; idx < child_path.size(); ++idx) {
                if (parent_idx < parent_path.size() &&
                    child_path[idx] == parent_path[parent_idx]) {
                    ++parent_idx;
                    continue;
                }
                score += target_edge_mutual_information[child_path[idx]];
            }
            return -score;
        };

          auto fill_candidates = [&](DeferredExpansion& expansion) {
              expansion.candidates.clear();
              expansion.plateau_candidate.reset();
              expansion.constructive_plateau_candidate.reset();
              expansion.constructive_descent_candidate.reset();
              expansion.tied_descent_candidates.clear();
              expansion.next_candidate = 0;
              expansion.plateau_expanded = false;
              expansion.full_restart_budget = false;
              // Only the actual word length is an exact budget bound. The
              // parent residual may improve by more than the insertion count
              // when all local rotations are jointly refitted.
              if (expansion.parent->path.size() + expansion.num_cnot >
                      level_limit) {
                  return;
              }
              std::set<GrayCodeCNOT> unique_candidates;
              generate_insertions(
                expansion.parent->path, topology, expansion.edge_order,
                  expansion.num_cnot,
                  [&](const GrayCodeCNOT& path) {
                      if (is_unique_structure(path, topology) &&
                          unique_candidates.insert(path).second) {
                          expansion.candidates.push_back(path);
                      }
                    return false;
                }
            );

            const GrayCodeCNOT& parent_path = expansion.parent->path;
            const std::vector<int>& edge_rank = expansion.edge_rank;
            const size_t topology_size = topology.size();
            auto inserted_edge_counts = [&](const GrayCodeCNOT& path) {
                std::vector<int> counts(topology_size, 0);
                for (int idx = 0; idx < path.size(); ++idx) {
                    ++counts[path[idx]];
                }
                for (int idx = 0; idx < parent_path.size(); ++idx) {
                    --counts[parent_path[idx]];
                }
                return counts;
            };
            auto insertion_motif_score = [&](const GrayCodeCNOT& path) {
                if (qbit_num < 4 || expansion.num_cnot < 2) {
                    return std::make_tuple(0, 0, 0, 0);
                }
                std::vector<int> positions;
                std::vector<int> edges;
                int parent_idx = 0;
                for (int child_idx = 0; child_idx < path.size(); ++child_idx) {
                    if (parent_idx < parent_path.size() &&
                        path[child_idx] == parent_path[parent_idx]) {
                        ++parent_idx;
                    } else {
                        positions.push_back(child_idx);
                        edges.push_back(path[child_idx]);
                    }
                }
                if (edges.size() !=
                    static_cast<size_t>(expansion.num_cnot)) {
                    return std::make_tuple(9, 0, 0, 0);
                }
                auto overlaps = [&](int lhs_edge, int rhs_edge) {
                    const int lhs_q0 = topology[lhs_edge][0];
                    const int lhs_q1 = topology[lhs_edge][1];
                    const int rhs_q0 = topology[rhs_edge][0];
                    const int rhs_q1 = topology[rhs_edge][1];
                    return lhs_q0 == rhs_q0 || lhs_q0 == rhs_q1 ||
                           lhs_q1 == rhs_q0 || lhs_q1 == rhs_q1;
                };
                const int span = positions.back() - positions.front();
                if (expansion.num_cnot > 4) {
                    std::vector<bool> motif_covered(edges.size(), false);
                    std::vector<bool> support(qbit_num, false);
                    int weave_core_count = 0;
                    int weave_pair_count = 0;
                    int alternating_count = 0;
                    int sandwich_count = 0;
                    int identical_pair_count = 0;
                    int overlap_transition_count = 0;
                    for (size_t idx = 0; idx < edges.size(); ++idx) {
                        support[topology[edges[idx]][0]] = true;
                        support[topology[edges[idx]][1]] = true;
                        if (idx + 1 < edges.size()) {
                            overlap_transition_count += static_cast<int>(
                                overlaps(edges[idx], edges[idx + 1])
                            );
                            if (edges[idx] == edges[idx + 1]) {
                                ++identical_pair_count;
                                motif_covered[idx] = true;
                                motif_covered[idx + 1] = true;
                            }
                        }
                        if (idx + 2 < edges.size() &&
                            edges[idx] == edges[idx + 2] &&
                            edges[idx] != edges[idx + 1] &&
                            overlaps(edges[idx], edges[idx + 1])) {
                            ++sandwich_count;
                            motif_covered[idx] = true;
                            motif_covered[idx + 1] = true;
                            motif_covered[idx + 2] = true;
                        }
                        if (idx + 3 < edges.size() &&
                            edges[idx] == edges[idx + 2] &&
                            edges[idx + 1] == edges[idx + 3] &&
                            edges[idx] != edges[idx + 1] &&
                            overlaps(edges[idx], edges[idx + 1])) {
                            ++alternating_count;
                            motif_covered[idx] = true;
                            motif_covered[idx + 1] = true;
                            motif_covered[idx + 2] = true;
                            motif_covered[idx + 3] = true;
                        }
                    }
                    // The constructive core A-B-C-B-A-C uses the three edges
                    // of a four-qubit star.  It is the prefix from which the
                    // complete weave below is formed, so retaining it avoids
                    // committing the optimistic search to an incompatible
                    // six-CNOT parent before the final D-D pair is visible.
                    for (size_t idx = 0; idx + 5 < edges.size(); ++idx) {
                        const int edge_a = edges[idx];
                        const int edge_b = edges[idx + 1];
                        const int edge_c = edges[idx + 2];
                        if (edges[idx + 3] != edge_b ||
                            edges[idx + 4] != edge_a ||
                            edges[idx + 5] != edge_c ||
                            edge_a == edge_b || edge_a == edge_c ||
                            edge_b == edge_c) {
                            continue;
                        }
                        int star_center = -1;
                        for (int qbit = 0; qbit < qbit_num; ++qbit) {
                            const auto contains_qbit = [&](int edge) {
                                return topology[edge][0] == qbit ||
                                       topology[edge][1] == qbit;
                            };
                            if (contains_qbit(edge_a) &&
                                contains_qbit(edge_b) &&
                                contains_qbit(edge_c)) {
                                star_center = qbit;
                                break;
                            }
                        }
                        if (star_center < 0) {
                            continue;
                        }
                        ++weave_core_count;
                        for (size_t covered = idx;
                             covered < idx + 6; ++covered) {
                            motif_covered[covered] = true;
                        }
                    }
                    // A four-qubit multiplexor/Toffoli family found in the
                    // exact mod5 audit has the entangler word
                    // A-B-C-B-A-C-D-D: A/B/C form a three-edge star and D
                    // joins two leaves.  It is a distinct constructive lift
                    // of the same cut cover as an alternating word and must
                    // not lose that tie merely because ABAB sorts first.
                    for (size_t idx = 0; idx + 7 < edges.size(); ++idx) {
                        const int edge_a = edges[idx];
                        const int edge_b = edges[idx + 1];
                        const int edge_c = edges[idx + 2];
                        const int edge_d = edges[idx + 6];
                        if (edges[idx + 3] != edge_b ||
                            edges[idx + 4] != edge_a ||
                            edges[idx + 5] != edge_c ||
                            edges[idx + 7] != edge_d ||
                            edge_a == edge_b || edge_a == edge_c ||
                            edge_b == edge_c || edge_d == edge_a ||
                            edge_d == edge_b || edge_d == edge_c) {
                            continue;
                        }
                        int star_center = -1;
                        for (int qbit = 0; qbit < qbit_num; ++qbit) {
                            const auto contains_qbit = [&](int edge) {
                                return topology[edge][0] == qbit ||
                                       topology[edge][1] == qbit;
                            };
                            if (contains_qbit(edge_a) &&
                                contains_qbit(edge_b) &&
                                contains_qbit(edge_c)) {
                                star_center = qbit;
                                break;
                            }
                        }
                        if (star_center < 0 ||
                            topology[edge_d][0] == star_center ||
                            topology[edge_d][1] == star_center) {
                            continue;
                        }
                        ++weave_pair_count;
                        for (size_t covered = idx;
                             covered < idx + 8; ++covered) {
                            motif_covered[covered] = true;
                        }
                    }
                    const int support_size = static_cast<int>(std::count(
                        support.begin(), support.end(), true
                    ));
                    const int motif_coverage = static_cast<int>(std::count(
                        motif_covered.begin(), motif_covered.end(), true
                    ));
                    int motif_class = 6;
                    if (support_size == qbit_num &&
                        (weave_core_count > 0 || weave_pair_count > 0))
                        motif_class = 0;
                    else if (support_size == qbit_num &&
                             alternating_count > 0)
                        motif_class = 1;
                    else if (support_size == qbit_num &&
                             sandwich_count > 0 &&
                             identical_pair_count > 0)
                        motif_class = 2;
                    else if (support_size == qbit_num && sandwich_count > 0)
                        motif_class = 3;
                    else if (support_size == qbit_num &&
                             identical_pair_count > 0)
                        motif_class = 4;
                    else if (support_size == qbit_num)
                        motif_class = 5;
                    return std::make_tuple(
                        motif_class, -motif_coverage,
                        -(4 * weave_pair_count + weave_core_count +
                          alternating_count +
                          sandwich_count +
                          identical_pair_count),
                        -overlap_transition_count
                    );
                }
                if (expansion.num_cnot == 2) {
                    const bool identical = edges[0] == edges[1];
                    if (identical && span == 1)
                        return std::make_tuple(0, span, 0, 0);
                    if (identical)
                        return std::make_tuple(1, span, 0, 0);
                    if (overlaps(edges[0], edges[1]))
                        return std::make_tuple(2, span, 0, 0);
                    return std::make_tuple(3, span, 0, 0);
                }

                if (expansion.num_cnot == 3) {
                    const bool contiguous = span == 2;
                    const bool completes_paired_block =
                        paired_block_count(path) >
                        paired_block_count(parent_path);
                    int paired_block_inserted_coverage = 0;
                    if (completes_paired_block) {
                        for (int block = 0; block + 3 < path.size();
                             ++block) {
                            if (path[block] != path[block + 1] ||
                                path[block + 2] != path[block + 3] ||
                                path[block] == path[block + 2]) {
                                continue;
                            }
                            int inserted_coverage = 0;
                            for (int pos : positions) {
                                if (pos >= block && pos < block + 4) {
                                    ++inserted_coverage;
                                }
                            }
                            paired_block_inserted_coverage = std::max(
                                paired_block_inserted_coverage,
                                inserted_coverage
                            );
                        }
                    }
                    const bool all_identical =
                        edges[0] == edges[1] && edges[1] == edges[2];
                    const bool sandwich =
                        edges[0] == edges[2] && edges[0] != edges[1] &&
                        overlaps(edges[0], edges[1]);
                    const int overlap_count =
                        static_cast<int>(overlaps(edges[0], edges[1])) +
                        static_cast<int>(overlaps(edges[0], edges[2])) +
                        static_cast<int>(overlaps(edges[1], edges[2]));
                    // Independent two-qubit interactions canonicalize to an
                    // A-A-B-B block.  Completing that block through an
                    // inherited A or B is constructive even though the newly
                    // inserted edge multiset is disconnected by itself.
                    if (completes_paired_block)
                        return std::make_tuple(
                            -1, -paired_block_inserted_coverage, span, 0
                        );
                    if (sandwich && contiguous)
                        return std::make_tuple(0, span, 0, 0);
                    if (all_identical && contiguous)
                        return std::make_tuple(1, span, 0, 0);
                    if (sandwich)
                        return std::make_tuple(2, span, 0, 0);
                    if (all_identical)
                        return std::make_tuple(3, span, 0, 0);
                    if (contiguous && overlap_count >= 2)
                        return std::make_tuple(4, span, -overlap_count, 0);
                    if (overlap_count >= 2)
                        return std::make_tuple(5, span, -overlap_count, 0);
                    if (overlap_count == 1)
                        return std::make_tuple(6, span, -overlap_count, 0);
                    return std::make_tuple(7, span, 0, 0);
                }

                const bool contiguous = span == 3;
                const bool alternating =
                    edges[0] == edges[2] && edges[1] == edges[3] &&
                    edges[0] != edges[1] &&
                    overlaps(edges[0], edges[1]);
                const bool nested =
                    edges[0] == edges[3] && edges[1] == edges[2] &&
                    edges[0] != edges[1] &&
                    overlaps(edges[0], edges[1]);
                const bool paired =
                    edges[0] == edges[1] && edges[2] == edges[3] &&
                    overlaps(edges[0], edges[2]);
                if (alternating && contiguous)
                    return std::make_tuple(0, span, 0, 0);
                if (nested && contiguous)
                    return std::make_tuple(1, span, 0, 0);
                if (paired && contiguous)
                    return std::make_tuple(2, span, 0, 0);
                if (alternating)
                    return std::make_tuple(3, span, 0, 0);
                if (nested)
                    return std::make_tuple(4, span, 0, 0);
                if (paired)
                    return std::make_tuple(5, span, 0, 0);
                return std::make_tuple(6, span, 0, 0);
            };
            auto full_weave_completion_score = [&](const GrayCodeCNOT& path) {
                return expansion.num_cnot == 2 && is_full_weave_path(path)
                    ? 0
                    : 1;
            };
            auto placement_connectivity_score = [&](
                const GrayCodeCNOT& path) {
                if (qbit_num < 4 || expansion.num_cnot < 2) {
                    return std::make_tuple(0, 0, 0, 0, 0, 0);
                }
                std::vector<bool> inserted(path.size(), true);
                int parent_idx = 0;
                for (int child_idx = 0;
                     child_idx < path.size() &&
                     parent_idx < parent_path.size(); ++child_idx) {
                    if (path[child_idx] == parent_path[parent_idx]) {
                        inserted[child_idx] = false;
                        ++parent_idx;
                    }
                }

                int best_growth = 0;
                int best_connected = 0;
                int longest_run = 0;
                int run_count = 0;
                int best_boundary_distance =
                    static_cast<int>(parent_path.size());
                int best_boundary_side = 2;
                int inherited_before = 0;
                for (int idx = 0; idx < path.size();) {
                    if (!inserted[idx]) {
                        ++inherited_before;
                        ++idx;
                        continue;
                    }
                    const int run_begin = idx;
                    while (idx < path.size() && inserted[idx])
                        ++idx;
                    const int run_end = idx;
                    ++run_count;
                    longest_run = std::max(longest_run, run_end - run_begin);
                    best_boundary_distance = std::min(
                        best_boundary_distance,
                        std::min(
                            inherited_before,
                            static_cast<int>(parent_path.size()) -
                                inherited_before
                        )
                    );
                    if (inherited_before == 0)
                        best_boundary_side = 0;
                    else if (inherited_before == parent_path.size())
                        best_boundary_side = std::min(best_boundary_side, 1);

                    auto score_from_boundary = [&](
                        int boundary_idx, int first, int last, int step) {
                        if (boundary_idx < 0 || boundary_idx >= path.size())
                            return std::make_pair(0, 0);
                        std::vector<bool> support(qbit_num, false);
                        const int boundary_edge = path[boundary_idx];
                        support[topology[boundary_edge][0]] = true;
                        support[topology[boundary_edge][1]] = true;
                        int growth = 0;
                        int connected = 0;
                        for (int pos = first; pos != last; pos += step) {
                            const int edge = path[pos];
                            const int q0 = topology[edge][0];
                            const int q1 = topology[edge][1];
                            if (!support[q0] && !support[q1])
                                break;
                            ++connected;
                            growth += static_cast<int>(!support[q0]) +
                                      static_cast<int>(!support[q1]);
                            support[q0] = true;
                            support[q1] = true;
                        }
                        return std::make_pair(growth, connected);
                    };

                    const std::pair<int, int> from_left =
                        score_from_boundary(
                            run_begin - 1, run_begin, run_end, 1
                        );
                    const std::pair<int, int> from_right =
                        score_from_boundary(
                            run_end, run_end - 1, run_begin - 1, -1
                        );
                    const std::pair<int, int> run_score =
                        std::max(from_left, from_right);
                    best_growth = std::max(best_growth, run_score.first);
                    best_connected = std::max(
                        best_connected, run_score.second
                    );
                }
                return std::make_tuple(
                    -best_growth, -best_connected, -longest_run,
                    run_count, best_boundary_distance, best_boundary_side
                );
            };

            // Try constructive placements first, but do not delete other
            // placements on the strength of a fitted cut-cover estimate.
            // Reoptimizing the inherited U3 gates can change that estimate,
            // so it is not a proof that a generic placement cannot descend.
            const bool prioritize_constructive_motif =
                qbit_num >= 4 && expansion.num_cnot >= 3;

            std::map<std::vector<int>, std::tuple<int, size_t, double>>
                continuation_scores;
            const bool lazy_deep_class =
                qbit_num >= 4 &&
                expansion.num_cnot > expansion.parent->get_min_cnots();
            const auto& parent_osr =
                expansion.parent->get_best_osr_result();
            const std::vector<std::vector<int>> minimum_edge_covers =
                lazy_deep_class
                    ? osr_bound_solver.enumerate_min_edge_covers(
                          std::get<3>(parent_osr),
                          expansion.parent->get_min_cnots()
                      )
                    : std::vector<std::vector<int>>();
            struct RankedCandidate {
                GrayCodeCNOT path;
                std::tuple<int, double, double> screening;
                std::tuple<int, size_t, double> continuation;
                std::tuple<int, int, int, int> motif;
                int weave_completion;
                std::tuple<int, int, int, int, int, int> placement;
                std::vector<int> inserted_counts;
                std::vector<int> edge_signature;
            };
            std::vector<RankedCandidate> ranked_candidates;
            ranked_candidates.reserve(expansion.candidates.size());
            for (const GrayCodeCNOT& path : expansion.candidates) {
                const std::vector<int> counts = inserted_edge_counts(path);
                auto continuation_it = continuation_scores.find(counts);
                if (continuation_it == continuation_scores.end()) {
                    continuation_it = continuation_scores.emplace(
                        counts,
                        osr_bound_solver.score_edge_multiset_for_search(
                            std::get<3>(parent_osr), counts
                        )
                    ).first;
                }
                std::tuple<int, double, double> screening;
                  if (lazy_deep_class) {
                      screening = std::make_tuple(0, 0.0, 0.0);
                  } else {
                      // Evaluate the inherited circuit with new U3 blocks at
                      // identity for a cheap placement-specific OSR screen.
                      SearchNode screened = evaluate_candidate_path(
                          path, expansion.parent.get(), false, -1, -1, -1
                      );
                      const auto& screened_osr =
                          screened.get_best_osr_result();
                      screening = std::make_tuple(
                          std::get<0>(screened_osr),
                          placement_objective_ordering
                              ? screened.screening_objective : 0.0,
                          std::get<1>(screened_osr)
                      );
                  }
                std::vector<int> signature;
                signature.reserve(expansion.num_cnot);
                for (size_t edge = 0; edge < topology_size; ++edge) {
                    for (int count = 0; count < counts[edge]; ++count) {
                        signature.push_back(edge_rank[edge]);
                    }
                }
                std::sort(signature.begin(), signature.end());
                ranked_candidates.push_back(RankedCandidate{
                    path,
                    screening,
                    continuation_it->second,
                    insertion_motif_score(path),
                    full_weave_completion_score(path),
                    placement_connectivity_score(path),
                    counts,
                    std::move(signature)
                });
            }

            std::vector<size_t> candidate_order;
            candidate_order.reserve(ranked_candidates.size());
            for (size_t idx = 0; idx < ranked_candidates.size(); ++idx) {
                candidate_order.push_back(idx);
            }
            // The inherited-identity screen is evaluated from a numerically
            // reconstructed target.  Equivalent circuits can differ at the
            // roundoff level, so use transitive bins before ordering 4q
            // placements.  Exact double comparisons made the greedy path
            // depend on sub-ulp changes to the same target unitary.
            const auto score_bin = [](double value) -> long long {
                if (!std::isfinite(value))
                    return std::numeric_limits<long long>::max();
                const double scaled = value * 1e12;
                if (scaled >= static_cast<double>(
                        std::numeric_limits<long long>::max()))
                    return std::numeric_limits<long long>::max();
                if (scaled <= static_cast<double>(
                        std::numeric_limits<long long>::min()))
                    return std::numeric_limits<long long>::min();
                return std::llround(scaled);
            };
            const auto screening_order = [&](
                const std::tuple<int, double, double>& score) {
                return std::make_tuple(
                    std::get<0>(score), score_bin(std::get<1>(score)),
                    score_bin(std::get<2>(score)));
            };
            std::stable_sort(
                candidate_order.begin(), candidate_order.end(),
                [&](size_t lhs_idx, size_t rhs_idx) {
                      const RankedCandidate& lhs =
                          ranked_candidates[lhs_idx];
                      const RankedCandidate& rhs =
                          ranked_candidates[rhs_idx];
                      if (lhs.weave_completion != rhs.weave_completion) {
                          return lhs.weave_completion < rhs.weave_completion;
                      }
                      const bool screening_differs = qbit_num >= 4
                          ? screening_order(lhs.screening) !=
                                screening_order(rhs.screening)
                          : lhs.screening != rhs.screening;
                      const bool screening_less = qbit_num >= 4
                          ? screening_order(lhs.screening) <
                                screening_order(rhs.screening)
                          : lhs.screening < rhs.screening;
                      if (screen_before_motif && screening_differs) {
                          return screening_less;
                      }
                      if (prioritize_constructive_motif &&
                          lhs.motif != rhs.motif) {
                          return lhs.motif < rhs.motif;
                      }
                      if (!screen_before_motif && screening_differs) {
                          return screening_less;
                      }
                    if (std::get<0>(lhs.continuation) !=
                        std::get<0>(rhs.continuation)) {
                        return std::get<0>(lhs.continuation) <
                               std::get<0>(rhs.continuation);
                    }
                    if (std::get<1>(lhs.continuation) !=
                        std::get<1>(rhs.continuation)) {
                        return std::get<1>(lhs.continuation) >
                               std::get<1>(rhs.continuation);
                    }
                    const auto lhs_continuation = qbit_num >= 4
                        ? score_bin(std::get<2>(lhs.continuation))
                        : 0;
                    const auto rhs_continuation = qbit_num >= 4
                        ? score_bin(std::get<2>(rhs.continuation))
                        : 0;
                    if (qbit_num >= 4) {
                        if (lhs_continuation != rhs_continuation)
                            return lhs_continuation < rhs_continuation;
                    } else if (std::get<2>(lhs.continuation) !=
                               std::get<2>(rhs.continuation)) {
                        return std::get<2>(lhs.continuation) <
                               std::get<2>(rhs.continuation);
                    }
                    if (!prioritize_constructive_motif &&
                        lhs.motif != rhs.motif) {
                        return lhs.motif < rhs.motif;
                    }
                    // Finish ties with locality around the inherited path.
                    if (lhs.placement != rhs.placement) {
                        return lhs.placement < rhs.placement;
                    }
                    return lhs.edge_signature < rhs.edge_signature;
                }
            );
            expansion.candidates.clear();
            expansion.candidates.reserve(ranked_candidates.size());
            if (lazy_deep_class) {
                // Integer cut cover can have several exactly degenerate
                // minimum edge compositions, and each cover can have several
                // inequivalent constructive lifts. Retain the best placement
                // for every (cover, motif-family) pair rather than allowing
                // an alternating lift to suppress a weave, sandwich, or pair
                // lift of the same exact cover.
                std::set<size_t> retained_indices;
                // A fitted minimum cover is guidance, not an admissible
                // exclusion proof.  Preserve one placement for each edge
                // multiset that completes a canonical A-A-B-B block through
                // the inherited path.  This finite O(E^2) motif family
                // captures paired controlled rotations without reopening the
                // full placement frontier.
                std::set<std::vector<int>> retained_paired_multisets;
                for (size_t idx : candidate_order) {
                    const RankedCandidate& candidate =
                        ranked_candidates[idx];
                    if (paired_block_count(candidate.path) <=
                        paired_block_count(parent_path)) {
                        continue;
                    }
                    if (retained_paired_multisets.insert(
                            candidate.inserted_counts).second) {
                        retained_indices.insert(idx);
                    }
                }
                for (const std::vector<int>& cover : minimum_edge_covers) {
                    bool retained_for_cover = false;
                    for (int motif_family = 0;
                         motif_family <= 4; ++motif_family) {
                        std::set<std::vector<int>> retained_multisets;
                        bool retained_for_family = false;
                        for (size_t idx : candidate_order) {
                            const RankedCandidate& candidate =
                                ranked_candidates[idx];
                            if (std::get<0>(candidate.motif) !=
                                motif_family) {
                                continue;
                            }
                            const std::vector<int>& counts =
                                candidate.inserted_counts;
                            bool lifts_cover = true;
                            for (size_t edge = 0;
                                 edge < cover.size(); ++edge) {
                                if (counts[edge] < cover[edge]) {
                                    lifts_cover = false;
                                    break;
                                }
                            }
                            if (!lifts_cover) {
                                continue;
                            }
                            const bool fully_constructive =
                                -std::get<1>(candidate.motif) >=
                                expansion.num_cnot;
                            // A count multiset identifies the complete weave's
                            // support, but not its gate order. A-B-C-B-A-C-D-D
                            // has six inequivalent A/B/C orderings with the
                            // same counts. Preserve those six- and eight-CNOT
                            // weave orderings only; broadening this exception
                            // to other motif families causes an eager explosion.
                            const bool retain_ordered_weave =
                                fully_constructive && motif_family == 0 &&
                                (expansion.num_cnot == 6 ||
                                 expansion.num_cnot == 8);
                            if (!retain_ordered_weave &&
                                !retained_multisets.insert(counts).second) {
                                continue;
                            }
                            if (retained_for_family &&
                                !fully_constructive) {
                                continue;
                            }
                            // Candidate order already resolves placement and
                            // continuation ties. Keep one placement for each
                            // distinct edge-count multiset. A complete weave
                            // keeps every ordered signature.
                            retained_indices.insert(idx);
                            retained_for_family = true;
                            retained_for_cover = true;
                        }
                    }
                    if (!retained_for_cover) {
                        for (size_t idx : candidate_order) {
                            const std::vector<int>& counts =
                                ranked_candidates[idx].inserted_counts;
                            bool lifts_cover = true;
                            for (size_t edge = 0; edge < cover.size(); ++edge) {
                                if (counts[edge] < cover[edge]) {
                                    lifts_cover = false;
                                    break;
                                }
                            }
                            if (lifts_cover) {
                                retained_indices.insert(idx);
                                break;
                            }
                        }
                    }
                }
                // Sparse topologies or uniqueness reduction can leave no
                // realizable lift for a formal cover. Preserve forward
                // progress with the globally best legal constructive path.
                if (retained_indices.empty() && !candidate_order.empty()) {
                    retained_indices.insert(candidate_order.front());
                }
                for (size_t idx : candidate_order) {
                    if (retained_indices.count(idx) > 0) {
                        expansion.candidates.push_back(
                            std::move(ranked_candidates[idx].path)
                        );
                    }
                }
            } else {
                for (size_t idx : candidate_order) {
                    expansion.candidates.push_back(
                        std::move(ranked_candidates[idx].path)
                    );
                }
            }
            if (qbit_num >= 4 && expansion.num_cnot == 2) {
                const int parent_paired_blocks =
                    paired_block_count(expansion.parent->path);
                std::stable_partition(
                    expansion.candidates.begin(),
                    expansion.candidates.end(),
                    [&](const GrayCodeCNOT& path) {
                        return paired_block_count(path) >
                            parent_paired_blocks;
                    }
                );
            }
            // Past the cut-cover lower bound, exhaustive local fitting no
            // longer follows an OSR descent signal and grows combinatorially.
            // Keep those legal deeper classes in the optimistic search, but
            // defer only their best structurally ranked candidate.
            // The small 3q topology remains exhaustive because its known
            // four-CNOT manifolds can sit beyond a two-CNOT cut-cover bound.
            // ``lazy_deep_class`` has already been reduced to a bounded set
            // of algebraically distinct motif representatives above.
        };

        auto make_expansion = [&](const std::shared_ptr<SearchNode>& parent) {
            if (qbit_num >= 4 && certified_terminal_completion) {
                *parent = complete_terminal_path(parent->path, *parent, true);
            }
            DeferredExpansion expansion;
            expansion.parent = parent;
            expansion.next_candidate = 0;
            expansion.num_cnot = 1;
            expansion.max_insertion_cnots = -1;
            expansion.plateau_expanded = false;
            expansion.full_restart_budget = false;
            const auto& best_osr = parent->get_best_osr_result();
            expansion.edge_order = osr_bound_solver.rank_edges_for_search(
                std::get<3>(best_osr), std::get<0>(best_osr),
                std::get<2>(best_osr)
            );
            // A zero OSR residual has no deficient cut from which to rank an
            // edge, but an uncertified zero-residual prefix can still need
            // CNOTs to reach the exact unitary manifold.  Continue over the
            // complete legal topology and let HS provide the ordering.
            if (expansion.edge_order.empty()) {
                expansion.edge_order = full_topo_filter;
            }
            expansion.edge_rank.resize(topology.size());
            for (size_t rank = 0; rank < expansion.edge_order.size(); ++rank) {
                expansion.edge_rank[expansion.edge_order[rank]] = rank;
            }
            if (certified_terminal == nullptr) fill_candidates(expansion);
            return expansion;
        };

        if (certify_zero_residual(*root)) {
            return record_certificate(*root);
        }
        if (root->path.size() < level_limit) {
            frontier.push_back(make_expansion(root));
        }

        while (!frontier.empty()) {
              if (certified_terminal != nullptr) {
                  return record_certificate(*certified_terminal);
              }
              DeferredExpansion& expansion = frontier.back();
              if (expansion.next_candidate >= expansion.candidates.size()) {
                  if (qbit_num >= 4 &&
                      expansion.tied_descent_candidates.size() > 1 &&
                      expansion.tied_descent_candidates.front()
                              ->get_min_cnots() == 1) {
                      // A residual-one cut bound can still require a
                      // two-CNOT constructive completion (for example a
                      // controlled-rotation pair).  All states retained here
                      // are equally good strict descents from the same exact
                      // minimum-cover class.  Keep that algebraic degeneracy
                      // until one forward completion reaches zero instead of
                      // selecting a core by a numerically insignificant HS
                      // tie and permanently losing the realizable path.
                      std::vector<std::shared_ptr<SearchNode>> tied_descents =
                          std::move(expansion.tied_descent_candidates);
                      std::shared_ptr<SearchNode> fallback_descent =
                          expansion.plateau_candidate;
                      frontier.clear();
                      if (fallback_descent != nullptr &&
                          fallback_descent->path.size() < level_limit) {
                          DeferredExpansion fallback =
                              make_expansion(fallback_descent);
                          fallback.num_cnot = 3;
                          fill_candidates(fallback);
                          frontier.push_back(std::move(fallback));
                      }
                      for (auto it = tied_descents.rbegin();
                           it != tied_descents.rend(); ++it) {
                          if ((*it)->path.size() < level_limit) {
                              DeferredExpansion completion =
                                  make_expansion(*it);
                              completion.max_insertion_cnots = 2;
                              frontier.push_back(std::move(completion));
                          }
                      }
                      continue;
                  }
                  // A single CNOT can give a misleading integer OSR drop
                  // even when the constructive route begins with a paired
                  // CRY/sandwich/overlap motif.  Compare the complete 1-CNOT
                  // and reduced 2-CNOT motif classes before making one
                  // irreversible 4q commitment.
                  bool shallow_integer_descent = false;
                  bool shallow_zero_residual_hs_progress = false;
                  if (expansion.plateau_candidate != nullptr) {
                      const int parent_residual =
                          expansion.parent->get_min_cnots();
                      const int candidate_residual =
                          expansion.plateau_candidate->get_min_cnots();
                      shallow_integer_descent =
                          candidate_residual < parent_residual;
                      const double parent_hs =
                          expansion.parent->screening_objective;
                      const double hs_epsilon = 1e-12 * std::max(
                          1.0, std::abs(parent_hs));
                      shallow_zero_residual_hs_progress =
                          parent_residual == 0 && candidate_residual == 0 &&
                          expansion.plateau_candidate
                                  ->screening_objective + hs_epsilon <
                              parent_hs;
                  }
                  if (qbit_num >= 4 && expansion.num_cnot == 1 &&
                      (shallow_integer_descent ||
                       shallow_zero_residual_hs_progress) &&
                      expansion.parent->path.size() + 2 <= level_limit) {
                      expansion.lookahead_candidate =
                          expansion.plateau_candidate;
                      ++expansion.num_cnot;
                      fill_candidates(expansion);
                      continue;
                  }
                  if (qbit_num >= 4 && expansion.num_cnot == 2 &&
                      expansion.lookahead_candidate != nullptr) {
                      if (expansion.plateau_candidate == nullptr ||
                          *expansion.plateau_candidate >
                              *expansion.lookahead_candidate) {
                          expansion.plateau_candidate =
                              expansion.lookahead_candidate;
                      }
                  }
                  if (expansion.constructive_descent_candidate != nullptr &&
                      (expansion.plateau_candidate == nullptr ||
                       expansion.constructive_descent_candidate
                               ->get_min_cnots() <=
                           expansion.plateau_candidate->get_min_cnots())) {
                      // The exact OSR count is the primary ordering.  When
                      // several placements attain the same count, prefer a
                      // completed algebraic two-CNOT block over an
                      // unstructured placement and use the fitted HS value
                      // only to rank representatives of that motif family.
                      std::shared_ptr<SearchNode> progress =
                          expansion.constructive_descent_candidate;
                      frontier.clear();
                      if (progress->path.size() < level_limit) {
                          frontier.push_back(make_expansion(progress));
                      }
                      continue;
                  }
                  if (expansion.constructive_plateau_candidate != nullptr) {
                      std::shared_ptr<SearchNode> progress =
                          expansion.constructive_plateau_candidate;
                      frontier.clear();
                      if (progress->path.size() < level_limit) {
                          frontier.push_back(make_expansion(progress));
                      }
                      continue;
                  }
                  if (expansion.plateau_candidate != nullptr) {
                      std::shared_ptr<SearchNode> progress =
                          expansion.plateau_candidate;
                      frontier.clear();
                      if (progress->path.size() < level_limit) {
                          frontier.push_back(make_expansion(progress));
                      }
                      continue;
                  }
                  if (deferred_profile_plateau_descent &&
                      expansion.fractional_plateau_candidate != nullptr &&
                      expansion.num_cnot >= std::min(
                          3, level_limit - static_cast<int>(
                              expansion.parent->path.size()))) {
                      // Three-CNOT blocks can realize a full two-qubit gate
                      // or a useful sandwich that two-CNOT projections miss.
                      // Preserve every original 1/2/3-CNOT class (up to the
                      // exact remaining budget) before using this auxiliary
                      // fractional bound. It never caps the complete search.
                      // Make one forward commitment, retaining no siblings or
                      // ancestor and imposing no synthesis/depth/time cap.
                      std::shared_ptr<SearchNode> progress =
                          expansion.fractional_plateau_candidate;
                      if (trace_candidate_placements) {
                          std::stringstream sstream;
                          sstream << "OSR deferred profile descent: parent_depth="
                                  << expansion.parent->path.size()
                                  << ", child_depth=" << progress->path.size()
                                  << ", residual=" << progress->get_min_cnots()
                                  << ", parent_distance="
                                  << expansion.parent->lower_budget_profile_distance
                                  << ", child_distance="
                                  << progress->lower_budget_profile_distance
                                  << std::endl;
                          print(sstream, 2);
                      }
                      frontier.clear();
                      if (progress->path.size() < level_limit) {
                          frontier.push_back(make_expansion(progress));
                      }
                      continue;
                  }
                // A lower surplus at the same exact residual is useful for
                // ordering, but it is not evidence that another CNOT has
                // reduced the synthesis bound.  Evaluate the complete
                // insertion class, then descend through its best plateau
                  // representative while retaining this parent as the exact
                  // fallback for larger insertion classes.
                if (expansion.max_insertion_cnots >= 0 &&
                    expansion.num_cnot >= expansion.max_insertion_cnots) {
                    frontier.pop_back();
                    continue;
                }
                ++expansion.num_cnot;
                if (expansion.parent->path.size() + expansion.num_cnot >
                    level_limit) {
                    frontier.pop_back();
                    continue;
                }
                if (geometric_completion_motifs && !geometric_root_checked &&
                    expansion.parent->path.size() == 0 &&
                    expansion.num_cnot > geometric_physical_floor + 1 &&
                    std::all_of(target_cut_requirements.begin(),
                                target_cut_requirements.end(),
                                [](int crossings) { return crossings >= 3; })) {
                    geometric_root_checked = true;
                    // Preserve the cover class and its one-edge extension
                    // before proposing whole-word charts on a spectrally
                    // saturated root plateau. This lets the established
                    // constructive descent distinguish degenerate covers.
                    // Structured lower-rank targets retain their insertion
                    // sequence; only verified smaller charts refine a fit.
                    // This is not a plateau/depth cap: failed proposals
                    // resume the same complete insertion frontier below.
                    auto completed = try_geometric_completion(level_limit);
                    if (completed != nullptr) {
                        return record_certificate(std::move(*completed));
                    }
                }
                fill_candidates(expansion);
                continue;
            }

              GrayCodeCNOT candidate_path =
                  expansion.candidates[expansion.next_candidate++];
              std::shared_ptr<SearchNode> parent = expansion.parent;
            if (trace_candidate_placements) {
                std::stringstream sstream;
                sstream << "OSR placement: parent_depth="
                        << parent->path.size()
                        << ", parent_residual="
                        << parent->get_min_cnots()
                        << ", insertions=" << expansion.num_cnot
                        << ", rank=" << expansion.next_candidate
                        << ", path=";
                for (int idx = 0; idx < candidate_path.size(); ++idx) {
                    if (idx > 0)
                        sstream << ',';
                    sstream << static_cast<int>(candidate_path[idx]);
                }
                sstream << std::endl;
                print(sstream, 2);
            }
              visited.clear();
              top_heap.reset();
              const SearchNode* optimization_warm_start = parent.get();
              int target_bound_override = -1;
              if (candidate_path.size() > level_limit ||
                  !is_unique_structure(candidate_path, topology)) {
                  continue;
              }
              const bool final_three_qubit_insertion_class =
                  rank_final_three_qubit_class && qbit_num == 3 &&
                  (expansion.num_cnot == 4 ||
                   parent->path.size() + expansion.num_cnot +
                       std::max(
                           0,
                           parent->get_min_cnots() - expansion.num_cnot
                       ) ==
                       level_limit);
              const bool completed_weave =
                  qbit_num >= 4 && is_full_weave_path(candidate_path);
              const bool completed_paired_block =
                  qbit_num >= 4 &&
                  paired_block_count(candidate_path) >
                      paired_block_count(parent->path);
              if (qbit_num >= 4 && expansion.num_cnot == 2 &&
                  !completed_paired_block) {
                  std::shared_ptr<SearchNode> motif_progress;
                  if (
                      expansion.constructive_descent_candidate != nullptr) {
                      motif_progress =
                          expansion.constructive_descent_candidate;
                  } else if (
                      expansion.constructive_plateau_candidate != nullptr) {
                      motif_progress =
                          expansion.constructive_plateau_candidate;
                  }
                  if (motif_progress != nullptr) {
                      // Paired motifs form a finite constructive prefix of
                      // the two-CNOT class.  Once that prefix contains the
                      // same plateau step the full class would prefer, or a
                      // strictly better total relaxed CNOT count than the complete
                      // one-CNOT class, evaluating unrelated
                      // placements cannot improve this forward motif
                      // decision.  Commit once and discard every sibling.
                      frontier.clear();
                      if (motif_progress->path.size() < level_limit) {
                          frontier.push_back(
                              make_expansion(motif_progress));
                      }
                      continue;
                  }
              }
              const bool degenerate_cover_insertion_class =
                  qbit_num >= 4 &&
                  expansion.num_cnot > parent->get_min_cnots();
              const bool use_full_restart_budget =
                  expansion.full_restart_budget || completed_weave ||
                  completed_paired_block ||
                  degenerate_cover_insertion_class;
              const bool use_final_three_qubit_basin =
                  final_three_qubit_insertion_class &&
                  final_three_qubit_basin_hops > 0;
              if (physical_cover_motifs && !physical_cover_motifs_checked &&
                  qbit_num >= 3 && qbit_num <= 4 &&
                  parent->path.size() == 0 && use_hilbert_schmidt_guidance) {
                  physical_cover_motifs_checked = true;
                  // A tree has exactly one crossing edge per bridge cut.
                  // Its sum of necessary bridge counts is a global CNOT floor.
                  // A verified word at that floor cannot worsen the result.
                  std::vector<std::pair<int, int>> edges;
                  std::vector<int> edge_digits;
                  std::vector<unsigned> supports(possible_target_qbits.size());
                  std::vector<unsigned> neighbors(qbit_num, 0);
                  for (int digit = 0; digit < possible_target_qbits.size(); ++digit) {
                      const int first = possible_target_qbits[digit];
                      const int second = possible_control_qbits[digit];
                      supports[digit] = (unsigned(1) << first) | (unsigned(1) << second);
                      const auto edge = std::minmax(first, second);
                      const std::pair<int, int> key(edge.first, edge.second);
                      if (std::find(edges.begin(), edges.end(), key) != edges.end()) continue;
                      edges.push_back(key);
                      edge_digits.push_back(static_cast<int>(digit));
                      neighbors[first] |= unsigned(1) << second;
                      neighbors[second] |= unsigned(1) << first;
                  }
                  const unsigned all = (unsigned(1) << qbit_num) - 1;
                  unsigned connected = 1;
                  for (int pass = 0; pass < qbit_num; ++pass) {
                      for (int qubit = 0; qubit < qbit_num; ++qubit) {
                          if (connected & (unsigned(1) << qubit)) connected |= neighbors[qubit];
                      }
                  }
                  if (edges.size() == static_cast<size_t>(qbit_num - 1) && connected == all) {
                      double acceptance = 1e-10;
                      if (config.count("synthesis_acceptance_tolerance") > 0) {
                          config["synthesis_acceptance_tolerance"].get_property(acceptance);
                      }
                      std::vector<osr_motifs::EdgeBlock> blocks;
                      int minimum = 0;
                      bool single_blocks_suffice = std::isfinite(acceptance) && acceptance > 0.0;
                      for (size_t edge = 0; edge < edges.size() && single_blocks_suffice; ++edge) {
                          const int first = edges[edge].first;
                          const int second = edges[edge].second;
                          unsigned component = unsigned(1) << first;
                          for (int pass = 0; pass < qbit_num; ++pass) {
                              for (int qubit = 0; qubit < qbit_num; ++qubit) {
                                  if (!(component & (unsigned(1) << qubit))) continue;
                                  unsigned adjacent = neighbors[qubit];
                                  if (qubit == first) adjacent &= ~(unsigned(1) << second);
                                  if (qubit == second) adjacent &= ~(unsigned(1) << first);
                                  component |= adjacent;
                              }
                          }
                          // Preserve the existing tall-matrix SVD invariant.
                          int count = 0;
                          for (int qubit = 0; qubit < qbit_num; ++qubit) {
                              if (component & (unsigned(1) << qubit)) ++count;
                          }
                          if (count < qbit_num - count) component = all ^ component;
                          std::vector<int> cut;
                          for (int qubit = 0; qubit < qbit_num; ++qubit) {
                              if (component & (unsigned(1) << qubit)) cut.push_back(qubit);
                          }
                          const int required = operator_schmidt_cnot_bound(
                              Umtx, qbit_num, cut, Fnorm, acceptance).first;
                          // A single contiguous block acts only on two qubits:
                          // it has at most three irreducible CNOTs. Larger bridge
                          // requirements need interleaving, left to graph search.
                          if (required > 3) {
                              single_blocks_suffice = false;
                              break;
                          }
                          minimum += required;
                          blocks.push_back({edge_digits[edge], supports[edge_digits[edge]], required});
                      }
                      if (single_blocks_suffice && minimum > 0 && minimum <= level_limit) {
                          const osr_motifs::CausalWitness witness(qbit_num,
                              [&](size_t row, size_t column) {
                                  const auto value = Umtx.get_data()[row * Umtx.stride + column];
                                  return std::complex<double>(value.real, value.imag);
                              });
                          const int8_t digit_limit = candidate_path.get_Limits()[0];
                          for (const auto& word : osr_motifs::block_orders(blocks)) {
                              const std::vector<int> reverse(word.rbegin(), word.rend());
                              // Gray words and inverse targets use opposite
                              // temporal conventions. Requiring both bounds to
                              // fail is conservative under either convention.
                              if (std::min(witness.lower_bound(word, supports),
                                           witness.lower_bound(reverse, supports)) > acceptance) continue;
                              matrix_base<int8_t> limits(1, word.size());
                              for (size_t index = 0; index < word.size(); ++index) {
                                  limits[index] = digit_limit;
                              }
                              GrayCodeCNOT motif(limits);
                              for (size_t index = 0; index < word.size(); ++index) {
                                  motif[index] = static_cast<int8_t>(word[index]);
                              }
                              SearchNode fitted = evaluate_candidate_path(
                                  motif, nullptr, true, -1, -1, 0, true);
                              if (fitted.get_min_cnots() == 0 && certify_zero_residual(fitted)) {
                                  return record_certificate(std::move(fitted));
                              }
                          }
                      }
                  }
              }
              SearchNode evaluated = evaluate_candidate_path(
                  candidate_path, optimization_warm_start, true,
                  use_final_three_qubit_basin
                      ? static_cast<int>(final_three_qubit_basin_hops)
                      : (use_full_restart_budget ? -1 : 1),
                  use_final_three_qubit_basin
                      ? -1
                      : (use_full_restart_budget
                            ? -1
                            : (use_hilbert_schmidt_guidance
                            ? static_cast<int>(std::max<long long>(
                                  hilbert_schmidt_guidance_inner_iterations, 1
                              ))
                            : 100)),
                  target_bound_override,
                  use_full_restart_budget
              );
              top_heap.reset(new SearchNode(std::move(evaluated)));
              if (top_heap == nullptr) {
                  continue;
              }
            std::shared_ptr<SearchNode> candidate(top_heap.release());
            if (trace_candidate_placements) {
                const auto& traced_osr =
                    candidate->get_best_osr_result();
                std::stringstream sstream;
                sstream << "OSR placement result: residual="
                        << candidate->get_min_cnots()
                        << ", surplus=" << std::get<1>(traced_osr)
                        << ", hs=" << candidate->screening_objective
                        << std::endl;
                print(sstream, 2);
            }
            if (*best > *candidate) {
                best = candidate;
            }
            const bool constructive_pair_lookahead_class =
                qbit_num >= 4 &&
                (expansion.num_cnot == 1 ||
                 (expansion.num_cnot == 2 &&
                  expansion.lookahead_candidate != nullptr) ||
                 (rank_constructive_pair_class &&
                  expansion.num_cnot == 2) ||
                 (completed_paired_block && expansion.num_cnot == 2));
            const bool rank_complete_insertion_class =
                final_three_qubit_insertion_class ||
                (rank_degenerate_cover_class &&
                 degenerate_cover_insertion_class) ||
                constructive_pair_lookahead_class;
            if (trace_candidate_placements) {
                std::stringstream sstream;
                sstream << "OSR placement result: parent_depth="
                        << parent->path.size()
                        << ", insertions=" << expansion.num_cnot
                        << ", rank=" << expansion.next_candidate
                        << ", residual=" << candidate->get_min_cnots()
                        << ", surplus="
                        << std::get<1>(candidate->get_best_osr_result())
                        << ", hs=" << candidate->screening_objective
                        << ", paired=" << completed_paired_block
                        << std::endl;
                print(sstream, 2);
            }
            const bool claims_zero_residual =
                candidate->get_min_cnots() == 0;
            if (claims_zero_residual && certify_zero_residual(*candidate)) {
                return record_certificate(*candidate);
            }
            // A completed paired block is a finite constructive move across
            // Growing the connected support can require an uphill surplus
            // step before a paired motif becomes useful.  Other plateau
            // pairs still require an OSR surplus decrease.
            const double parent_surplus = std::get<1>(
                parent->get_best_osr_result());
            const double candidate_surplus = std::get<1>(
                candidate->get_best_osr_result());
            // This surrogate is obtained after iterative fitting and SVDs.
            // A near-zero change is not a constructive plateau signal: tiny
            // differences in equivalent target matrices can otherwise make
            // the optimistic search discard the rest of an insertion class.
            // Keep the established 3q comparison unchanged.
            const double surplus_epsilon =
                (qbit_num >= 4
                    ? std::sqrt(std::numeric_limits<double>::epsilon())
                    : 1e-12) *
                std::max(1.0, std::abs(parent_surplus));
            const bool grows_connected_support =
                qbit_num == 4 &&
                std::get<0>(inherited_connectivity_score(
                    parent->path, candidate_path)) < 0;
            const double hs_epsilon = 1e-12 * std::max(
                1.0, std::abs(parent->screening_objective));
            const double profile_epsilon =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(parent_surplus));
            const bool profile_plateau_progress =
                profile_plateau_descent &&
                parent->get_min_cnots() > 0 &&
                candidate->get_min_cnots() == parent->get_min_cnots() &&
                candidate_surplus + profile_epsilon < parent_surplus;
            if (profile_plateau_progress) {
                // Finish this shallow insertion class before making one
                // forward-only commitment. Preserve any strict integer OSR
                // descent, and otherwise choose the strongest lower-budget
                // cut-profile improvement, not merely a lower HS error.
                if (expansion.plateau_candidate == nullptr ||
                    (expansion.plateau_candidate->get_min_cnots() ==
                         parent->get_min_cnots() &&
                     candidate_surplus < std::get<1>(
                         expansion.plateau_candidate->get_best_osr_result()))) {
                    expansion.plateau_candidate = candidate;
                }
                continue;
            }
            const bool projection_plateau_progress =
                projection_plateau_descent &&
                use_hilbert_schmidt_guidance &&
                parent->get_min_cnots() > 0 &&
                candidate->get_min_cnots() == parent->get_min_cnots() &&
                candidate->screening_objective + hs_epsilon <
                    parent->screening_objective;
            if (projection_plateau_progress) {
                // Do not commit the first lucky local fit, and do not retain
                // siblings after the class is complete. Compare projections
                // only within the same integer residual tier; an already
                // recorded strict OSR descent cannot be replaced by a tie.
                if (expansion.plateau_candidate == nullptr ||
                    (expansion.plateau_candidate->get_min_cnots() ==
                         parent->get_min_cnots() &&
                     candidate->screening_objective <
                         expansion.plateau_candidate->screening_objective)) {
                    expansion.plateau_candidate = candidate;
                }
                continue;
            }
            // OSR zero is a lower-bound statement, not an HS certificate.
            // Once both nodes have zero residual, the exact HS objective is
            // the remaining constructive signal and must be allowed to guide
            // further insertions until certification succeeds.
            const bool zero_residual_hs_progress =
                parent->get_min_cnots() == 0 &&
                candidate->get_min_cnots() == 0 &&
                candidate->screening_objective + hs_epsilon <
                    parent->screening_objective;
            const bool constructive_plateau_progress =
                completed_paired_block &&
                candidate->get_min_cnots() == parent->get_min_cnots() &&
                (candidate_surplus + surplus_epsilon < parent_surplus ||
                 grows_connected_support);
            bool better_constructive_descent =
                expansion.constructive_descent_candidate == nullptr;
            if (!better_constructive_descent) {
                const int incumbent_residual =
                    expansion.constructive_descent_candidate
                        ->get_min_cnots();
                const auto candidate_connectivity =
                    inherited_connectivity_score(
                        parent->path, candidate_path
                    );
                const auto incumbent_connectivity =
                    inherited_connectivity_score(
                        parent->path,
                        expansion.constructive_descent_candidate->path
                    );
                const double candidate_mutual_information =
                    insertion_mutual_information_score(
                        parent->path, candidate_path
                    );
                const double incumbent_mutual_information =
                    insertion_mutual_information_score(
                        parent->path,
                        expansion.constructive_descent_candidate->path
                    );
                better_constructive_descent =
                    candidate->get_min_cnots() < incumbent_residual ||
                    (candidate->get_min_cnots() == incumbent_residual &&
                     (candidate_connectivity < incumbent_connectivity ||
                      (candidate_connectivity == incumbent_connectivity &&
                       (candidate_mutual_information <
                            incumbent_mutual_information ||
                        (candidate_mutual_information ==
                             incumbent_mutual_information &&
                         candidate->screening_objective <
                             expansion.constructive_descent_candidate
                                 ->screening_objective)))));
            }
            if (completed_paired_block &&
                candidate->get_min_cnots() < parent->get_min_cnots() &&
                better_constructive_descent) {
                expansion.constructive_descent_candidate = candidate;
            }
            const bool improves_parent =
                candidate->get_min_cnots() < parent->get_min_cnots() ||
                constructive_plateau_progress ||
                zero_residual_hs_progress;
            if (constructive_plateau_progress) {
                const auto candidate_connectivity =
                    inherited_connectivity_score(
                        parent->path, candidate_path
                    );
                const double candidate_mutual_information =
                    insertion_mutual_information_score(
                        parent->path, candidate_path
                    );
                const bool better_connectivity =
                    expansion.constructive_plateau_candidate == nullptr ||
                    candidate_connectivity < inherited_connectivity_score(
                        parent->path,
                        expansion.constructive_plateau_candidate->path
                    );
                const bool equal_connectivity =
                    expansion.constructive_plateau_candidate != nullptr &&
                    candidate_connectivity == inherited_connectivity_score(
                        parent->path,
                        expansion.constructive_plateau_candidate->path
                    );
                const double incumbent_mutual_information =
                    expansion.constructive_plateau_candidate == nullptr
                        ? std::numeric_limits<double>::infinity()
                        : insertion_mutual_information_score(
                              parent->path,
                              expansion.constructive_plateau_candidate->path
                          );
                if (better_connectivity ||
                    (equal_connectivity &&
                     (candidate_mutual_information <
                          incumbent_mutual_information ||
                      (candidate_mutual_information ==
                           incumbent_mutual_information &&
                       candidate->screening_objective <
                           expansion.constructive_plateau_candidate
                               ->screening_objective)))) {
                    expansion.constructive_plateau_candidate = candidate;
                }
                continue;
            }
            if (improves_parent) {
                if (rank_complete_insertion_class) {
                    if (expansion.tied_descent_candidates.empty() ||
                        candidate->get_min_cnots() <
                            expansion.tied_descent_candidates.front()
                                ->get_min_cnots()) {
                        expansion.tied_descent_candidates.clear();
                        expansion.tied_descent_candidates.push_back(candidate);
                    } else if (
                        candidate->get_min_cnots() ==
                        expansion.tied_descent_candidates.front()
                            ->get_min_cnots()) {
                        expansion.tied_descent_candidates.push_back(candidate);
                    }
                    // These finite classes contain several exactly
                    // degenerate cut-cover descents.  ``fill_candidates`` has
                    // already reduced the 4q class to one representative per
                    // constructive (cover, motif, ordering) lift.  Committing
                    // to its first integer-bound decrease would throw those
                    // representatives away before their target-specific HS
                    // projections can distinguish them.  Rank the complete
                    // reduced class, then make one forward-only commitment;
                    // no ancestor or sibling is retained afterward.
                    if (expansion.plateau_candidate == nullptr ||
                        *expansion.plateau_candidate > *candidate) {
                        expansion.plateau_candidate = candidate;
                    }
                    continue;
                }
                // The candidates are already ordered by placement-specific
                // OSR/HS screening. Commit the first exact residual decrease
                // and permanently discard every ancestor and sibling.
                frontier.clear();
                if (candidate->path.size() < level_limit) {
                    frontier.push_back(make_expansion(candidate));
                }
                continue;
            }
            if (deferred_profile_plateau_descent && expansion.num_cnot <= 2 &&
                parent->get_min_cnots() > 0 &&
                candidate->get_min_cnots() == parent->get_min_cnots() &&
                std::isfinite(parent->lower_budget_profile_distance) &&
                std::isfinite(candidate->lower_budget_profile_distance)) {
                const double distance_epsilon =
                    std::max(
                        64.0 * std::numeric_limits<double>::epsilon(),
                        use_hilbert_schmidt_guidance
                            ? osr_guidance_tolerance_loc
                            : osr_optimization_tolerance_loc) *
                    std::max(1.0, std::abs(parent->lower_budget_profile_distance));
                // A sub-resolution tail change is not evidence worth an
                // irreversible CNOT insertion. This uses the existing rank
                // guidance scale; final unitary acceptance is unchanged.
                if (candidate->lower_budget_profile_distance + distance_epsilon <
                    parent->lower_budget_profile_distance &&
                    (expansion.fractional_plateau_candidate == nullptr ||
                     candidate->lower_budget_profile_distance <
                         expansion.fractional_plateau_candidate
                             ->lower_budget_profile_distance)) {
                    expansion.fractional_plateau_candidate = candidate;
                }
            }
        }
        best_first_solution_path = best->path.copy();
        best_first_solution_parameters = best->optimized_parameters;
        return best->path;
    }

    while (top_heap != nullptr) {
        std::unique_ptr<SearchNode> cur(top_heap.release());
        visited.clear(); // clear visited to save memory, relying on the fact that we won't revisit nodes anyway
        if (certify_zero_residual(*cur)) {
            return record_certificate(*cur);
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
                best_first_solution_path = cur->path.copy();
                best_first_solution_parameters = cur->optimized_parameters;
                return cur->path; // best solution found within level limit, return immediately
            }

            generate_insertions(cur->path, topology, topo_filter, num_cnot,
                [&](const GrayCodeCNOT& newpath) {
                    if (add_to_heap(newpath, cur.get(), true)) {
                        return top_heap->get_min_cnots() == 0;
                    }
                    return false;
                });

            // Every generated insertion can be rejected as a duplicate or a
            // non-unique structure. In that case no successor exists at this
            // insertion count; do not dereference the empty greedy frontier.
            if (top_heap == nullptr) {
                ++num_cnot;
                continue;
            }

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
    return startpath; // single qubit fall-through case
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
    bool stop_first_solution = true;
    if (config.count("stop_first_solution") > 0) {
        config["stop_first_solution"].get_property(stop_first_solution);
    }
    GrayCodeCNOT best_solution;
    std::atomic<bool> found_optimal_solution{false};

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
    double osr_optimization_tolerance_loc =
        DEFAULT_OSR_OPTIMIZATION_TOLERANCE;
    if (config.count("osr_optimization_tolerance") > 0) {
        config["osr_optimization_tolerance"].get_property(
            osr_optimization_tolerance_loc
        );
    }
    bool use_hilbert_schmidt_guidance = qbit_num >= 2;
    if (config.count("osr_use_hilbert_schmidt_guidance") > 0) {
        config["osr_use_hilbert_schmidt_guidance"].get_property(
            use_hilbert_schmidt_guidance
        );
    }
    double osr_guidance_tolerance_loc = 1e-4;
    if (config.count("osr_guidance_tolerance") > 0) {
        config["osr_guidance_tolerance"].get_property(
            osr_guidance_tolerance_loc
        );
    }
    const double osr_tol = std::sqrt(
        use_hilbert_schmidt_guidance
            ? osr_guidance_tolerance_loc
            : osr_optimization_tolerance_loc
    );

    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads;
    concurrency = concurrency < iteration_max ? concurrency : iteration_max;
    int parallel = get_parallel_configuration();
    bool deterministic_random_seed = config.count("random_seed") > 0;
    long long configured_random_seed = 0;
    if (deterministic_random_seed) {
        config["random_seed"].get_property(configured_random_seed);
    }

    auto process_job_range = [&](int64_t begin, int64_t end) {
        N_Qubit_Decomposition_custom&& cDecomp_custom_random = perform_optimization(nullptr, true);
        cDecomp_custom_random.set_cost_function_variant(OSR_ENTANGLEMENT);
        std::mt19937 ts_gen(std::random_device{}());
        std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);

        for (int64_t job_idx = begin; job_idx < end; ++job_idx) {

            if (deterministic_random_seed) {
                std::seed_seq seed_sequence{
                    static_cast<std::uint32_t>(configured_random_seed),
                    static_cast<std::uint32_t>(
                        static_cast<unsigned long long>(configured_random_seed) >> 32
                    ),
                    static_cast<std::uint32_t>(level_num),
                    static_cast<std::uint32_t>(job_idx),
                };
                ts_gen.seed(seed_sequence);
            }

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
                if (stop_first_solution &&
                    found_optimal_solution.load(std::memory_order_acquire)) {
                    break;
                }
                const GrayCodeCNOT& solution = all_pairs[iter_idx];

                SearchNode sn = evaluate_path(cDecomp_custom_random, osr_bound_solver, all_cuts, Fnorm, osr_tol, distrib_real, ts_gen, solution);
                increment_num_iters(
                    cDecomp_custom_random
                        .get_num_iters()); // retrieve the number of iterations spent on optimization

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
                         std::get<1>(osr_result) > std::get<1>(prefix_osr_result) + 1e-3)) {
                        isWorse = true;
                        break;
                    }
                }
                int cnot_lower_bound = std::get<0>(osr_result);
                if (cnot_lower_bound <= level_limit - level_num && !isWorse) {
                    tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};
                    all_osr_results.emplace(std::move(sn));
                    if (cnot_lower_bound == 0) {
                        found_optimal_solution.store(true, std::memory_order_release);
                        successful_solutions.push_back(solution.copy());
                    }
                }

                /*for( int gcode_idx=0; gcode_idx<solution.size(); gcode_idx++ ) {
                    std::cout << solution[gcode_idx] << ", ";
                }
                std::cout << current_minimum << std::endl;*/
            }
        }
    };

    if (parallel == 0) {
        process_job_range(0, concurrency);
    }
    else {
        int64_t work_batch = 1;
        // std::cout << "levels " << level_num << std::endl;
        tbb::parallel_for(
            tbb::blocked_range<int64_t>((int64_t)0, concurrency, work_batch), [&](tbb::blocked_range<int64_t> r) {
                process_job_range(r.begin(), r.end());
        });
    }

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

        increment_num_iters(
            cDecomp_custom_random.get_num_iters()); // retrieve the number of iterations spent on optimization

        double current_minimum_tmp = cDecomp_custom_random.get_current_minimum();
        sstream.str("");
        sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp;
        print(sstream, 1);

        if (current_minimum_tmp < current_minimum) {
            current_minimum = current_minimum_tmp;
            optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
            sync_optimized_parameters_float();
        }

        // std::cout << "iiiiiiiiiiiiiiiiii " << current_minimum_tmp << std::endl;
        delete (gate_structure_loc);
        return gcode;
    }

    GrayCodeCNOT gcode_best_solution;
    std::atomic<bool> found_optimal_solution{false};

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

    auto process_job_range = [&](int64_t begin, int64_t end) {
        for (int64_t job_idx = begin; job_idx < end; ++job_idx) {

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

                if (found_optimal_solution.load(std::memory_order_acquire)) {
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

                increment_num_iters(cDecomp_custom_random
                                       .get_num_iters()); // retrieve the number of iterations spent on optimization

                double current_minimum_tmp = cDecomp_custom_random.get_current_minimum();
                sstream.str("");
                sstream << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp;
                print(sstream, 1);

                // std::cout << "Optimization with " << level_num << " levels converged to " << current_minimum_tmp
                // << std::endl;

                {
                    tbb::spin_mutex::scoped_lock tree_search_lock{tree_search_mutex};

                    if (current_minimum_tmp < current_minimum &&
                        !found_optimal_solution.load(std::memory_order_relaxed)) {

                        current_minimum = current_minimum_tmp;
                        gcode_best_solution = gcode;

                        optimized_parameters_mtx = cDecomp_custom_random.get_optimized_parameters();
                        sync_optimized_parameters_float();
                    }

                    if (current_minimum < optimization_tolerance_loc &&
                        !found_optimal_solution.load(std::memory_order_relaxed)) {
                        found_optimal_solution.store(true, std::memory_order_release);
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
    };

    if (parallel == 0) {
        process_job_range(0, concurrency);
    }
    else {
        int64_t work_batch = 1;
        // std::cout << "levels " << level_num << std::endl;
        tbb::parallel_for(
            tbb::blocked_range<int64_t>((int64_t)0, concurrency, work_batch), [&](tbb::blocked_range<int64_t> r) {
                process_job_range(r.begin(), r.end());
        });
    }

    return gcode_best_solution;
}

/**
@brief Call to perform the optimization on the given gate structure
@param gate_structure_loc The gate structure to be optimized (can be nullptr)
@param osr_scoring If true, configure this optimizer for OSR scoring, including
the OSR convergence tolerance and optional float32 target. False configures it
for Hilbert-Schmidt synthesis using the user-specified tolerance and float64 target.
@return Returns an instance of N_Qubit_Decomposition_custom with optimized parameters
*/
N_Qubit_Decomposition_custom N_Qubit_Decomposition_Tree_Search::perform_optimization(
    Gates_block* gate_structure_loc,
    bool osr_scoring
) {

    double hilbert_schmidt_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(hilbert_schmidt_tolerance_loc);
    } else {
        hilbert_schmidt_tolerance_loc = optimization_tolerance;
    }

    // Only calls made for OSR cost-function scoring pass osr_scoring=true.
    // Keep that optimizer's numerical convergence
    // independent from the user-specified Hilbert-Schmidt synthesis target.
    // The fixed-structure candidate optimization in determine_gate_structure
    // calls this with false and therefore always uses the HS tolerance.
    double cost_tolerance_loc = osr_scoring
        ? DEFAULT_OSR_OPTIMIZATION_TOLERANCE
        : hilbert_schmidt_tolerance_loc;
    if (osr_scoring && config.count("osr_optimization_tolerance") > 0) {
        config["osr_optimization_tolerance"].get_property(cost_tolerance_loc);
    }

    // OSR structure scoring may use the float32 target, but Hilbert-Schmidt
    // refinement must use the original float64 target. Promoting Umtx_float
    // back to double leaves a slightly nonunitary, quantized target and creates
    // an artificial fidelity floor.
    std::map<std::string, Config_Element> optimization_config = config;
    bool optimization_uses_float =
        use_float && osr_scoring && qbit_num >= 5;
    optimization_config["use_float"].set_property(
        "use_float", optimization_uses_float
    );
    optimization_config["optimization_tolerance"].set_property(
        "optimization_tolerance", cost_tolerance_loc
    );
    if (qbit_num == 4) {
        long long max_inner_iterations_4q = 1000;
        if (config.count("max_inner_iterations_bfgs2_4q") > 0) {
            config["max_inner_iterations_bfgs2_4q"].get_property(
                max_inner_iterations_4q
            );
        }
        optimization_config["max_inner_iterations_bfgs2"].set_property(
            "max_inner_iterations_bfgs2", max_inner_iterations_4q
        );
    }

    N_Qubit_Decomposition_custom cDecomp_custom_random;
    if ( optimization_uses_float ) {
        cDecomp_custom_random =
            N_Qubit_Decomposition_custom(Umtx_float.copy(), qbit_num, false, optimization_config, RANDOM, accelerator_num);
    }
    else {
        cDecomp_custom_random =
            N_Qubit_Decomposition_custom(Umtx.copy(), qbit_num, false, optimization_config, RANDOM, accelerator_num);
    }
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
    cDecomp_custom_random.set_optimization_tolerance(cost_tolerance_loc);
    cDecomp_custom_random.set_trace_offset(trace_offset);
    cDecomp_custom_random.set_optimizer(alg);
    cDecomp_custom_random.set_project_name(project_name);
    if (alg == ADAM || alg == BFGS2) {
        int max_inner_iterations_loc = 1000;
        if (gate_structure_loc != nullptr) {
            int param_num_loc = gate_structure_loc->get_parameter_num();
            // This limit applies independently to every local solve in BFGS2
            // basin hopping.  The old 10,000,000-at-852-parameters scaling
            // allowed one rejected four-qubit plateau candidate to run for
            // hours after per-solve counters were corrected.
            const int scaled_budget = (param_num_loc * 10000 + 851) / 852;
            if (scaled_budget > max_inner_iterations_loc) {
                max_inner_iterations_loc = scaled_budget;
            }
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
    if ( use_float ) {
        Umtx_float = Umtx_new.to_float32();
    }
}

void N_Qubit_Decomposition_Tree_Search::set_unitary(Matrix_float& Umtx_new) {

    Umtx_float = Umtx_new;
    Umtx = Umtx_new.to_float64();
    use_float = true;
}
