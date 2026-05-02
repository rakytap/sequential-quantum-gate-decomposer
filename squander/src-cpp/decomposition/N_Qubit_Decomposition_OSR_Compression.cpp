/*
Created on Sat May 02 2026
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file N_Qubit_Decomposition_OSR_Compression.cpp
    \brief OSR-guided top-down compression for an existing gate structure.
*/

#include "N_Qubit_Decomposition_OSR_Compression.h"
#include "N_Qubit_Decomposition_Cost_Function.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <utility>

namespace {

struct CompressionCandidate {
    std::vector<int> removed_ids;
    Matrix_real initial_parameters;
    N_Qubit_Decomposition_OSR_Compression_Score score;
    int entangling_gate_num;
    std::shared_ptr<Gates_block> gate_structure;
    std::string key;
};

static bool is_entangling_gate_type(gate_type type) {
    switch (type) {
    case CNOT_OPERATION:
    case CZ_OPERATION:
    case CH_OPERATION:
    case SYC_OPERATION:
    case CRY_OPERATION:
    case CRX_OPERATION:
    case CRZ_OPERATION:
    case CP_OPERATION:
    case CR_OPERATION:
    case CROT_OPERATION:
    case CZ_NU_OPERATION:
    case CU_OPERATION:
    case ADAPTIVE_OPERATION:
    case RXX_OPERATION:
    case RYY_OPERATION:
    case RZZ_OPERATION:
    case SWAP_OPERATION:
    case CSWAP_OPERATION:
    case CCX_OPERATION:
        return true;
    default:
        return false;
    }
}

static bool is_entangling_gate(Gate* gate) {
    if (gate == NULL || gate->get_type() == BLOCK_OPERATION) {
        return false;
    }
    if (is_entangling_gate_type(gate->get_type())) {
        return true;
    }
    return gate->get_involved_qubits().size() > 1;
}

static void collect_entangling_gate_paths(Gates_block* block,
                                          std::vector<int>& prefix,
                                          std::vector<OSRGatePath>& out) {
    if (block == NULL) {
        return;
    }

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        prefix.push_back(idx);

        if (is_entangling_gate(gate)) {
            OSRGatePath path;
            path.indices = prefix;
            out.push_back(path);
        }

        if (gate != NULL && gate->get_type() == BLOCK_OPERATION) {
            collect_entangling_gate_paths(static_cast<Gates_block*>(gate), prefix, out);
        }

        prefix.pop_back();
    }
}

static std::vector<OSRGatePath> collect_entangling_gate_paths(Gates_block* block) {
    std::vector<OSRGatePath> ret;
    std::vector<int> prefix;
    collect_entangling_gate_paths(block, prefix, ret);
    return ret;
}

static void append_int_vector_signature(std::stringstream& sstream,
                                        const std::vector<int>& values) {
    sstream << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx > 0) {
            sstream << ",";
        }
        sstream << values[idx];
    }
    sstream << "]";
}

static void append_gate_structure_signature(Gates_block* block,
                                            std::stringstream& sstream) {
    if (block == NULL) {
        sstream << "NULL";
        return;
    }

    sstream << "B" << block->get_gate_num() << "(";
    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            sstream << "NULL;";
            continue;
        }

        sstream << static_cast<int>(gate->get_type()) << ":T";
        std::vector<int> targets = gate->get_target_qbits();
        if (targets.empty() && gate->get_target_qbit() >= 0) {
            targets.push_back(gate->get_target_qbit());
        }
        append_int_vector_signature(sstream, targets);

        sstream << ":C";
        std::vector<int> controls = gate->get_control_qbits();
        if (controls.empty() && gate->get_control_qbit() >= 0) {
            controls.push_back(gate->get_control_qbit());
        }
        append_int_vector_signature(sstream, controls);

        sstream << ":P" << gate->get_parameter_num();
        if (gate->get_type() == BLOCK_OPERATION) {
            sstream << "{";
            append_gate_structure_signature(static_cast<Gates_block*>(gate), sstream);
            sstream << "}";
        }
        sstream << ";";
    }
    sstream << ")";
}

static std::string gate_structure_signature(Gates_block* block) {
    std::stringstream sstream;
    append_gate_structure_signature(block, sstream);
    return sstream.str();
}

static Gate* gate_at_path(Gates_block* block, const OSRGatePath& path) {
    Gates_block* current_block = block;
    for (size_t depth = 0; depth < path.indices.size(); ++depth) {
        if (current_block == NULL) {
            return NULL;
        }

        int gate_idx = path.indices[depth];
        if (gate_idx < 0 || gate_idx >= current_block->get_gate_num()) {
            return NULL;
        }

        Gate* gate = current_block->get_gate(gate_idx);
        if (depth == path.indices.size() - 1) {
            return gate;
        }
        if (gate == NULL || gate->get_type() != BLOCK_OPERATION) {
            return NULL;
        }
        current_block = static_cast<Gates_block*>(gate);
    }
    return NULL;
}

static bool get_two_qubit_endpoint_pair(Gate* gate, int& q0, int& q1) {
    if (gate == NULL) {
        return false;
    }

    std::vector<int> involved = gate->get_involved_qubits();
    std::sort(involved.begin(), involved.end());
    involved.erase(std::unique(involved.begin(), involved.end()), involved.end());
    if (involved.size() != 2) {
        return false;
    }

    q0 = involved[0];
    q1 = involved[1];
    return true;
}

static bool gate_endpoint_sets_are_disjoint(Gate* lhs, Gate* rhs) {
    int lhs_q0 = 0;
    int lhs_q1 = 0;
    int rhs_q0 = 0;
    int rhs_q1 = 0;
    if (!get_two_qubit_endpoint_pair(lhs, lhs_q0, lhs_q1) ||
        !get_two_qubit_endpoint_pair(rhs, rhs_q0, rhs_q1)) {
        return false;
    }

    return lhs_q0 != rhs_q0 && lhs_q0 != rhs_q1 &&
           lhs_q1 != rhs_q0 && lhs_q1 != rhs_q1;
}

static bool gate_type_is_directional(gate_type type) {
    switch (type) {
    case CZ_OPERATION:
    case SWAP_OPERATION:
    case RXX_OPERATION:
    case RYY_OPERATION:
    case RZZ_OPERATION:
        return false;
    default:
        return true;
    }
}

static bool rewire_two_qubit_gate(Gate* gate, int new_target, int new_control) {
    if (gate == NULL || new_target == new_control) {
        return false;
    }

    int old_q0 = 0;
    int old_q1 = 0;
    if (!get_two_qubit_endpoint_pair(gate, old_q0, old_q1)) {
        return false;
    }

    std::vector<int> controls = gate->get_control_qbits();
    if (!controls.empty() || gate->get_control_qbit() >= 0) {
        gate->set_target_qbit(new_target);
        gate->set_control_qbit(new_control);
        return true;
    }

    std::vector<int> targets = gate->get_target_qbits();
    if (targets.size() >= 2) {
        std::vector<int> new_targets;
        new_targets.push_back(new_target);
        new_targets.push_back(new_control);
        gate->set_target_qbits(new_targets);
        return true;
    }

    return false;
}

static bool path_has_prefix(const OSRGatePath& path, const std::vector<int>& prefix) {
    if (path.indices.size() < prefix.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), path.indices.begin());
}

static bool path_equals_prefix(const OSRGatePath& path, const std::vector<int>& prefix) {
    return path.indices.size() == prefix.size() && path_has_prefix(path, prefix);
}

static bool subtree_contains_removed_path(const std::set<OSRGatePath>& removed_paths,
                                          const std::vector<int>& prefix) {
    for (std::set<OSRGatePath>::const_iterator it = removed_paths.begin(); it != removed_paths.end(); ++it) {
        if (path_has_prefix(*it, prefix)) {
            return true;
        }
    }
    return false;
}

static Gates_block* clone_without_removed_paths(Gates_block* block,
                                                const std::set<OSRGatePath>& removed_paths,
                                                std::vector<int>& prefix) {
    Gates_block* ret = new Gates_block(block->get_qbit_num());

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        prefix.push_back(idx);

        bool remove_gate = false;
        for (std::set<OSRGatePath>::const_iterator it = removed_paths.begin(); it != removed_paths.end(); ++it) {
            if (path_equals_prefix(*it, prefix)) {
                remove_gate = true;
                break;
            }
        }

        if (!remove_gate) {
            if (gate->get_type() == BLOCK_OPERATION && subtree_contains_removed_path(removed_paths, prefix)) {
                Gates_block* cloned_block = clone_without_removed_paths(
                    static_cast<Gates_block*>(gate), removed_paths, prefix);
                ret->add_gate(cloned_block);
            } else {
                ret->add_gate(gate->clone());
            }
        }

        prefix.pop_back();
    }

    return ret;
}

static Gates_block* clone_without_removed_paths(Gates_block* block,
                                                const std::vector<OSRGatePath>& all_paths,
                                                const std::vector<int>& removed_ids) {
    std::set<OSRGatePath> removed_paths;
    for (size_t idx = 0; idx < removed_ids.size(); ++idx) {
        removed_paths.insert(all_paths[removed_ids[idx]]);
    }

    std::vector<int> prefix;
    return clone_without_removed_paths(block, removed_paths, prefix);
}

static Gates_block* clone_with_rewired_gate_path(Gates_block* block,
                                                 const OSRGatePath& path,
                                                 int depth,
                                                 int new_target,
                                                 int new_control) {
    Gates_block* ret = new Gates_block(block->get_qbit_num());

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            continue;
        }

        if (depth < static_cast<int>(path.indices.size()) &&
            idx == path.indices[depth]) {
            if (depth == static_cast<int>(path.indices.size()) - 1) {
                Gate* cloned_gate = gate->clone();
                if (!rewire_two_qubit_gate(cloned_gate, new_target, new_control)) {
                    delete cloned_gate;
                    delete ret;
                    return NULL;
                }
                ret->add_gate(cloned_gate);
            } else {
                if (gate->get_type() != BLOCK_OPERATION) {
                    delete ret;
                    return NULL;
                }

                Gates_block* rewired_block = clone_with_rewired_gate_path(
                    static_cast<Gates_block*>(gate), path, depth + 1,
                    new_target, new_control);
                if (rewired_block == NULL) {
                    delete ret;
                    return NULL;
                }
                ret->add_gate(rewired_block);
            }
        } else {
            ret->add_gate(gate->clone());
        }
    }

    return ret;
}

static Gates_block* clone_with_rewired_gate_path(Gates_block* block,
                                                 const OSRGatePath& path,
                                                 int new_target,
                                                 int new_control) {
    return clone_with_rewired_gate_path(block, path, 0, new_target, new_control);
}

static Gates_block* clone_with_swapped_sibling_gates(Gates_block* block,
                                                     const std::vector<int>& parent_path,
                                                     int depth,
                                                     int first_idx,
                                                     int second_idx) {
    Gates_block* ret = new Gates_block(block->get_qbit_num());

    if (depth == static_cast<int>(parent_path.size())) {
        for (int idx = 0; idx < block->get_gate_num(); ++idx) {
            int source_idx = idx;
            if (idx == first_idx) {
                source_idx = second_idx;
            } else if (idx == second_idx) {
                source_idx = first_idx;
            }
            Gate* source_gate = block->get_gate(source_idx);
            if (source_gate != NULL) {
                ret->add_gate(source_gate->clone());
            }
        }
        return ret;
    }

    int selected_idx = parent_path[depth];
    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            continue;
        }

        if (idx == selected_idx) {
            if (gate->get_type() != BLOCK_OPERATION) {
                delete ret;
                return NULL;
            }
            Gates_block* swapped_block = clone_with_swapped_sibling_gates(
                static_cast<Gates_block*>(gate), parent_path, depth + 1,
                first_idx, second_idx);
            if (swapped_block == NULL) {
                delete ret;
                return NULL;
            }
            ret->add_gate(swapped_block);
        } else {
            ret->add_gate(gate->clone());
        }
    }

    return ret;
}

static Gates_block* clone_with_swapped_sibling_gates(Gates_block* block,
                                                     const OSRGatePath& first_path,
                                                     const OSRGatePath& second_path) {
    if (first_path.indices.size() != second_path.indices.size() ||
        first_path.indices.empty()) {
        return NULL;
    }

    std::vector<int> first_parent(
        first_path.indices.begin(), first_path.indices.end() - 1);
    std::vector<int> second_parent(
        second_path.indices.begin(), second_path.indices.end() - 1);
    if (first_parent != second_parent) {
        return NULL;
    }

    int first_idx = first_path.indices.back();
    int second_idx = second_path.indices.back();
    if (first_idx == second_idx) {
        return NULL;
    }
    if (second_idx < first_idx) {
        std::swap(first_idx, second_idx);
    }

    return clone_with_swapped_sibling_gates(
        block, first_parent, 0, first_idx, second_idx);
}

static bool parameter_interval_for_path(Gates_block* block,
                                        const OSRGatePath& path,
                                        int depth,
                                        int offset,
                                        int& start,
                                        int& length) {
    if (block == NULL || depth >= static_cast<int>(path.indices.size())) {
        return false;
    }

    int gate_idx = path.indices[depth];
    Gate* gate = block->get_gate(gate_idx);
    if (gate == NULL) {
        return false;
    }

    int gate_offset = offset + gate->get_parameter_start_idx();
    if (depth == static_cast<int>(path.indices.size()) - 1) {
        start = gate_offset;
        length = gate->get_parameter_num();
        return true;
    }

    if (gate->get_type() != BLOCK_OPERATION) {
        return false;
    }

    return parameter_interval_for_path(
        static_cast<Gates_block*>(gate), path, depth + 1, gate_offset, start, length);
}

static Matrix_real reduced_parameters_without_paths(
    Gates_block* original_gate_structure,
    const std::vector<OSRGatePath>& removed_paths,
    const Matrix_real& original_parameters) {
    if (original_parameters.size() == 0 ||
        original_parameters.size() != original_gate_structure->get_parameter_num()) {
        return Matrix_real(0, 0);
    }

    std::vector<std::pair<int, int>> intervals;
    intervals.reserve(removed_paths.size());
    for (size_t idx = 0; idx < removed_paths.size(); ++idx) {
        int start = 0;
        int length = 0;
        if (parameter_interval_for_path(
                original_gate_structure, removed_paths[idx], 0, 0, start, length) &&
            length > 0) {
            intervals.push_back(std::make_pair(start, start + length));
        }
    }

    if (intervals.empty()) {
        return original_parameters.copy();
    }

    std::sort(intervals.begin(), intervals.end());
    std::vector<std::pair<int, int>> merged;
    for (size_t idx = 0; idx < intervals.size(); ++idx) {
        if (merged.empty() || intervals[idx].first > merged.back().second) {
            merged.push_back(intervals[idx]);
        } else {
            merged.back().second = std::max(merged.back().second, intervals[idx].second);
        }
    }

    int removed_parameter_num = 0;
    for (size_t idx = 0; idx < merged.size(); ++idx) {
        removed_parameter_num += merged[idx].second - merged[idx].first;
    }

    Matrix_real reduced_parameters(1, original_parameters.size() - removed_parameter_num);
    int src = 0;
    int dst = 0;
    for (size_t idx = 0; idx < merged.size(); ++idx) {
        int keep_num = merged[idx].first - src;
        if (keep_num > 0) {
            std::memcpy(reduced_parameters.get_data() + dst,
                        original_parameters.get_data() + src,
                        keep_num * sizeof(double));
            dst += keep_num;
        }
        src = merged[idx].second;
    }

    if (src < original_parameters.size()) {
        int keep_num = original_parameters.size() - src;
        std::memcpy(reduced_parameters.get_data() + dst,
                    original_parameters.get_data() + src,
                    keep_num * sizeof(double));
    }

    return reduced_parameters;
}

static Matrix_real reduced_parameters_without_removed_paths(
    Gates_block* original_gate_structure,
    const std::vector<OSRGatePath>& all_paths,
    const std::vector<int>& removed_ids,
    const Matrix_real& original_parameters) {
    std::vector<OSRGatePath> removed_paths;
    removed_paths.reserve(removed_ids.size());
    for (size_t idx = 0; idx < removed_ids.size(); ++idx) {
        removed_paths.push_back(all_paths[removed_ids[idx]]);
    }
    return reduced_parameters_without_paths(
        original_gate_structure, removed_paths, original_parameters);
}

static void add_topology_edge(std::set<std::pair<int, int>>& edges, int q0, int q1) {
    if (q0 == q1) {
        return;
    }
    if (q1 < q0) {
        std::swap(q0, q1);
    }
    edges.insert(std::make_pair(q0, q1));
}

static void collect_topology_edges(Gates_block* block, std::set<std::pair<int, int>>& edges) {
    if (block == NULL) {
        return;
    }

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            continue;
        }

        if (gate->get_type() == BLOCK_OPERATION) {
            collect_topology_edges(static_cast<Gates_block*>(gate), edges);
            continue;
        }

        if (!is_entangling_gate(gate)) {
            continue;
        }

        std::vector<int> involved = gate->get_involved_qubits();
        for (size_t q0_idx = 0; q0_idx < involved.size(); ++q0_idx) {
            for (size_t q1_idx = q0_idx + 1; q1_idx < involved.size(); ++q1_idx) {
                add_topology_edge(edges, involved[q0_idx], involved[q1_idx]);
            }
        }
    }
}

static std::vector<matrix_base<int>> topology_from_gate_structure(Gates_block* gate_structure, int qbit_num) {
    std::set<std::pair<int, int>> edges;
    collect_topology_edges(gate_structure, edges);

    if (edges.empty() && qbit_num > 1) {
        for (int q0 = 0; q0 < qbit_num; ++q0) {
            for (int q1 = q0 + 1; q1 < qbit_num; ++q1) {
                edges.insert(std::make_pair(q0, q1));
            }
        }
    }

    std::vector<matrix_base<int>> topology;
    topology.reserve(edges.size());
    for (std::set<std::pair<int, int>>::const_iterator it = edges.begin(); it != edges.end(); ++it) {
        matrix_base<int> edge(2, 1);
        edge[0] = it->first;
        edge[1] = it->second;
        topology.push_back(edge);
    }
    return topology;
}

static std::vector<std::pair<int, int>> topology_pairs_from_matrices(
    const std::vector<matrix_base<int>>& topology) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(topology.size());
    for (size_t idx = 0; idx < topology.size(); ++idx) {
        int q0 = topology[idx][0];
        int q1 = topology[idx][1];
        if (q0 == q1) {
            continue;
        }
        if (q1 < q0) {
            std::swap(q0, q1);
        }
        std::pair<int, int> edge(q0, q1);
        if (std::find(pairs.begin(), pairs.end(), edge) == pairs.end()) {
            pairs.push_back(edge);
        }
    }
    return pairs;
}

static std::vector<std::pair<int, int>> complete_topology_pairs(int qbit_num) {
    std::vector<std::pair<int, int>> pairs;
    for (int q0 = 0; q0 < qbit_num; ++q0) {
        for (int q1 = q0 + 1; q1 < qbit_num; ++q1) {
            pairs.push_back(std::make_pair(q0, q1));
        }
    }
    return pairs;
}

static Gates_block* construct_cnot_skeleton_gate_structure(
    int qbit_num,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& sequence) {
    Gates_block* gate_structure = new Gates_block(qbit_num);

    for (size_t idx = 0; idx < sequence.size(); ++idx) {
        int edge_idx = sequence[idx];
        if (edge_idx < 0 || edge_idx >= static_cast<int>(edges.size())) {
            delete gate_structure;
            return NULL;
        }

        Gates_block* layer = new Gates_block(qbit_num);
        int target = edges[edge_idx].first;
        int control = edges[edge_idx].second;
        layer->add_u3(target);
        layer->add_u3(control);
        layer->add_cnot(target, control);
        gate_structure->add_gate(layer);
    }

    Gates_block* final_layer = new Gates_block(qbit_num);
    for (int qbit = 0; qbit < qbit_num; ++qbit) {
        final_layer->add_u3(qbit);
    }
    gate_structure->add_gate(final_layer);

    return gate_structure;
}

static int64_t limited_integer_power(int base, int exponent, int64_t limit) {
    int64_t value = 1;
    for (int idx = 0; idx < exponent; ++idx) {
        if (base <= 0 || value > limit / base) {
            return limit + 1;
        }
        value *= base;
    }
    return value;
}

static std::vector<CompressionCandidate> generate_cnot_skeleton_candidates(
    int qbit_num,
    int original_entangling_gate_num,
    const std::vector<std::pair<int, int>>& edges,
    const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    std::vector<CompressionCandidate> candidates;
    if (!options.enable_skeleton_search || edges.empty() ||
        options.skeleton_max_candidates <= 0 || original_entangling_gate_num <= 0) {
        return candidates;
    }

    int target_depth = options.skeleton_target_cnots;
    if (target_depth < 0) {
        int removed = options.max_removed_gates >= 0 ? options.max_removed_gates : 2;
        target_depth = original_entangling_gate_num - removed;
    }
    if (target_depth < 0 || target_depth >= original_entangling_gate_num) {
        return candidates;
    }

    int edge_num = static_cast<int>(edges.size());
    int64_t combination_num = limited_integer_power(
        edge_num, target_depth, static_cast<int64_t>(options.skeleton_max_candidates));

    if (combination_num > options.skeleton_max_candidates) {
        return candidates;
    }

    std::set<std::string> seen;
    for (int64_t state = 0; state < combination_num; ++state) {
        int64_t value = state;
        std::vector<int> sequence(target_depth, 0);
        for (int depth = target_depth - 1; depth >= 0; --depth) {
            sequence[depth] = static_cast<int>(value % edge_num);
            value /= edge_num;
        }

        std::shared_ptr<Gates_block> gate_structure(
            construct_cnot_skeleton_gate_structure(qbit_num, edges, sequence));
        if (!gate_structure) {
            continue;
        }

        std::string key = gate_structure_signature(gate_structure.get());
        if (!seen.insert(key).second) {
            continue;
        }

        CompressionCandidate candidate;
        candidate.entangling_gate_num = target_depth;
        candidate.gate_structure = gate_structure;
        candidate.key = key;
        candidate.initial_parameters = Matrix_real(0, 0);
        candidates.push_back(candidate);
    }

    return candidates;
}

static double residual_sum(const std::vector<std::pair<int, double>>& cut_bounds) {
    return std::accumulate(cut_bounds.begin(), cut_bounds.end(), 0.0,
        [&cut_bounds](double acc, const std::pair<int, double>& item) {
            return acc + item.first * cut_bounds.size() + item.second;
        });
}

static bool score_less(const N_Qubit_Decomposition_OSR_Compression_Score& lhs,
                       const N_Qubit_Decomposition_OSR_Compression_Score& rhs) {
    if (lhs.min_remaining_cnots != rhs.min_remaining_cnots) {
        return lhs.min_remaining_cnots < rhs.min_remaining_cnots;
    }
    if (lhs.kappa != rhs.kappa) {
        return lhs.kappa < rhs.kappa;
    }
    return lhs.residual < rhs.residual;
}

static bool beam_candidate_less(const CompressionCandidate& lhs,
                                const CompressionCandidate& rhs) {
    if (score_less(lhs.score, rhs.score)) {
        return true;
    }
    if (score_less(rhs.score, lhs.score)) {
        return false;
    }
    return lhs.key < rhs.key;
}

static bool final_candidate_less(const CompressionCandidate& lhs,
                                 const CompressionCandidate& rhs) {
    if (lhs.entangling_gate_num != rhs.entangling_gate_num) {
        return lhs.entangling_gate_num < rhs.entangling_gate_num;
    }
    return beam_candidate_less(lhs, rhs);
}

static std::string compression_candidate_key(const CompressionCandidate& candidate) {
    if (!candidate.key.empty()) {
        return candidate.key;
    }

    std::stringstream sstream;
    sstream << "removed:";
    for (size_t idx = 0; idx < candidate.removed_ids.size(); ++idx) {
        if (idx > 0) {
            sstream << ",";
        }
        sstream << candidate.removed_ids[idx];
    }
    return sstream.str();
}

static void sort_unique_candidates(std::vector<CompressionCandidate>& candidates, bool final_sort) {
    std::sort(candidates.begin(), candidates.end(),
              final_sort ? final_candidate_less : beam_candidate_less);

    std::set<std::string> seen;
    std::vector<CompressionCandidate> unique_candidates;
    unique_candidates.reserve(candidates.size());
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        std::string key = compression_candidate_key(candidates[idx]);
        if (seen.insert(key).second) {
            unique_candidates.push_back(candidates[idx]);
        }
    }
    candidates.swap(unique_candidates);
}

static Gates_block* clone_gate_structure_for_candidate(
    Gates_block* original_gate_structure,
    const std::vector<OSRGatePath>& original_paths,
    const CompressionCandidate& candidate) {
    if (candidate.gate_structure) {
        return candidate.gate_structure->clone();
    }
    return clone_without_removed_paths(
        original_gate_structure, original_paths, candidate.removed_ids);
}

static bool edge_shares_endpoint(const std::pair<int, int>& edge, int q0, int q1) {
    return edge.first == q0 || edge.first == q1 ||
           edge.second == q0 || edge.second == q1;
}

static bool same_undirected_edge(const std::pair<int, int>& edge, int q0, int q1) {
    int a = q0;
    int b = q1;
    if (b < a) {
        std::swap(a, b);
    }
    return edge.first == a && edge.second == b;
}

static bool same_parent_and_adjacent(const OSRGatePath& lhs,
                                     const OSRGatePath& rhs) {
    if (lhs.indices.size() != rhs.indices.size() || lhs.indices.empty()) {
        return false;
    }

    for (size_t idx = 0; idx + 1 < lhs.indices.size(); ++idx) {
        if (lhs.indices[idx] != rhs.indices[idx]) {
            return false;
        }
    }

    return std::abs(lhs.indices.back() - rhs.indices.back()) == 1;
}

static void append_candidate_if_new(std::vector<CompressionCandidate>& out,
                                    std::set<std::string>& seen,
                                    CompressionCandidate& candidate) {
    if (!candidate.gate_structure) {
        return;
    }

    candidate.key = gate_structure_signature(candidate.gate_structure.get());
    if (seen.insert(candidate.key).second) {
        candidate.entangling_gate_num =
            static_cast<int>(collect_entangling_gate_paths(candidate.gate_structure.get()).size());
        out.push_back(candidate);
    }
}

static std::vector<CompressionCandidate> generate_local_mutation_candidates(
    Gates_block* base_structure,
    const Matrix_real& base_parameters,
    const CompressionCandidate& parent,
    const std::vector<std::pair<int, int>>& mutation_edges,
    const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    std::vector<CompressionCandidate> candidates;
    if (!options.enable_mutations || options.mutation_candidates <= 0 ||
        base_structure == NULL) {
        return candidates;
    }

    std::vector<OSRGatePath> paths = collect_entangling_gate_paths(base_structure);
    if (paths.empty()) {
        return candidates;
    }

    std::set<std::string> seen;
    seen.insert(gate_structure_signature(base_structure));

    for (size_t idx = 0; idx + 1 < paths.size() &&
                         static_cast<int>(candidates.size()) < options.mutation_candidates; ++idx) {
        const OSRGatePath& lhs_path = paths[idx];
        const OSRGatePath& rhs_path = paths[idx + 1];
        if (!same_parent_and_adjacent(lhs_path, rhs_path)) {
            continue;
        }

        Gate* lhs_gate = gate_at_path(base_structure, lhs_path);
        Gate* rhs_gate = gate_at_path(base_structure, rhs_path);
        if (!gate_endpoint_sets_are_disjoint(lhs_gate, rhs_gate)) {
            continue;
        }

        std::shared_ptr<Gates_block> swapped(
            clone_with_swapped_sibling_gates(base_structure, lhs_path, rhs_path));
        if (!swapped) {
            continue;
        }

        CompressionCandidate child = parent;
        child.gate_structure = swapped;
        if (base_parameters.size() == base_structure->get_parameter_num() &&
            swapped->get_parameter_num() == base_parameters.size()) {
            child.initial_parameters = base_parameters.copy();
        } else {
            child.initial_parameters = Matrix_real(0, 0);
        }
        append_candidate_if_new(candidates, seen, child);
    }

    for (size_t path_idx = 0; path_idx < paths.size() &&
                              static_cast<int>(candidates.size()) < options.mutation_candidates; ++path_idx) {
        Gate* gate = gate_at_path(base_structure, paths[path_idx]);
        if (gate == NULL) {
            continue;
        }

        int old_q0 = 0;
        int old_q1 = 0;
        if (!get_two_qubit_endpoint_pair(gate, old_q0, old_q1)) {
            continue;
        }

        bool directional = gate_type_is_directional(gate->get_type()) &&
            (!gate->get_control_qbits().empty() || gate->get_control_qbit() >= 0);

        for (int pass = 0; pass < 2 &&
                           static_cast<int>(candidates.size()) < options.mutation_candidates; ++pass) {
            for (size_t edge_idx = 0; edge_idx < mutation_edges.size() &&
                                      static_cast<int>(candidates.size()) < options.mutation_candidates; ++edge_idx) {
                const std::pair<int, int>& edge = mutation_edges[edge_idx];
                if (same_undirected_edge(edge, old_q0, old_q1)) {
                    continue;
                }

                bool shares_endpoint = edge_shares_endpoint(edge, old_q0, old_q1);
                if ((pass == 0 && !shares_endpoint) || (pass == 1 && shares_endpoint)) {
                    continue;
                }

                int orientation_count = directional ? 2 : 1;
                for (int orientation = 0;
                     orientation < orientation_count &&
                     static_cast<int>(candidates.size()) < options.mutation_candidates;
                     ++orientation) {
                    int new_target = (orientation == 0) ? edge.first : edge.second;
                    int new_control = (orientation == 0) ? edge.second : edge.first;

                    std::shared_ptr<Gates_block> rewired(
                        clone_with_rewired_gate_path(
                            base_structure, paths[path_idx], new_target, new_control));
                    if (!rewired) {
                        continue;
                    }

                    CompressionCandidate child = parent;
                    child.gate_structure = rewired;
                    if (base_parameters.size() == base_structure->get_parameter_num() &&
                        rewired->get_parameter_num() == base_parameters.size()) {
                        child.initial_parameters = base_parameters.copy();
                    } else {
                        child.initial_parameters = Matrix_real(0, 0);
                    }
                    append_candidate_if_new(candidates, seen, child);
                }
            }
        }
    }

    return candidates;
}

static bool candidate_is_osr_admissible(const CompressionCandidate& candidate,
                                        const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    return candidate.score.min_remaining_cnots <= options.osr_bound_limit;
}

} // namespace

N_Qubit_Decomposition_OSR_Compression_Result::N_Qubit_Decomposition_OSR_Compression_Result()
    : current_minimum(std::numeric_limits<double>::infinity()),
      original_entangling_gate_num(0),
      compressed_entangling_gate_num(0),
      validated(false),
      reached_tolerance(false),
      decomposition_error(std::numeric_limits<double>::infinity()) {}

N_Qubit_Decomposition_OSR_Compression::N_Qubit_Decomposition_OSR_Compression()
    : N_Qubit_Decomposition_custom() {
    name = "OSR_Compression";
}

N_Qubit_Decomposition_OSR_Compression::N_Qubit_Decomposition_OSR_Compression(
    Matrix Umtx_in,
    int qbit_num_in,
    std::map<std::string, Config_Element>& config,
    int accelerator_num)
    : N_Qubit_Decomposition_custom(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {
    name = "OSR_Compression";
}

N_Qubit_Decomposition_OSR_Compression::N_Qubit_Decomposition_OSR_Compression(
    Matrix Umtx_in,
    int qbit_num_in,
    std::vector<matrix_base<int>> topology_in,
    std::map<std::string, Config_Element>& config,
    int accelerator_num)
    : N_Qubit_Decomposition_custom(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num),
      topology(std::move(topology_in)) {
    name = "OSR_Compression";
}

N_Qubit_Decomposition_OSR_Compression::~N_Qubit_Decomposition_OSR_Compression() {}

void N_Qubit_Decomposition_OSR_Compression::start_decomposition() {
    std::unique_ptr<Gates_block> original_gate_structure(clone());
    Matrix_real original_parameters = optimized_parameters_mtx.size() > 0
        ? optimized_parameters_mtx.copy()
        : Matrix_real(0, 0);

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    N_Qubit_Decomposition_OSR_Compression_Result result =
        compress_gate_structure(original_gate_structure.get());

    if (!result.reached_tolerance &&
        original_parameters.size() == original_gate_structure->get_parameter_num()) {
        double original_cost = optimization_problem(original_parameters);
        if (original_cost < optimization_tolerance_loc ||
            original_cost < result.current_minimum) {
            result.gate_structure.reset(original_gate_structure->clone());
            result.optimized_parameters = original_parameters.copy();
            result.current_minimum = original_cost;
            result.decomposition_error = original_cost;
            result.compressed_entangling_gate_num =
                result.original_entangling_gate_num;
            result.removed_gate_paths.clear();
            result.validated = true;
            result.reached_tolerance = original_cost < optimization_tolerance_loc;
        }
    }

    release_gates();
    combine(result.gate_structure.get());

    if (result.validated && result.optimized_parameters.size() == get_parameter_num()) {
        optimized_parameters_mtx = result.optimized_parameters.copy();
        current_minimum = result.current_minimum;
        decomposition_error = result.decomposition_error;
    } else {
        N_Qubit_Decomposition_custom::start_decomposition();
    }
}

N_Qubit_Decomposition_OSR_Compression_Options
N_Qubit_Decomposition_OSR_Compression::get_osr_compression_options() {
    N_Qubit_Decomposition_OSR_Compression_Options options;

    long long int_value;
    double double_value;

    if (config.count("osr_compression_beam") > 0) {
        config["osr_compression_beam"].get_property(int_value);
        options.beam_width = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_max_removed") > 0) {
        config["osr_compression_max_removed"].get_property(int_value);
        options.max_removed_gates = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_bound_limit") > 0) {
        config["osr_compression_bound_limit"].get_property(int_value);
        options.osr_bound_limit = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_validation_trials") > 0) {
        config["osr_compression_validation_trials"].get_property(int_value);
        options.validation_trials = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_validate") > 0) {
        config["osr_compression_validate"].get_property(int_value);
        options.validate_final = int_value != 0;
    }
    if (config.count("osr_compression_osr_tolerance") > 0) {
        config["osr_compression_osr_tolerance"].get_property(double_value);
        options.osr_tolerance = double_value;
    }
    if (config.count("osr_compression_enable_mutations") > 0) {
        config["osr_compression_enable_mutations"].get_property(int_value);
        options.enable_mutations = int_value != 0;
    }
    if (config.count("osr_compression_mutation_rounds") > 0) {
        config["osr_compression_mutation_rounds"].get_property(int_value);
        options.mutation_rounds = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_mutation_candidates") > 0) {
        config["osr_compression_mutation_candidates"].get_property(int_value);
        options.mutation_candidates = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_mutate_full_topology") > 0) {
        config["osr_compression_mutate_full_topology"].get_property(int_value);
        options.mutate_full_topology = int_value != 0;
    }
    if (config.count("osr_compression_enable_skeleton_search") > 0) {
        config["osr_compression_enable_skeleton_search"].get_property(int_value);
        options.enable_skeleton_search = int_value != 0;
    }
    if (config.count("osr_compression_skeleton_target_cnots") > 0) {
        config["osr_compression_skeleton_target_cnots"].get_property(int_value);
        options.skeleton_target_cnots = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_skeleton_max_candidates") > 0) {
        config["osr_compression_skeleton_max_candidates"].get_property(int_value);
        options.skeleton_max_candidates = static_cast<int>(int_value);
    }

    options.beam_width = std::max(options.beam_width, 1);
    options.validation_trials = std::max(options.validation_trials, 1);
    options.osr_bound_limit = std::max(options.osr_bound_limit, 0);
    options.mutation_rounds = std::max(options.mutation_rounds, 0);
    options.mutation_candidates = std::max(options.mutation_candidates, 0);
    options.skeleton_max_candidates = std::max(options.skeleton_max_candidates, 0);

    return options;
}

N_Qubit_Decomposition_custom
N_Qubit_Decomposition_OSR_Compression::prepare_custom_optimizer(
    Gates_block* gate_structure_in,
    cost_function_type cost_function_variant) {
    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    N_Qubit_Decomposition_custom cDecomp_custom_random =
        N_Qubit_Decomposition_custom(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    cDecomp_custom_random.set_custom_gate_structure(gate_structure_in);
    cDecomp_custom_random.set_optimization_blocks(gate_structure_in->get_gate_num());
    cDecomp_custom_random.set_max_iteration(max_outer_iterations);
#ifndef __DFE__
    cDecomp_custom_random.set_verbose(verbose);
#else
    cDecomp_custom_random.set_verbose(0);
#endif
    cDecomp_custom_random.set_cost_function_variant(cost_function_variant);
    cDecomp_custom_random.set_debugfile("");
    cDecomp_custom_random.set_optimization_tolerance(optimization_tolerance_loc);
    cDecomp_custom_random.set_trace_offset(trace_offset);
    cDecomp_custom_random.set_optimizer(alg);

    if (alg == ADAM || alg == BFGS2) {
        int max_inner_iterations_loc = 10000;
        int param_num_loc = gate_structure_in->get_parameter_num();
        max_inner_iterations_loc = static_cast<int>((double)param_num_loc / 852 * 10000000.0);
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
        cDecomp_custom_random.set_random_shift_count_max(5);
    } else if (alg == ADAM_BATCHED) {
        int max_inner_iterations_loc = 2000;
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
        cDecomp_custom_random.set_random_shift_count_max(5);
    } else if (alg == BFGS) {
        int max_inner_iterations_loc = 10000;
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
    }

    return cDecomp_custom_random;
}

N_Qubit_Decomposition_OSR_Compression_Score
N_Qubit_Decomposition_OSR_Compression::evaluate_gate_structure_osr(
    Gates_block* gate_structure_in,
    const Matrix_real& initial_parameters,
    MinCnotBoundSolver& osr_bound_solver,
    std::vector<std::vector<int>>& all_cuts) {
    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();
    N_Qubit_Decomposition_OSR_Compression_Score best_score;
    best_score.min_remaining_cnots = std::numeric_limits<int>::max();
    best_score.kappa = std::numeric_limits<double>::infinity();
    best_score.residual = std::numeric_limits<double>::infinity();

    if (qbit_num <= 1 || all_cuts.empty()) {
        best_score.min_remaining_cnots = 0;
        best_score.kappa = 0.0;
        best_score.residual = 0.0;
        return best_score;
    }

    double Fnorm = std::sqrt(static_cast<double>(1 << qbit_num));
    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);

    N_Qubit_Decomposition_custom cDecomp_custom_random =
        prepare_custom_optimizer(gate_structure_in, OSR_ENTANGLEMENT);
    std::vector<double> optimized_parameters(cDecomp_custom_random.get_parameter_num());
    if (initial_parameters.size() == cDecomp_custom_random.get_parameter_num()) {
        std::copy(initial_parameters.get_data(),
                  initial_parameters.get_data() + initial_parameters.size(),
                  optimized_parameters.begin());
    } else if (optimized_parameters_mtx.size() == cDecomp_custom_random.get_parameter_num()) {
        std::copy(optimized_parameters_mtx.get_data(),
                  optimized_parameters_mtx.get_data() + optimized_parameters_mtx.size(),
                  optimized_parameters.begin());
    } else {
        for (size_t idx = 0; idx < optimized_parameters.size(); ++idx) {
            optimized_parameters[idx] = distrib_real(gen);
        }
    }
    if (!optimized_parameters.empty()) {
        cDecomp_custom_random.set_optimized_parameters(
            optimized_parameters.data(), static_cast<int>(optimized_parameters.size()));
    }

    for (const std::vector<int>& cut : all_cuts) {
        if (cut.size() != 1) {
            continue;
        }

        int cut_size = static_cast<int>(cut.size());
        int max_rank = 2 * std::min(cut_size, qbit_num - cut_size);
        max_rank = std::max(max_rank, 1);

        for (int rank = max_rank - 1; rank >= 0; --rank) {
            cDecomp_custom_random.set_osr_params({cut}, rank, false);
            cDecomp_custom_random.start_decomposition();

            Matrix U = Umtx.copy();
            Matrix_real params = cDecomp_custom_random.get_optimized_parameters();
            cDecomp_custom_random.apply_to(params, U);

            std::vector<std::pair<int, double>> osr_result;
            osr_result.reserve(all_cuts.size());
            int newrank = rank;
            for (const std::vector<int>& eval_cut : all_cuts) {
                osr_result.emplace_back(
                    operator_schmidt_rank(U, qbit_num, eval_cut, Fnorm, options.osr_tolerance));
                if (cut == eval_cut) {
                    newrank = osr_result.back().first;
                }
            }

            double kappa = std::numeric_limits<double>::infinity();
            std::vector<int> edge_counts;
            int min_cnots = osr_bound_solver.solve_min_cnots(osr_result, kappa, edge_counts);

            N_Qubit_Decomposition_OSR_Compression_Score score;
            score.min_remaining_cnots = min_cnots;
            score.kappa = kappa;
            score.residual = residual_sum(osr_result);
            score.edge_counts = edge_counts;
            score.cut_bounds = osr_result;

            if (score_less(score, best_score)) {
                best_score = score;
            }

            if (newrank > rank) {
                break;
            }
            rank = std::min(rank, newrank);
        }
    }

    if (best_score.min_remaining_cnots == std::numeric_limits<int>::max()) {
        std::vector<std::pair<int, double>> osr_result(all_cuts.size(), std::make_pair(0, 0.0));
        double kappa = std::numeric_limits<double>::infinity();
        std::vector<int> edge_counts;
        best_score.min_remaining_cnots = osr_bound_solver.solve_min_cnots(osr_result, kappa, edge_counts);
        best_score.kappa = kappa;
        best_score.residual = 0.0;
        best_score.edge_counts = edge_counts;
        best_score.cut_bounds = osr_result;
    }

    return best_score;
}

void N_Qubit_Decomposition_OSR_Compression::validate_compressed_gate_structure(
    Gates_block* gate_structure_in,
    const Matrix_real& initial_parameters,
    N_Qubit_Decomposition_OSR_Compression_Result& result) {
    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    result.validated = true;
    result.current_minimum = std::numeric_limits<double>::infinity();
    result.decomposition_error = std::numeric_limits<double>::infinity();

    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);
    for (int iter = 0; iter < options.validation_trials; ++iter) {
        N_Qubit_Decomposition_custom cDecomp_custom_random =
            prepare_custom_optimizer(gate_structure_in, cost_fnc);

        std::vector<double> optimized_parameters(cDecomp_custom_random.get_parameter_num());
        if (iter == 0 && initial_parameters.size() == cDecomp_custom_random.get_parameter_num()) {
            std::copy(initial_parameters.get_data(),
                      initial_parameters.get_data() + initial_parameters.size(),
                      optimized_parameters.begin());
        } else if (iter == 0 && optimized_parameters_mtx.size() == cDecomp_custom_random.get_parameter_num()) {
            std::copy(optimized_parameters_mtx.get_data(),
                      optimized_parameters_mtx.get_data() + optimized_parameters_mtx.size(),
                      optimized_parameters.begin());
        } else {
            for (size_t idx = 0; idx < optimized_parameters.size(); ++idx) {
                optimized_parameters[idx] = distrib_real(gen);
            }
        }
        if (!optimized_parameters.empty()) {
            cDecomp_custom_random.set_optimized_parameters(
                optimized_parameters.data(), static_cast<int>(optimized_parameters.size()));
        }

        cDecomp_custom_random.start_decomposition();
        Matrix_real optimized_parameters_tmp = cDecomp_custom_random.get_optimized_parameters();
        double current_minimum_tmp = cDecomp_custom_random.optimization_problem(optimized_parameters_tmp);
        if (current_minimum_tmp < result.current_minimum) {
            result.current_minimum = current_minimum_tmp;
            result.optimized_parameters = optimized_parameters_tmp.copy();
            result.decomposition_error = cDecomp_custom_random.get_decomposition_error();
        }
        if (current_minimum_tmp < optimization_tolerance_loc &&
            cDecomp_custom_random.get_decomposition_error() < optimization_tolerance_loc) {
            result.reached_tolerance = true;
            break;
        }
    }
}

N_Qubit_Decomposition_OSR_Compression_Result
N_Qubit_Decomposition_OSR_Compression::compress_gate_structure(
    Gates_block* gate_structure_in) {
    if (gate_structure_in == NULL) {
        std::string err("N_Qubit_Decomposition_OSR_Compression::compress_gate_structure: gate_structure is null");
        throw err;
    }

    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();
    std::vector<OSRGatePath> removable_paths = collect_entangling_gate_paths(gate_structure_in);

    std::vector<std::vector<int>> all_cuts = unique_cuts(qbit_num);
    std::sort(all_cuts.begin(), all_cuts.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    });

    std::vector<matrix_base<int>> active_topology = !this->topology.empty()
        ? this->topology
        : topology_from_gate_structure(gate_structure_in, qbit_num);
    std::vector<std::pair<int, int>> mutation_edges = options.mutate_full_topology
        ? complete_topology_pairs(qbit_num)
        : topology_pairs_from_matrices(active_topology);
    MinCnotBoundSolver osr_bound_solver(qbit_num, all_cuts, active_topology);

    CompressionCandidate root;
    root.entangling_gate_num = static_cast<int>(removable_paths.size());
    root.key = gate_structure_signature(gate_structure_in);
    if (optimized_parameters_mtx.size() == gate_structure_in->get_parameter_num()) {
        root.initial_parameters = optimized_parameters_mtx.copy();
    }
    root.score = evaluate_gate_structure_osr(
        gate_structure_in, root.initial_parameters, osr_bound_solver, all_cuts);

    CompressionCandidate best = root;
    std::vector<CompressionCandidate> beam(1, root);

    int max_removed = options.max_removed_gates < 0
        ? static_cast<int>(removable_paths.size())
        : std::min(options.max_removed_gates, static_cast<int>(removable_paths.size()));

    for (int depth = 1; depth <= max_removed; ++depth) {
        std::vector<CompressionCandidate> next_candidates;

        for (size_t beam_idx = 0; beam_idx < beam.size(); ++beam_idx) {
            const CompressionCandidate& parent = beam[beam_idx];
            int start_id = parent.removed_ids.empty() ? 0 : parent.removed_ids.back() + 1;

            for (int remove_id = start_id; remove_id < static_cast<int>(removable_paths.size()); ++remove_id) {
                CompressionCandidate child;
                child.removed_ids = parent.removed_ids;
                child.removed_ids.push_back(remove_id);
                child.entangling_gate_num =
                    static_cast<int>(removable_paths.size()) - static_cast<int>(child.removed_ids.size());

                std::unique_ptr<Gates_block> candidate_gate_structure(
                    clone_without_removed_paths(gate_structure_in, removable_paths, child.removed_ids));
                child.initial_parameters = reduced_parameters_without_removed_paths(
                    gate_structure_in, removable_paths, child.removed_ids, optimized_parameters_mtx);
                child.key = gate_structure_signature(candidate_gate_structure.get());
                child.score = evaluate_gate_structure_osr(
                    candidate_gate_structure.get(), child.initial_parameters, osr_bound_solver, all_cuts);

                if (candidate_is_osr_admissible(child, options)) {
                    next_candidates.push_back(child);
                }
            }
        }

        if (next_candidates.empty()) {
            break;
        }

        sort_unique_candidates(next_candidates, false);
        if (static_cast<int>(next_candidates.size()) > options.beam_width) {
            next_candidates.resize(options.beam_width);
        }

        for (size_t idx = 0; idx < next_candidates.size(); ++idx) {
            if (candidate_is_osr_admissible(next_candidates[idx], options) &&
                final_candidate_less(next_candidates[idx], best)) {
                best = next_candidates[idx];
            }
        }

        beam.swap(next_candidates);
    }

    std::vector<CompressionCandidate> validation_pool = beam;
    validation_pool.push_back(root);
    validation_pool.push_back(best);

    if (options.enable_mutations && options.mutation_rounds > 0 &&
        options.mutation_candidates > 0 && !mutation_edges.empty()) {
        std::vector<CompressionCandidate> mutation_seeds = validation_pool;
        sort_unique_candidates(mutation_seeds, true);
        if (static_cast<int>(mutation_seeds.size()) > options.beam_width) {
            mutation_seeds.resize(options.beam_width);
        }

        for (int round = 0; round < options.mutation_rounds; ++round) {
            std::vector<CompressionCandidate> round_mutations;

            for (size_t seed_idx = 0; seed_idx < mutation_seeds.size(); ++seed_idx) {
                const CompressionCandidate& seed = mutation_seeds[seed_idx];
                std::unique_ptr<Gates_block> seed_gate_structure(
                    clone_gate_structure_for_candidate(
                        gate_structure_in, removable_paths, seed));

                std::vector<CompressionCandidate> local_mutations =
                    generate_local_mutation_candidates(
                        seed_gate_structure.get(), seed.initial_parameters, seed,
                        mutation_edges, options);

                for (size_t mut_idx = 0; mut_idx < local_mutations.size(); ++mut_idx) {
                    CompressionCandidate& mutation = local_mutations[mut_idx];
                    mutation.score = evaluate_gate_structure_osr(
                        mutation.gate_structure.get(), mutation.initial_parameters,
                        osr_bound_solver, all_cuts);
                    if (candidate_is_osr_admissible(mutation, options)) {
                        round_mutations.push_back(mutation);
                    }
                }
            }

            if (round_mutations.empty()) {
                break;
            }

            sort_unique_candidates(round_mutations, false);
            if (static_cast<int>(round_mutations.size()) > options.mutation_candidates) {
                round_mutations.resize(options.mutation_candidates);
            }

            validation_pool.insert(
                validation_pool.end(), round_mutations.begin(), round_mutations.end());
            mutation_seeds.swap(round_mutations);
        }
    }

    if (options.enable_skeleton_search) {
        std::vector<std::pair<int, int>> skeleton_edges =
            (options.mutate_full_topology || qbit_num <= 3)
                ? complete_topology_pairs(qbit_num)
                : mutation_edges;
        std::vector<CompressionCandidate> skeleton_candidates =
            generate_cnot_skeleton_candidates(
                qbit_num, static_cast<int>(removable_paths.size()),
                skeleton_edges, options);
        validation_pool.insert(
            validation_pool.end(), skeleton_candidates.begin(), skeleton_candidates.end());
    }

    sort_unique_candidates(validation_pool, true);
    int validation_pool_limit = options.beam_width;
    if (options.enable_mutations) {
        validation_pool_limit += options.mutation_candidates;
    }
    if (options.enable_skeleton_search) {
        validation_pool_limit += options.skeleton_max_candidates;
    }
    validation_pool_limit = std::max(validation_pool_limit, options.beam_width);
    if (static_cast<int>(validation_pool.size()) > validation_pool_limit) {
        validation_pool.resize(validation_pool_limit);
    }
    bool has_root_candidate = false;
    for (size_t idx = 0; idx < validation_pool.size(); ++idx) {
        if (compression_candidate_key(validation_pool[idx]) == root.key) {
            has_root_candidate = true;
            break;
        }
    }
    if (!has_root_candidate) {
        validation_pool.push_back(root);
    }

    N_Qubit_Decomposition_OSR_Compression_Result result;
    result.original_entangling_gate_num = static_cast<int>(removable_paths.size());

    if (!options.validate_final) {
        result.gate_structure.reset(
            clone_gate_structure_for_candidate(gate_structure_in, removable_paths, best));
        result.osr_score = best.score;
        for (size_t idx = 0; idx < best.removed_ids.size(); ++idx) {
            result.removed_gate_paths.push_back(removable_paths[best.removed_ids[idx]]);
        }
        result.compressed_entangling_gate_num = best.entangling_gate_num;
        return result;
    }

    bool selected_validated_candidate = false;
    N_Qubit_Decomposition_OSR_Compression_Result best_validated_result;

    best_validated_result.gate_structure.reset(
        clone_gate_structure_for_candidate(gate_structure_in, removable_paths, root));
    best_validated_result.osr_score = root.score;
    best_validated_result.original_entangling_gate_num = static_cast<int>(removable_paths.size());
    best_validated_result.compressed_entangling_gate_num = root.entangling_gate_num;
    validate_compressed_gate_structure(
        best_validated_result.gate_structure.get(), root.initial_parameters, best_validated_result);
    selected_validated_candidate = true;

    for (size_t idx = 0; idx < validation_pool.size(); ++idx) {
        const CompressionCandidate& candidate = validation_pool[idx];
        if (compression_candidate_key(candidate) == root.key) {
            continue;
        }
        N_Qubit_Decomposition_OSR_Compression_Result candidate_result;
        candidate_result.gate_structure.reset(
            clone_gate_structure_for_candidate(gate_structure_in, removable_paths, candidate));
        candidate_result.osr_score = candidate.score;
        candidate_result.original_entangling_gate_num = static_cast<int>(removable_paths.size());
        candidate_result.compressed_entangling_gate_num = candidate.entangling_gate_num;
        for (size_t removed_idx = 0; removed_idx < candidate.removed_ids.size(); ++removed_idx) {
            candidate_result.removed_gate_paths.push_back(removable_paths[candidate.removed_ids[removed_idx]]);
        }

        validate_compressed_gate_structure(
            candidate_result.gate_structure.get(), candidate.initial_parameters, candidate_result);

        if ((candidate_result.reached_tolerance && !best_validated_result.reached_tolerance) ||
            (candidate_result.reached_tolerance && best_validated_result.reached_tolerance &&
             candidate_result.compressed_entangling_gate_num < best_validated_result.compressed_entangling_gate_num) ||
            (candidate_result.reached_tolerance && best_validated_result.reached_tolerance &&
             candidate_result.compressed_entangling_gate_num == best_validated_result.compressed_entangling_gate_num &&
             candidate_result.current_minimum < best_validated_result.current_minimum) ||
            (!candidate_result.reached_tolerance && !best_validated_result.reached_tolerance &&
             candidate_result.current_minimum < best_validated_result.current_minimum)) {
            best_validated_result = std::move(candidate_result);
        }

        if (best_validated_result.reached_tolerance &&
            best_validated_result.compressed_entangling_gate_num == validation_pool.front().entangling_gate_num) {
            break;
        }
    }

    if (selected_validated_candidate) {
        return best_validated_result;
    }

    result.gate_structure.reset(
        clone_gate_structure_for_candidate(gate_structure_in, removable_paths, best));
    result.osr_score = best.score;
    for (size_t idx = 0; idx < best.removed_ids.size(); ++idx) {
        result.removed_gate_paths.push_back(removable_paths[best.removed_ids[idx]]);
    }
    result.compressed_entangling_gate_num = best.entangling_gate_num;
    return result;
}
