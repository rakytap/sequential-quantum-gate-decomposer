#pragma once
/*
Copyright 2025 SQUANDER Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

C++ backend for the SABRE-style partition-aware routing engine.
Ported from squander/synthesis/PartAM.py and PartAM_utils.py.
*/

#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace squander::routing {

// ---------------------------------------------------------------------------
// Data structures (flattened from Python objects)
// ---------------------------------------------------------------------------

struct Edge {
    int u, v;
};

struct CandidateData {
    int partition_idx;
    int topology_idx;
    int permutation_idx;
    int candidate_idx = -1;
    int cnot_count;

    // Permutations within the reduced (q*) space
    // P_i[v] = position in Q* space for input routing
    // P_o[v] = position in Q* space for output placement
    std::vector<int> P_i;
    std::vector<int> P_o;

    // node_mapping_flat[Q*_idx] = Q (physical qubit)
    // Dense array indexed by Q* index
    std::vector<int> node_mapping_flat;

    // qbit_map: original circuit qubit q -> reduced qubit q*
    std::vector<int> qbit_map_keys;
    std::vector<int> qbit_map_vals;

    // Original circuit qubits involved in this partition
    std::vector<int> involved_qbits;

    // Precomputed routing helpers.
    std::vector<int> P_i_inv;
    std::vector<int> P_o_inv;
    std::vector<int> qbit_map_keys_sorted;
    std::vector<int> qbit_map_vals_sorted;
    std::vector<int> qstar_to_q;
};

struct CanonicalEntry {
    std::vector<int> edges_u; // virtual qubit indices
    std::vector<int> edges_v;
    int cnot = 0;
};

struct LayoutPartInfo {
    bool is_single;
    std::vector<int> involved_qbits;
};

struct SabreConfig {
    int prefilter_top_k = 50;
    int max_E_size = 20;
    int max_lookahead = 4;
    double E_weight = 0.5;
    double E_alpha = 1.0; // LightSABRE uses no per-depth decay; set <1 for SQUANDER-style decay
    double cnot_cost = 1.0 / 3.0; // weight on candidate.cnot_count; swap cost is fixed at 1.0 (1 SWAP = 3 CNOTs)
    int sabre_iterations = 1;
    int n_layout_trials = 1;
    int random_seed = 42;
    double decay_delta = 0.001; // Qiskit LightSABRE DECAY_RATE
    int swap_burst_budget = 5; // Qiskit LightSABRE DECAY_RESET_INTERVAL
    double path_tiebreak_weight = 0.2;
};

struct RouteStep {
    int type = 0; // 0=swap, 1=partition, 2=single
    int partition_idx = -1;
    int candidate_idx = -1;
    int physical_qubit = -1;
    std::vector<std::pair<int,int>> swaps;
};

struct ForwardRouteResult {
    std::vector<int> pi_initial;
    std::vector<int> pi;
    int cnot_count = 0;
    std::vector<RouteStep> steps;
};

struct TrialResult {
    std::vector<int> pi;
    double total_cost;
};

struct NeighborEdge {
    int u_idx;
    int v_idx;
    double weight;
};

struct NeighborInfo {
    std::vector<int> neighbor_vqs;
    std::vector<int> initial_pos;
    std::vector<NeighborEdge> edges;
    double weight = 0.0;

    bool uses_tiebreak() const {
        return weight > 0.0 && !edges.empty();
    }
};

// ---------------------------------------------------------------------------
// Swap cache key for deduplication within a single heuristic_search call
// ---------------------------------------------------------------------------

struct SwapCacheKey {
    int64_t pi_snapshot;
    int64_t targets;
    int k;
    // 0 when the neighbor tiebreak is inactive; otherwise a stable hash of
    // (edges, initial_pos, weight) from NeighborInfo so that two calls with
    // the same active future context share cache entries.
    uint64_t neighbor_hash;

    bool operator==(const SwapCacheKey& o) const {
        return pi_snapshot == o.pi_snapshot && targets == o.targets
            && k == o.k && neighbor_hash == o.neighbor_hash;
    }
};

struct SwapCacheKeyHash {
    size_t operator()(const SwapCacheKey& k) const {
        size_t h = static_cast<size_t>(k.pi_snapshot);
        h ^= static_cast<size_t>(k.targets) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.neighbor_hash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

using SwapList = std::vector<std::pair<int,int>>;
using SwapCache = std::unordered_map<SwapCacheKey, SwapList, SwapCacheKeyHash>;

// ---------------------------------------------------------------------------
// A* state packing helpers
// ---------------------------------------------------------------------------

// For k <= 4 partition qubits on N <= 64 physical qubits, pack state into int64_t
// State = sum(positions[i] * N^i), fits in 64 bits when N <= 64 and k <= 4
inline int64_t pack_state(const std::vector<int>& positions, int N) {
    int64_t s = 0;
    int64_t stride = 1;
    for (size_t i = 0; i < positions.size(); i++) {
        s += static_cast<int64_t>(positions[i]) * stride;
        stride *= N;
    }
    return s;
}

inline std::vector<int> unpack_state(int64_t packed, int k, int N) {
    std::vector<int> positions(k);
    for (int i = 0; i < k; i++) {
        positions[i] = static_cast<int>(packed % N);
        packed /= N;
    }
    return positions;
}

// ---------------------------------------------------------------------------
// SabreRouter class
// ---------------------------------------------------------------------------

class SabreRouter {
public:
    SabreRouter(
        const SabreConfig& config,
        int N,
        std::vector<double> D,
        std::vector<std::vector<int>> adj,
        std::vector<std::vector<int>> DAG,
        std::vector<std::vector<int>> IDAG,
        std::vector<std::vector<CandidateData>> candidate_cache,
        std::vector<LayoutPartInfo> layout_partitions,
        std::unordered_map<int, CanonicalEntry> canonical_data_fwd,
        std::unordered_map<int, CanonicalEntry> canonical_data_rev
    );

    // Thread-safe: all mutable state is stack-local
    ForwardRouteResult route_forward(
        const std::vector<int>& pi
    ) const;

    TrialResult run_trial(
        int trial_idx,
        const std::vector<int>& seeded_pi,
        int n_iterations,
        int n_trials
    ) const;

private:
    // Distance lookup (flat row-major)
    inline double dist(int phys_u, int phys_v) const {
        return D_[phys_u * N_ + phys_v];
    }

    // Heuristic search (port of _heuristic_search_layout_only)
    // children_graph/parents_graph: swapped for backward passes
    std::pair<std::vector<int>, double> heuristic_search(
        const std::vector<int>& F_init,
        std::vector<int> pi,
        bool reverse,
        std::mt19937* rng,
        const std::unordered_map<int, CanonicalEntry>& canonical_data,
        const std::vector<std::vector<int>>& children_graph,
        const std::vector<std::vector<int>>& parents_graph,
        ForwardRouteResult* route_trace = nullptr
    ) const;

    // A* constrained swap search (port of find_constrained_swaps_partial)
    std::pair<std::vector<std::pair<int,int>>, std::vector<int>>
    find_constrained_swaps(
        const std::vector<int>& pi,
        const std::vector<int>& qbit_map_keys,
        const std::vector<int>& qbit_map_vals,
        const std::vector<int>& node_mapping_flat,
        const std::vector<int>& P_route_inv,
        SwapCache* swap_cache,
        const NeighborInfo* neighbor_info = nullptr
    ) const;

    // Lower-bound swap estimate (port of estimate_swap_count)
    int estimate_swap_count(
        const CandidateData& cand,
        const std::vector<int>& pi,
        bool reverse
    ) const;

    // BFS lookahead (port of generate_extended_set)
    std::vector<std::pair<int,int>> generate_extended_set(
        const std::vector<int>& F,
        const std::vector<uint8_t>& resolved,
        const std::vector<std::vector<int>>& children_graph,
        const std::vector<std::vector<int>>& parents_graph
    ) const;

    // Pre-resolved canonical entries for an F-step (avoids hash lookups per candidate)
    struct ResolvedEntry {
        int partition_idx;
        const CanonicalEntry* entry; // may be null
        double alpha; // 1.0 for F, alpha^depth for E
    };

    // LightSABRE scoring (port of score_partition_candidate)
    double score_candidate(
        const CandidateData& cand,
        const std::vector<int>& F_snapshot,
        const std::vector<int>& pi,
        const std::vector<std::pair<int,int>>& E,
        bool reverse,
        const std::unordered_map<int, CanonicalEntry>& canonical_data,
        SwapCache* swap_cache,
        const std::vector<double>* decay = nullptr,
        std::vector<std::pair<int,int>>* out_swaps = nullptr,
        std::vector<int>* out_pi_new = nullptr,
        const std::vector<ResolvedEntry>* resolved_F = nullptr,
        const std::vector<ResolvedEntry>* resolved_E = nullptr,
        const NeighborInfo* cached_neighbor_info = nullptr
    ) const;

    // Route and update layout for a candidate (port of transform_pi)
    std::pair<std::vector<std::pair<int,int>>, std::vector<int>>
    transform_pi(
        const CandidateData& cand,
        const std::vector<int>& pi,
        bool reverse,
        SwapCache* swap_cache,
        const NeighborInfo* neighbor_info = nullptr
    ) const;

    NeighborInfo build_neighbor_info(
        int exclude_partition_idx,
        const std::vector<int>& F_snapshot,
        const std::vector<std::pair<int,int>>& E,
        const std::vector<int>& pi,
        const std::unordered_map<int, CanonicalEntry>& canonical_data
    ) const;

    double decay_factor_for_swaps(
        const std::vector<std::pair<int,int>>& swaps,
        const std::vector<double>& decay
    ) const;

    double routing_objective(
        double route_cost,
        int cnot_count,
        double cnot_weight = 1.0,
        double decay_factor = 1.0
    ) const;

    void apply_decay_for_swaps(
        const std::vector<std::pair<int,int>>& swaps,
        std::vector<double>& decay
    ) const;

    void reset_decay(std::vector<double>& decay) const;

    std::vector<int> bfs_shortest_path(int src, int dst) const;

    std::pair<std::vector<std::pair<int,int>>, std::vector<int>> release_valve(
        const std::vector<int>& F,
        const std::vector<int>& pi,
        const std::unordered_map<int, CanonicalEntry>& canonical_data
    ) const;

    // Apply a list of SWAPs to pi
    std::vector<int> apply_swaps_to_pi(
        const std::vector<int>& pi,
        const std::vector<std::pair<int,int>>& swaps
    ) const;

    // Get initial layer (partitions with no unresolved parents)
    std::vector<int> get_initial_layer() const;

    // Get final layer (partitions with no children)
    std::vector<int> get_final_layer() const;

    // Prefilter candidates by cheap swap estimate
    std::vector<const CandidateData*> prefilter_candidates(
        const std::vector<const CandidateData*>& candidates,
        const std::vector<int>& pi,
        int top_k,
        const std::vector<int>& F_snapshot,
        const std::vector<std::pair<int,int>>& E,
        bool reverse,
        const std::unordered_map<int, CanonicalEntry>& canonical_data
    ) const;

    // Select best candidate with optional stochastic tie-breaking
    const CandidateData& select_best_candidate(
        const std::vector<const CandidateData*>& candidates,
        const std::vector<double>& scores,
        std::mt19937* rng
    ) const;

    // Check if partition is single-qubit
    inline bool partition_is_single(int partition_idx) const {
        return layout_partitions_[partition_idx].is_single;
    }

    // Gather all candidates for partitions in F
    std::vector<const CandidateData*> obtain_partition_candidates(
        const std::vector<int>& F
    ) const;

    // Random permutation of [0..N-1]
    std::vector<int> random_permutation(int n, std::mt19937& rng) const;

    // Apply a small random walk on topology edges to diversify a seeded layout.
    std::vector<int> perturb_layout(
        const std::vector<int>& base,
        int num_swaps,
        std::mt19937& rng
    ) const;

    // Stratified initial-layout sampling with the same total trial budget.
    std::vector<int> sample_initial_layout(
        int trial_idx,
        int n_trials,
        const std::vector<int>& seeded_pi,
        std::mt19937& rng
    ) const;

    // Build P_route_inv: the inverse permutation used for routing
    std::vector<int> build_route_inv(const std::vector<int>& P, bool reverse) const;

    // Build target dict for A*: {qbit_map_key -> node_mapping[P_route_inv[qbit_map_val]]}
    void build_target_positions(
        const CandidateData& cand,
        bool reverse,
        std::vector<int>& out_keys,
        std::vector<int>& out_targets
    ) const;

    // Compute routing cost for canonical edges under a given pi
    double compute_routing_cost(
        const std::vector<int>& pi,
        int exclude_partition_idx,
        const std::vector<int>& partition_indices,
        const std::unordered_map<int, CanonicalEntry>& canonical_data
    ) const;

    // Compute lookahead cost with alpha^depth decay
    double compute_lookahead_cost(
        const std::vector<int>& pi,
        int exclude_partition_idx,
        const std::vector<std::pair<int,int>>& E,
        const std::unordered_map<int, CanonicalEntry>& canonical_data
    ) const;

    double entry_future_cost(
        const CanonicalEntry& entry,
        const std::vector<int>& pi
    ) const;

    double future_context_cost(
        int exclude_partition_idx,
        const std::vector<int>& pi,
        const std::vector<int>& F_snapshot,
        const std::vector<std::pair<int,int>>& E,
        bool reverse,
        const std::unordered_map<int, CanonicalEntry>& canonical_data
    ) const;

    std::vector<int> estimate_candidate_output_layout(
        const CandidateData& cand,
        const std::vector<int>& pi,
        bool reverse
    ) const;

    // Immutable data members
    SabreConfig config_;
    int N_; // number of physical qubits
    int num_partitions_;
    std::vector<double> D_; // flat N*N distance matrix (owned copy)
    std::vector<std::vector<int>> adj_;
    // CSR view of adj_ for tight inner loops
    std::vector<int> adj_offsets_;
    std::vector<int> adj_flat_;
    std::vector<std::vector<int>> DAG_;
    std::vector<std::vector<int>> IDAG_;
    std::vector<std::vector<CandidateData>> candidate_cache_;
    std::vector<LayoutPartInfo> layout_partitions_;
    std::unordered_map<int, CanonicalEntry> canonical_data_fwd_;
    std::unordered_map<int, CanonicalEntry> canonical_data_rev_;
    std::vector<double> alpha_weights_;
    double max_finite_distance_ = 1.0;
};

} // namespace squander::routing
