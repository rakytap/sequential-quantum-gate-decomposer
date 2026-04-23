/*
Copyright 2025 SQUANDER Contributors

C++ backend for the SABRE-style partition-aware routing engine.
*/

#include "sabre_router.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace squander::routing {

namespace {

std::vector<int> invert_permutation(const std::vector<int>& P) {
    std::vector<int> inv(P.size());
    for (size_t i = 0; i < P.size(); i++) {
        inv[P[i]] = static_cast<int>(i);
    }
    return inv;
}

void prepare_candidate(CandidateData& cand) {
    cand.P_i_inv = invert_permutation(cand.P_i);
    cand.P_o_inv = invert_permutation(cand.P_o);

    const int k = static_cast<int>(cand.qbit_map_keys.size());
    std::vector<int> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return cand.qbit_map_keys[a] < cand.qbit_map_keys[b];
    });

    cand.qbit_map_keys_sorted.resize(k);
    cand.qbit_map_vals_sorted.resize(k);
    int max_qstar = -1;
    for (int i = 0; i < k; i++) {
        const int src_idx = order[i];
        const int qstar = cand.qbit_map_vals[src_idx];
        cand.qbit_map_keys_sorted[i] = cand.qbit_map_keys[src_idx];
        cand.qbit_map_vals_sorted[i] = qstar;
        if (qstar > max_qstar) max_qstar = qstar;
    }

    const int dense_size = std::max(
        {max_qstar + 1,
         static_cast<int>(cand.P_i.size()),
         static_cast<int>(cand.P_o.size()),
         static_cast<int>(cand.node_mapping_flat.size())}
    );
    cand.qstar_to_q.assign(dense_size, -1);
    for (size_t i = 0; i < cand.qbit_map_keys.size(); i++) {
        const int qstar = cand.qbit_map_vals[i];
        if (qstar >= 0) {
            if (qstar >= static_cast<int>(cand.qstar_to_q.size())) {
                cand.qstar_to_q.resize(qstar + 1, -1);
            }
            cand.qstar_to_q[qstar] = cand.qbit_map_keys[i];
        }
    }
}

inline void unpack_state_into(int64_t packed, int k, int N, std::vector<int>& positions) {
    positions.resize(k);
    for (int i = 0; i < k; i++) {
        positions[i] = static_cast<int>(packed % N);
        packed /= N;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SabreRouter::SabreRouter(
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
)
    : config_(config)
    , N_(N)
    , num_partitions_(static_cast<int>(DAG.size()))
    , D_(std::move(D))
    , adj_(std::move(adj))
    , DAG_(std::move(DAG))
    , IDAG_(std::move(IDAG))
    , candidate_cache_(std::move(candidate_cache))
    , layout_partitions_(std::move(layout_partitions))
    , canonical_data_fwd_(std::move(canonical_data_fwd))
    , canonical_data_rev_(std::move(canonical_data_rev))
{
    if (static_cast<int>(D_.size()) != N_ * N_) {
        throw std::invalid_argument("Distance matrix D must be N x N");
    }
    for (auto& partition_candidates : candidate_cache_) {
        for (auto& cand : partition_candidates) {
            prepare_candidate(cand);
        }
    }

    const int max_depth = std::max(0, config_.max_lookahead);
    alpha_weights_.resize(max_depth + 1);
    if (!alpha_weights_.empty()) {
        alpha_weights_[0] = 1.0;
        for (int depth = 1; depth <= max_depth; depth++) {
            alpha_weights_[depth] = alpha_weights_[depth - 1] * config_.E_alpha;
        }
    }
}

// ---------------------------------------------------------------------------
// run_trial (stub for Phase A)
// ---------------------------------------------------------------------------

// run_trial implemented below (after all private methods)

// ---------------------------------------------------------------------------
// Helper: random permutation
// ---------------------------------------------------------------------------

std::vector<int> SabreRouter::random_permutation(int n, std::mt19937& rng) const {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
}

// ---------------------------------------------------------------------------
// Helper: build P_route_inv
// ---------------------------------------------------------------------------

std::vector<int> SabreRouter::build_route_inv(const std::vector<int>& P, bool /*reverse*/) const {
    // P_route_inv[i] = index of i in P (inverse permutation)
    int k = static_cast<int>(P.size());
    std::vector<int> inv(k);
    for (int i = 0; i < k; i++) {
        inv[P[i]] = i;
    }
    return inv;
}

// ---------------------------------------------------------------------------
// Helper: build target positions for A*
// ---------------------------------------------------------------------------

void SabreRouter::build_target_positions(
    const CandidateData& cand,
    bool reverse,
    std::vector<int>& out_keys,
    std::vector<int>& out_targets
) const {
    const std::vector<int>& P_route_inv = reverse ? cand.P_o_inv : cand.P_i_inv;

    out_keys = cand.qbit_map_keys;
    out_targets.resize(cand.qbit_map_keys.size());
    for (size_t i = 0; i < cand.qbit_map_keys.size(); i++) {
        int v = cand.qbit_map_vals[i];
        out_targets[i] = cand.node_mapping_flat[P_route_inv[v]];
    }
}

// ---------------------------------------------------------------------------
// BFS shortest path
// ---------------------------------------------------------------------------

std::vector<int> SabreRouter::bfs_shortest_path(int src, int dst) const {
    if (src == dst) return {src};

    std::vector<int> parent(N_, -1);
    std::vector<uint8_t> visited(N_, 0);
    std::deque<int> queue;
    queue.push_back(src);
    visited[src] = 1;

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();
        for (int nb : adj_[node]) {
            if (!visited[nb]) {
                visited[nb] = 1;
                parent[nb] = node;
                if (nb == dst) {
                    // Reconstruct path
                    std::vector<int> path;
                    int cur = dst;
                    while (cur != src) {
                        path.push_back(cur);
                        cur = parent[cur];
                    }
                    path.push_back(src);
                    std::reverse(path.begin(), path.end());
                    return path;
                }
                queue.push_back(nb);
            }
        }
    }
    return {}; // unreachable
}

// ---------------------------------------------------------------------------
// apply_swaps_to_pi
// ---------------------------------------------------------------------------

std::vector<int> SabreRouter::apply_swaps_to_pi(
    const std::vector<int>& pi,
    const std::vector<std::pair<int,int>>& swaps
) const {
    std::vector<int> result(pi);
    std::vector<int> p2v(N_);
    for (int q = 0; q < N_; q++) p2v[result[q]] = q;

    for (auto [P1, P2] : swaps) {
        int q1 = p2v[P1];
        int q2 = p2v[P2];
        p2v[P1] = q2;
        p2v[P2] = q1;
        result[q1] = P2;
        result[q2] = P1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// get_initial_layer / get_final_layer
// ---------------------------------------------------------------------------

std::vector<int> SabreRouter::get_initial_layer() const {
    std::vector<int> layer;
    std::vector<uint8_t> covered(N_, 0);
    int uncovered = N_;
    for (int p = 0; p < num_partitions_ && uncovered > 0; p++) {
        if (IDAG_[p].empty()) {
            layer.push_back(p);
            for (int q : layout_partitions_[p].involved_qbits) {
                if (q < N_ && !covered[q]) {
                    covered[q] = 1;
                    uncovered--;
                }
            }
        }
    }
    return layer;
}

std::vector<int> SabreRouter::get_final_layer() const {
    std::vector<int> layer;
    std::vector<uint8_t> covered(N_, 0);
    int uncovered = N_;
    for (int p = num_partitions_ - 1; p >= 0 && uncovered > 0; p--) {
        if (DAG_[p].empty()) {
            layer.push_back(p);
            for (int q : layout_partitions_[p].involved_qbits) {
                if (q < N_ && !covered[q]) {
                    covered[q] = 1;
                    uncovered--;
                }
            }
        }
    }
    return layer;
}

// ---------------------------------------------------------------------------
// estimate_swap_count
// ---------------------------------------------------------------------------

int SabreRouter::estimate_swap_count(
    const CandidateData& cand,
    const std::vector<int>& pi,
    bool reverse
) const {
    const std::vector<int>& P_route_inv = reverse ? cand.P_o_inv : cand.P_i_inv;

    double total = 0.0;
    for (size_t i = 0; i < cand.qbit_map_keys.size(); i++) {
        int k = cand.qbit_map_keys[i];
        int v = cand.qbit_map_vals[i];
        int target_P = cand.node_mapping_flat[P_route_inv[v]];
        int current_P = pi[k];
        double d = dist(current_P, target_P);
        if (d < std::numeric_limits<double>::infinity()) {
            total += d;
        }
    }
    return static_cast<int>(total / 2.0);
}

// ---------------------------------------------------------------------------
// find_constrained_swaps (A* over k-dimensional state space)
// Port of find_constrained_swaps_partial from PartAM_utils.py
// ---------------------------------------------------------------------------

std::pair<std::vector<std::pair<int,int>>, std::vector<int>>
SabreRouter::find_constrained_swaps(
    const std::vector<int>& pi,
    const std::vector<int>& qbit_map_keys,
    const std::vector<int>& qbit_map_vals,
    const std::vector<int>& node_mapping_flat,
    const std::vector<int>& P_route_inv,
    SwapCache* swap_cache
) const {
    int k = static_cast<int>(qbit_map_keys.size());
    std::vector<int> target_positions(k);
    std::vector<int> initial_positions(k);

    for (int i = 0; i < k; i++) {
        const int q = qbit_map_keys[i];
        const int v = qbit_map_vals[i];
        target_positions[i] = node_mapping_flat[P_route_inv[v]];
        initial_positions[i] = pi[q];
    }

    // Check if already at target
    bool already_there = true;
    for (int i = 0; i < k; i++) {
        if (initial_positions[i] != target_positions[i]) {
            already_there = false;
            break;
        }
    }
    if (already_there) {
        return {{}, pi};
    }

    int64_t initial_packed = pack_state(initial_positions, N_);
    int64_t target_packed = pack_state(target_positions, N_);
    const SwapCacheKey cache_key{initial_packed, target_packed, k};

    if (swap_cache) {
        auto it = swap_cache->find(cache_key);
        if (it != swap_cache->end()) {
            // Replay cached swaps on current pi
            auto result_pi = apply_swaps_to_pi(pi, it->second);
            return {it->second, result_pi};
        }
    }

    // A* search over k-dimensional state space
    // State: vector of physical positions for each partition qubit
    // Heuristic: sum(D[pos_i][target_i]) / 2

    // Compute initial heuristic
    double h0 = 0.0;
    for (int i = 0; i < k; i++) {
        h0 += dist(initial_positions[i], target_positions[i]);
    }
    h0 /= 2.0;

    // Priority queue: (f_score, g_score, counter, packed_state)
    // Counter provides FIFO tie-breaking, matching Python's counter variable
    using PQEntry = std::tuple<double, int, uint64_t, int64_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
    uint64_t counter = 0;

    // Visited: packed_state -> best g_score
    std::unordered_map<int64_t, int> visited;
    // Parent: packed_state -> (parent_packed_state, swap)
    std::unordered_map<int64_t, std::pair<int64_t, std::pair<int,int>>> parent;
    visited.reserve(256);
    parent.reserve(256);

    pq.push({h0, 0, counter++, initial_packed});
    visited[initial_packed] = 0;
    parent[initial_packed] = {-1, {-1, -1}};

    std::vector<int> positions;
    std::vector<int> new_positions(k);
    positions.reserve(k);

    while (!pq.empty()) {
        auto entry = pq.top();
        pq.pop();
        int g = std::get<1>(entry);
        int64_t packed = std::get<3>(entry);

        if (packed == target_packed) {
            // Reconstruct swap path
            std::vector<std::pair<int,int>> path;
            int64_t cur = packed;
            while (parent[cur].first != -1) {
                path.push_back(parent[cur].second);
                cur = parent[cur].first;
            }
            std::reverse(path.begin(), path.end());

            // Replay swaps on full pi
            auto result_pi = apply_swaps_to_pi(pi, path);

            // Store in cache
            if (swap_cache) {
                (*swap_cache)[cache_key] = path;
            }

            return {path, result_pi};
        }

        // Skip if we've found a better path to this state
        auto vis_it = visited.find(packed);
        if (vis_it != visited.end() && vis_it->second < g) {
            continue;
        }

        unpack_state_into(packed, k, N_, positions);

        // Try every SWAP that moves at least one partition qubit
        for (int i = 0; i < k; i++) {
            int p = positions[i];
            for (int nb : adj_[p]) {
                std::copy(positions.begin(), positions.end(), new_positions.begin());
                new_positions[i] = nb;
                // If neighbor also holds a partition qubit, swap it
                for (int j = 0; j < k; j++) {
                    if (positions[j] == nb) {
                        new_positions[j] = p;
                        break;
                    }
                }

                int64_t new_packed = pack_state(new_positions, N_);
                int new_g = g + 1;

                auto new_vis = visited.find(new_packed);
                if (new_vis != visited.end() && new_vis->second <= new_g) {
                    continue;
                }

                // Compute heuristic
                double h = 0.0;
                for (int j = 0; j < k; j++) {
                    h += dist(new_positions[j], target_positions[j]);
                }
                h /= 2.0;

                visited[new_packed] = new_g;
                parent[new_packed] = {packed, {std::min(p, nb), std::max(p, nb)}};
                pq.push({new_g + h, new_g, counter++, new_packed});
            }
        }
    }

    // Failed to route (should not happen on a connected graph)
    return {{}, pi};
}

// ---------------------------------------------------------------------------
// transform_pi
// ---------------------------------------------------------------------------

std::pair<std::vector<std::pair<int,int>>, std::vector<int>>
SabreRouter::transform_pi(
    const CandidateData& cand,
    const std::vector<int>& pi,
    bool reverse,
    SwapCache* swap_cache
) const {
    const std::vector<int>& P_route_inv = reverse ? cand.P_o_inv : cand.P_i_inv;

    // Route qubits to input positions
    auto [swaps, pi_routed] = find_constrained_swaps(
        pi,
        cand.qbit_map_keys_sorted,
        cand.qbit_map_vals_sorted,
        cand.node_mapping_flat,
        P_route_inv,
        swap_cache
    );

    // Update output positions using P_exit
    const std::vector<int>& P_exit = reverse ? cand.P_i : cand.P_o;
    std::vector<int> pi_output = pi_routed;

    for (size_t q_star = 0; q_star < P_exit.size(); q_star++) {
        if (q_star < cand.qstar_to_q.size()) {
            int k = cand.qstar_to_q[q_star];
            if (k < 0) continue;
            pi_output[k] = cand.node_mapping_flat[P_exit[q_star]];
        }
    }

    return {swaps, pi_output};
}

// ---------------------------------------------------------------------------
// generate_extended_set (BFS lookahead)
// ---------------------------------------------------------------------------

std::vector<std::pair<int,int>> SabreRouter::generate_extended_set(
    const std::vector<int>& F,
    const std::vector<uint8_t>& resolved,
    const std::vector<std::vector<int>>& children_graph,
    const std::vector<std::vector<int>>& parents_graph
) const {
    std::vector<std::pair<int,int>> E;
    std::vector<uint8_t> in_E(num_partitions_, 0);
    std::vector<uint8_t> in_F(num_partitions_, 0);
    for (int p : F) in_F[p] = 1;

    struct BFSNode {
        int partition;
        int depth;
    };

    for (int front_idx : F) {
        if (static_cast<int>(E.size()) >= config_.max_E_size) break;

        std::deque<BFSNode> queue;
        // EXACT Python logic: No pre-checks before pushing!
        for (int child : children_graph[front_idx]) {
            queue.push_back({child, 1});
        }

        while (!queue.empty() && static_cast<int>(E.size()) < config_.max_E_size) {
            auto [part, depth] = queue.front();
            queue.pop_front();

            if (depth > config_.max_lookahead) continue;
            if (in_E[part] || in_F[part] || resolved[part]) continue;

            bool parents_ok = true;
            for (int par : parents_graph[part]) {
                if (!resolved[par] && !in_F[par]) {
                    parents_ok = false;
                    break;
                }
            }
            if (!parents_ok) continue;

            if (layout_partitions_[part].is_single) {
                // EXACT Python logic: blindly push grandchildren!
                for (int child : children_graph[part]) {
                    queue.push_back({child, depth});
                }
                continue;
            }

            E.push_back({part, depth});
            in_E[part] = 1;

            if (depth < config_.max_lookahead) {
                for (int child : children_graph[part]) {
                    queue.push_back({child, depth + 1});
                }
            }
        }
    }

    return E;
}

// ---------------------------------------------------------------------------
// Routing cost helpers
// ---------------------------------------------------------------------------

double SabreRouter::compute_routing_cost(
    const std::vector<int>& pi,
    int exclude_partition_idx,
    const std::vector<int>& partition_indices,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    double total = 0.0;
    for (int p_idx : partition_indices) {
        if (p_idx == exclude_partition_idx) continue;
        auto it = canonical_data.find(p_idx);
        if (it == canonical_data.end()) continue;
        const auto& entry = it->second;
        if (entry.edges_u.empty()) continue;
        for (size_t i = 0; i < entry.edges_u.size(); i++) {
            int u = entry.edges_u[i];
            int v = entry.edges_v[i];
            double d = dist(pi[u], pi[v]);
            double cost = d - 1.0;
            if (cost > 0.0) total += config_.swap_cost * cost;
        }
    }
    return total;
}

double SabreRouter::compute_lookahead_cost(
    const std::vector<int>& pi,
    int exclude_partition_idx,
    const std::vector<std::pair<int,int>>& E,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    if (E.empty()) return 0.0;
    double total = 0.0;
    for (auto [p_idx, depth] : E) {
        if (p_idx == exclude_partition_idx) continue;
        auto it = canonical_data.find(p_idx);
        if (it == canonical_data.end()) continue;
        const auto& entry = it->second;
        if (entry.edges_u.empty()) continue;
        double d_cost = 0.0;
        for (size_t i = 0; i < entry.edges_u.size(); i++) {
            int u = entry.edges_u[i];
            int v = entry.edges_v[i];
            double d = dist(pi[u], pi[v]);
            double cost = d - 1.0;
            if (cost > 0.0) d_cost += config_.swap_cost * cost;
        }
        const double alpha = (depth >= 0 && depth < static_cast<int>(alpha_weights_.size()))
            ? alpha_weights_[depth]
            : std::pow(config_.E_alpha, depth);
        total += alpha * d_cost;
    }
    return config_.E_weight * total / static_cast<double>(E.size());
}

// ---------------------------------------------------------------------------
// score_candidate (LightSABRE scoring)
// ---------------------------------------------------------------------------

double SabreRouter::score_candidate(
    const CandidateData& cand,
    const std::vector<int>& F_snapshot,
    const std::vector<int>& pi,
    const std::vector<std::pair<int,int>>& E,
    bool reverse,
    const std::unordered_map<int, CanonicalEntry>& canonical_data,
    SwapCache* swap_cache
) const {
    auto [swaps, output_perm] = transform_pi(cand, pi, reverse, swap_cache);

    double score = config_.swap_cost * static_cast<double>(swaps.size());
    score += config_.local_cost_weight * static_cast<double>(cand.cnot_count);

    // F cost: average routing cost over F \ {cand}
    int cand_idx = cand.partition_idx;
    int n_other = 0;
    double f_sum = 0.0;
    for (int p_idx : F_snapshot) {
        if (p_idx == cand_idx) continue;
        auto it = canonical_data.find(p_idx);
        if (it == canonical_data.end()) continue;
        const auto& entry = it->second;
        n_other++;
        if (entry.edges_u.empty()) continue;
        for (size_t i = 0; i < entry.edges_u.size(); i++) {
            int u = entry.edges_u[i];
            int v = entry.edges_v[i];
            double d = dist(output_perm[u], output_perm[v]);
            double cost = d - 1.0;
            if (cost > 0.0) f_sum += config_.swap_cost * cost;
        }
    }
    if (n_other > 0) score += f_sum / static_cast<double>(n_other);

    // E cost: alpha^depth-decayed lookahead
    if (!E.empty()) {
        double e_sum = 0.0;
        for (auto [p_idx, depth] : E) {
            if (p_idx == cand_idx) continue;
            auto it = canonical_data.find(p_idx);
            if (it == canonical_data.end()) continue;
            const auto& entry = it->second;
            if (entry.edges_u.empty()) continue;
            double d_cost = 0.0;
            for (size_t i = 0; i < entry.edges_u.size(); i++) {
                int u = entry.edges_u[i];
                int v = entry.edges_v[i];
                double d = dist(output_perm[u], output_perm[v]);
                double cost = d - 1.0;
                if (cost > 0.0) d_cost += config_.swap_cost * cost;
            }
            const double alpha = (depth >= 0 && depth < static_cast<int>(alpha_weights_.size()))
                ? alpha_weights_[depth]
                : std::pow(config_.E_alpha, depth);
            e_sum += alpha * d_cost;
        }
        score += config_.E_weight * e_sum / static_cast<double>(E.size());
    }

    return score;
}

// ---------------------------------------------------------------------------
// obtain_partition_candidates
// ---------------------------------------------------------------------------

std::vector<const CandidateData*> SabreRouter::obtain_partition_candidates(
    const std::vector<int>& F
) const {
    std::vector<const CandidateData*> result;
    for (int p_idx : F) {
        if (p_idx < 0 || p_idx >= num_partitions_) continue;
        for (const auto& cand : candidate_cache_[p_idx]) {
            result.push_back(&cand);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// prefilter_candidates
// ---------------------------------------------------------------------------

std::vector<const CandidateData*> SabreRouter::prefilter_candidates(
    const std::vector<const CandidateData*>& candidates,
    const std::vector<int>& pi,
    int top_k,
    bool reverse
) const {
    if (static_cast<int>(candidates.size()) <= top_k) return candidates;
    if (top_k <= 0) return {};

    using Pair = std::pair<double, const CandidateData*>;
    std::vector<Pair> estimated;
    estimated.reserve(candidates.size());
    for (const auto* cand : candidates) {
        double est = estimate_swap_count(*cand, pi, reverse) * config_.swap_cost
                     + config_.local_cost_weight * cand->cnot_count;
        estimated.push_back({est, cand});
    }

    std::nth_element(
        estimated.begin(),
        estimated.begin() + top_k,
        estimated.end(),
        [](const Pair& a, const Pair& b) {
            return a.first < b.first;
        }
    );

    std::vector<const CandidateData*> result;
    result.reserve(top_k);
    for (int i = 0; i < top_k; i++) {
        result.push_back(estimated[i].second);
    }
    return result;
}

// ---------------------------------------------------------------------------
// select_best_candidate
// ---------------------------------------------------------------------------

const CandidateData& SabreRouter::select_best_candidate(
    const std::vector<const CandidateData*>& candidates,
    const std::vector<double>& scores,
    std::mt19937* rng
) const {
    // Find minimum score
    double min_score = scores[0];
    size_t min_idx = 0;
    for (size_t i = 1; i < scores.size(); i++) {
        if (scores[i] < min_score) {
            min_score = scores[i];
            min_idx = i;
        }
    }

    if (rng && min_score > 0.0) {
        std::vector<size_t> near_best;
        double threshold = min_score * (1.0 + config_.score_tolerance);
        for (size_t i = 0; i < scores.size(); i++) {
            if (scores[i] <= threshold) near_best.push_back(i);
        }
        if (near_best.size() > 1) {
            std::uniform_int_distribution<size_t> dist(0, near_best.size() - 1);
            return *candidates[near_best[dist(*rng)]];
        }
    }

    return *candidates[min_idx];
}

// ---------------------------------------------------------------------------
// release_valve
// ---------------------------------------------------------------------------

std::pair<std::vector<std::pair<int,int>>, std::vector<int>>
SabreRouter::release_valve(
    const std::vector<int>& F,
    const std::vector<int>& pi,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    // Find the F partition whose worst-pair distance is smallest
    int best_d = std::numeric_limits<int>::max();
    int best_p = -1;
    int best_u = -1, best_v = -1;

    for (int p_idx : F) {
        auto it = canonical_data.find(p_idx);
        if (it == canonical_data.end()) continue;
        const auto& entry = it->second;
        if (entry.edges_u.empty()) continue;
        for (size_t i = 0; i < entry.edges_u.size(); i++) {
            int u = entry.edges_u[i];
            int v = entry.edges_v[i];
            double d = dist(pi[u], pi[v]);
            int di = static_cast<int>(d);
            if (di > 1 && (di < best_d || (di == best_d && p_idx < best_p))) {
                best_d = di;
                best_p = p_idx;
                best_u = u;
                best_v = v;
            }
        }
    }

    if (best_p < 0) return {{}, pi};

    auto path = bfs_shortest_path(pi[best_u], pi[best_v]);
    if (static_cast<int>(path.size()) < 2) return {{}, pi};

    int k = static_cast<int>(path.size()) - 1;
    int m = k / 2;
    std::vector<std::pair<int,int>> swaps;
    for (int i = 0; i < m; i++) {
        swaps.push_back({path[i], path[i + 1]});
    }
    for (int i = k; i > m; i--) {
        swaps.push_back({path[i], path[i - 1]});
    }

    auto pi_new = apply_swaps_to_pi(pi, swaps);
    return {swaps, pi_new};
}

// ---------------------------------------------------------------------------
// heuristic_search (main loop)
// ---------------------------------------------------------------------------

std::pair<std::vector<int>, int> SabreRouter::heuristic_search(
    const std::vector<int>& F_init,
    std::vector<int> pi,
    bool reverse,
    std::mt19937* rng,
    const std::unordered_map<int, CanonicalEntry>& canonical_data,
    const std::vector<std::vector<int>>& cg,
    const std::vector<std::vector<int>>& pg
) const {
    std::vector<int> F;
    std::vector<int> queue;
    std::vector<uint8_t> resolved(num_partitions_, 0);
    std::vector<uint8_t> in_F(num_partitions_, 0);
    int total_swaps = 0;

    // Split F_init into F (multi-qubit) and queue (single-qubit)
    for (int p : F_init) {
        if (layout_partitions_[p].is_single) {
            queue.push_back(p);
        } else {
            F.push_back(p);
            in_F[p] = 1;
        }
    }

    // Flush initial single-qubit partitions
    while (!queue.empty()) {
        int p = queue.back();
        queue.pop_back();

        if (resolved[p]) continue;
        resolved[p] = 1;

        for (int child : cg[p]) {
            if (!resolved[child] && !in_F[child]) {
                bool parents_ok = true;
                for (int par : pg[child]) {
                    if (!resolved[par]) { parents_ok = false; break; }
                }
                if (parents_ok) {
                    if (layout_partitions_[child].is_single) {
                        queue.push_back(child);
                    } else {
                        F.push_back(child);
                        in_F[child] = 1;
                    }
                }
            }
        }
    }

    // Swap cache for this search call (thread-local, on stack)
    SwapCache swap_cache;

    // Main search loop
    while (!F.empty()) {
        auto all_candidates = obtain_partition_candidates(F);
        if (all_candidates.empty()) break;

        // Prefilter
        auto candidates = prefilter_candidates(
            all_candidates, pi, config_.prefilter_top_k, reverse);

        // Generate extended set
        auto E = generate_extended_set(F, resolved, cg, pg);

        // Score all candidates
        std::vector<double> scores;
        scores.reserve(candidates.size());
        for (const auto* cand : candidates) {
            scores.push_back(score_candidate(
                *cand, F, pi, E, reverse, canonical_data, &swap_cache));
        }

        // Select best
        const auto& best = select_best_candidate(candidates, scores, rng);

        // Remove from F and mark resolved
        F.erase(std::remove(F.begin(), F.end(), best.partition_idx), F.end());
        in_F[best.partition_idx] = 0;
        resolved[best.partition_idx] = 1;

        // Apply transform
        auto [swaps, pi_new] = transform_pi(best, pi, reverse, &swap_cache);
        total_swaps += static_cast<int>(swaps.size());
        pi = std::move(pi_new);

        // Update F with newly eligible children
        for (int child : cg[best.partition_idx]) {
            if (!resolved[child] && !in_F[child]) {
                bool parents_ok = true;
                for (int par : pg[child]) {
                    if (!resolved[par]) { parents_ok = false; break; }
                }
                
                if (parents_ok) {
                    if (layout_partitions_[child].is_single) {
                        resolved[child] = 1;
                        std::vector<int> stack;
                        for (int gc : cg[child]) stack.push_back(gc);
                        
                        while (!stack.empty()) {
                            int gc = stack.back();
                            stack.pop_back();
                            
                            if (!resolved[gc] && !in_F[gc]) {
                                bool gc_parents_ok = true;
                                for (int p_gc : pg[gc]) {
                                    if (!resolved[p_gc]) { gc_parents_ok = false; break; }
                                }
                                if (gc_parents_ok) {
                                    if (layout_partitions_[gc].is_single) {
                                        resolved[gc] = 1;
                                        for (int ggc : cg[gc]) stack.push_back(ggc);
                                    } else {
                                        F.push_back(gc);
                                        in_F[gc] = 1;
                                    }
                                }
                            }
                        }
                    } else {
                        F.push_back(child);
                        in_F[child] = 1;
                    }
                }
            }
        }
    }

    return {pi, total_swaps};
}

// ---------------------------------------------------------------------------
// run_trial (full implementation)
// ---------------------------------------------------------------------------

TrialResult SabreRouter::run_trial(
    int trial_idx,
    const std::vector<int>& seeded_pi,
    int n_iterations,
    int n_trials
) const {
    // RNG setup
    std::mt19937 rng_gen(config_.random_seed + trial_idx);
    std::mt19937* rng = (n_trials > 1) ? &rng_gen : nullptr;

    // vf2_cutoff: first 5% of trials use seeded layout
    int vf2_cutoff = std::max(1, static_cast<int>(n_trials * 0.05));
    std::vector<int> pi;
    if (trial_idx < vf2_cutoff) {
        pi = seeded_pi;
    } else {
        pi = random_permutation(N_, rng_gen);
    }

    auto F_rev = get_final_layer();
    auto F_fwd = get_initial_layer();

    // Forward-backward-forward iterations
    for (int iteration = 0; iteration < n_iterations; iteration++) {
        // Backward pass: swap DAG/IDAG
        auto bwd_result = heuristic_search(F_rev, pi, true, rng, canonical_data_rev_, IDAG_, DAG_);
        pi = std::move(bwd_result.first);

        // Forward pass (skip on last iteration)
        if (iteration < n_iterations - 1) {
            auto fwd_result = heuristic_search(F_fwd, pi, false, rng, canonical_data_fwd_, DAG_, IDAG_);
            pi = std::move(fwd_result.first);
        }
    }

    // Final evaluation pass (deterministic, no RNG)
    auto eval_result = heuristic_search(F_fwd, pi, false, nullptr, canonical_data_fwd_, DAG_, IDAG_); // Evaluates cost using a copy under the hood
    int cost = eval_result.second;

    return TrialResult{std::move(pi), cost}; // Return the pi from AFTER the backward pass, BEFORE the eval pass
}

} // namespace squander::routing
