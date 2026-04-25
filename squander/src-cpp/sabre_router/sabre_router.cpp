/*
Copyright 2025 SQUANDER Contributors

C++ backend for the SABRE-style partition-aware routing engine.
*/

#include "sabre_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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
    // Build CSR view of adj_
    adj_offsets_.resize(N_ + 1);
    adj_offsets_[0] = 0;
    for (int i = 0; i < N_; i++) {
        adj_offsets_[i + 1] = adj_offsets_[i] + static_cast<int>(adj_[i].size());
    }
    adj_flat_.resize(adj_offsets_[N_]);
    for (int i = 0; i < N_; i++) {
        for (size_t j = 0; j < adj_[i].size(); j++) {
            adj_flat_[adj_offsets_[i] + j] = adj_[i][j];
        }
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

    max_finite_distance_ = 1.0;
    for (double d : D_) {
        if (std::isfinite(d) && d > max_finite_distance_) {
            max_finite_distance_ = d;
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

std::vector<int> SabreRouter::perturb_layout(
    const std::vector<int>& base,
    int num_swaps,
    std::mt19937& rng
) const {
    if (num_swaps <= 0 || adj_.empty()) {
        return base;
    }

    std::vector<std::pair<int, int>> swaps;
    swaps.reserve(num_swaps);
    std::uniform_int_distribution<int> phys_dist(0, N_ - 1);

    for (int step = 0; step < num_swaps; step++) {
        int phys = phys_dist(rng);
        int retries = 0;
        while (adj_[phys].empty() && retries < N_) {
            phys = (phys + 1) % N_;
            retries++;
        }
        if (adj_[phys].empty()) {
            break;
        }
        std::uniform_int_distribution<int> nb_dist(
            0, static_cast<int>(adj_[phys].size()) - 1
        );
        int nb = adj_[phys][nb_dist(rng)];
        swaps.push_back({std::min(phys, nb), std::max(phys, nb)});
    }

    if (swaps.empty()) {
        return base;
    }

    return apply_swaps_to_pi(base, swaps);
}

std::vector<int> SabreRouter::sample_initial_layout(
    int trial_idx,
    int n_trials,
    const std::vector<int>& seeded_pi,
    std::mt19937& rng
) const {
    if (n_trials <= 1) {
        return seeded_pi;
    }

    std::vector<int> mirrored_pi(N_);
    for (int q = 0; q < N_; q++) {
        mirrored_pi[q] = (N_ - 1) - seeded_pi[q];
    }

    if (trial_idx == 0) {
        return seeded_pi;
    }
    if (trial_idx == 1) {
        return mirrored_pi;
    }

    const int local_cutoff = std::max(
        3, static_cast<int>(std::ceil(n_trials * 0.6))
    );
    if (trial_idx < local_cutoff) {
        const int local_idx = trial_idx - 2;
        const int band_idx = local_idx / 2;
        const int local_budget = std::max(1, local_cutoff - 2);
        const double phase = static_cast<double>(band_idx)
            / std::max(1, local_budget / 2);
        const int num_swaps = (phase < 0.5)
            ? (1 + (band_idx % 3))
            : (4 + (band_idx % 5));
        const std::vector<int>& base =
            (local_idx % 2 == 0) ? seeded_pi : mirrored_pi;
        return perturb_layout(base, num_swaps, rng);
    }

    return random_permutation(N_, rng);
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
// apply_swaps_to_pi
// ---------------------------------------------------------------------------

std::vector<int> SabreRouter::apply_swaps_to_pi(
    const std::vector<int>& pi,
    const std::vector<std::pair<int,int>>& swaps
) const {
    std::vector<int> result(pi);
    thread_local std::vector<int> p2v;
    if (static_cast<int>(p2v.size()) < N_) p2v.assign(N_, 0);
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

NeighborInfo SabreRouter::build_neighbor_info(
    int exclude_partition_idx,
    const std::vector<int>& F_snapshot,
    const std::vector<std::pair<int,int>>& E,
    const std::vector<int>& pi,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    (void)canonical_data;
    NeighborInfo info;
    info.weight = config_.path_tiebreak_weight;
    if (info.weight <= 0.0) {
        return info;
    }

    // Per-call scratch via thread_local, reset by tracking touched entries
    thread_local std::vector<int> q_to_idx;
    thread_local std::vector<int> q_touched;
    if (static_cast<int>(q_to_idx.size()) < N_) q_to_idx.assign(N_, -1);
    q_touched.clear();

    auto ensure_qubit = [&](int q) -> int {
        int idx = q_to_idx[q];
        if (idx >= 0) return idx;
        idx = static_cast<int>(info.neighbor_vqs.size());
        q_to_idx[q] = idx;
        q_touched.push_back(q);
        info.neighbor_vqs.push_back(q);
        info.initial_pos.push_back(pi[q]);
        return idx;
    };

    // edges: parallel arrays keyed by (lo, hi) — small linear scan dedup
    thread_local std::vector<int> ekey_lo;
    thread_local std::vector<int> ekey_hi;
    thread_local std::vector<int> eu_idx;
    thread_local std::vector<int> ev_idx;
    thread_local std::vector<double> ew;
    ekey_lo.clear(); ekey_hi.clear();
    eu_idx.clear(); ev_idx.clear(); ew.clear();

    auto add_edge = [&](int u, int v, double weight) {
        const int u_idx = ensure_qubit(u);
        const int v_idx = ensure_qubit(v);
        const int lo = std::min(u, v);
        const int hi = std::max(u, v);
        for (size_t i = 0; i < ekey_lo.size(); i++) {
            if (ekey_lo[i] == lo && ekey_hi[i] == hi) {
                ew[i] += weight;
                return;
            }
        }
        ekey_lo.push_back(lo);
        ekey_hi.push_back(hi);
        eu_idx.push_back(u_idx);
        ev_idx.push_back(v_idx);
        ew.push_back(weight);
    };

    auto add_partition_edges = [&](int partition_idx, double weight) {
        if (partition_idx == exclude_partition_idx || weight <= 0.0) return;
        if (
            partition_idx < 0
            || partition_idx >= static_cast<int>(layout_partitions_.size())
        ) return;
        const auto& involved = layout_partitions_[partition_idx].involved_qbits;
        if (involved.size() < 2) return;
        for (size_t i = 0; i < involved.size(); i++) {
            for (size_t j = i + 1; j < involved.size(); j++) {
                add_edge(involved[i], involved[j], weight);
            }
        }
    };

    for (int partition_idx : F_snapshot) {
        add_partition_edges(partition_idx, 1.0);
    }
    for (auto [partition_idx, depth] : E) {
        const double alpha =
            (depth >= 0 && depth < static_cast<int>(alpha_weights_.size()))
                ? alpha_weights_[depth]
                : std::pow(config_.E_alpha, depth);
        add_partition_edges(partition_idx, config_.E_weight * alpha);
    }

    info.edges.reserve(ew.size());
    for (size_t i = 0; i < ew.size(); i++) {
        info.edges.push_back(NeighborEdge{eu_idx[i], ev_idx[i], ew[i]});
    }

    // Reset q_to_idx via touched-list (avoids O(N) clear)
    for (int q : q_touched) q_to_idx[q] = -1;

    return info;
}

double SabreRouter::decay_factor_for_swaps(
    const std::vector<std::pair<int,int>>& swaps,
    const std::vector<double>& decay
) const {
    double factor = 1.0;
    for (auto [u, v] : swaps) {
        factor = std::max(factor, std::max(decay[u], decay[v]));
    }
    return factor;
}

double SabreRouter::routing_objective(
    double route_cost,
    int cnot_count,
    double cnot_weight,
    double decay_factor
) const {
    return decay_factor * (
        route_cost
        + cnot_weight * config_.cnot_cost * static_cast<double>(cnot_count)
    );
}

void SabreRouter::apply_decay_for_swaps(
    const std::vector<std::pair<int,int>>& swaps,
    std::vector<double>& decay
) const {
    if (config_.decay_delta <= 0.0) {
        return;
    }
    for (auto [u, v] : swaps) {
        decay[u] += config_.decay_delta;
        decay[v] += config_.decay_delta;
    }
}

void SabreRouter::reset_decay(std::vector<double>& decay) const {
    std::fill(decay.begin(), decay.end(), 1.0);
}

std::vector<int> SabreRouter::bfs_shortest_path(int src, int dst) const {
    if (src == dst) {
        return {src};
    }

    std::vector<int> parent(N_, -1);
    std::vector<uint8_t> visited(N_, 0);
    std::deque<int> queue;
    queue.push_back(src);
    visited[src] = 1;

    while (!queue.empty()) {
        const int node = queue.front();
        queue.pop_front();
        for (int nb : adj_[node]) {
            if (visited[nb]) {
                continue;
            }
            visited[nb] = 1;
            parent[nb] = node;
            if (nb == dst) {
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

    return {};
}

std::pair<std::vector<std::pair<int,int>>, std::vector<int>> SabreRouter::release_valve(
    const std::vector<int>& F,
    const std::vector<int>& pi,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    double best_worst_dist = -std::numeric_limits<double>::infinity();
    int best_partition_idx = -1;
    int best_u = -1;
    int best_v = -1;

    for (int partition_idx : F) {
        auto it = canonical_data.find(partition_idx);
        if (it == canonical_data.end()) continue;
        const auto& entry = it->second;
        if (entry.edges_u.empty()) continue;

        double worst_dist = 0.0;
        int worst_u = -1;
        int worst_v = -1;
        for (size_t i = 0; i < entry.edges_u.size(); i++) {
            const int u = entry.edges_u[i];
            const int v = entry.edges_v[i];
            const double d = dist(pi[u], pi[v]);
            if (d > worst_dist) {
                worst_dist = d;
                worst_u = u;
                worst_v = v;
            }
        }

        if (worst_dist <= 1.0 || worst_u < 0) continue;

        if (
            worst_dist > best_worst_dist
            || (worst_dist == best_worst_dist
                && (best_partition_idx < 0 || partition_idx < best_partition_idx))
        ) {
            best_worst_dist = worst_dist;
            best_partition_idx = partition_idx;
            best_u = worst_u;
            best_v = worst_v;
        }
    }

    if (best_u < 0) {
        return {{}, pi};
    }

    const auto path = bfs_shortest_path(pi[best_u], pi[best_v]);
    if (path.size() < 2) {
        return {{}, pi};
    }

    const int k = static_cast<int>(path.size()) - 1;
    const int m = k / 2;
    std::vector<std::pair<int,int>> swaps;
    for (int i = 0; i < m; i++) {
        swaps.push_back({path[i], path[i + 1]});
    }
    for (int i = k; i > m + 1; i--) {
        swaps.push_back({path[i], path[i - 1]});
    }

    auto pi_new = apply_swaps_to_pi(pi, swaps);
    return {swaps, pi_new};
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
    SwapCache* swap_cache,
    const NeighborInfo* neighbor_info
) const {
    const int k = static_cast<int>(qbit_map_keys.size());

    // ---- Setup: target/initial positions, pow_N, h0 ----
    thread_local std::vector<int> target_positions;
    thread_local std::vector<int> initial_positions;
    thread_local std::vector<int64_t> pow_N;
    target_positions.resize(k);
    initial_positions.resize(k);
    pow_N.resize(k);
    {
        int64_t s = 1;
        for (int i = 0; i < k; i++) { pow_N[i] = s; s *= N_; }
    }

    bool already_there = true;
    double h0_sum = 0.0;
    int64_t initial_packed = 0;
    int64_t target_packed = 0;
    for (int i = 0; i < k; i++) {
        const int q = qbit_map_keys[i];
        const int v = qbit_map_vals[i];
        const int t = node_mapping_flat[P_route_inv[v]];
        const int ip = pi[q];
        target_positions[i] = t;
        initial_positions[i] = ip;
        if (ip != t) already_there = false;
        h0_sum += dist(ip, t);
        initial_packed += static_cast<int64_t>(ip) * pow_N[i];
        target_packed  += static_cast<int64_t>(t)  * pow_N[i];
    }
    if (already_there) {
        return {{}, pi};
    }

    const bool use_neighbor =
        neighbor_info != nullptr && neighbor_info->uses_tiebreak();

    auto mix64 = [](uint64_t h, uint64_t v) -> uint64_t {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    };

    uint64_t neighbor_hash = 0;
    if (use_neighbor) {
        neighbor_hash = 0xcbf29ce484222325ULL;
        for (const auto& edge : neighbor_info->edges) {
            const int lo = std::min(edge.u_idx, edge.v_idx);
            const int hi = std::max(edge.u_idx, edge.v_idx);
            uint64_t w_bits;
            std::memcpy(&w_bits, &edge.weight, sizeof(w_bits));
            neighbor_hash = mix64(neighbor_hash, static_cast<uint64_t>(lo));
            neighbor_hash = mix64(neighbor_hash, static_cast<uint64_t>(hi));
            neighbor_hash = mix64(neighbor_hash, w_bits);
        }
        for (int p : neighbor_info->initial_pos) {
            neighbor_hash = mix64(neighbor_hash, static_cast<uint64_t>(p));
        }
        uint64_t weight_bits;
        const double weight_val = neighbor_info->weight;
        std::memcpy(&weight_bits, &weight_val, sizeof(weight_bits));
        neighbor_hash = mix64(neighbor_hash, weight_bits);
    }

    const SwapCacheKey cache_key{initial_packed, target_packed, k, neighbor_hash};

    if (swap_cache) {
        auto it = swap_cache->find(cache_key);
        if (it != swap_cache->end()) {
            auto result_pi = apply_swaps_to_pi(pi, it->second);
            return {it->second, result_pi};
        }
    }

    // ---- Neighbor heuristic setup ----
    double total_edge_weight = 0.0;
    if (use_neighbor) {
        for (const auto& edge : neighbor_info->edges) {
            total_edge_weight += edge.weight;
        }
    }
    const double neighbor_norm = std::max(
        1.0, total_edge_weight * std::max(1.0, max_finite_distance_)
    );
    const double neighbor_scale =
        use_neighbor ? (neighbor_info->weight / neighbor_norm) : 0.0;

    auto compute_nb_total = [&](const std::vector<int>& pos_nb) {
        double total = 0.0;
        for (const auto& edge : neighbor_info->edges) {
            total += edge.weight * dist(pos_nb[edge.u_idx], pos_nb[edge.v_idx]);
        }
        return total;
    };

    double initial_nb_total = 0.0;
    if (use_neighbor) {
        initial_nb_total = compute_nb_total(neighbor_info->initial_pos);
    }

    // ---- Arena + open-addressed hash table (replaces visited+parent maps) ----
    struct Node {
        int64_t packed;
        int parent_idx;
        int g;
        int sw_lo, sw_hi;
        double h_sum;       // sum(dist(pos[i], target[i])) — twice the admissible h
        double nb_total;    // sum(edge.weight * dist(...)) — pre-scale
        int nb_arena_idx;   // -1 if !use_neighbor; else index into nb_arena
    };
    thread_local std::vector<Node> arena;
    thread_local std::vector<int32_t> table;
    thread_local std::vector<std::vector<int>> nb_arena;
    arena.clear();
    nb_arena.clear();
    arena.reserve(1024);

    // table size: power of 2, ~2x expected entries
    size_t cap = 1024;
    table.assign(cap, -1);

    auto hash_packed = [](int64_t v) -> uint64_t {
        uint64_t x = static_cast<uint64_t>(v);
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    };

    auto table_grow = [&]() {
        std::vector<int32_t> new_table(table.size() * 2, -1);
        const size_t mask = new_table.size() - 1;
        for (int32_t idx : table) {
            if (idx < 0) continue;
            size_t i = hash_packed(arena[idx].packed) & mask;
            while (new_table[i] >= 0) i = (i + 1) & mask;
            new_table[i] = idx;
        }
        table = std::move(new_table);
    };

    // Returns slot index in `table`. *slot is -1 if empty.
    auto table_slot = [&](int64_t packed) -> size_t {
        const size_t mask = table.size() - 1;
        size_t i = hash_packed(packed) & mask;
        while (true) {
            int32_t idx = table[i];
            if (idx < 0) return i;
            if (arena[idx].packed == packed) return i;
            i = (i + 1) & mask;
        }
    };

    // ---- Push initial node ----
    {
        Node n;
        n.packed = initial_packed;
        n.parent_idx = -1;
        n.g = 0;
        n.sw_lo = -1; n.sw_hi = -1;
        n.h_sum = h0_sum;
        n.nb_total = initial_nb_total;
        n.nb_arena_idx = -1;
        if (use_neighbor) {
            n.nb_arena_idx = static_cast<int>(nb_arena.size());
            nb_arena.push_back(neighbor_info->initial_pos);
        }
        arena.push_back(n);
        table[table_slot(initial_packed)] = 0;
    }

    // PQ entry: (f, g, counter, arena_idx)
    using PQEntry = std::tuple<double, int, uint64_t, int32_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
    uint64_t counter = 0;
    pq.push({0.5 * h0_sum + neighbor_scale * initial_nb_total, 0, counter++, 0});

    thread_local std::vector<int> positions;
    positions.resize(k);

    while (!pq.empty()) {
        auto [f, g_e, ctr, idx] = pq.top();
        pq.pop();
        (void)f; (void)ctr;
        const int g = g_e;
        const int64_t packed = arena[idx].packed;

        // A state can be reinserted with a lower g-cost after this queue entry
        // was pushed. The hash table always points at the current best arena
        // node for a packed state, so discard stale superseded nodes before
        // accepting a target or expanding neighbors.
        if (table[table_slot(packed)] != idx) {
            continue;
        }

        if (packed == target_packed) {
            // Reconstruct path
            std::vector<std::pair<int,int>> path;
            int cur = idx;
            while (arena[cur].parent_idx != -1) {
                path.push_back({arena[cur].sw_lo, arena[cur].sw_hi});
                cur = arena[cur].parent_idx;
            }
            std::reverse(path.begin(), path.end());

            auto result_pi = apply_swaps_to_pi(pi, path);
            if (swap_cache) {
                (*swap_cache)[cache_key] = path;
            }
            return {path, result_pi};
        }

        // Stale entry?
        if (arena[idx].g < g) continue;

        // Unpack positions for this state
        {
            int64_t p = packed;
            for (int i = 0; i < k; i++) {
                positions[i] = static_cast<int>(p % N_);
                p /= N_;
            }
        }
        const double cur_h_sum = arena[idx].h_sum;
        const double cur_nb_total = arena[idx].nb_total;
        const int cur_nb_arena_idx = arena[idx].nb_arena_idx;

        // Expand: every SWAP that moves at least one partition qubit
        for (int i = 0; i < k; i++) {
            const int p = positions[i];
            const int t_i = target_positions[i];
            const int adj_lo = adj_offsets_[p];
            const int adj_hi = adj_offsets_[p + 1];
            for (int nb_idx = adj_lo; nb_idx < adj_hi; nb_idx++) {
                const int nb = adj_flat_[nb_idx];
                // Find j such that positions[j] == nb (if any)
                int j_swap = -1;
                for (int j = 0; j < k; j++) {
                    if (positions[j] == nb) { j_swap = j; break; }
                }

                // Incremental packed
                int64_t new_packed = packed + static_cast<int64_t>(nb - p) * pow_N[i];
                if (j_swap >= 0) {
                    new_packed += static_cast<int64_t>(p - nb) * pow_N[j_swap];
                }

                // Incremental h_sum
                double new_h_sum = cur_h_sum
                    - dist(p, t_i) + dist(nb, t_i);
                if (j_swap >= 0) {
                    const int t_j = target_positions[j_swap];
                    new_h_sum += -dist(nb, t_j) + dist(p, t_j);
                }

                const int new_g = g + 1;
                const size_t slot = table_slot(new_packed);
                const int32_t existing = table[slot];
                if (existing >= 0 && arena[existing].g <= new_g) {
                    continue;
                }

                // Neighbor heuristic: simple recompute (cheaper than incremental for small edge counts)
                double new_nb_total = cur_nb_total;
                int new_nb_arena_idx = -1;
                if (use_neighbor) {
                    std::vector<int> new_pos_nb = nb_arena[cur_nb_arena_idx];
                    int idx_nb = -1, idx_p = -1;
                    for (size_t z = 0; z < new_pos_nb.size(); z++) {
                        const int phys = new_pos_nb[z];
                        if (phys == nb) idx_nb = static_cast<int>(z);
                        else if (phys == p) idx_p = static_cast<int>(z);
                        if (idx_nb >= 0 && idx_p >= 0) break;
                    }
                    if (idx_nb >= 0 || idx_p >= 0) {
                        if (idx_nb >= 0) new_pos_nb[idx_nb] = p;
                        if (idx_p >= 0)  new_pos_nb[idx_p]  = nb;
                        new_nb_total = compute_nb_total(new_pos_nb);
                        new_nb_arena_idx = static_cast<int>(nb_arena.size());
                        nb_arena.push_back(std::move(new_pos_nb));
                    } else {
                        new_nb_arena_idx = cur_nb_arena_idx;
                    }
                }

                // Insert/update node
                Node n;
                n.packed = new_packed;
                n.parent_idx = idx;
                n.g = new_g;
                const int lo = std::min(p, nb);
                const int hi = std::max(p, nb);
                n.sw_lo = lo; n.sw_hi = hi;
                n.h_sum = new_h_sum;
                n.nb_total = new_nb_total;
                n.nb_arena_idx = new_nb_arena_idx;

                int32_t new_idx = static_cast<int32_t>(arena.size());
                arena.push_back(n);

                // Re-find slot if arena grew (table didn't, but slot is still valid
                // since we didn't grow `table`); just write
                table[slot] = new_idx;

                // Grow table if load factor too high (> 0.5)
                if (arena.size() * 2 > table.size()) {
                    table_grow();
                }

                const double f_new = static_cast<double>(new_g)
                                   + 0.5 * new_h_sum
                                   + neighbor_scale * new_nb_total;
                pq.push({f_new, new_g, counter++, new_idx});
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
    SwapCache* swap_cache,
    const NeighborInfo* neighbor_info
) const {
    const std::vector<int>& P_route_inv = reverse ? cand.P_o_inv : cand.P_i_inv;

    // Route qubits to input positions
    auto [swaps, pi_routed] = find_constrained_swaps(
        pi,
        cand.qbit_map_keys_sorted,
        cand.qbit_map_vals_sorted,
        cand.node_mapping_flat,
        P_route_inv,
        swap_cache,
        neighbor_info
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

double SabreRouter::entry_future_cost(
    const CanonicalEntry& entry,
    const std::vector<int>& pi
) const {
    double total = 0.0;
    for (size_t i = 0; i < entry.edges_u.size(); i++) {
        const double d = dist(pi[entry.edges_u[i]], pi[entry.edges_v[i]]);
        if (d > 1.0) total += d - 1.0;
    }
    return total;
}

double SabreRouter::future_context_cost(
    int exclude_partition_idx,
    const std::vector<int>& pi,
    const std::vector<int>& F_snapshot,
    const std::vector<std::pair<int,int>>& E,
    bool reverse,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    (void)reverse;

    // BQSKit-style cost: sum max(0, dist - 1) over each canonical gate edge,
    // with no candidate-permutation enumeration. Same shape for F and E so
    // the future signal is monotone in distance instead of flickering with
    // whichever candidate happens to win the lower bound.
    double f_sum = 0.0;
    int n_other = 0;
    for (int p_idx : F_snapshot) {
        if (p_idx == exclude_partition_idx) continue;
        auto it = canonical_data.find(p_idx);
        if (it == canonical_data.end()) continue;
        f_sum += entry_future_cost(it->second, pi);
        n_other++;
    }

    double score = n_other > 0
        ? f_sum / static_cast<double>(n_other)
        : 0.0;

    if (!E.empty()) {
        double e_sum = 0.0;
        for (auto [p_idx, depth] : E) {
            if (p_idx == exclude_partition_idx) continue;
            auto it = canonical_data.find(p_idx);
            if (it == canonical_data.end()) continue;
            const double alpha =
                (depth >= 0 && depth < static_cast<int>(alpha_weights_.size()))
                    ? alpha_weights_[depth]
                    : std::pow(config_.E_alpha, depth);
            e_sum += alpha * entry_future_cost(it->second, pi);
        }
        score += config_.E_weight * e_sum / static_cast<double>(E.size());
    }

    return score;
}

std::vector<int> SabreRouter::estimate_candidate_output_layout(
    const CandidateData& cand,
    const std::vector<int>& pi,
    bool reverse
) const {
    const std::vector<int>& P_exit = reverse ? cand.P_i : cand.P_o;
    std::vector<int> pi_output = pi;

    for (size_t q_star = 0; q_star < P_exit.size(); q_star++) {
        if (q_star < cand.qstar_to_q.size()) {
            int k = cand.qstar_to_q[q_star];
            if (k < 0) continue;
            pi_output[k] = cand.node_mapping_flat[P_exit[q_star]];
        }
    }

    return pi_output;
}

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
        total += entry_future_cost(it->second, pi);
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
        const double d_cost = entry_future_cost(it->second, pi);
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
    SwapCache* swap_cache,
    const std::vector<double>* decay,
    std::vector<std::pair<int,int>>* out_swaps,
    std::vector<int>* out_pi_new,
    const std::vector<ResolvedEntry>* resolved_F,
    const std::vector<ResolvedEntry>* resolved_E,
    const NeighborInfo* cached_neighbor_info
) const {
    NeighborInfo local_neighbor_info;
    const NeighborInfo* neighbor_ptr;
    if (cached_neighbor_info) {
        neighbor_ptr = cached_neighbor_info->uses_tiebreak() ? cached_neighbor_info : nullptr;
    } else {
        local_neighbor_info = build_neighbor_info(
            cand.partition_idx, F_snapshot, E, pi, canonical_data);
        neighbor_ptr = local_neighbor_info.uses_tiebreak() ? &local_neighbor_info : nullptr;
    }
    auto [swaps, output_perm] = transform_pi(
        cand,
        pi,
        reverse,
        swap_cache,
        neighbor_ptr
    );

    double decay_factor = 1.0;
    if (decay != nullptr && !swaps.empty()) {
        decay_factor = decay_factor_for_swaps(swaps, *decay);
    }
    double score = routing_objective(
        static_cast<double>(swaps.size()),
        cand.cnot_count,
        1.0,
        decay_factor
    );

    const int cand_idx = cand.partition_idx;
    score += future_context_cost(
        cand_idx,
        output_perm,
        F_snapshot,
        E,
        reverse,
        canonical_data
    );
    (void)resolved_F;
    (void)resolved_E;

    if (out_swaps) *out_swaps = std::move(swaps);
    if (out_pi_new) *out_pi_new = std::move(output_perm);
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
    const std::vector<int>& F_snapshot,
    const std::vector<std::pair<int,int>>& E,
    bool reverse,
    const std::unordered_map<int, CanonicalEntry>& canonical_data
) const {
    if (static_cast<int>(candidates.size()) <= top_k) return candidates;
    if (top_k <= 0) return {};

    using Pair = std::pair<double, const CandidateData*>;
    std::vector<Pair> estimated;
    estimated.reserve(candidates.size());
    for (const auto* cand : candidates) {
        const auto approx_output = estimate_candidate_output_layout(
            *cand, pi, reverse);
        const double est = routing_objective(
            static_cast<double>(estimate_swap_count(*cand, pi, reverse)),
            cand->cnot_count
        ) + future_context_cost(
            cand->partition_idx, approx_output, F_snapshot, E, reverse,
            canonical_data);
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
    (void)rng;

    // Find minimum score
    double min_score = scores[0];
    size_t min_idx = 0;
    for (size_t i = 1; i < scores.size(); i++) {
        if (scores[i] < min_score) {
            min_score = scores[i];
            min_idx = i;
        }
    }

    return *candidates[min_idx];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// heuristic_search (main loop)
// ---------------------------------------------------------------------------

std::pair<std::vector<int>, double> SabreRouter::heuristic_search(
    const std::vector<int>& F_init,
    std::vector<int> pi,
    bool reverse,
    std::mt19937* rng,
    const std::unordered_map<int, CanonicalEntry>& canonical_data,
    const std::vector<std::vector<int>>& cg,
    const std::vector<std::vector<int>>& pg,
    ForwardRouteResult* route_trace
) const {
    std::vector<int> F;
    std::vector<int> queue;
    std::vector<uint8_t> resolved(num_partitions_, 0);
    std::vector<uint8_t> in_F(num_partitions_, 0);
    double total_cost = 0.0;

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
        if (route_trace) {
            RouteStep step;
            step.type = 2;
            step.partition_idx = p;
            if (!layout_partitions_[p].involved_qbits.empty()) {
                step.physical_qubit = pi[layout_partitions_[p].involved_qbits[0]];
            }
            route_trace->steps.push_back(std::move(step));
        }

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
    std::vector<double> decay(N_, 1.0);
    int swap_heavy_partitions = 0;

    // Main search loop
    while (!F.empty()) {
        if (
            config_.swap_burst_budget > 0
            && swap_heavy_partitions >= config_.swap_burst_budget
        ) {
            auto [valve_swaps, pi_bridged] = release_valve(
                F,
                pi,
                canonical_data
            );
            if (!valve_swaps.empty()) {
                total_cost += routing_objective(
                    static_cast<double>(valve_swaps.size()),
                    0,
                    1.0,
                    decay_factor_for_swaps(valve_swaps, decay)
                );
                if (route_trace) {
                    RouteStep step;
                    step.type = 0;
                    step.swaps = valve_swaps;
                    route_trace->cnot_count += static_cast<int>(valve_swaps.size()) * 3;
                    route_trace->steps.push_back(std::move(step));
                }
                apply_decay_for_swaps(valve_swaps, decay);
                pi = std::move(pi_bridged);
                swap_heavy_partitions = 0;
                continue;
            }
            reset_decay(decay);
            swap_heavy_partitions = 0;
        }

        auto all_candidates = obtain_partition_candidates(F);
        if (all_candidates.empty()) break;

        // Generate extended set
        auto E = generate_extended_set(F, resolved, cg, pg);

        // Prefilter with a cheap estimate of the candidate's future context.
        auto candidates = prefilter_candidates(
            all_candidates, pi, config_.prefilter_top_k, F, E, reverse,
            canonical_data);

        // Pre-resolve canonical entries for F and E once per F-step
        std::vector<ResolvedEntry> resolved_F;
        resolved_F.reserve(F.size());
        for (int p_idx : F) {
            auto it = canonical_data.find(p_idx);
            const CanonicalEntry* ent = (it != canonical_data.end()) ? &it->second : nullptr;
            resolved_F.push_back({p_idx, ent, 1.0});
        }
        std::vector<ResolvedEntry> resolved_E;
        resolved_E.reserve(E.size());
        for (auto [p_idx, depth] : E) {
            auto it = canonical_data.find(p_idx);
            const CanonicalEntry* ent = (it != canonical_data.end()) ? &it->second : nullptr;
            const double alpha = (depth >= 0 && depth < static_cast<int>(alpha_weights_.size()))
                ? alpha_weights_[depth]
                : std::pow(config_.E_alpha, depth);
            resolved_E.push_back({p_idx, ent, alpha});
        }

        // Group candidates by partition_idx so build_neighbor_info is shared
        std::vector<size_t> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return candidates[a]->partition_idx < candidates[b]->partition_idx;
        });

        // Score all candidates and cache each one's transform output
        std::vector<double> scores(candidates.size());
        std::vector<std::vector<std::pair<int,int>>> cached_swaps(candidates.size());
        std::vector<std::vector<int>> cached_pi(candidates.size());
        int prev_partition_idx = -1;
        NeighborInfo cached_ni;
        for (size_t k_ord = 0; k_ord < order.size(); k_ord++) {
            const size_t ci = order[k_ord];
            const int p_idx = candidates[ci]->partition_idx;
            if (p_idx != prev_partition_idx) {
                cached_ni = build_neighbor_info(p_idx, F, E, pi, canonical_data);
                prev_partition_idx = p_idx;
            }
            scores[ci] = score_candidate(
                *candidates[ci],
                F, pi, E, reverse, canonical_data,
                &swap_cache, &decay,
                &cached_swaps[ci], &cached_pi[ci],
                &resolved_F, &resolved_E,
                &cached_ni
            );
        }

        // Select best
        const auto& best = select_best_candidate(candidates, scores, rng);
        // Find selected index to retrieve cached transform
        size_t best_ci = 0;
        for (size_t ci = 0; ci < candidates.size(); ci++) {
            if (candidates[ci] == &best) { best_ci = ci; break; }
        }

        // Remove from F and mark resolved
        F.erase(std::remove(F.begin(), F.end(), best.partition_idx), F.end());
        in_F[best.partition_idx] = 0;
        resolved[best.partition_idx] = 1;

        // Reuse cached transform from scoring (F_snapshot \ {best} == F_after_erase
        // because exclude_partition_idx == best.partition_idx in both cases)
        std::vector<std::pair<int,int>> swaps = std::move(cached_swaps[best_ci]);
        std::vector<int> pi_new = std::move(cached_pi[best_ci]);
        const double decay_factor = swaps.empty()
            ? 1.0
            : decay_factor_for_swaps(swaps, decay);
        total_cost += routing_objective(
            static_cast<double>(swaps.size()),
            best.cnot_count,
            1.0,
            decay_factor
        );
        if (route_trace) {
            if (!swaps.empty()) {
                RouteStep swap_step;
                swap_step.type = 0;
                swap_step.swaps = swaps;
                route_trace->cnot_count += static_cast<int>(swaps.size()) * 3;
                route_trace->steps.push_back(std::move(swap_step));
            }
            RouteStep part_step;
            part_step.type = 1;
            part_step.partition_idx = best.partition_idx;
            part_step.candidate_idx = best.candidate_idx;
            route_trace->cnot_count += best.cnot_count;
            route_trace->steps.push_back(std::move(part_step));
        }
        pi = std::move(pi_new);
        apply_decay_for_swaps(swaps, decay);
        if (swaps.empty()) {
            swap_heavy_partitions = 0;
            reset_decay(decay);
        } else {
            swap_heavy_partitions++;
        }

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
                        if (route_trace) {
                            RouteStep step;
                            step.type = 2;
                            step.partition_idx = child;
                            if (!layout_partitions_[child].involved_qbits.empty()) {
                                step.physical_qubit = pi[layout_partitions_[child].involved_qbits[0]];
                            }
                            route_trace->steps.push_back(std::move(step));
                        }
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
                                        if (route_trace) {
                                            RouteStep step;
                                            step.type = 2;
                                            step.partition_idx = gc;
                                            if (!layout_partitions_[gc].involved_qbits.empty()) {
                                                step.physical_qubit = pi[layout_partitions_[gc].involved_qbits[0]];
                                            }
                                            route_trace->steps.push_back(std::move(step));
                                        }
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

    return {pi, total_cost};
}

ForwardRouteResult SabreRouter::route_forward(
    const std::vector<int>& pi
) const {
    ForwardRouteResult result;
    result.pi_initial = pi;
    auto F_fwd = get_initial_layer();
    auto routed = heuristic_search(
        F_fwd,
        pi,
        false,
        nullptr,
        canonical_data_fwd_,
        DAG_,
        IDAG_,
        &result
    );
    result.pi = std::move(routed.first);
    return result;
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

    std::vector<int> pi = sample_initial_layout(
        trial_idx, n_trials, seeded_pi, rng_gen
    );

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
    double cost = eval_result.second;

    return TrialResult{std::move(pi), cost}; // Return the pi from AFTER the backward pass, BEFORE the eval pass
}

} // namespace squander::routing
