#include "include/OSR_Parity_Network.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <map>

namespace squander_osr {
namespace {
struct LinearGroup {
    unsigned width;
    std::vector<unsigned> codes;
    std::vector<std::pair<unsigned, unsigned>> gates;
    std::vector<std::array<unsigned, 12>> next;
};

std::shared_ptr<const LinearGroup> linear_group(unsigned width, unsigned couplers) {
    // At most 64 physical graphs at width four; no target-dependent cache.
    static std::mutex mutex;
    static std::map<std::pair<unsigned, unsigned>, std::shared_ptr<const LinearGroup>> groups;
    const unsigned valid_mask = [&] {
        unsigned result = 0;
        for (unsigned a = 0; a < width; ++a)
            for (unsigned b = a + 1; b < width; ++b) result |= 1U << (a * width + b);
        return result;
    }();
    couplers &= valid_mask;
    std::lock_guard<std::mutex> guard(mutex);
    const auto key = std::make_pair(width, couplers);
    auto found = groups.find(key);
    if (found != groups.end()) return found->second;
    auto group = std::make_shared<LinearGroup>();
    group->width = width;
    const unsigned row_mask = (1U << width) - 1;
    unsigned identity = 0;
    for (unsigned q = 0; q < width; ++q) identity |= (1U << q) << (width * q);
    for (unsigned control = 0; control < width; ++control)
        for (unsigned target = 0; target < width; ++target)
            if (control != target && (couplers & (1U << (std::min(control, target) * width + std::max(control, target)))))
                group->gates.push_back({control, target});
    std::vector<int> ids(1U << (width * width), -1);
    group->codes.push_back(identity);
    ids[identity] = 0;
    for (unsigned index = 0; index < group->codes.size(); ++index) {
        const unsigned code = group->codes[index];
        std::array<unsigned, 12> transitions{};
        for (unsigned edge = 0; edge < group->gates.size(); ++edge) {
            const auto gate = group->gates[edge];
            const unsigned row = (code >> (width * gate.first)) & row_mask;
            const unsigned following = code ^ (row << (width * gate.second));
            if (ids[following] < 0) {
                ids[following] = static_cast<int>(group->codes.size());
                group->codes.push_back(following);
            }
            transitions[edge] = static_cast<unsigned>(ids[following]);
        }
        group->next.push_back(transitions);
    }
    groups.emplace(key, group);
    return group;
}
}

std::vector<ParityNetwork> parity_networks(unsigned width, unsigned request,
                                          unsigned couplers, int cnot_budget) {
    std::vector<ParityNetwork> result;
    if (width < 2 || width > 4 || cnot_budget < 0) return result;
    const unsigned controller_terms = (1U << (width - 1)) - 1;
    if (request >= (1U << controller_terms)) return result;
    std::vector<unsigned> parities;
    for (unsigned subset = 1; subset <= controller_terms; ++subset)
        if (request & (1U << (subset - 1))) parities.push_back(1U | (subset << 1));
    // One gate introduces at most one requested row. The last gate must
    // restore root separation, so cannot introduce a requested root parity.
    if (!parities.empty() && cnot_budget < static_cast<int>(parities.size() + 1)) return result;
    const auto group = linear_group(width, couplers);
    const unsigned row_mask = (1U << width) - 1;
    const unsigned bits = static_cast<unsigned>(parities.size());
    const unsigned seen_space = 1U << bits, full = seen_space - 1;
    std::vector<unsigned> hits(group->codes.size(), 0);
    std::vector<bool> terminal(group->codes.size(), false);
    for (unsigned index = 0; index < group->codes.size(); ++index) {
        const unsigned code = group->codes[index];
        bool separates_root = (code & row_mask) == 1;
        for (unsigned q = 0; q < width; ++q) {
            const unsigned row = (code >> (width * q)) & row_mask;
            if (q && (row & 1)) separates_root = false;
            for (unsigned bit = 0; bit < bits; ++bit)
                if (row == parities[bit]) hits[index] |= 1U << bit;
        }
        terminal[index] = separates_root;
    }
    const unsigned states = static_cast<unsigned>(group->codes.size()) * seen_space;
    std::vector<int> parent(states, -2);
    std::vector<unsigned char> edge_used(states, 0);
    std::vector<unsigned short> distance(states, 0);
    std::vector<unsigned> queue;
    queue.reserve(std::min(states, 100000U));
    const unsigned initial = hits[0];
    parent[initial] = -1;
    queue.push_back(initial);
    int minimum = -1;
    std::vector<unsigned> goals;
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const unsigned state = queue[head], linear = state >> bits, seen = state & full;
        const int depth = distance[state];
        if (minimum >= 0 && depth > minimum) break;
        if (seen == full && terminal[linear]) {
            if (minimum < 0) minimum = depth;
            goals.push_back(state);
            continue;
        }
        if (depth >= cnot_budget || (minimum >= 0 && depth >= minimum)) continue;
        for (unsigned edge = 0; edge < group->gates.size(); ++edge) {
            const unsigned following = group->next[linear][edge];
            const unsigned next_state = (following << bits) | (seen | hits[following]);
            if (parent[next_state] != -2) continue;
            parent[next_state] = static_cast<int>(state);
            edge_used[next_state] = static_cast<unsigned char>(edge);
            distance[next_state] = static_cast<unsigned short>(depth + 1);
            queue.push_back(next_state);
        }
    }
    for (unsigned state : goals) {
        ParityNetwork network;
        const unsigned code = group->codes[state >> bits];
        for (unsigned q = 0; q < width; ++q) network.end_rows.push_back((code >> (width * q)) & row_mask);
        for (int current = static_cast<int>(state); parent[current] >= 0; current = parent[current])
            network.moves.push_back(group->gates[edge_used[current]]);
        std::reverse(network.moves.begin(), network.moves.end());
        result.push_back(std::move(network));
    }
    return result;
}
}

extern "C" int qgd_osr_parity_networks(unsigned width, unsigned request,
        unsigned couplers, int cnot_budget, unsigned* rows, unsigned* moves,
        int network_capacity, int* minimum, int* network_count) {
    if (!minimum || !network_count || network_capacity < 0) return -1;
    try {
        auto networks = squander_osr::parity_networks(width, request, couplers, cnot_budget);
        *network_count = static_cast<int>(networks.size());
        *minimum = networks.empty() ? -1 : static_cast<int>(networks.front().moves.size());
        if (!rows && !moves) return 0;
        if (!rows || (!moves && *minimum != 0) || network_capacity < *network_count) return -1;
        for (std::size_t index = 0; index < networks.size(); ++index) {
            for (unsigned q = 0; q < width; ++q) rows[index * width + q] = networks[index].end_rows[q];
            for (std::size_t step = 0; step < networks[index].moves.size(); ++step) {
                moves[2 * (index * *minimum + step)] = networks[index].moves[step].first;
                moves[2 * (index * *minimum + step) + 1] = networks[index].moves[step].second;
            }
        }
        return 0;
    } catch (...) { return -1; }
}
