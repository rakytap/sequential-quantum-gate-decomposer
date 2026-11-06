#ifndef SQUANDER_OSR_FAN_ECHO_MOTIFS_H
#define SQUANDER_OSR_FAN_ECHO_MOTIFS_H

#include <algorithm>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace squander_osr {

// A repeated interaction conjugates a neighboring fan. An optional triangle
// edge precedes or follows the echo. Local U3 layers make CNOT direction a
// choice of chart, so retain one available edge for each undirected support.
// This finite proposal family is neither an exhaustive library nor a global
// depth restriction. It is generated uniformly for every supported width.
inline std::vector<std::vector<int>> fan_echo_motifs(
    int qubit_count, const std::vector<unsigned>& supports, int cnot_bound
) {
    std::set<std::vector<int>> distinct;
    if (qubit_count < 2 ||
        qubit_count >= std::numeric_limits<unsigned>::digits ||
        cnot_bound < 2) {
        return {};
    }
    std::vector<std::vector<int>> edge(
        qubit_count, std::vector<int>(qubit_count, -1)
    );
    for (int first = 0; first < qubit_count; ++first) {
        for (int second = first + 1; second < qubit_count; ++second) {
            const unsigned support = (unsigned(1) << first) |
                                     (unsigned(1) << second);
            const auto found = std::find(supports.begin(), supports.end(), support);
            if (found != supports.end()) {
                const int index = static_cast<int>(found - supports.begin());
                edge[first][second] = edge[second][first] = index;
            }
        }
    }
    for (int first = 0; first < qubit_count; ++first) {
        for (int hub = 0; hub < qubit_count; ++hub) {
            const int repeated = edge[first][hub];
            if (first == hub || repeated < 0) continue;
            std::vector<int> others;
            for (int qubit = 0; qubit < qubit_count; ++qubit) {
                if (qubit != first && qubit != hub) others.push_back(qubit);
            }
            const unsigned subset_count = unsigned(1) << others.size();
            for (unsigned mask = 0; mask < subset_count; ++mask) {
                std::vector<int> fan;
                for (size_t index = 0; index < others.size(); ++index) {
                    if (mask & (unsigned(1) << index)) fan.push_back(others[index]);
                }
                if (static_cast<int>(fan.size()) + 2 > cnot_bound) continue;
                do {
                    std::vector<int> word{repeated};
                    bool available = true;
                    for (int qubit : fan) {
                        const int interaction = edge[hub][qubit];
                        if (interaction < 0) {
                            available = false;
                            break;
                        }
                        word.push_back(interaction);
                    }
                    if (!available) continue;
                    word.push_back(repeated);
                    distinct.insert(word);
                    if (static_cast<int>(word.size()) + 1 > cnot_bound) continue;
                    for (int qubit : fan) {
                        const int closing = edge[first][qubit];
                        if (closing < 0) continue;
                        std::vector<int> before{closing};
                        before.insert(before.end(), word.begin(), word.end());
                        distinct.insert(std::move(before));
                        std::vector<int> after = word;
                        after.push_back(closing);
                        distinct.insert(std::move(after));
                    }
                } while (std::next_permutation(fan.begin(), fan.end()));
            }
        }
    }
    std::vector<std::vector<int>> result(distinct.begin(), distinct.end());
    std::sort(result.begin(), result.end(), [](
        const std::vector<int>& first, const std::vector<int>& second
    ) {
        return first.size() != second.size() ? first.size() < second.size()
                                             : first < second;
    });
    return result;
}

} // namespace squander_osr

#endif
