#include "include/OSR_Phase_Polynomial.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
int population(unsigned value) {
    int count = 0;
    while (value) { value &= value - 1; ++count; }
    return count;
}

int residue(int value, int modulus) {
    return (value % modulus + modulus) % modulus;
}

std::vector<int> anf(const std::vector<int>& truth, int width, int modulus) {
    auto result = truth;
    for (int bit = 0; bit < width; ++bit)
        for (int mask = 0; mask < (1 << width); ++mask)
            if (mask & (1 << bit))
                result[mask] = residue(result[mask] - result[mask ^ (1 << bit)], modulus);
    return result;
}

struct Polynomial {
    int width;
    std::vector<int> truth;
    std::vector<unsigned> odd_patterns;
    std::array<unsigned, 16> clifford_columns{};
    int clifford_dimension = 0;

    Polynomial(int n, const int* values) : width(n), truth(values, values + (1 << n)) {
        const int variables = (1 << width) - 1;
        const auto coefficients = anf(truth, width, 8);
        std::vector<unsigned> equations;
        for (int subset = 1; subset <= variables; ++subset) {
            const int degree = population(static_cast<unsigned>(subset));
            if (degree > 3) {
                if (coefficients[subset] != 0) return;
                continue;
            }
            const int divisor = 1 << (degree - 1);
            if (coefficients[subset] % divisor) return;
            unsigned row = 0;
            for (int mask = 1; mask <= variables; ++mask)
                if ((mask & subset) == subset) row |= 1U << (mask - 1);
            row |= static_cast<unsigned>((coefficients[subset] / divisor) & 1) << variables;
            equations.push_back(row);
        }
        std::vector<int> pivots;
        int rank = 0;
        for (int column = 0; column < variables; ++column) {
            int pivot = rank;
            while (pivot < static_cast<int>(equations.size()) && !(equations[pivot] & (1U << column))) ++pivot;
            if (pivot == static_cast<int>(equations.size())) continue;
            std::swap(equations[rank], equations[pivot]);
            for (int row = 0; row < static_cast<int>(equations.size()); ++row)
                if (row != rank && (equations[row] & (1U << column))) equations[row] ^= equations[rank];
            pivots.push_back(column);
            ++rank;
        }
        for (int row = rank; row < static_cast<int>(equations.size()); ++row)
            if (equations[row] & (1U << variables)) return;
        std::vector<int> free_columns;
        for (int column = 0; column < variables; ++column) {
            bool pivot = false;
            for (int value : pivots) if (value == column) pivot = true;
            if (!pivot) free_columns.push_back(column);
        }
        for (unsigned choice = 0; choice < (1U << free_columns.size()); ++choice) {
            unsigned solution = 0;
            for (unsigned bit = 0; bit < free_columns.size(); ++bit)
                if (choice & (1U << bit)) solution |= 1U << free_columns[bit];
            for (int row = 0; row < rank; ++row) {
                const unsigned rhs = (equations[row] >> variables) & 1U;
                const unsigned overlap = population(equations[row] & solution) & 1U;
                if (rhs ^ overlap) solution |= 1U << pivots[row];
            }
            odd_patterns.push_back(solution);
        }
        for (int subset = 1; subset <= variables; ++subset) {
            if (population(static_cast<unsigned>(subset)) > 2) continue;
            for (int mask = 1; mask <= variables; ++mask)
                if ((mask & subset) == subset) clifford_columns[mask] |= 1U << clifford_dimension;
            ++clifford_dimension;
        }
    }

    bool solve(unsigned available, std::array<int, 16>& output) const {
        const int dimension = 1 << width;
        for (unsigned odd : odd_patterns) {
            if ((available & odd) != odd) continue;
            std::array<int, 16> coefficients{};
            std::vector<int> residual(dimension);
            bool valid = true;
            for (int mask = 1; mask < dimension; ++mask)
                if (odd & (1U << (mask - 1))) coefficients[mask] = 1;
            for (int x = 0; x < dimension; ++x) {
                int value = 0;
                for (int mask = 1; mask < dimension; ++mask)
                    if ((population(static_cast<unsigned>(x & mask)) & 1) && coefficients[mask]) ++value;
                value = residue(truth[x] - value, 8);
                if (value & 1) { valid = false; break; }
                residual[x] = value / 2;
            }
            if (!valid) continue;
            const auto residual_anf = anf(residual, width, 4);
            unsigned syndrome = 0;
            int position = 0;
            for (int subset = 1; subset < dimension; ++subset) {
                const int degree = population(static_cast<unsigned>(subset));
                if (degree > 2) continue;
                const int divisor = 1 << (degree - 1);
                if (residual_anf[subset] % divisor) { valid = false; break; }
                syndrome |= static_cast<unsigned>((residual_anf[subset] / divisor) & 1) << position++;
            }
            if (!valid) continue;
            std::array<unsigned, 10> basis{}, combinations{};
            for (int mask = 1; mask < dimension; ++mask) {
                if (!(available & (1U << (mask - 1)))) continue;
                unsigned value = clifford_columns[mask], combination = 1U << (mask - 1);
                for (int bit = clifford_dimension - 1; bit >= 0; --bit) {
                    if (!(value & (1U << bit))) continue;
                    if (basis[bit]) { value ^= basis[bit]; combination ^= combinations[bit]; }
                    else { basis[bit] = value; combinations[bit] = combination; break; }
                }
            }
            unsigned selected = 0;
            for (int bit = clifford_dimension - 1; bit >= 0; --bit) {
                if (!(syndrome & (1U << bit))) continue;
                if (!basis[bit]) { valid = false; break; }
                syndrome ^= basis[bit]; selected ^= combinations[bit];
            }
            if (!valid) continue;
            for (int mask = 1; mask < dimension; ++mask)
                if (selected & (1U << (mask - 1))) coefficients[mask] += 2;
            for (int bit = 0; bit < width; ++bit) {
                const int x = 1 << bit;
                int value = 0;
                for (int mask = 1; mask < dimension; ++mask)
                    if (population(static_cast<unsigned>(x & mask)) & 1) value += coefficients[mask];
                const int correction = residue(truth[x] - value, 8);
                if (correction != 0 && correction != 4) { valid = false; break; }
                coefficients[x] += correction;
            }
            if (!valid) continue;
            for (int x = 0; x < dimension; ++x) {
                int value = 0;
                for (int mask = 1; mask < dimension; ++mask)
                    if (population(static_cast<unsigned>(x & mask)) & 1) value += coefficients[mask];
                if (residue(value, 8) != truth[x]) { valid = false; break; }
            }
            if (valid) { output = coefficients; return true; }
        }
        return false;
    }
};

struct State {
    unsigned rows;
    unsigned available;
    std::vector<unsigned char> word;
};

std::vector<State> advance(const std::vector<State>& previous, const int* controls,
                          const int* targets, int edges) {
    std::vector<State> result;
    std::unordered_map<unsigned, std::size_t> seen;
    for (const auto& state : previous) {
        for (int edge = 0; edge < edges; ++edge) {
            const unsigned control = (state.rows >> (4 * controls[edge])) & 15U;
            const unsigned target = (state.rows >> (4 * targets[edge])) & 15U;
            const unsigned rows = state.rows ^ (control << (4 * targets[edge]));
            const unsigned available = state.available | (1U << ((control ^ target) - 1));
            const unsigned key = rows | (available << 16);
            if (!seen.emplace(key, result.size()).second) continue;
            auto word = state.word;
            word.push_back(static_cast<unsigned char>(edge));
            result.push_back({rows, available, std::move(word)});
        }
    }
    return result;
}
} // namespace

extern "C" int qgd_osr_phase_parity_network(int width, const int* final_rows,
        const int* truth, int budget, const int* controls, const int* targets,
        int edges, int* word, int* coefficients, unsigned long long* statistics) {
    if (width < 3 || width > 4 || budget < 0 || !final_rows || !truth ||
        !controls || !targets || edges < 0 || edges > 12 || !word || !coefficients || !statistics) return -1;
    for (int edge = 0; edge < edges; ++edge)
        if (controls[edge] < 0 || controls[edge] >= width || targets[edge] < 0 ||
            targets[edge] >= width || controls[edge] == targets[edge]) return -1;
    Polynomial polynomial(width, truth);
    if (polynomial.odd_patterns.empty()) return -2;
    unsigned initial = 0, final = 0, initial_available = 0, final_available = 0;
    for (int bit = 0; bit < width; ++bit) {
        if (final_rows[bit] <= 0 || final_rows[bit] >= (1 << width)) return -1;
        initial |= (1U << bit) << (4 * bit);
        final |= static_cast<unsigned>(final_rows[bit]) << (4 * bit);
        initial_available |= 1U << ((1 << bit) - 1);
        final_available |= 1U << (final_rows[bit] - 1);
    }
    std::vector<std::vector<State>> forward{{State{initial, initial_available, {}}}};
    std::vector<std::vector<State>> backward{{State{final, final_available, {}}}};
    struct Cached { signed char status = 0; std::array<int, 16> coefficients{}; };
    std::vector<Cached> cache(1U << ((1 << width) - 1));
    statistics[0] = statistics[1] = statistics[2] = 0;
    for (int count = 0; count <= budget; ++count) {
        const int before = count / 2, after = count - before;
        while (static_cast<int>(forward.size()) <= before)
            forward.push_back(advance(forward.back(), controls, targets, edges));
        while (static_cast<int>(backward.size()) <= after)
            backward.push_back(advance(backward.back(), controls, targets, edges));
        std::unordered_map<unsigned, std::vector<const State*>> endings;
        for (const auto& state : backward[after]) endings[state.rows].push_back(&state);
        for (const auto& start : forward[before]) {
            auto found = endings.find(start.rows);
            if (found == endings.end()) continue;
            for (const auto* end : found->second) {
                ++statistics[1];
                const unsigned available = start.available | end->available;
                auto& entry = cache[available];
                if (entry.status == 0) {
                    ++statistics[2];
                    entry.status = polynomial.solve(available, entry.coefficients) ? 1 : -1;
                }
                if (entry.status < 0) continue;
                int position = 0;
                for (unsigned char edge : start.word) word[position++] = edge;
                for (auto edge = end->word.rbegin(); edge != end->word.rend(); ++edge) word[position++] = *edge;
                for (int mask = 0; mask < (1 << width); ++mask) coefficients[mask] = entry.coefficients[mask];
                for (const auto& level : forward) statistics[0] += level.size();
                for (const auto& level : backward) statistics[0] += level.size();
                return count;
            }
        }
    }
    for (const auto& level : forward) statistics[0] += level.size();
    for (const auto& level : backward) statistics[0] += level.size();
    return -3;
}
