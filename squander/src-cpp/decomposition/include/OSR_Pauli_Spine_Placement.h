#ifndef SQUANDER_OSR_PAULI_SPINE_PLACEMENT_H
#define SQUANDER_OSR_PAULI_SPINE_PLACEMENT_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace squander_osr {

struct PauliEncoderRequirement {
    int minimum_cnots = 0;
    bool single_crossing_possible = true;
    double rank_one_tail_lower_bound = 0.0;
    double local_pauli_distance = 0.0;
    double spectator_distance = 0.0;
    double spine_distance = 0.0;
    double projection_threshold = 0.0;
};

struct PauliSpinePlacement {
    PauliEncoderRequirement before;
    PauliEncoderRequirement after;
};

namespace pauli_spine_detail {

using Complex = std::complex<double>;

inline int insert_bit(int index, int wire, int bit) {
    const int low = (1 << wire) - 1;
    return (index & low) | ((index & ~low) << 1) | (bit << wire);
}

template <class Reader>
std::vector<std::vector<Complex>> interaction_columns(
    int width, int root, const Reader& read
) {
    const int dimension = 1 << (width - 1);
    std::vector<std::vector<Complex>> blocks(
        4, std::vector<Complex>(dimension * dimension)
    );
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            auto& block = blocks[2 * a + b];
            for (int row = 0; row < dimension; ++row) {
                for (int col = 0; col < dimension; ++col) {
                    block[row * dimension + col] = read(
                        insert_bit(row, root, a), insert_bit(col, root, b)
                    );
                }
            }
        }
    }
    std::vector<std::vector<Complex>> columns(
        16, std::vector<Complex>(dimension * dimension)
    );
    const double scale = 0.5 / std::sqrt(static_cast<double>(dimension));
    for (int first = 0; first < 4; ++first) {
        for (int second = 0; second < 4; ++second) {
            auto& column = columns[4 * first + second];
            Complex trace(0.0, 0.0);
            for (int row = 0; row < dimension; ++row) {
                for (int col = 0; col < dimension; ++col) {
                    Complex value(0.0, 0.0);
                    for (int inner = 0; inner < dimension; ++inner) {
                        value += blocks[first][row * dimension + inner] *
                            std::conj(blocks[second][col * dimension + inner]);
                    }
                    column[row * dimension + col] = value;
                    if (row == col) trace += value;
                }
            }
            trace /= dimension;
            for (int row = 0; row < dimension; ++row) {
                column[row * dimension + row] -= trace;
            }
            for (auto& value : column) value *= scale;
        }
    }
    return columns;
}

template <class Reader>
PauliEncoderRequirement encoder_requirement(
    int width, int root, double tolerance, const Reader& read
) {
    const int body_width = width - 1;
    const int dimension = 1 << body_width;
    const int rest_dimension = dimension / 2;
    const auto columns = interaction_columns(width, root, read);
    std::vector<double> norms(columns.size(), 0.0);
    double total = 0.0;
    for (size_t idx = 0; idx < columns.size(); ++idx) {
        for (const auto& value : columns[idx]) norms[idx] += std::norm(value);
        total += norms[idx];
    }

    // For a single crossing CNOT these columns have rank at most one.
    // e_2(C^*C)/tr(C^*C) lower-bounds its rank-one tail. Evaluate the
    // column-pair wedge areas as squared residuals, not by subtraction.
    double wedge_sum = 0.0;
    for (size_t first = 0; first < columns.size(); ++first) {
        for (size_t second = first + 1; second < columns.size(); ++second) {
            const size_t pivot = norms[first] >= norms[second] ? first : second;
            const size_t other = pivot == first ? second : first;
            if (norms[pivot] == 0.0) continue;
            Complex inner(0.0, 0.0);
            for (size_t idx = 0; idx < columns[pivot].size(); ++idx) {
                inner += std::conj(columns[pivot][idx]) * columns[other][idx];
            }
            const Complex coefficient = inner / norms[pivot];
            double residual = 0.0;
            for (size_t idx = 0; idx < columns[pivot].size(); ++idx) {
                residual += std::norm(
                    columns[other][idx] - coefficient * columns[pivot][idx]
                );
            }
            wedge_sum += norms[pivot] * residual;
        }
    }

    double local_distance = std::numeric_limits<double>::infinity();
    double spectator_distance = local_distance;
    double spine_distance = local_distance;
    for (int wire = 0; wire < body_width; ++wire) {
        double identity_energy = 0.0;
        double traceless_energy = 0.0;
        double local_residual = 0.0;
        for (const auto& column : columns) {
            Complex partial[4]{};
            for (int rest = 0; rest < rest_dimension; ++rest) {
                for (int a = 0; a < 2; ++a) {
                    for (int b = 0; b < 2; ++b) {
                        partial[2 * a + b] += column[
                            insert_bit(rest, wire, a) * dimension +
                            insert_bit(rest, wire, b)
                        ];
                    }
                }
            }
            const Complex half_trace = (partial[0] + partial[3]) * 0.5;
            partial[0] -= half_trace;
            partial[3] -= half_trace;
            for (auto& value : partial) value /= rest_dimension;
            for (int row = 0; row < rest_dimension; ++row) {
                for (int col = 0; col < rest_dimension; ++col) {
                    const Complex diagonal_zero = column[
                        insert_bit(row, wire, 0) * dimension + insert_bit(col, wire, 0)
                    ];
                    const Complex diagonal_one = column[
                        insert_bit(row, wire, 1) * dimension + insert_bit(col, wire, 1)
                    ];
                    identity_energy += 0.5 * std::norm(diagonal_zero + diagonal_one);
                    traceless_energy += 0.5 * std::norm(diagonal_zero - diagonal_one);
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            const Complex value = column[
                                insert_bit(row, wire, a) * dimension +
                                insert_bit(col, wire, b)
                            ];
                            if (a != b) traceless_energy += std::norm(value);
                            const Complex projected = row == col ? partial[2 * a + b] :
                                Complex(0.0, 0.0);
                            local_residual += std::norm(value - projected);
                        }
                    }
                }
            }
        }
        local_distance = std::min(local_distance, local_residual);
        spectator_distance = std::min(spectator_distance, traceless_energy);
        spine_distance = std::min(spine_distance,
                                 std::min(identity_energy, traceless_energy));
    }

    // C(U) is 2-Lipschitz in normalized unitary Frobenius distance.
    // e = 2T/(1+sqrt(1-T)) converts process infidelity T without cancellation.
    // Include rounding at amplitude level before squaring; this is a guard
    // on a necessary bound, never a relaxation of final certification.
    const double distance = 2.0 * tolerance / (1.0 + std::sqrt(1.0 - tolerance));
    const double rounding = 64.0 * std::numeric_limits<double>::epsilon() *
        dimension * dimension * dimension;
    const double radius = 2.0 * std::sqrt(distance) + rounding;
    PauliEncoderRequirement result;
    result.projection_threshold = radius * radius;
    result.rank_one_tail_lower_bound = total > 0.0 ? wedge_sum / total : 0.0;
    result.single_crossing_possible =
        result.rank_one_tail_lower_bound <= result.projection_threshold;
    result.local_pauli_distance = local_distance;
    result.spectator_distance = spectator_distance;
    result.spine_distance = spine_distance;
    if (local_distance > result.projection_threshold) result.minimum_cnots = 1;
    if (spectator_distance > result.projection_threshold) {
        result.minimum_cnots = std::max(result.minimum_cnots, body_width - 1);
    }
    // With at most m-1 body CNOTs an encoded local Pauli has either an
    // identity spectator or a traceless original-qubit spine. Violating
    // both alternatives requires at least m body CNOTs.
    if (spine_distance > result.projection_threshold) {
        result.minimum_cnots = std::max(result.minimum_cnots, body_width);
    }
    return result;
}

} // namespace pauli_spine_detail

template <class Reader>
std::vector<PauliSpinePlacement> pauli_spine_placements(
    int width, double tolerance, const Reader& desired_unitary
) {
    if (width < 2 || width > 4 || !std::isfinite(tolerance) ||
        tolerance <= 0.0 || tolerance >= 1.0) return {};
    std::vector<PauliSpinePlacement> result(width);
    const auto adjoint = [&](int row, int col) {
        return std::conj(desired_unitary(col, row));
    };
    for (int wire = 0; wire < width; ++wire) {
        result[wire].before = pauli_spine_detail::encoder_requirement(
            width, wire, tolerance, adjoint
        );
        result[wire].after = pauli_spine_detail::encoder_requirement(
            width, wire, tolerance, desired_unitary
        );
    }
    return result;
}

inline bool admits_pauli_spine_placement(
    const std::vector<int>& word, const std::vector<unsigned>& supports,
    const std::vector<PauliSpinePlacement>& requirements
) {
    for (size_t wire = 0; wire < requirements.size(); ++wire) {
        int crossings = 0;
        int position = -1;
        for (size_t idx = 0; idx < word.size(); ++idx) {
            if (supports[word[idx]] & (unsigned(1) << wire)) {
                ++crossings;
                position = static_cast<int>(idx);
            }
        }
        if (crossings != 1) continue;
        const auto& requirement = requirements[wire];
        if (!requirement.before.single_crossing_possible ||
            !requirement.after.single_crossing_possible ||
            position < requirement.before.minimum_cnots ||
            static_cast<int>(word.size()) - position - 1 < requirement.after.minimum_cnots) {
            return false;
        }
    }
    return true;
}

} // namespace squander_osr

#endif
