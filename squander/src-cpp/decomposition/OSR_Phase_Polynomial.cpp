#include "include/OSR_Phase_Polynomial.h"

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace {
using Complex = std::complex<double>;
using Local = std::array<Complex, 4>;
using Dense = std::vector<Complex>;
const Local identity{{1.0, 0.0, 0.0, 1.0}};
const std::array<Local, 3> paulis{{
    Local{{0.0, 1.0, 1.0, 0.0}},
    Local{{0.0, Complex(0.0, -1.0), Complex(0.0, 1.0), 0.0}},
    Local{{1.0, 0.0, 0.0, -1.0}}
}};

Local adjoint(const Local& a) {
    return {{std::conj(a[0]), std::conj(a[2]), std::conj(a[1]), std::conj(a[3])}};
}

void left_local(Dense& matrix, int width, int wire, const Local& gate) {
    const int dimension = 1 << width, bit = 1 << wire;
    for (int row = 0; row < dimension; ++row) {
        if (row & bit) continue;
        for (int column = 0; column < dimension; ++column) {
            const Complex a = matrix[row * dimension + column];
            const Complex b = matrix[(row | bit) * dimension + column];
            matrix[row * dimension + column] = gate[0] * a + gate[1] * b;
            matrix[(row | bit) * dimension + column] = gate[2] * a + gate[3] * b;
        }
    }
}

void right_local(Dense& matrix, int width, int wire, const Local& gate) {
    const int dimension = 1 << width, bit = 1 << wire;
    for (int column = 0; column < dimension; ++column) {
        if (column & bit) continue;
        for (int row = 0; row < dimension; ++row) {
            const Complex a = matrix[row * dimension + column];
            const Complex b = matrix[row * dimension + (column | bit)];
            matrix[row * dimension + column] = a * gate[0] + b * gate[2];
            matrix[row * dimension + (column | bit)] = a * gate[1] + b * gate[3];
        }
    }
}

Dense multiply(const Dense& a, const Dense& b, int dimension, bool adjoint_a, bool adjoint_b) {
    Dense result(dimension * dimension);
    for (int row = 0; row < dimension; ++row)
        for (int k = 0; k < dimension; ++k) {
            const Complex value = adjoint_a ? std::conj(a[k * dimension + row]) : a[row * dimension + k];
            for (int column = 0; column < dimension; ++column)
                result[row * dimension + column] += value *
                    (adjoint_b ? std::conj(b[column * dimension + k]) : b[k * dimension + column]);
        }
    return result;
}

double error(const Dense& target, const Dense& candidate, int dimension) {
    Complex overlap(0.0, 0.0);
    for (std::size_t index = 0; index < target.size(); ++index)
        overlap += std::conj(target[index]) * candidate[index];
    if (!std::isfinite(std::abs(overlap)) || std::abs(overlap) == 0.0)
        return std::numeric_limits<double>::infinity();
    const Complex phase = overlap / std::abs(overlap);
    double chord = 0.0;
    for (std::size_t index = 0; index < target.size(); ++index)
        chord += std::norm(candidate[index] - phase * target[index]);
    chord /= dimension;
    // Only used near a unitary certificate, not as a metric for arbitrary data.
    return chord <= 2.0 ? chord * (1.0 - 0.25 * chord)
                        : std::numeric_limits<double>::infinity();
}

Local axis_basis(const Eigen::Vector3d& axis) {
    Eigen::Matrix2cd observable;
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 2; ++column) {
            observable(row, column) = 0.0;
            for (int component = 0; component < 3; ++component)
                observable(row, column) += axis[component] * paulis[component][2 * row + column];
        }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> solver(observable);
    Local result{};
    if (solver.info() != Eigen::Success) return identity;
    for (int row = 0; row < 2; ++row) {
        result[2 * row] = solver.eigenvectors()(row, 1);
        result[2 * row + 1] = solver.eigenvectors()(row, 0);
    }
    return result;
}

std::vector<Local> commuting_bases(const Dense& relative, int width) {
    const int dimension = 1 << width;
    std::vector<Local> result;
    for (int wire = 0; wire < width; ++wire) {
        std::array<Dense, 3> commutators;
        for (int component = 0; component < 3; ++component) {
            Dense a = relative, b = relative;
            right_local(a, width, wire, paulis[component]);
            left_local(b, width, wire, paulis[component]);
            commutators[component].resize(relative.size());
            for (std::size_t index = 0; index < relative.size(); ++index)
                commutators[component][index] = a[index] - b[index];
        }
        Eigen::Matrix3d gram;
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column) {
                double value = 0.0;
                for (std::size_t index = 0; index < relative.size(); ++index)
                    value += std::real(std::conj(commutators[row][index]) * commutators[column][index]);
                gram(row, column) = value / dimension;
            }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(gram);
        if (solver.info() != Eigen::Success) return {};
        // A spectator's null space is degenerate. Keep its computational axis
        // when it already commutes, rather than choosing an arbitrary rotation.
        const double rounding = 128.0 * std::numeric_limits<double>::epsilon();
        const Eigen::Vector3d axis = gram(2, 2) <= rounding
            ? Eigen::Vector3d(0.0, 0.0, 1.0) : Eigen::Vector3d(solver.eigenvectors().col(0));
        result.push_back(axis_basis(axis));
    }
    return result;
}

int expand(int compact, int root, int root_bit, int width) {
    int result = root_bit << root, position = 0;
    for (int wire = 0; wire < width; ++wire)
        if (wire != root) result |= ((compact >> position++) & 1) << wire;
    return result;
}

int parity(unsigned value) {
    int result = 0;
    while (value) { result ^= 1; value &= value - 1; }
    return result;
}

void apply_prefix(Dense& matrix, int width, const squander_osr::LogicalFactor& factor) {
    const int dimension = 1 << width;
    for (const auto& operation : factor.prefix) {
        if (operation.control < 0) left_local(matrix, width, operation.target, operation.matrix);
        else {
            for (int row = 0; row < dimension; ++row) {
                if (!(row & (1 << operation.control)) || (row & (1 << operation.target))) continue;
                const int other = row | (1 << operation.target);
                for (int column = 0; column < dimension; ++column)
                    std::swap(matrix[row * dimension + column], matrix[other * dimension + column]);
            }
        }
    }
}
} // namespace

namespace squander_osr {
LogicalFactor phase_polynomial_factor(const Complex* input, int width, int stride,
        int root, unsigned couplers, double tolerance, int cnot_budget) {
    LogicalFactor best;
    if (!input || width < 3 || width > 4 || stride < (1 << width) ||
        root < 0 || root >= width || cnot_budget < 0 ||
        !std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) return best;
    const int dimension = 1 << width, half = dimension / 2;
    const double quarter = std::acos(-1.0) / 4.0;
    Dense target(dimension * dimension);
    for (int row = 0; row < dimension; ++row)
        for (int column = 0; column < dimension; ++column) {
            target[row * dimension + column] = input[row * stride + column];
            if (!std::isfinite(std::abs(target[row * dimension + column]))) return best;
        }
    std::vector<int> controls, targets;
    for (int control = 0; control < width; ++control)
        for (int wire = 0; wire < width; ++wire) {
            if (control == wire) continue;
            const int a = std::min(control, wire), b = std::max(control, wire);
            if (couplers & (1U << (a * width + b))) {
                controls.push_back(control); targets.push_back(wire);
            }
        }
    std::vector<int> active_input;
    for (int wire = 0; wire < width; ++wire) if (wire != root) active_input.push_back(wire);
    for (int output_root = 0; output_root < width; ++output_root) {
        Eigen::Matrix3d channel;
        for (int in = 0; in < 3; ++in)
            for (int out = 0; out < 3; ++out) {
                Dense transformed = target;
                right_local(transformed, width, root, paulis[in]);
                left_local(transformed, width, output_root, paulis[out]);
                Complex value(0.0, 0.0);
                for (std::size_t index = 0; index < target.size(); ++index)
                    value += std::conj(target[index]) * transformed[index];
                channel(out, in) = value.real() / dimension;
            }
        Eigen::JacobiSVD<Eigen::Matrix3d> singular(channel, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // Screening only: the actual acceptance uses the complete raw matrix.
        if (singular.info() != Eigen::Success ||
            singular.singularValues()[0] < 1.0 - 8.0 * std::sqrt(tolerance)) continue;
        const Local incoming = axis_basis(singular.matrixV().col(0));
        const Local outgoing = axis_basis(singular.matrixU().col(0));
        Dense controlled = target;
        right_local(controlled, width, root, incoming);
        left_local(controlled, width, output_root, adjoint(outgoing));
        Dense first(half * half), second(half * half);
        for (int row = 0; row < half; ++row)
            for (int column = 0; column < half; ++column) {
                first[row * half + column] = controlled[
                    expand(row, output_root, 0, width) * dimension + expand(column, root, 0, width)];
                second[row * half + column] = controlled[
                    expand(row, output_root, 1, width) * dimension + expand(column, root, 1, width)];
            }
        const auto input_bases = commuting_bases(multiply(first, second, half, true, false), width - 1);
        const auto output_bases = commuting_bases(multiply(second, first, half, false, true), width - 1);
        if (input_bases.size() != std::size_t(width - 1) ||
            output_bases.size() != std::size_t(width - 1)) continue;
        std::vector<int> active_output;
        for (int wire = 0; wire < width; ++wire) if (wire != output_root) active_output.push_back(wire);
        Dense canonical = controlled;
        for (int wire = 0; wire < width - 1; ++wire) {
            right_local(canonical, width, active_input[wire], input_bases[wire]);
            left_local(canonical, width, active_output[wire], adjoint(output_bases[wire]));
        }
        std::vector<int> permutation(dimension), used(dimension, 0);
        std::vector<Complex> phases(dimension);
        Dense projection(dimension * dimension);
        bool valid = true;
        for (int column = 0; column < dimension; ++column) {
            int row = 0;
            for (int next = 1; next < dimension; ++next)
                if (std::norm(canonical[next * dimension + column]) >
                    std::norm(canonical[row * dimension + column])) row = next;
            if (used[row]++ || std::abs(canonical[row * dimension + column]) == 0.0) { valid = false; break; }
            permutation[column] = row;
            phases[column] = canonical[row * dimension + column] / std::abs(canonical[row * dimension + column]);
            projection[row * dimension + column] = phases[column];
        }
        if (!valid || error(canonical, projection, dimension) > tolerance) continue;
        const int offset = permutation[0];
        std::array<int, 4> columns{}, rows{};
        for (int bit = 0; bit < width; ++bit) columns[bit] = permutation[1 << bit] ^ offset;
        for (int x = 0; x < dimension; ++x) {
            int predicted = offset;
            for (int bit = 0; bit < width; ++bit) if (x & (1 << bit)) predicted ^= columns[bit];
            if (predicted != permutation[x]) valid = false;
        }
        if (!valid) continue;
        unsigned free = 0;
        for (int wire = 0; wire < width; ++wire) {
            for (int bit = 0; bit < width; ++bit) rows[wire] |= ((columns[bit] >> wire) & 1) << bit;
            free |= 1U << ((1 << wire) - 1);
            free |= 1U << (rows[wire] - 1);
        }
        std::array<double, 16> continuous{};
        for (int mask = 1; mask < dimension; ++mask) {
            if (!(free & (1U << (mask - 1)))) continue;
            double coefficient = 0.0;
            for (int x = 0; x < dimension; ++x)
                coefficient += (parity(mask & x) ? -1.0 : 1.0) * std::arg(phases[x] / phases[0]);
            coefficient *= -2.0 / dimension;
            continuous[mask] = coefficient - quarter * std::floor(coefficient / quarter + 0.5);
        }
        std::array<int, 16> truth{};
        double phase_chord = 0.0;
        for (int x = 0; x < dimension; ++x) {
            Complex corrected = phases[x] / phases[0];
            for (int mask = 1; mask < dimension; ++mask)
                if (parity(mask & x)) corrected *= std::exp(Complex(0.0, -continuous[mask]));
            const int value = static_cast<int>(std::llround(std::arg(corrected) / quarter));
            truth[x] = (value % 8 + 8) % 8;
            phase_chord += std::norm(corrected - std::exp(Complex(0.0, quarter * truth[x])));
        }
        if (phase_chord / dimension > tolerance) continue;
        const int budget = best.valid ? std::min(cnot_budget, best.prefix_cnots - 1) : cnot_budget;
        if (budget < 0) break;
        std::vector<int> word(std::max(1, budget));
        std::array<int, 16> coefficients{};
        unsigned long long statistics[3]{};
        int empty = 0;
        const int count = qgd_osr_phase_parity_network(width, rows.data(), truth.data(), budget,
            controls.empty() ? &empty : controls.data(), targets.empty() ? &empty : targets.data(),
            int(controls.size()), word.data(), coefficients.data(), statistics);
        if (count < 0) continue;
        LogicalFactor factor;
        factor.root = root;
        factor.prefix_cnots = count;
        factor.local_adjoint = identity;
        factor.child_adjoint.resize(half * half);
        for (int index = 0; index < half; ++index) factor.child_adjoint[index * half + index] = 1.0;
        auto local = [&](int wire, const Local& gate) { factor.prefix.push_back({wire, -1, gate}); };
        local(root, adjoint(incoming));
        for (int wire = 0; wire < width - 1; ++wire) local(active_input[wire], adjoint(input_bases[wire]));
        std::array<int, 4> current{};
        unsigned applied = 0;
        for (int wire = 0; wire < width; ++wire) current[wire] = 1 << wire;
        auto apply_phases = [&]() {
            for (int wire = 0; wire < width; ++wire) {
                const int mask = current[wire];
                if (applied & (1U << (mask - 1))) continue;
                applied |= 1U << (mask - 1);
                const double angle = quarter * coefficients[mask] + continuous[mask];
                if (angle != 0.0) local(wire, {{1.0, 0.0, 0.0, std::exp(Complex(0.0, angle))}});
            }
        };
        apply_phases();
        for (int index = 0; index < count; ++index) {
            const int edge = word[index];
            factor.prefix.push_back({targets[edge], controls[edge], Local{}});
            current[targets[edge]] ^= current[controls[edge]];
            apply_phases();
        }
        const Local x{{0.0, 1.0, 1.0, 0.0}};
        for (int wire = 0; wire < width; ++wire) if (offset & (1 << wire)) local(wire, x);
        for (int wire = 0; wire < width - 1; ++wire) local(active_output[wire], output_bases[wire]);
        local(output_root, outgoing);
        Dense rebuilt(dimension * dimension);
        for (int index = 0; index < dimension; ++index) rebuilt[index * dimension + index] = 1.0;
        apply_prefix(rebuilt, width, factor);
        factor.process_infidelity = error(target, rebuilt, dimension);
        if (std::isfinite(factor.process_infidelity) && factor.process_infidelity <= tolerance) {
            factor.valid = true;
            best = std::move(factor);
        }
    }
    return best;
}
} // namespace squander_osr

extern "C" int qgd_osr_phase_polynomial_factor(
        const double* input, int width, int stride, int root, unsigned couplers,
        double tolerance, int budget, int capacity, int* controls, int* targets,
        double* matrices, double* process_infidelity) {
    if (!input || width < 3 || width > 4 || stride < (1 << width) || capacity < 0 ||
        !controls || !targets || !matrices || !process_infidelity) return -1;
    const int dimension = 1 << width;
    std::vector<std::complex<double>> matrix(dimension * dimension);
    for (int row = 0; row < dimension; ++row)
        for (int column = 0; column < dimension; ++column)
            matrix[row * dimension + column] = {input[2 * (row * stride + column)], input[2 * (row * stride + column) + 1]};
    const auto factor = squander_osr::phase_polynomial_factor(
        matrix.data(), width, dimension, root, couplers, tolerance, budget);
    if (!factor.valid) return -2;
    if (factor.prefix.size() > std::size_t(capacity)) return -3;
    for (std::size_t index = 0; index < factor.prefix.size(); ++index) {
        controls[index] = factor.prefix[index].control;
        targets[index] = factor.prefix[index].target;
        for (int entry = 0; entry < 4; ++entry) {
            matrices[8 * index + 2 * entry] = factor.prefix[index].matrix[entry].real();
            matrices[8 * index + 2 * entry + 1] = factor.prefix[index].matrix[entry].imag();
        }
    }
    *process_infidelity = factor.process_infidelity;
    return int(factor.prefix.size());
}
