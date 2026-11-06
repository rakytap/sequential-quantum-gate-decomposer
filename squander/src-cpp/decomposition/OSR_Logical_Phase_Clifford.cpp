#include "include/OSR_Logical_Phase_Clifford.h"
#include "include/OSR_Parity_Network.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace squander_osr {
namespace {
using Complex = std::complex<double>;
using Local = std::array<Complex, 4>;

void local(std::vector<LogicalFactorOperation>& operations, int wire,
           const Local& matrix) {
    LogicalFactorOperation operation;
    operation.target = wire;
    operation.matrix = matrix;
    operations.push_back(operation);
}

void cnot(std::vector<LogicalFactorOperation>& operations, int control, int target) {
    LogicalFactorOperation operation;
    operation.control = control;
    operation.target = target;
    operations.push_back(operation);
}

Local rz(double half_angle) {
    return {{std::polar(1.0, -half_angle), Complex(0), Complex(0),
             std::polar(1.0, half_angle)}};
}

std::vector<Complex> prefix_matrix(
        const std::vector<LogicalFactorOperation>& operations, int width) {
    const int dimension = 1 << width;
    std::vector<Complex> result(dimension * dimension);
    for (int index = 0; index < dimension; ++index)
        result[index * dimension + index] = 1.0;
    for (const auto& operation : operations) {
        const int bit = 1 << operation.target;
        for (int row = 0; row < dimension; ++row) {
            if (row & bit) continue;
            if (operation.control >= 0 && !(row & (1 << operation.control))) continue;
            for (int column = 0; column < dimension; ++column) {
                auto& first = result[row * dimension + column];
                auto& second = result[(row | bit) * dimension + column];
                if (operation.control >= 0) std::swap(first, second);
                else {
                    const Complex a = first, b = second;
                    first = operation.matrix[0] * a + operation.matrix[1] * b;
                    second = operation.matrix[2] * a + operation.matrix[3] * b;
                }
            }
        }
    }
    return result;
}

unsigned body_mask(unsigned physical, const std::vector<int>& active) {
    unsigned result = 0;
    for (unsigned index = 0; index < active.size(); ++index)
        if (physical & (1U << active[index])) result |= 1U << index;
    return result;
}

void compile_residue(const Complex* target, int width, int stride, int root,
                     unsigned couplers, double tolerance, int budget,
                     const LogicalFactor& original, double rounding,
                     const std::function<void(LogicalFactor&&)>& emit) {
    const int dimension = 1 << width, half = dimension / 2;
    std::vector<int> active;
    for (int wire = 0; wire < width; ++wire)
        if (wire != root) active.push_back(wire);
    std::vector<int> wires{root};
    wires.insert(wires.end(), active.begin(), active.end());
    std::vector<unsigned> rows(width);
    for (int wire = 0; wire < width; ++wire) rows[wire] = 1U << wire;
    std::vector<LogicalFactorOperation> leading;
    std::vector<double> coefficients(half, 0.0);
    bool network_started = false, extracted = false;
    unsigned cz_mask = 0;
    double root_phase = 0.0;
    const double quarter = std::acos(-1.0) / 4.0;
    const double epsilon = 64.0 * std::numeric_limits<double>::epsilon();
    for (const auto& operation : original.prefix) {
        if (operation.control >= 0) {
            network_started = true;
            rows[operation.target] ^= rows[operation.control];
            continue;
        }
        if (!network_started) {
            leading.push_back(operation);
            continue;
        }
        if (std::abs(operation.matrix[1]) > epsilon ||
            std::abs(operation.matrix[2]) > epsilon) return;
        const unsigned parity = rows[operation.target];
        if (!(parity & (1U << root))) return;
        const unsigned mask = body_mask(parity, active);
        const double angle = 0.5 * std::arg(operation.matrix[3] *
                                           std::conj(operation.matrix[0]));
        if (mask && std::abs(std::abs(angle) - quarter) <= rounding) {
            extracted = true;
            cz_mask ^= mask;
            root_phase += angle > 0.0 ? 2.0 * quarter : -2.0 * quarter;
        } else coefficients[mask] += angle;
    }
    if (!extracted) return;
    int cz_count = 0;
    for (unsigned index = 0; index < active.size(); ++index) {
        if (!(cz_mask & (1U << index))) continue;
        const int a = std::min(root, active[index]), b = std::max(root, active[index]);
        if (!(couplers & (1U << (a * width + b)))) return;
        ++cz_count;
    }
    if (cz_count > budget) return;
    unsigned logical_couplers = 0;
    for (int a = 0; a < width; ++a)
        for (int b = a + 1; b < width; ++b) {
            const int low = std::min(wires[a], wires[b]), high = std::max(wires[a], wires[b]);
            if (couplers & (1U << (low * width + high)))
                logical_couplers |= 1U << (a * width + b);
        }
    unsigned request = 0;
    for (int mask = 1; mask < half; ++mask)
        if (coefficients[mask] != 0.0) request |= 1U << (mask - 1);
    const auto networks = parity_networks(width, request, logical_couplers, budget - cz_count);
    for (const auto& network : networks) {
        auto operations = leading;
        local(operations, root, {{Complex(1), Complex(0), Complex(0),
                                  std::polar(1.0, root_phase)}});
        const double h = 1.0 / std::sqrt(2.0);
        const Local hadamard{{h, h, h, -h}};
        for (unsigned index = 0; index < active.size(); ++index) {
            if (!(cz_mask & (1U << index))) continue;
            local(operations, root, hadamard);
            cnot(operations, active[index], root);
            local(operations, root, hadamard);
        }
        if (coefficients[0] != 0.0) local(operations, root, rz(coefficients[0]));
        std::vector<unsigned> logical_rows(width);
        for (int wire = 0; wire < width; ++wire) logical_rows[wire] = 1U << wire;
        unsigned emitted = 0;
        for (const auto& move : network.moves) {
            cnot(operations, wires[move.first], wires[move.second]);
            logical_rows[move.second] ^= logical_rows[move.first];
            const unsigned parity = logical_rows[move.second];
            if (!(parity & 1U) || parity == 1U) continue;
            const unsigned mask = parity >> 1;
            const unsigned term = 1U << (mask - 1);
            if ((request & term) && !(emitted & term)) {
                local(operations, wires[move.second], rz(coefficients[mask]));
                emitted |= term;
            }
        }
        if (emitted != request) continue;
        const auto prefix = prefix_matrix(operations, width);
        std::vector<Complex> remainder(dimension * dimension);
        for (int row = 0; row < dimension; ++row)
            for (int column = 0; column < dimension; ++column)
                for (int index = 0; index < dimension; ++index)
                    remainder[row * dimension + column] +=
                        target[row * stride + index] * std::conj(prefix[column * dimension + index]);
        std::vector<int> neighbors(active.size(), root);
        auto factor = logical_factor(remainder.data(), width, dimension, root,
                                     active.data(), neighbors.data(), couplers, tolerance, 0);
        if (!factor.valid || !factor.prefix.empty()) continue;
        factor.prefix = std::move(operations);
        factor.prefix_cnots = cz_count + static_cast<int>(network.moves.size());
        emit(std::move(factor));
    }
}
} // namespace

void logical_phase_clifford_factors(
        const Complex* target, int width, int stride, int root,
        unsigned couplers, double tolerance, int budget,
        const std::function<void(LogicalFactor&&)>& emit) {
    if (width < 2 || width > 4 || root < 0 || root >= width || !target ||
        stride < (1 << width) || budget < 0 || !std::isfinite(tolerance) ||
        tolerance <= 0.0 || tolerance >= 1.0) return;
    // A constructive ceiling, not a synthesis cap: compute/uncompute each
    // parity separately, routing each remote CNOT with SWAPs and restoring
    // the intermediate wires. Every shortest parity program fits below it.
    const int controls = width - 1;
    const int remote_cnot_cost = 6 * (width - 2) + 1;
    const int constructive_ceiling = controls * (1 << controls) * remote_cnot_cost;
    logical_phase_factors(target, width, stride, root, couplers, tolerance,
        std::max(budget, constructive_ceiling), [&](LogicalFactor&& original) {
            const double exact = 64.0 * std::numeric_limits<double>::epsilon();
            compile_residue(target, width, stride, root, couplers, tolerance,
                            budget, original, exact, emit);
            const double relaxed = std::max(exact, std::asin(std::sqrt(tolerance)) / (1 << width));
            if (relaxed > exact)
                compile_residue(target, width, stride, root, couplers, tolerance,
                                budget, original, relaxed, emit);
        });
}

} // namespace squander_osr
