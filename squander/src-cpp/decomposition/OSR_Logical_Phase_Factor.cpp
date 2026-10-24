#include "include/OSR_Logical_Factor.h"
#include "include/OSR_Parity_Network.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace squander_osr {
namespace {
using Complex = std::complex<double>;
using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 16, 16>;
using Body = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 8, 8>;
using Local = Eigen::Matrix<Complex, 2, 2, Eigen::RowMajor>;
const Complex imaginary(0.0, 1.0);

Local pauli(int axis) {
    Local result = Local::Zero();
    if (axis == 0) result(0, 1) = result(1, 0) = 1.0;
    else if (axis == 1) { result(0, 1) = -imaginary; result(1, 0) = imaginary; }
    else { result(0, 0) = 1.0; result(1, 1) = -1.0; }
    return result;
}

Matrix expanded(const Local& local, int wire, int width) {
    const int dimension = 1 << width, bit = 1 << wire;
    Matrix result = Matrix::Zero(dimension, dimension);
    for (int base = 0; base < dimension; ++base) {
        if (base & bit) continue;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) result(base | (a << wire), base | (b << wire)) = local(a, b);
    }
    return result;
}

int position(int index, const std::vector<int>& wires) {
    int result = 0;
    for (std::size_t q = 0; q < wires.size(); ++q) if (index & (1 << q)) result |= 1 << wires[q];
    return result;
}

Local align_z(const Local& axis) {
    Eigen::SelfAdjointEigenSolver<Local> solver(axis);
    if (solver.info() != Eigen::Success) return Local::Zero();
    Local result;
    result.row(0) = solver.eigenvectors().col(1).adjoint();
    result.row(1) = solver.eigenvectors().col(0).adjoint();
    return result;
}

Body reduced(const Body& matrix, const std::vector<int>& active, int width) {
    std::vector<int> spectators;
    for (int wire = 0; wire < width; ++wire)
        if (std::find(active.begin(), active.end(), wire) == active.end()) spectators.push_back(wire);
    const int dimension = 1 << active.size(), copies = 1 << spectators.size();
    Body result = Body::Zero(dimension, dimension);
    for (int s = 0; s < copies; ++s)
        for (int row = 0; row < dimension; ++row)
            for (int col = 0; col < dimension; ++col)
                result(row, col) += matrix(position(row, active) | position(s, spectators),
                                          position(col, active) | position(s, spectators)) / double(copies);
    return result;
}

double support_error(const Body& matrix, const Body& small,
                     const std::vector<int>& active, int width) {
    std::vector<int> spectators;
    for (int wire = 0; wire < width; ++wire)
        if (std::find(active.begin(), active.end(), wire) == active.end()) spectators.push_back(wire);
    Body projected = Body::Zero(matrix.rows(), matrix.cols());
    for (int s = 0; s < (1 << spectators.size()); ++s)
        for (int row = 0; row < small.rows(); ++row)
            for (int col = 0; col < small.cols(); ++col)
                projected(position(row, active) | position(s, spectators),
                          position(col, active) | position(s, spectators)) = small(row, col);
    return (matrix - projected).squaredNorm() / matrix.rows();
}

void append_local(LogicalFactor& output, const Local& matrix, int wire) {
    LogicalFactorOperation operation;
    operation.target = wire;
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) operation.matrix[2 * row + col] = matrix(row, col);
    output.prefix.push_back(operation);
}

void append_rz(LogicalFactor& output, double angle, int wire) {
    Local matrix = Local::Zero();
    matrix(0, 0) = std::exp(-0.5 * imaginary * angle);
    matrix(1, 1) = std::exp(0.5 * imaginary * angle);
    append_local(output, matrix, wire);
}

unsigned parity(unsigned value) {
    unsigned result = 0;
    for (; value; value &= value - 1) result ^= 1U;
    return result;
}

Matrix prefix_matrix(const LogicalFactor& factor, int width) {
    const int dimension = 1 << width;
    Matrix result = Matrix::Identity(dimension, dimension);
    for (const auto& operation : factor.prefix) {
        if (operation.control < 0) {
            Local local;
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col) local(row, col) = operation.matrix[2 * row + col];
            result = (expanded(local, operation.target, width) * result).eval();
        } else {
            for (int row = 0; row < dimension; ++row)
                if ((row & (1 << operation.control)) && !(row & (1 << operation.target)))
                    result.row(row).swap(result.row(row | (1 << operation.target)));
        }
    }
    return result;
}
} // namespace

void logical_phase_factors(const Complex* data, int width, int stride, int root,
                          unsigned couplers, double tolerance, int cnot_budget,
                          const std::function<void(LogicalFactor&&)>& emit) {
    if (!data || !emit || width < 2 || width > 4 || root < 0 || root >= width ||
        stride < (1 << width) || !std::isfinite(tolerance) || tolerance <= 0.0 ||
        tolerance >= 1.0 || cnot_budget < 0) return;
    const int dimension = 1 << width, half = dimension / 2, body_width = width - 1;
    const double pi = std::acos(-1.0), epsilon = std::numeric_limits<double>::epsilon();
    Matrix target(dimension, dimension);
    for (int row = 0; row < dimension; ++row)
        for (int col = 0; col < dimension; ++col) target(row, col) = data[row * stride + col];
    if (!target.allFinite() || (target.adjoint() * target - Matrix::Identity(dimension, dimension)).norm() >
        64.0 * epsilon * dimension * dimension * dimension) return;
    const Matrix inverse = target.adjoint();
    std::array<Matrix, 3> images;
    Eigen::Matrix3d transfer;
    for (int b = 0; b < 3; ++b) {
        images[b] = inverse * expanded(pauli(b), root, width) * target;
        for (int a = 0; a < 3; ++a)
            transfer(a, b) = (expanded(pauli(a), root, width).conjugate().cwiseProduct(images[b])).sum().real() / dimension;
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> root_svd(transfer, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (root_svd.info() != Eigen::Success) return;
    const Eigen::Vector3d source_axis = root_svd.matrixV().col(0), target_axis = root_svd.matrixU().col(0);
    Local source_pauli = Local::Zero(), target_pauli = Local::Zero();
    Matrix difference = Matrix::Zero(dimension, dimension);
    for (int axis = 0; axis < 3; ++axis) {
        source_pauli += source_axis(axis) * pauli(axis);
        target_pauli += target_axis(axis) * pauli(axis);
        difference += source_axis(axis) * images[axis];
    }
    difference -= expanded(target_pauli, root, width);
    if (difference.squaredNorm() / dimension > 4.0 * tolerance) return;
    const Local source_frame = align_z(source_pauli), target_frame = align_z(target_pauli);
    const Matrix aligned = expanded(target_frame, root, width) * inverse *
                           expanded(source_frame.adjoint(), root, width);
    std::vector<int> outside;
    for (int wire = 0; wire < width; ++wire) if (wire != root) outside.push_back(wire);
    Body plus(half, half), minus(half, half);
    for (int row = 0; row < half; ++row)
        for (int col = 0; col < half; ++col) {
            const int a = position(row, outside), b = position(col, outside);
            plus(row, col) = aligned(a, b);
            minus(row, col) = aligned(a | (1 << root), b | (1 << root));
        }
    const Body relative = minus * plus.adjoint();
    std::vector<bool> supported(1 << body_width, false);
    // C++14's default allocator need not satisfy Eigen's AVX alignment.
    std::vector<Body, Eigen::aligned_allocator<Body>> reductions(1 << body_width);
    for (int mask = 0; mask < (1 << body_width); ++mask) {
        std::vector<int> active;
        for (int wire = 0; wire < body_width; ++wire) if (mask & (1 << wire)) active.push_back(wire);
        reductions[mask] = reduced(relative, active, body_width);
        supported[mask] = support_error(relative, reductions[mask], active, body_width) <= 4.0 * tolerance;
    }
    std::vector<int> wires{root};
    wires.insert(wires.end(), outside.begin(), outside.end());
    unsigned compact_couplers = 0;
    for (int a = 0; a < width; ++a)
        for (int b = a + 1; b < width; ++b) {
            const int lo = std::min(wires[a], wires[b]), hi = std::max(wires[a], wires[b]);
            if (couplers & (1U << (lo * width + hi))) compact_couplers |= 1U << (a * width + b);
        }
    std::map<unsigned, std::vector<ParityNetwork>> books;
    for (int mask = 1; mask < (1 << body_width); ++mask) {
        if (!supported[mask]) continue;
        bool minimal = true;
        for (int subset = 0; subset < mask; ++subset)
            if ((subset & mask) == subset && supported[subset]) minimal = false;
        if (!minimal) continue;
        std::vector<int> active;
        for (int wire = 0; wire < body_width; ++wire) if (mask & (1 << wire)) active.push_back(wire);
        const int count = static_cast<int>(active.size()), small_dimension = 1 << count;
        Eigen::JacobiSVD<Body> polar(reductions[mask], Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (polar.info() != Eigen::Success) continue;
        const Body unitary = polar.matrixU() * polar.matrixV().adjoint();
        std::vector<Local, Eigen::aligned_allocator<Local>> frames(
            body_width, Local::Identity());
        bool valid = true;
        for (int wire = 0; wire < count; ++wire) {
            std::array<Matrix, 3> commutators;
            Eigen::Matrix3d gram;
            for (int axis = 0; axis < 3; ++axis) {
                const Matrix local_axis = expanded(pauli(axis), wire, count);
                commutators[axis] = local_axis * unitary - unitary * local_axis;
            }
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    gram(a, b) = (commutators[a].conjugate().cwiseProduct(commutators[b])).sum().real();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> axes(gram);
            if (axes.info() != Eigen::Success) { valid = false; break; }
            const Eigen::Vector3d coefficients = axes.eigenvectors().col(0);
            Matrix error = Matrix::Zero(small_dimension, small_dimension);
            Local axis = Local::Zero();
            for (int a = 0; a < 3; ++a) {
                error += coefficients(a) * commutators[a];
                axis += coefficients(a) * pauli(a);
            }
            if (error.squaredNorm() / small_dimension > 4.0 * tolerance) { valid = false; break; }
            frames[active[wire]] = align_z(axis);
        }
        if (!valid) continue;
        Matrix frame = Matrix::Identity(small_dimension, small_dimension);
        for (int q = 0; q < count; ++q) frame = (expanded(frames[active[q]], q, count) * frame).eval();
        const Matrix diagonalized = frame * unitary * frame.adjoint();
        double off_diagonal = 0.0;
        for (int row = 0; row < small_dimension; ++row)
            for (int col = 0; col < small_dimension; ++col)
                if (row != col) off_diagonal += std::norm(diagonalized(row, col));
        if (off_diagonal / small_dimension > 4.0 * tolerance) continue;
        std::vector<double> phases(half);
        for (int index = 0; index < half; ++index) {
            int compact = 0;
            for (int q = 0; q < count; ++q) if (index & (1 << active[q])) compact |= 1 << q;
            phases[index] = std::arg(diagonalized(compact, compact));
        }
        std::vector<int> monomials;
        for (int term = 0; term < half; ++term) {
            int population = 0;
            for (int bit = 0; bit < body_width; ++bit) population += (term >> bit) & 1;
            if (population >= 2) monomials.push_back(term);
        }
        for (int branch = 0; branch < (1 << monomials.size()); ++branch) {
            std::vector<double> values(half), coefficients(half), multiples(half);
            double scale = 0.0;
            for (int index = 0; index < half; ++index) {
                unsigned flip = 0;
                for (std::size_t term = 0; term < monomials.size(); ++term)
                    if ((branch & (1 << term)) && (index & monomials[term]) == monomials[term]) flip ^= 1U;
                values[index] = -phases[index] + (flip ? 2.0 * pi : 0.0);
                scale += std::abs(values[index]);
            }
            for (int term = 0; term < half; ++term) {
                double value = 0.0;
                for (int index = 0; index < half; ++index)
                    value += (parity(index & term) ? -1.0 : 1.0) * values[index] / half;
                multiples[term] = std::nearbyint(value / pi);
                coefficients[term] = value - multiples[term] * pi;
            }
            std::vector<double> previous;
            for (int sparsity = 0; sparsity < 2; ++sparsity) {
                auto proposed = coefficients;
                const double rounding = 128.0 * epsilon * std::max(1.0, scale);
                const double threshold = sparsity == 0 ? rounding :
                    std::max(rounding, std::asin(std::sqrt(tolerance)) / half);
                unsigned request = 0;
                for (int term = 1; term < half; ++term) {
                    if (std::abs(proposed[term]) <= threshold) proposed[term] = 0.0;
                    if (proposed[term] != 0.0) request |= 1U << (term - 1);
                }
                if (proposed == previous) continue;
                previous = proposed;
                auto found = books.find(request);
                if (found == books.end()) found = books.emplace(request,
                    parity_networks(width, request, compact_couplers, cnot_budget)).first;
                for (const auto& network : found->second) {
                    LogicalFactor factor;
                    factor.root = root;
                    append_local(factor, target_frame, root);
                    for (int wire = 0; wire < body_width; ++wire) append_local(factor, frames[wire], outside[wire]);
                    unsigned z_mask = 0;
                    for (int term = 0; term < half; ++term)
                        if (std::fmod(std::abs(multiples[term]), 2.0) == 1.0) z_mask ^= 1U | (unsigned(term) << 1);
                    for (int wire = 0; wire < width; ++wire) if (z_mask & (1U << wire)) append_local(factor, pauli(2), wires[wire]);
                    if (proposed[0] != 0.0) append_rz(factor, proposed[0], root);
                    std::vector<unsigned> rows(width);
                    for (int wire = 0; wire < width; ++wire) rows[wire] = 1U << wire;
                    unsigned emitted = 0;
                    for (const auto& move : network.moves) {
                        LogicalFactorOperation operation;
                        operation.control = wires[move.first]; operation.target = wires[move.second];
                        factor.prefix.push_back(operation);
                        ++factor.prefix_cnots;
                        rows[move.second] ^= rows[move.first];
                        const unsigned row = rows[move.second];
                        if ((row & 1U) && row != 1U) {
                            const unsigned term = row >> 1, bit = 1U << (term - 1);
                            if ((request & bit) && !(emitted & bit)) {
                                append_rz(factor, proposed[term], wires[move.second]); emitted |= bit;
                            }
                        }
                    }
                    if (emitted != request || rows != network.end_rows) continue;
                    const Matrix prefix = prefix_matrix(factor, width);
                    const Matrix remainder = target * prefix.adjoint();
                    std::vector<int> neighbors(outside.size(), root);
                    auto tail = logical_factor(remainder.data(), width, dimension, root,
                        outside.data(), neighbors.data(), couplers, tolerance, 0);
                    if (!tail.valid || !tail.prefix.empty()) continue;
                    factor.valid = true;
                    factor.process_infidelity = tail.process_infidelity;
                    factor.child_adjoint = std::move(tail.child_adjoint);
                    factor.local_adjoint = tail.local_adjoint;
                    emit(std::move(factor));
                }
            }
        }
    }
}
} // namespace squander_osr
