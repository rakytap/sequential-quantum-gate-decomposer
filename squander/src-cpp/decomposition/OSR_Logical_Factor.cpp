#include "include/OSR_Logical_Factor.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>

namespace squander_osr {
namespace {

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor, 16, 16>;
using Body = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor, 8, 8>;
using Local = Eigen::Matrix<Complex, 2, 2, Eigen::RowMajor>;
using Triple = std::array<Matrix, 3>;

const Complex I(0.0, 1.0);

Local pauli(int axis) {
    Local result = Local::Zero();
    if (axis == 0) {
        result(0, 1) = result(1, 0) = 1.0;
    } else if (axis == 1) {
        result(0, 1) = -I;
        result(1, 0) = I;
    } else {
        result(0, 0) = 1.0;
        result(1, 1) = -1.0;
    }
    return result;
}

Matrix expanded(const Local& local, int wire, int width) {
    const int dimension = 1 << width;
    Matrix result = Matrix::Zero(dimension, dimension);
    const int bit = 1 << wire;
    for (int base = 0; base < dimension; ++base) {
        if (base & bit) continue;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                result(base | (a << wire), base | (b << wire)) = local(a, b);
    }
    return result;
}

int insert_zero_bit(int index, int wire) {
    const int low = (1 << wire) - 1;
    return (index & low) | ((index & ~low) << 1);
}

double leakage(const Triple& images, int leaf, int dimension) {
    double error = 0.0;
    const int half = dimension / 2;
    for (const auto& image : images) {
        for (int a = 0; a < half; ++a) {
            const int row = insert_zero_bit(a, leaf);
            for (int b = 0; b < half; ++b) {
                const int col = insert_zero_bit(b, leaf);
                const Complex mean = 0.5 * (image(row, col) +
                    image(row | (1 << leaf), col | (1 << leaf)));
                error += std::norm(image(row, col) - mean);
                error += std::norm(image(row | (1 << leaf), col | (1 << leaf)) - mean);
                error += std::norm(image(row, col | (1 << leaf)));
                error += std::norm(image(row | (1 << leaf), col));
            }
        }
    }
    return error / (3.0 * dimension);
}

Triple images_of(const Matrix& residual, const Triple& root_paulis) {
    Triple images;
    for (int axis = 0; axis < 3; ++axis)
        images[axis] = residual * root_paulis[axis] * residual.adjoint();
    return images;
}

template<int Size>
Eigen::Matrix<double, Size, 1> least_branch(
        const std::array<Triple, Size>& branches, double& error) {
    Eigen::Matrix<double, Size, Size> gram;
    for (int a = 0; a < Size; ++a) {
        for (int b = a; b < Size; ++b) {
            double product = 0.0;
            for (int axis = 0; axis < 3; ++axis)
                product += (branches[a][axis].conjugate().cwiseProduct(
                    branches[b][axis])).sum().real();
            gram(a, b) = gram(b, a) = product;
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Size, Size>> eigensolver(gram);
    if (eigensolver.info() != Eigen::Success) {
        error = std::numeric_limits<double>::infinity();
        return Eigen::Matrix<double, Size, 1>::Zero();
    }
    auto vector = eigensolver.eigenvectors().col(0).eval();
    Eigen::Index largest;
    vector.cwiseAbs().maxCoeff(&largest);
    if (vector(largest) < 0.0) vector = -vector;
    // Evaluate the residual directly, not a nearly-zero Gram eigenvalue.
    error = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        Matrix residual = Matrix::Zero(branches[0][axis].rows(), branches[0][axis].cols());
        for (int a = 0; a < Size; ++a) residual += vector(a) * branches[a][axis];
        error += residual.squaredNorm();
    }
    return vector;
}

Local align_z(const Local& axis) {
    Eigen::SelfAdjointEigenSolver<Local> eigensolver(axis);
    if (eigensolver.info() != Eigen::Success) return Local::Zero();
    Local rotation;
    rotation.row(0) = eigensolver.eigenvectors().col(1).adjoint();
    rotation.row(1) = eigensolver.eigenvectors().col(0).adjoint();
    return rotation;
}

Local rz(double angle) {
    Local result = Local::Zero();
    result(0, 0) = std::exp(-0.5 * I * angle);
    result(1, 1) = std::exp(0.5 * I * angle);
    return result;
}

Local ry(double angle) {
    const double c = std::cos(angle / 2.0), s = std::sin(angle / 2.0);
    Local result;
    result << c, -s, s, c;
    return result;
}

void append_local(LogicalFactor& output, const Local& local, int wire) {
    LogicalFactorOperation operation;
    operation.target = wire;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) operation.matrix[2 * a + b] = local(a, b);
    output.prefix.push_back(operation);
}

void append_cnot(LogicalFactor& output, int control, int target) {
    LogicalFactorOperation operation;
    operation.target = target;
    operation.control = control;
    output.prefix.push_back(operation);
    ++output.prefix_cnots;
}

bool append_controlled(LogicalFactor& output, const Local& unitary,
                       int control, int target, bool single,
                       const Local& axis) {
    const double pi = std::acos(-1.0);
    if (single) {
        Local hadamard;
        hadamard << 1.0, 1.0, 1.0, -1.0;
        hadamard /= std::sqrt(2.0);
        const Local frame = hadamard * align_z(axis);
        if ((unitary + I * axis).norm() > 1e-10) return false;
        append_local(output, frame, target);
        append_cnot(output, control, target);
        append_local(output, frame.adjoint(), target);
        Local phase = Local::Identity();
        phase(1, 1) = std::exp(-0.5 * I * pi);
        append_local(output, phase, control);
        return true;
    }
    const double theta = 2.0 * std::atan2(std::abs(unitary(1, 0)), std::abs(unitary(0, 0)));
    const double alpha = std::arg(unitary(1, 0)) - std::arg(unitary(0, 0));
    const double gamma = -std::arg(unitary(1, 0)) - std::arg(unitary(0, 0));
    const Local reconstructed = rz(alpha) * ry(theta) * rz(gamma);
    const double phase = std::arg((reconstructed.conjugate().cwiseProduct(unitary)).sum());
    if ((std::exp(I * phase) * reconstructed - unitary).norm() > 1e-10) return false;
    const Local a = rz(alpha) * ry(theta / 2.0);
    const Local b = ry(-theta / 2.0) * rz(-(alpha + gamma) / 2.0);
    const Local c = rz((gamma - alpha) / 2.0);
    if ((a * b * c - Local::Identity()).norm() > 1e-10 ||
        (a * pauli(0) * b * pauli(0) * c - reconstructed).norm() > 1e-10) return false;
    append_local(output, c, target);
    append_cnot(output, control, target);
    append_local(output, b, target);
    append_cnot(output, control, target);
    append_local(output, a, target);
    Local phase_gate = Local::Identity();
    phase_gate(1, 1) = std::exp(I * phase);
    append_local(output, phase_gate, control);
    return true;
}

Matrix prefix_matrix(const LogicalFactor& factor, int width) {
    const int dimension = 1 << width;
    Matrix result = Matrix::Identity(dimension, dimension);
    for (const auto& operation : factor.prefix) {
        if (operation.control < 0) {
            Local local;
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) local(a, b) = operation.matrix[2 * a + b];
            result = (expanded(local, operation.target, width) * result).eval();
        } else {
            const int c = 1 << operation.control, t = 1 << operation.target;
            for (int row = 0; row < dimension; ++row)
                if ((row & c) && !(row & t)) result.row(row).swap(result.row(row | t));
        }
    }
    return result;
}

double infidelity(const Matrix& target, const Matrix& actual) {
    const Complex overlap = (actual.conjugate().cwiseProduct(target)).sum();
    if (std::abs(overlap) == 0.0) return 1.0;
    const Complex phase = overlap / std::abs(overlap);
    const double chord = (target - phase * actual).squaredNorm() / target.rows();
    return chord * (1.0 - 0.25 * chord);
}

} // namespace

LogicalFactor logical_factor(const Complex* data, int width, int stride, int root,
                             const int* leaf_order, const int* neighbors,
                             unsigned couplers, double tolerance,
                             int prefix_cnot_budget) {
    LogicalFactor output;
    if (!data || !leaf_order || !neighbors || width < 2 || width > 4 ||
        root < 0 || root >= width || stride < (1 << width) ||
        !std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0 ||
        prefix_cnot_budget < 0) return output;
    const int dimension = 1 << width, half = dimension / 2;
    Matrix target(dimension, dimension);
    for (int row = 0; row < dimension; ++row)
        for (int col = 0; col < dimension; ++col) target(row, col) = data[row * stride + col];
    if (!target.allFinite()) return output;
    const double epsilon = std::numeric_limits<double>::epsilon();
    if ((target.adjoint() * target - Matrix::Identity(dimension, dimension)).norm() >
        64.0 * epsilon * dimension * dimension * dimension) return output;
    Matrix residual = target.adjoint();
    Triple root_paulis;
    for (int axis = 0; axis < 3; ++axis) root_paulis[axis] = expanded(pauli(axis), root, width);
    unsigned active = (1U << width) - 1U;
    for (int step = 0; step < width - 1; ++step) {
        const int leaf = leaf_order[step], neighbor = neighbors[step];
        if (leaf < 0 || leaf >= width || leaf == root || !(active & (1U << leaf)) ||
            neighbor < 0 || neighbor >= width || neighbor == leaf || !(active & (1U << neighbor)))
            return LogicalFactor{};
        const Triple images = images_of(residual, root_paulis);
        active &= ~(1U << leaf);
        if (leakage(images, leaf, dimension) <= 4.0 * tolerance) continue;
        const int lo = std::min(leaf, neighbor), hi = std::max(leaf, neighbor);
        if (!(couplers & (1U << (lo * width + hi)))) return LogicalFactor{};
        std::array<Triple, 3> commutators;
        for (int a = 0; a < 3; ++a) {
            const Matrix local_axis = expanded(pauli(a), leaf, width);
            for (int b = 0; b < 3; ++b)
                commutators[a][b] = local_axis * images[b] - images[b] * local_axis;
        }
        double error;
        const auto coefficients = least_branch<3>(commutators, error);
        if (error / (12.0 * dimension) > 4.0 * tolerance) return LogicalFactor{};
        Local axis = Local::Zero();
        for (int a = 0; a < 3; ++a) axis += coefficients(a) * pauli(a);
        const Local rotation = align_z(axis);
        const Matrix frame = expanded(rotation, leaf, width);
        Triple plus, minus;
        for (int a = 0; a < 3; ++a) {
            const Matrix aligned = frame * images[a] * frame.adjoint();
            plus[a].resize(half, half);
            minus[a].resize(half, half);
            for (int row = 0; row < half; ++row) {
                const int r = insert_zero_bit(row, leaf);
                for (int col = 0; col < half; ++col) {
                    const int c = insert_zero_bit(col, leaf);
                    plus[a](row, col) = aligned(r, c);
                    minus[a](row, col) = aligned(r | (1 << leaf), c | (1 << leaf));
                }
            }
        }
        std::array<Triple, 4> intertwiners;
        const int compact_neighbor = neighbor - (neighbor > leaf ? 1 : 0);
        for (int a = 0; a < 4; ++a) {
            const Matrix basis = a == 0 ? Matrix::Identity(half, half).eval() :
                (I * expanded(pauli(a - 1), compact_neighbor, width - 1)).eval();
            for (int b = 0; b < 3; ++b)
                intertwiners[a][b] = minus[b] * basis - basis * plus[b];
        }
        std::array<Triple, 3> pure{{intertwiners[1], intertwiners[2], intertwiners[3]}};
        const auto pure_coefficients = least_branch<3>(pure, error);
        Eigen::Vector4d quaternion;
        const bool single = error / (3.0 * dimension) <= 4.0 * tolerance;
        if (single) {
            quaternion(0) = 0.0;
            quaternion.tail<3>() = pure_coefficients;
        } else {
            quaternion = least_branch<4>(intertwiners, error);
            if (error / (3.0 * dimension) > 4.0 * tolerance) return LogicalFactor{};
        }
        Local unitary = quaternion(0) * Local::Identity();
        Local pure_axis = Local::Zero();
        for (int a = 0; a < 3; ++a) pure_axis += quaternion(a + 1) * pauli(a);
        unitary += I * pure_axis;
        const Matrix body = expanded(unitary.adjoint(), compact_neighbor, width - 1);
        Matrix controlled = Matrix::Identity(dimension, dimension);
        for (int row = 0; row < half; ++row) {
            const int r = insert_zero_bit(row, leaf) | (1 << leaf);
            for (int col = 0; col < half; ++col)
                controlled(r, insert_zero_bit(col, leaf) | (1 << leaf)) = body(row, col);
        }
        residual = (controlled * frame * residual).eval();
        if (leakage(images_of(residual, root_paulis), leaf, dimension) > 4.0 * tolerance)
            return LogicalFactor{};
        append_local(output, rotation, leaf);
        if (!append_controlled(output, unitary.adjoint(), leaf, neighbor, single, pure_axis) ||
            output.prefix_cnots > prefix_cnot_budget) return LogicalFactor{};
    }
    const Matrix prefix = prefix_matrix(output, width);
    if ((residual - prefix * target.adjoint()).norm() > 1e-10) return LogicalFactor{};
    std::array<Body, 4> blocks;
    int best = 0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            Body& block = blocks[2 * a + b];
            block.resize(half, half);
            for (int row = 0; row < half; ++row)
                for (int col = 0; col < half; ++col)
                    block(row, col) = residual(insert_zero_bit(row, root) | (a << root),
                                              insert_zero_bit(col, root) | (b << root));
            if (block.squaredNorm() > blocks[best].squaredNorm()) best = 2 * a + b;
        }
    }
    Eigen::JacobiSVD<Body> child_svd(blocks[best], Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (child_svd.info() != Eigen::Success) return LogicalFactor{};
    const Body child = child_svd.matrixU() * child_svd.matrixV().adjoint();
    Local local;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            local(a, b) = (child.conjugate().cwiseProduct(blocks[2 * a + b])).sum() / double(half);
    Eigen::JacobiSVD<Local> local_svd(local, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (local_svd.info() != Eigen::Success) return LogicalFactor{};
    local = local_svd.matrixU() * local_svd.matrixV().adjoint();
    Matrix product(dimension, dimension);
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int row = 0; row < half; ++row)
                for (int col = 0; col < half; ++col)
                    product(insert_zero_bit(row, root) | (a << root),
                            insert_zero_bit(col, root) | (b << root)) = local(a, b) * child(row, col);
    output.process_infidelity = infidelity(target, product.adjoint() * prefix);
    if (!std::isfinite(output.process_infidelity) || output.process_infidelity > tolerance)
        return LogicalFactor{};
    output.root = root;
    output.child_adjoint.resize(half * half);
    for (int row = 0; row < half; ++row)
        for (int col = 0; col < half; ++col)
            output.child_adjoint[row * half + col] = std::conj(child(col, row));
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) output.local_adjoint[2 * a + b] = std::conj(local(b, a));
    output.valid = true;
    return output;
}

} // namespace squander_osr

extern "C" int qgd_osr_logical_factor(
        const double* target, int width, int stride, int root,
        const int* leaf_order, const int* neighbors, unsigned couplers,
        double tolerance, int prefix_cnot_budget,
        double* child_adjoint, double* local_adjoint,
        int* operation_targets, int* operation_controls, double* operation_matrices,
        int operation_capacity, int* operation_count, int* prefix_cnots,
        double* process_infidelity) {
    if (!target || width < 2 || width > 4 || stride < (1 << width) ||
        !child_adjoint || !local_adjoint || !operation_targets || !operation_controls ||
        !operation_matrices || !operation_count || !prefix_cnots || !process_infidelity ||
        operation_capacity < 0) return -1;
    try {
        const int dimension = 1 << width;
        std::vector<std::complex<double>> matrix(dimension * dimension);
        for (int row = 0; row < dimension; ++row)
            for (int col = 0; col < dimension; ++col) {
                const int source = 2 * (row * stride + col);
                matrix[row * dimension + col] = {target[source], target[source + 1]};
            }
        const auto factor = squander_osr::logical_factor(
            matrix.data(), width, dimension, root, leaf_order, neighbors,
            couplers, tolerance, prefix_cnot_budget);
        if (!factor.valid) return 0;
        if (factor.prefix.size() > static_cast<std::size_t>(operation_capacity)) return -1;
        for (std::size_t i = 0; i < factor.child_adjoint.size(); ++i) {
            child_adjoint[2 * i] = factor.child_adjoint[i].real();
            child_adjoint[2 * i + 1] = factor.child_adjoint[i].imag();
        }
        for (int i = 0; i < 4; ++i) {
            local_adjoint[2 * i] = factor.local_adjoint[i].real();
            local_adjoint[2 * i + 1] = factor.local_adjoint[i].imag();
        }
        for (std::size_t i = 0; i < factor.prefix.size(); ++i) {
            operation_targets[i] = factor.prefix[i].target;
            operation_controls[i] = factor.prefix[i].control;
            for (int j = 0; j < 4; ++j) {
                operation_matrices[8 * i + 2 * j] = factor.prefix[i].matrix[j].real();
                operation_matrices[8 * i + 2 * j + 1] = factor.prefix[i].matrix[j].imag();
            }
        }
        *operation_count = static_cast<int>(factor.prefix.size());
        *prefix_cnots = factor.prefix_cnots;
        *process_infidelity = factor.process_infidelity;
        return 1;
    } catch (...) {
        return -1;
    }
}
