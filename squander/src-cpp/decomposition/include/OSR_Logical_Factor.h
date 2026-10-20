#ifndef SQUANDER_OSR_LOGICAL_FACTOR_H
#define SQUANDER_OSR_LOGICAL_FACTOR_H

#include <array>
#include <complex>
#include <functional>
#include <vector>

namespace squander_osr {

// Chronological gates of a matrix-derived decoder. All matrices are row-major.
struct LogicalFactorOperation {
    int target = -1;
    int control = -1; // -1 means a one-qubit unitary, otherwise a CNOT.
    std::array<std::complex<double>, 4> matrix{};
};

struct LogicalFactor {
    bool valid = false;
    int root = -1;
    int prefix_cnots = 0;
    double process_infidelity = 0.0;
    std::vector<LogicalFactorOperation> prefix;
    // U = (local_adjoint on root) (child_adjoint on other wires) prefix.
    std::array<std::complex<double>, 4> local_adjoint{};
    std::vector<std::complex<double>> child_adjoint;
};

// A bounded causal motif, not an exhaustive synthesis or optimality claim.
// `target` is U, not U.adjoint(); stride is measured in complex elements.
// An undirected coupler (a,b) occupies bit (a * width + b), a < b.
LogicalFactor logical_factor(const std::complex<double>* target,
                             int width, int stride, int root,
                             const int* leaf_order, const int* neighbors,
                             unsigned couplers, double tolerance,
                             int prefix_cnot_budget);

// Locally commuting controller frames and exact relative-parity networks.
// Every emitted factor is verified against the original matrix and tolerance.
void logical_phase_factors(const std::complex<double>* target,
                          int width, int stride, int root,
                          unsigned couplers, double tolerance, int cnot_budget,
                          const std::function<void(LogicalFactor&&)>& emit);

} // namespace squander_osr

#if defined(_WIN32) && defined(qgd_EXPORTS)
#define SQUANDER_LOGICAL_FACTOR_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
#define SQUANDER_LOGICAL_FACTOR_EXPORT __declspec(dllimport)
#else
#define SQUANDER_LOGICAL_FACTOR_EXPORT
#endif

// Interleaved real/imaginary arrays provide a layout-independent testing ABI.
// Output buffers need 2*(2^(width-1))^2 child doubles, 8 local doubles,
// operation_capacity integer entries per gate array and 8 doubles per matrix.
extern "C" SQUANDER_LOGICAL_FACTOR_EXPORT int qgd_osr_logical_factor(
    const double* target, int width, int stride, int root,
    const int* leaf_order, const int* neighbors, unsigned couplers,
    double tolerance, int prefix_cnot_budget,
    double* child_adjoint, double* local_adjoint,
    int* operation_targets, int* operation_controls, double* operation_matrices,
    int operation_capacity, int* operation_count, int* prefix_cnots,
    double* process_infidelity);

#undef SQUANDER_LOGICAL_FACTOR_EXPORT
#endif
