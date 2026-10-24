#ifndef SQUANDER_OSR_PHASE_POLYNOMIAL_H
#define SQUANDER_OSR_PHASE_POLYNOMIAL_H

#include "OSR_Logical_Factor.h"

namespace squander_osr {
// A complete, target-derived affine phase-polynomial certificate. Unsupported
// targets return an invalid factor; no numerical optimizer or reference circuit
// is used, and the caller still certifies its native gate representation.
LogicalFactor phase_polynomial_factor(
    const std::complex<double>* target, int width, int stride, int root,
    unsigned couplers, double tolerance, int cnot_budget);
}

// Exact constructive parity motifs on 3/4 wires. The budget is a CNOT bound,
// not a runtime/depth cap. Equivalent (linear map, phase support) states retain
// one constructive representative. Returns a CNOT count or a negative status.
extern "C" int qgd_osr_phase_parity_network(
    int width, const int* final_rows, const int* truth, int budget,
    const int* controls, const int* targets, int edges, int* word,
    int* coefficients, unsigned long long* statistics);

extern "C" int qgd_osr_phase_polynomial_factor(
    const double* target, int width, int stride, int root, unsigned couplers,
    double tolerance, int budget, int capacity, int* controls, int* targets,
    double* matrices, double* process_infidelity);

#endif
