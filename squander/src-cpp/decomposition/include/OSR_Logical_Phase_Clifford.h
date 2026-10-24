#ifndef OSR_LOGICAL_PHASE_CLIFFORD_H
#define OSR_LOGICAL_PHASE_CLIFFORD_H

#include "OSR_Logical_Factor.h"

namespace squander_osr {

// Extract Clifford residues of the same root-relative phase polynomial.
// Controller-only factors are absorbed into the independently solved child.
void logical_phase_clifford_factors(
    const std::complex<double>* target, int width, int stride, int root,
    unsigned couplers, double tolerance, int budget,
    const std::function<void(LogicalFactor&&)>& emit);

} // namespace squander_osr

#endif
