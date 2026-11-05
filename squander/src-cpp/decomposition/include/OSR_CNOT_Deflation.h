#ifndef SQUANDER_OSR_CNOT_DEFLATION_H
#define SQUANDER_OSR_CNOT_DEFLATION_H

#include "QGDTypes.h"

#if defined(_WIN32) && defined(qgd_EXPORTS)
#define QGD_OSR_DEFLATION_API __declspec(dllexport)
#elif defined(_WIN32)
#define QGD_OSR_DEFLATION_API __declspec(dllimport)
#else
#define QGD_OSR_DEFLATION_API
#endif

// Refit exactly one deletion from a native pre-CX-U3 chart (2--4 qubits).
// The input matrix is the target ADJOINT, row-major with the supplied stride.
// Parent angles use theta/2: six per CX, followed by three per final wire.
// Returns 1 only for an independently certified child, 0 for a failed fit,
// and -1 for invalid arguments/errors. No circuit is returned on failure.
extern "C" QGD_OSR_DEFLATION_API int qgd_osr_refit_cnot_deletion(
    const QGD_Complex16* target_adjoint, int qubit_count, int stride,
    const int* targets, const int* controls, int parent_length,
    const double* parent_parameters, int parameter_count,
    int removed, int branch, int basin_hops, int evaluations, int seed,
    double tolerance, double* child_parameters, int child_parameter_count,
    double* hs, double* process_infidelity, int trace);

#undef QGD_OSR_DEFLATION_API
#endif
