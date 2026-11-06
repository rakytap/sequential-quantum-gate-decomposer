#ifndef SQUANDER_OSR_CUT_CAPACITY_H
#define SQUANDER_OSR_CUT_CAPACITY_H

#include "QGDTypes.h"

namespace squander_osr {

// Necessary cut constraints, not a sufficient circuit-synthesis criterion.
// Unsupported inputs or numerical failures leave the cut unrestricted.
struct CutCapacity {
    int minimum_crossing_cnots = 0;
    bool valid = false;
    double transport_tail_lower_bound = 0.0;
    double transport_distance_bound = 0.0;
    double controlled_projection_distance = 0.0;
    double paired_spectrum_distance = 0.0;
    double paired_distance_bound = 0.0;
    double channel_gap = 0.0;
    bool spectrum_checked = false;
};

// tolerance is process infidelity. Data is row-major, with an explicit stride.
// adjoint changes the reader, without materializing or transposing the matrix.
CutCapacity cut_capacity(const QGD_Complex16* data, int qubit_count, int stride,
                         int root, double tolerance, bool adjoint = false);

} // namespace squander_osr

#endif
