#ifndef SQUANDER_OSR_PARITY_NETWORK_H
#define SQUANDER_OSR_PARITY_NETWORK_H

#include <utility>
#include <vector>

namespace squander_osr {
struct ParityNetwork {
    std::vector<unsigned> end_rows;
    std::vector<std::pair<unsigned, unsigned>> moves;
};

// Row zero is the root. Request bit (mask-1) denotes root XOR mask of
// controllers. The root is restored; controller-only end maps remain free.
// One shortest representative is retained for every minimum-cost end map.
std::vector<ParityNetwork> parity_networks(unsigned width, unsigned request,
                                          unsigned couplers, int cnot_budget);
}

#if defined(_WIN32) && defined(qgd_EXPORTS)
#define SQUANDER_PARITY_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
#define SQUANDER_PARITY_EXPORT __declspec(dllimport)
#else
#define SQUANDER_PARITY_EXPORT
#endif

// Pass null output arrays to query the common minimum length and count.
// Rows need count*width entries; moves need count*minimum*2 entries.
extern "C" SQUANDER_PARITY_EXPORT int qgd_osr_parity_networks(
    unsigned width, unsigned request, unsigned couplers, int cnot_budget,
    unsigned* rows, unsigned* moves, int network_capacity,
    int* minimum, int* network_count);
#undef SQUANDER_PARITY_EXPORT
#endif
