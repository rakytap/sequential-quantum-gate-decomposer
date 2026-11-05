#ifndef OSR_LOGICAL_FACTOR_INVERSE_H
#define OSR_LOGICAL_FACTOR_INVERSE_H

#include <algorithm>
#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

namespace squander_osr {

template<class Parameters>
auto factor_parameter_buffer(Parameters& parameters)
    -> decltype(parameters.get_data()) {
    return parameters.get_data();
}

inline double* factor_parameter_buffer(std::vector<double>& parameters) {
    return parameters.data();
}

// Invert the chronological U3/CNOT stream. Local gates on untouched wires
// commute through a CNOT; gates on either endpoint are flushed before it.
// This preserves both the physical couplers and the exact CNOT count.
template<class Word, class Parameters, class Targets, class Controls,
         class AppendAngles, class NativeU3>
void reverse_native_factor_word(Word& word, Parameters& parameters,
                                const Targets& targets, const Controls& controls,
                                int width, AppendAngles append_angles,
                                NativeU3 native_u3) {
    using Complex = std::complex<double>;
    using Local = std::array<Complex, 4>;
    const int count = static_cast<int>(word.size());
    const size_t expected = static_cast<size_t>(6 * count + 3 * width);
    if (parameters.size() != expected)
        throw std::invalid_argument("Invalid native factor parameter layout");
    double* values = factor_parameter_buffer(parameters);
    const std::vector<double> original(values, values + expected);
    auto adjoint = [](const Local& matrix) -> Local {
        return {{std::conj(matrix[0]), std::conj(matrix[2]),
                 std::conj(matrix[1]), std::conj(matrix[3])}};
    };
    const Local identity{{Complex(1), Complex(0), Complex(0), Complex(1)}};
    std::vector<Local> pending(width, identity);
    for (int wire = 0; wire < width; ++wire)
        pending[wire] = adjoint(native_u3(original.data() + 6 * count + 3 * wire));
    std::vector<double> inverse;
    inverse.reserve(expected);
    for (int index = count - 1; index >= 0; --index) {
        const int digit = word[index];
        const int target = targets.get_data()[digit];
        const int control = controls.get_data()[digit];
        append_angles(pending[target], inverse);
        append_angles(pending[control], inverse);
        pending[target] = adjoint(native_u3(original.data() + 6 * index));
        pending[control] = adjoint(native_u3(original.data() + 6 * index + 3));
    }
    for (const auto& matrix : pending)
        append_angles(matrix, inverse);
    std::copy(inverse.begin(), inverse.end(), values);
    for (int index = 0; index < count / 2; ++index)
        std::swap(word[index], word[count - 1 - index]);
}

} // namespace squander_osr

extern "C" int qgd_osr_reverse_logical_factor_word(
    int width, int count, int* word, int coupler_count,
    const int* targets, const int* controls, double* parameters, int parameter_count);

#endif
