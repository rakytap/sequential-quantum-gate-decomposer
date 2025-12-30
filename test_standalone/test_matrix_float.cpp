#include "matrix_float.h"
#include "matrix.h"
#include "QGDTypes.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // Basic instantiation
    MatrixFloat m(100, 100);
    assert(m.get_data() != nullptr);

    // Memory size check
    assert(sizeof(QGD_Complex8) == sizeof(QGD_Complex16) / 2);

    // Copy constructor and reference counting
    MatrixFloat m2(m);
    assert(m2.get_data() == m.get_data());

    // Element access
    m[0].real = 1.0f;
    m[0].imag = 2.0f;
    assert(m[0].real == 1.0f);

    // NaN propagation (requires IEEE 754 compliance, no -ffast-math)
    m[1].real = NAN;
    assert(m.isnan());

    // Conversion roundtrip
    Matrix md(2, 2);
    md[0].real = 3.14159265358979;
    md[0].imag = 2.71828182845905;
    MatrixFloat mf = md.to_float32();
    Matrix md2 = mf.to_float64();
    double rel_error_real = std::abs((md[0].real - md2[0].real) / md[0].real);
    double rel_error_imag = std::abs((md[0].imag - md2[0].imag) / md[0].imag);
    assert(rel_error_real < 2e-6 && rel_error_imag < 2e-6);

    // Overflow handling
    Matrix large(1, 1);
    large[0].real = 1e308;
    MatrixFloat overflow = large.to_float32();
    assert(std::isinf(overflow[0].real));

    // Copy semantics
    MatrixFloat m3(mf);
    assert(m3.get_data() != nullptr);

    // Alignment check
    assert(reinterpret_cast<uintptr_t>(m.get_data()) % CACHELINE == 0);
    m.ensure_aligned();
    assert(reinterpret_cast<uintptr_t>(m.get_data()) % CACHELINE == 0);

    // Edge cases
    MatrixFloat zero_mat(0, 0);
    assert(zero_mat.size() == 0);

    MatrixFloat single(1, 1);
    single[0].real = 42.0f;
    assert(single[0].real == 42.0f);

    MatrixFloat rect1(10, 100);
    MatrixFloat rect2(100, 10);
    assert(rect1.size() == rect2.size());

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
