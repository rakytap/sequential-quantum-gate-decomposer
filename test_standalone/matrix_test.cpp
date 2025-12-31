#include "matrix.h"
#include "QGDTypes.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // Basic instantiation
    Matrix m1(3, 3);
    assert(m1.get_data() != nullptr);

    // Size check
    assert(sizeof(Matrix) == CACHELINE);

    // Initialize data
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i * m1.stride + j;
            m1.data[idx].real = i + j;
            m1.data[idx].imag = i - j;
        }
    }

    // Copy constructor (shallow copy - reference counted)
    Matrix m2(m1);
    assert(m2.get_data() == m1.get_data());

    // Deep copy
    Matrix m3 = m1.copy();
    assert(m3.get_data() != m1.get_data());
    assert(m3.rows == m1.rows && m3.cols == m1.cols);

    // Verify data is copied
    for (int i = 0; i < m1.rows * m1.cols; i++) {
        assert(m3.data[i].real == m1.data[i].real);
        assert(m3.data[i].imag == m1.data[i].imag);
    }

    // Move constructor
    Matrix m4(std::move(m3));
    assert(m4.rows == m1.rows && m4.cols == m1.cols);

    // Assignment operator (reference counted)
    Matrix m5(2, 2);
    m5 = m1;
    assert(m5.get_data() == m1.get_data());

    // NaN detection
    Matrix m6(2, 2);
    m6[0].real = NAN;
    assert(m6.isnan());

    // Alignment check
    assert(reinterpret_cast<uintptr_t>(m1.get_data()) % CACHELINE == 0);
    m1.ensure_aligned();
    assert(reinterpret_cast<uintptr_t>(m1.get_data()) % CACHELINE == 0);

    // Edge cases
    Matrix zero_mat(0, 0);
    assert(zero_mat.size() == 0);

    Matrix single(1, 1);
    single[0].real = 42.0;
    assert(single[0].real == 42.0);

    Matrix rect1(10, 100);
    Matrix rect2(100, 10);
    assert(rect1.size() == rect2.size());

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
