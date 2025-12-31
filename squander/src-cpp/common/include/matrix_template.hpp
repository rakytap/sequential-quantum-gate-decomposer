#ifndef MATRIX_TEMPLATE_HPP
#define MATRIX_TEMPLATE_HPP

#include "matrix_base.hpp"
#include <utility>

template<typename ComplexType>
class Matrix_T : public matrix_base<ComplexType> {
private:
    static constexpr size_t base_size = sizeof(matrix_base<ComplexType>);
    static constexpr size_t padding_size =
        (CACHELINE - (base_size % CACHELINE)) % CACHELINE;

    static_assert(base_size <= CACHELINE, "matrix_base exceeds CACHELINE");

    char padding_[padding_size];

public:
    Matrix_T();
    Matrix_T(ComplexType* data_in, int rows_in, int cols_in);
    Matrix_T(ComplexType* data_in, int rows_in, int cols_in, int stride_in);
    Matrix_T(int rows_in, int cols_in);
    Matrix_T(int rows_in, int cols_in, int stride_in);

    Matrix_T(const Matrix_T &other);
    Matrix_T(Matrix_T &&other) noexcept;
    Matrix_T& operator=(const Matrix_T &other);
    Matrix_T& operator=(Matrix_T &&other) noexcept;
    ~Matrix_T() = default;

    Matrix_T copy() const;
    bool isnan();
    void print_matrix() const;
};

#endif // MATRIX_TEMPLATE_HPP
