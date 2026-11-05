#ifndef OSR_EIGEN_SVD_H
#define OSR_EIGEN_SVD_H

#include "QGDTypes.h"

#include <complex>
#include <utility>
#include <vector>

template<class ComplexT>
struct OSRTriplet {
    std::vector<double> singulars;
    std::vector<ComplexT> left_factors;
    std::vector<ComplexT> right_factors;

    OSRTriplet() = default;

    OSRTriplet(std::vector<double> singulars_in,
               std::vector<ComplexT> left_factors_in,
               std::vector<ComplexT> right_factors_in)
        : singulars(std::move(singulars_in)),
          left_factors(std::move(left_factors_in)),
          right_factors(std::move(right_factors_in)) {}
};

std::vector<double> osr_eigen_singular_values(
    const std::vector<std::complex<double>>& input,
    int rows, int cols, double frobenius_norm);
std::vector<double> osr_eigen_singular_values(
    const std::vector<std::complex<float>>& input,
    int rows, int cols, double frobenius_norm);

OSRTriplet<QGD_Complex16> osr_eigen_triplet(
    const std::vector<std::complex<double>>& input,
    int rows, int cols, double frobenius_norm);
OSRTriplet<QGD_Complex8> osr_eigen_triplet(
    const std::vector<std::complex<float>>& input,
    int rows, int cols, double frobenius_norm);

#endif
