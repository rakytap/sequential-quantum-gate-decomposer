#include "OSR_Eigen_SVD.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
template<int Rows, int Cols, class RealT>
static std::vector<double> eigen_osr_fixed(
    const std::vector<std::complex<RealT>>& input, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    typedef Eigen::Matrix<Scalar, Rows, Cols, Eigen::ColMajor> MatrixType;
    Eigen::Map<const MatrixType> matrix(input.data());
    Eigen::JacobiSVD<MatrixType> decomposition(matrix);
    const auto& singulars = decomposition.singularValues();
    std::vector<double> normalized(singulars.size());
    for (int idx = 0; idx < singulars.size(); ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;
    return normalized;
}

template<int Rows, class RealT>
static std::vector<double> eigen_osr_gram_fixed(
    const std::vector<std::complex<RealT>>& input, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    typedef Eigen::Matrix<Scalar, Rows, 4, Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor> GramType;
    Eigen::Map<const MatrixType> matrix(input.data());
    const GramType gram = matrix.adjoint() * matrix;
    Eigen::SelfAdjointEigenSolver<GramType> decomposition(
        gram, Eigen::EigenvaluesOnly);
    if (decomposition.info() != Eigen::Success)
        throw std::runtime_error("OSR Gram eigensolver failed");

    std::vector<double> normalized(4);
    for (int idx = 0; idx < 4; ++idx) {
        const RealT eigenvalue = std::max(
            RealT(0), decomposition.eigenvalues()[3 - idx]);
        normalized[idx] = static_cast<double>(std::sqrt(eigenvalue)) / Fnorm;
    }
    return normalized;
}

template<int Rows, int Cols, class RealT>
static std::vector<double> eigen_osr_bdc_fixed(
    const std::vector<std::complex<RealT>>& input, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor, Rows, Cols> MatrixType;
    Eigen::Map<const MatrixType> matrix(input.data(), Rows, Cols);
    Eigen::BDCSVD<MatrixType> decomposition(matrix);
    const auto& singulars = decomposition.singularValues();
    std::vector<double> normalized(singulars.size());
    for (int idx = 0; idx < singulars.size(); ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;
    return normalized;
}

template<class RealT>
static std::vector<double> eigen_osr_dynamic(
    const std::vector<std::complex<RealT>>& input,
    int rows, int cols, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    if (cols == 4) {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 4,
                              Eigen::ColMajor> MatrixType;
        typedef Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor> GramType;
        Eigen::Map<const MatrixType> matrix(input.data(), rows, 4);
        const GramType gram = matrix.adjoint() * matrix;
        Eigen::SelfAdjointEigenSolver<GramType> decomposition(
            gram, Eigen::EigenvaluesOnly);
        if (decomposition.info() != Eigen::Success)
            throw std::runtime_error("OSR dynamic Gram eigensolver failed");
        std::vector<double> normalized(4);
        for (int idx = 0; idx < 4; ++idx) {
            const RealT eigenvalue = std::max(
                RealT(0), decomposition.eigenvalues()[3 - idx]);
            normalized[idx] =
                static_cast<double>(std::sqrt(eigenvalue)) / Fnorm;
        }
        return normalized;
    }

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor> MatrixType;
    Eigen::Map<const MatrixType> matrix(input.data(), rows, cols);
    Eigen::BDCSVD<MatrixType> decomposition(matrix);
    const auto& singulars = decomposition.singularValues();
    std::vector<double> normalized(singulars.size());
    for (int idx = 0; idx < singulars.size(); ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;
    return normalized;
}

template<class RealT>
static std::vector<double> osr(
    const std::vector<std::complex<RealT>>& A,
    int m_rows, int m_cols, double Fnorm)
{
    if (m_rows == 4 && m_cols == 4)
        return eigen_osr_gram_fixed<4, RealT>(A, Fnorm);
    if (m_rows == 16 && m_cols == 4)
        return eigen_osr_gram_fixed<16, RealT>(A, Fnorm);
    if (m_rows == 64 && m_cols == 4)
        return eigen_osr_gram_fixed<64, RealT>(A, Fnorm);
    if (m_rows == 16 && m_cols == 16)
        return eigen_osr_bdc_fixed<16, 16, RealT>(A, Fnorm);
    return eigen_osr_dynamic<RealT>(A, m_rows, m_cols, Fnorm);
}

template<int Rows, int Cols, class ComplexT, class RealT>
static OSRTriplet<ComplexT> eigen_osr_triplet_fixed(
    const std::vector<std::complex<RealT>>& input, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    // Dynamic dimensions permit thin U/V, while fixed maxima keep every
    // matrix and Jacobi workspace bounded at compile time for this shape.
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor, Rows, Cols> MatrixType;
    Eigen::Map<const MatrixType> matrix(input.data(), Rows, Cols);
    Eigen::JacobiSVD<MatrixType> decomposition(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& singulars = decomposition.singularValues();
    const auto& left = decomposition.matrixU();
    const auto& right_adjoint = decomposition.matrixV().adjoint();

    std::vector<double> normalized(singulars.size());
    for (int idx = 0; idx < singulars.size(); ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;

    std::vector<ComplexT> left_factors(left.size());
    for (int idx = 0; idx < left.size(); ++idx)
        left_factors[idx] = ComplexT{
            static_cast<RealT>(left.data()[idx].real()),
            static_cast<RealT>(left.data()[idx].imag())};

    std::vector<ComplexT> right_factors(right_adjoint.size());
    for (int col = 0; col < right_adjoint.cols(); ++col) {
        for (int row = 0; row < right_adjoint.rows(); ++row) {
            const Scalar value = right_adjoint(row, col);
            right_factors[row + col * right_adjoint.rows()] = ComplexT{
                static_cast<RealT>(value.real()),
                static_cast<RealT>(value.imag())};
        }
    }
    return OSRTriplet<ComplexT>(std::move(normalized),
                                std::move(left_factors),
                                std::move(right_factors));
}

template<int Rows, class ComplexT, class RealT>
static OSRTriplet<ComplexT> eigen_osr_triplet_gram_fixed(
    const std::vector<std::complex<RealT>>& input, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    typedef Eigen::Matrix<Scalar, Rows, 4, Eigen::ColMajor> MatrixType;
    typedef Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor> GramType;
    Eigen::Map<const MatrixType> matrix(input.data());
    const GramType gram = matrix.adjoint() * matrix;
    Eigen::SelfAdjointEigenSolver<GramType> decomposition(gram);
    if (decomposition.info() != Eigen::Success)
        throw std::runtime_error("OSR Gram eigensolver failed");

    Eigen::Matrix<RealT, 4, 1> singulars;
    GramType right_vectors;
    for (int idx = 0; idx < 4; ++idx) {
        singulars[idx] = std::sqrt(std::max(
            RealT(0), decomposition.eigenvalues()[3 - idx]));
        right_vectors.col(idx) = decomposition.eigenvectors().col(3 - idx);
    }

    MatrixType left = matrix * right_vectors;
    const RealT cutoff = Eigen::NumTraits<RealT>::epsilon() *
        static_cast<RealT>(Rows) * singulars[0];
    for (int idx = 0; idx < 4; ++idx) {
        if (singulars[idx] > cutoff)
            left.col(idx) /= singulars[idx];
        else
            left.col(idx).setZero();
    }

    std::vector<double> normalized(4);
    for (int idx = 0; idx < 4; ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;

    std::vector<ComplexT> left_factors(Rows * 4);
    for (int idx = 0; idx < left.size(); ++idx)
        left_factors[idx] = ComplexT{
            static_cast<RealT>(left.data()[idx].real()),
            static_cast<RealT>(left.data()[idx].imag())};

    std::vector<ComplexT> right_factors(16);
    const GramType right_adjoint = right_vectors.adjoint();
    for (int idx = 0; idx < right_adjoint.size(); ++idx)
        right_factors[idx] = ComplexT{
            static_cast<RealT>(right_adjoint.data()[idx].real()),
            static_cast<RealT>(right_adjoint.data()[idx].imag())};

    return OSRTriplet<ComplexT>(std::move(normalized),
                                std::move(left_factors),
                                std::move(right_factors));
}

template<int Rows, int Cols, class ComplexT, class RealT>
static OSRTriplet<ComplexT> eigen_osr_triplet_bdc_fixed(
    const std::vector<std::complex<RealT>>& input, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor, Rows, Cols> MatrixType;
    Eigen::Map<const MatrixType> matrix(input.data(), Rows, Cols);
    Eigen::BDCSVD<MatrixType,
                  Eigen::ComputeThinU | Eigen::ComputeThinV> decomposition(
        matrix);
    const auto& singulars = decomposition.singularValues();
    const auto& left = decomposition.matrixU();
    const auto right_adjoint = decomposition.matrixV().adjoint().eval();

    std::vector<double> normalized(singulars.size());
    for (int idx = 0; idx < singulars.size(); ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;
    std::vector<ComplexT> left_factors(left.size());
    for (int idx = 0; idx < left.size(); ++idx)
        left_factors[idx] = ComplexT{
            static_cast<RealT>(left.data()[idx].real()),
            static_cast<RealT>(left.data()[idx].imag())};
    std::vector<ComplexT> right_factors(right_adjoint.size());
    for (int idx = 0; idx < right_adjoint.size(); ++idx)
        right_factors[idx] = ComplexT{
            static_cast<RealT>(right_adjoint.data()[idx].real()),
            static_cast<RealT>(right_adjoint.data()[idx].imag())};
    return OSRTriplet<ComplexT>(std::move(normalized),
                                std::move(left_factors),
                                std::move(right_factors));
}

template<class ComplexT, class RealT>
static OSRTriplet<ComplexT> eigen_osr_triplet_dynamic(
    const std::vector<std::complex<RealT>>& input,
    int rows, int cols, double Fnorm)
{
    typedef std::complex<RealT> Scalar;
    if (cols == 4) {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 4,
                              Eigen::ColMajor> MatrixType;
        typedef Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor> GramType;
        Eigen::Map<const MatrixType> matrix(input.data(), rows, 4);
        const GramType gram = matrix.adjoint() * matrix;
        Eigen::SelfAdjointEigenSolver<GramType> decomposition(gram);
        if (decomposition.info() != Eigen::Success)
            throw std::runtime_error("OSR dynamic Gram eigensolver failed");

        Eigen::Matrix<RealT, 4, 1> singulars;
        GramType right_vectors;
        for (int idx = 0; idx < 4; ++idx) {
            singulars[idx] = std::sqrt(std::max(
                RealT(0), decomposition.eigenvalues()[3 - idx]));
            right_vectors.col(idx) =
                decomposition.eigenvectors().col(3 - idx);
        }

        MatrixType left = matrix * right_vectors;
        const RealT cutoff = Eigen::NumTraits<RealT>::epsilon() *
            static_cast<RealT>(rows) * singulars[0];
        for (int idx = 0; idx < 4; ++idx) {
            if (singulars[idx] > cutoff)
                left.col(idx) /= singulars[idx];
            else
                left.col(idx).setZero();
        }

        std::vector<double> normalized(4);
        for (int idx = 0; idx < 4; ++idx)
            normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;
        std::vector<ComplexT> left_factors(left.size());
        for (int idx = 0; idx < left.size(); ++idx)
            left_factors[idx] = ComplexT{
                static_cast<RealT>(left.data()[idx].real()),
                static_cast<RealT>(left.data()[idx].imag())};
        std::vector<ComplexT> right_factors(16);
        const GramType right_adjoint = right_vectors.adjoint();
        for (int idx = 0; idx < right_adjoint.size(); ++idx)
            right_factors[idx] = ComplexT{
                static_cast<RealT>(right_adjoint.data()[idx].real()),
                static_cast<RealT>(right_adjoint.data()[idx].imag())};
        return OSRTriplet<ComplexT>(std::move(normalized),
                                    std::move(left_factors),
                                    std::move(right_factors));
    }

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor> MatrixType;
    Eigen::Map<const MatrixType> matrix(input.data(), rows, cols);
    Eigen::BDCSVD<MatrixType,
                  Eigen::ComputeThinU | Eigen::ComputeThinV> decomposition(
        matrix);
    const auto& singulars = decomposition.singularValues();
    const auto& left = decomposition.matrixU();
    const auto& right_adjoint = decomposition.matrixV().adjoint();

    std::vector<double> normalized(singulars.size());
    for (int idx = 0; idx < singulars.size(); ++idx)
        normalized[idx] = static_cast<double>(singulars[idx]) / Fnorm;

    std::vector<ComplexT> left_factors(left.size());
    for (int idx = 0; idx < left.size(); ++idx)
        left_factors[idx] = ComplexT{
            static_cast<RealT>(left.data()[idx].real()),
            static_cast<RealT>(left.data()[idx].imag())};

    std::vector<ComplexT> right_factors(right_adjoint.size());
    for (int col = 0; col < right_adjoint.cols(); ++col) {
        for (int row = 0; row < right_adjoint.rows(); ++row) {
            const Scalar value = right_adjoint(row, col);
            right_factors[row + col * right_adjoint.rows()] = ComplexT{
                static_cast<RealT>(value.real()),
                static_cast<RealT>(value.imag())};
        }
    }
    return OSRTriplet<ComplexT>(std::move(normalized),
                                std::move(left_factors),
                                std::move(right_factors));
}

template<class ComplexT, class RealT>
static OSRTriplet<ComplexT> eigen_osr_triplet(
    const std::vector<std::complex<RealT>>& input,
    int rows, int cols, double Fnorm)
{
    if (rows == 4 && cols == 4)
        return eigen_osr_triplet_gram_fixed<4, ComplexT, RealT>(input, Fnorm);
    if (rows == 16 && cols == 4)
        return eigen_osr_triplet_gram_fixed<16, ComplexT, RealT>(input, Fnorm);
    if (rows == 64 && cols == 4)
        return eigen_osr_triplet_gram_fixed<64, ComplexT, RealT>(input, Fnorm);
    if (rows == 16 && cols == 16)
        return eigen_osr_triplet_bdc_fixed<16, 16, ComplexT, RealT>(
            input, Fnorm);
    return eigen_osr_triplet_dynamic<ComplexT, RealT>(
        input, rows, cols, Fnorm);
}

} // namespace

std::vector<double> osr_eigen_singular_values(
    const std::vector<std::complex<double>>& input,
    int rows, int cols, double frobenius_norm)
{
    return osr<double>(input, rows, cols, frobenius_norm);
}

std::vector<double> osr_eigen_singular_values(
    const std::vector<std::complex<float>>& input,
    int rows, int cols, double frobenius_norm)
{
    return osr<float>(input, rows, cols, frobenius_norm);
}

OSRTriplet<QGD_Complex16> osr_eigen_triplet(
    const std::vector<std::complex<double>>& input,
    int rows, int cols, double frobenius_norm)
{
    return eigen_osr_triplet<QGD_Complex16, double>(
        input, rows, cols, frobenius_norm);
}

OSRTriplet<QGD_Complex8> osr_eigen_triplet(
    const std::vector<std::complex<float>>& input,
    int rows, int cols, double frobenius_norm)
{
    return eigen_osr_triplet<QGD_Complex8, float>(
        input, rows, cols, frobenius_norm);
}
