#include "OSR_Eigen_SVD.h"
#include "OSR_Cut_Capacity.h"
#include <array>
#include <limits>

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
    if (m_rows == 8 && m_cols == 8)
        return eigen_osr_fixed<8, 8, RealT>(A, Fnorm);
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
    // Evaluating an adjoint with auto produces row-major storage. The public
    // factors below are column-major, so retain that layout explicitly.
    const MatrixType right_adjoint = decomposition.matrixV().adjoint();

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
    if (rows == 8 && cols == 8)
        return eigen_osr_triplet_fixed<8, 8, ComplexT, RealT>(input, Fnorm);
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

namespace {

using CutComplex = std::complex<double>;
using CutBody = Eigen::Matrix<CutComplex, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::ColMajor, 8, 8>;

int cut_embed(int index, int root, int bit) {
    const int low = (1 << root) - 1;
    return (index & low) | ((index & ~low) << 1) | (bit << root);
}

Eigen::Matrix2cd cut_axis_basis(const Eigen::Vector3d& axis) {
    CutComplex a, b;
    if (axis.z() >= 0.0) {
        a = std::sqrt(0.5 * (1.0 + axis.z()));
        b = CutComplex(axis.x(), axis.y()) / std::sqrt(2.0 * (1.0 + axis.z()));
    } else {
        a = CutComplex(axis.x(), -axis.y()) / std::sqrt(2.0 * (1.0 - axis.z()));
        b = std::sqrt(0.5 * (1.0 - axis.z()));
    }
    const double norm = std::sqrt(std::norm(a) + std::norm(b));
    a /= norm;
    b /= norm;
    Eigen::Matrix2cd result;
    result << a, -std::conj(b), b, std::conj(a);
    return result;
}

bool cut_polar(const CutBody& matrix, CutBody& polar) {
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
    Eigen::JacobiSVD<CutBody, Eigen::ComputeFullU | Eigen::ComputeFullV> solver(matrix);
#else
    Eigen::JacobiSVD<CutBody> solver(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
#endif
    if (solver.info() != Eigen::Success) return false;
    polar.noalias() = solver.matrixU() * solver.matrixV().adjoint();
    return polar.allFinite();
}

// Exact normalized Frobenius distance to conjugate-paired eigenphase spectra,
// allowing a common phase and arbitrary body conjugation. D <= 8 makes the
// perfect-matchings/sign enumeration small (at most 105 * 8 choices).
struct CutPairedSpectrum {
    int dimension;
    std::array<CutComplex, 8> values;
    std::array<int, 8> pairs{}, best_pairs{};
    double best_score = -1.0;
    double phase = 0.0;

    void visit(unsigned remaining, int depth) {
        if (remaining == 0) {
            std::array<CutComplex, 4> midpoints;
            if (depth < 1 || depth > static_cast<int>(midpoints.size()))
                throw std::logic_error("Paired cut spectrum exceeds its leaf workspace");
            for (int pair = 0; pair < depth; ++pair) {
                midpoints[pair] = std::sqrt(values[pairs[2 * pair]] *
                                           values[pairs[2 * pair + 1]]);
            }
            for (unsigned signs = 0; signs < (1u << (depth - 1)); ++signs) {
                CutComplex sum = midpoints[0];
                for (int pair = 1; pair < depth; ++pair) {
                    sum += ((signs >> (pair - 1)) & 1u ? -1.0 : 1.0) * midpoints[pair];
                }
                const double score = std::abs(sum);
                if (score > best_score) {
                    best_score = score;
                    best_pairs = pairs;
                    phase = std::arg(sum);
                }
            }
            return;
        }
        if (depth < 0 || depth >= static_cast<int>(pairs.size() / 2))
            throw std::logic_error("Paired cut spectrum exceeds its matching workspace");
        int first = 0;
        while ((remaining & (1u << first)) == 0) ++first;
        remaining ^= 1u << first;
        for (int second = first + 1; second < dimension; ++second) {
            if ((remaining & (1u << second)) == 0) continue;
            pairs[2 * depth] = first;
            pairs[2 * depth + 1] = second;
            visit(remaining ^ (1u << second), depth + 1);
        }
    }

    double distance() {
        if (dimension < 2 || dimension > static_cast<int>(values.size()) || (dimension & 1))
            throw std::logic_error("Paired cut spectrum requires an even body dimension up to eight");
        visit((1u << dimension) - 1u, 0);
        const CutComplex forward = std::polar(1.0, phase);
        double squared = 0.0;
        for (int pair = 0; pair < dimension / 2; ++pair) {
            const CutComplex a = values[best_pairs[2 * pair]];
            const CutComplex b = values[best_pairs[2 * pair + 1]];
            const double angle = std::arg(a * std::conj(forward) + std::conj(b) * forward);
            // Direct residuals avoid subtractive cancellation near zero.
            squared += std::norm(a - std::polar(1.0, phase + angle));
            squared += std::norm(b - std::polar(1.0, phase - angle));
        }
        return std::sqrt(squared / dimension);
    }
};

} // namespace

namespace squander_osr {

CutCapacity cut_capacity(const QGD_Complex16* data, int qubit_count, int stride,
                         int root, double tolerance, bool adjoint) {
    CutCapacity result;
    if (!data || qubit_count < 2 || qubit_count > 4 || root < 0 || root >= qubit_count ||
        !std::isfinite(tolerance) || tolerance < 0.0 || tolerance >= 1.0) return result;
    const int dimension = 1 << qubit_count;
    const int body = dimension / 2;
    if (stride < dimension) return result;
    const auto read = [&](int row, int col) -> CutComplex {
        const auto& value = data[(adjoint ? col : row) * stride + (adjoint ? row : col)];
        return CutComplex(value.real, adjoint ? -value.imag : value.imag);
    };
    const double rounding = 64.0 * std::numeric_limits<double>::epsilon() *
                            dimension * dimension * dimension;
    double unitarity_squared = 0.0;
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            CutComplex gram = 0.0;
            for (int inner = 0; inner < dimension; ++inner) {
                gram += std::conj(read(inner, row)) * read(inner, col);
            }
            if (row == col) gram -= 1.0;
            unitarity_squared += std::norm(gram);
        }
    }
    const double unitarity_distance = std::sqrt(unitarity_squared / dimension);
    if (!std::isfinite(unitarity_distance) || unitarity_distance > rounding) return result;
    const double requested = std::sqrt(2.0 * tolerance / (1.0 + std::sqrt(1.0 - tolerance)));
    const double radius = requested + 2.0 * unitarity_distance + rounding;
    const double channel_lipschitz = 2.0 * std::sqrt(3.0);
    result.transport_distance_bound = channel_lipschitz * radius;

    std::array<CutBody, 3> transport;
    Eigen::Matrix3d channel;
    for (int pauli = 0; pauli < 3; ++pauli) {
        transport[pauli].resize(body, body);
        const auto coefficient = [&](int index) -> CutComplex {
            const bool bit = (index & (1 << root)) != 0;
            if (pauli == 0) return 1.0;
            if (pauli == 1) return CutComplex(0.0, bit ? -1.0 : 1.0);
            return bit ? -1.0 : 1.0;
        };
        for (int row = 0; row < body; ++row) {
            for (int col = 0; col < body; ++col) {
                CutComplex partial = 0.0;
                for (int bit = 0; bit < 2; ++bit) {
                    for (int input = 0; input < dimension; ++input) {
                        const int source = pauli == 2 ? input : (input ^ (1 << root));
                        partial += coefficient(input) * read(cut_embed(row, root, bit), source) *
                                   std::conj(read(cut_embed(col, root, bit), input));
                    }
                }
                transport[pauli](row, col) = 0.5 * partial;
            }
        }
        CutComplex g00 = 0.0, g11 = 0.0, g01 = 0.0;
        for (int index = 0; index < body; ++index) {
            const int zero = cut_embed(index, root, 0);
            const int one = cut_embed(index, root, 1);
            for (int input = 0; input < dimension; ++input) {
                const int source = pauli == 2 ? input : (input ^ (1 << root));
                const CutComplex factor = coefficient(input);
                g00 += factor * read(zero, source) * std::conj(read(zero, input));
                g11 += factor * read(one, source) * std::conj(read(one, input));
                g01 += factor * read(zero, source) * std::conj(read(one, input));
            }
        }
        channel(0, pauli) = 2.0 * g01.real() / dimension;
        channel(1, pauli) = -2.0 * g01.imag() / dimension;
        channel(2, pauli) = (g00.real() - g11.real()) / dimension;
    }

    // With <= 2 crossing CNOTs, root-to-body Pauli transport has rank <= 1.
    // e2/trace is a lower bound on its squared distance to rank one. Computing
    // e2 through column-pair residuals is stable even for exact rank-one cuts.
    double norm_sum = 0.0, exterior_squared = 0.0;
    for (int left = 0; left < 3; ++left) {
        const double norm = transport[left].squaredNorm();
        norm_sum += norm;
        if (norm == 0.0) continue;
        for (int right = left + 1; right < 3; ++right) {
            CutComplex inner = 0.0;
            for (int index = 0; index < body * body; ++index) {
                inner += std::conj(transport[left].data()[index]) * transport[right].data()[index];
            }
            const CutComplex projection = inner / norm;
            double residual_squared = 0.0;
            for (int index = 0; index < body * body; ++index) {
                residual_squared += std::norm(transport[right].data()[index] -
                                              projection * transport[left].data()[index]);
            }
            exterior_squared += norm * residual_squared;
        }
    }
    result.valid = true;
    result.transport_tail_lower_bound = norm_sum == 0.0 ? 0.0 :
        std::sqrt(exterior_squared / (body * norm_sum));
    if (std::sqrt(norm_sum / body) > result.transport_distance_bound) {
        result.minimum_crossing_cnots = 2;
    }
    if (result.transport_tail_lower_bound > result.transport_distance_bound) {
        result.minimum_crossing_cnots = 3;
        return result;
    }

    // Two crossings have root-channel singular values {|gamma|,|gamma*t|,|t|}.
    // Near a controlled cut, |gamma| or |t| must be near one. The resulting
    // controlled relative body unitary is a product of balanced reflections,
    // and therefore has conjugate-paired eigenphases, up to a common phase.
#if EIGEN_VERSION_AT_LEAST(5, 0, 0)
    Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV> axes(channel);
#else
    Eigen::JacobiSVD<Eigen::Matrix3d> axes(channel, Eigen::ComputeFullU | Eigen::ComputeFullV);
#endif
    if (axes.info() != Eigen::Success) return result;
    const Eigen::Matrix2cd output_basis = cut_axis_basis(axes.matrixU().col(0));
    const Eigen::Matrix2cd input_basis = cut_axis_basis(axes.matrixV().col(0));
    std::array<CutBody, 4> framed;
    for (int out = 0; out < 2; ++out) {
        for (int in = 0; in < 2; ++in) {
            auto& block = framed[2 * out + in];
            block.resize(body, body);
            for (int row = 0; row < body; ++row) {
                for (int col = 0; col < body; ++col) {
                    CutComplex value = 0.0;
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            value += std::conj(output_basis(a, out)) *
                                     read(cut_embed(row, root, a), cut_embed(col, root, b)) *
                                     input_basis(b, in);
                        }
                    }
                    block(row, col) = value;
                }
            }
        }
    }
    CutBody zero(body, body), one(body, body);
    if (!cut_polar(framed[0], zero) || !cut_polar(framed[3], one)) return result;
    result.controlled_projection_distance = std::sqrt(
        ((framed[0] - zero).squaredNorm() + (framed[3] - one).squaredNorm() +
         framed[1].squaredNorm() + framed[2].squaredNorm()) / dimension);
    const CutBody relative = zero.adjoint() * one;
    result.channel_gap = 1.0 - std::min(1.0, std::abs(relative.trace()) / body);
    const double controlled_radius = radius + result.controlled_projection_distance;
    const double two_crossing_projection = controlled_radius +
        std::sqrt(channel_lipschitz * controlled_radius);
    result.paired_distance_bound = result.channel_gap <= rounding ? 2.0 :
        2.0 * (two_crossing_projection + 2.0 * std::sqrt(2.0) *
               channel_lipschitz * two_crossing_projection / result.channel_gap);
    // All normalized unitary distances are <= 2. Avoid an unnecessary Schur
    // decomposition when the conservative perturbation bound cannot exclude.
    if (result.paired_distance_bound + rounding >= 2.0) return result;
    Eigen::ComplexSchur<CutBody> schur(relative, false);
    if (schur.info() != Eigen::Success) return result;
    CutPairedSpectrum spectrum;
    spectrum.dimension = body;
    for (int index = 0; index < body; ++index) {
        const CutComplex value = schur.matrixT()(index, index);
        const double norm = std::abs(value);
        if (!std::isfinite(norm) || norm == 0.0) return result;
        spectrum.values[index] = value / norm;
    }
    result.paired_spectrum_distance = spectrum.distance();
    result.spectrum_checked = true;
    if (result.paired_spectrum_distance > result.paired_distance_bound + rounding) {
        result.minimum_crossing_cnots = 3;
    }
    return result;
}

} // namespace squander_osr
