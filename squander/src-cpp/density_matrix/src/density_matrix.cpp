/*
Copyright 2025 SQUANDER Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "density_matrix.h"
#include "dot.h" // BLAS wrapper from existing SQUANDER
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

extern "C" {
// LAPACK eigenvalue solver for Hermitian matrices
void zheev_(char *jobz, char *uplo, int *n, QGD_Complex16 *a, int *lda,
            double *w, QGD_Complex16 *work, int *lwork, double *rwork,
            int *info);
}

namespace squander {
namespace density {

// ================================================================
// Helper Functions
// ================================================================

static int calculate_qbit_num(int dim) {
  int qbit_num = 0;
  int temp = dim;
  while (temp > 1) {
    if (temp % 2 != 0) {
      throw std::invalid_argument(
          "DensityMatrix: dimension must be power of 2");
    }
    temp /= 2;
    qbit_num++;
  }
  return qbit_num;
}

// ================================================================
// Constructors
// ================================================================

DensityMatrix::DensityMatrix(int qbit_num)
    : matrix_base<QGD_Complex16>(1 << qbit_num, 1 << qbit_num),
      qbit_num_(qbit_num) {
  if (qbit_num < 1) {
    throw std::invalid_argument("DensityMatrix: qbit_num must be >= 1");
  }

  int dim = 1 << qbit_num; // 2^qbit_num

  // Matrix already allocated by base class constructor
  // references and reference_mutex already created by base class
  // Just initialize to |0⟩⟨0| (ground state) - only ρ(0,0) = 1
  memset(data, 0, dim * dim * sizeof(QGD_Complex16));
  data[0].real = 1.0;
  data[0].imag = 0.0;
}

DensityMatrix::DensityMatrix(const matrix_base<QGD_Complex16> &state_vector)
    : DensityMatrix(calculate_qbit_num(state_vector.rows)) {
  // Validate input
  if (state_vector.cols != 1) {
    throw std::invalid_argument(
        "DensityMatrix: state_vector must be column vector (cols=1)");
  }

  // Compute outer product: ρ(i,j) = ψ(i) * conj(ψ(j))
  int dim = state_vector.rows;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      QGD_Complex16 psi_i = state_vector.data[i * state_vector.stride];
      QGD_Complex16 psi_j = state_vector.data[j * state_vector.stride];

      // Complex multiplication: ψ(i) * conj(ψ(j))
      data[i * stride + j].real =
          psi_i.real * psi_j.real + psi_i.imag * psi_j.imag;
      data[i * stride + j].imag =
          psi_i.imag * psi_j.real - psi_i.real * psi_j.imag;
    }
  }
}

DensityMatrix::DensityMatrix(QGD_Complex16 *data_in, int dim)
    : matrix_base<QGD_Complex16>(data_in, dim, dim) {
  // Calculate qbit_num
  qbit_num_ = 0;
  int temp = dim;
  while (temp > 1) {
    if (temp % 2 != 0) {
      throw std::invalid_argument(
          "DensityMatrix: dimension must be power of 2");
    }
    temp /= 2;
    qbit_num_++;
  }
}

DensityMatrix::DensityMatrix(const DensityMatrix &other)
    : matrix_base<QGD_Complex16>(other), qbit_num_(other.qbit_num_) {}

DensityMatrix::DensityMatrix(DensityMatrix &&other) noexcept
    : matrix_base<QGD_Complex16>(std::move(other)), qbit_num_(other.qbit_num_) {
  other.qbit_num_ = 0;
}

// ================================================================
// Assignment Operators
// ================================================================

DensityMatrix &DensityMatrix::operator=(const DensityMatrix &other) {
  if (this != &other) {
    matrix_base<QGD_Complex16>::operator=(other);
    qbit_num_ = other.qbit_num_;
  }
  return *this;
}

DensityMatrix &DensityMatrix::operator=(DensityMatrix &&other) noexcept {
  if (this != &other) {
    matrix_base<QGD_Complex16>::operator=(std::move(other));
    qbit_num_ = other.qbit_num_;
    other.qbit_num_ = 0;
  }
  return *this;
}

QGD_Complex16 &DensityMatrix::operator()(int i, int j) {
  if (i < 0 || i >= rows || j < 0 || j >= cols) {
    throw std::out_of_range("DensityMatrix: index out of bounds");
  }
  return data[i * stride + j];
}

const QGD_Complex16 &DensityMatrix::operator()(int i, int j) const {
  if (i < 0 || i >= rows || j < 0 || j >= cols) {
    throw std::out_of_range("DensityMatrix: index out of bounds");
  }
  return data[i * stride + j];
}

// ================================================================
// Quantum Properties
// ================================================================

QGD_Complex16 DensityMatrix::trace() const {
  QGD_Complex16 tr;
  tr.real = 0.0;
  tr.imag = 0.0;

  // Sum diagonal elements
  for (int i = 0; i < rows; i++) {
    tr.real += data[i * stride + i].real;
    tr.imag += data[i * stride + i].imag;
  }

  return tr;
}

double DensityMatrix::purity() const {
  // Compute ρ² using BLAS
  Matrix rho_matrix(data, rows, cols, stride);

  // Use existing SQUANDER BLAS wrapper: ρ² = ρ * ρ
  Matrix rho_squared = dot(rho_matrix, rho_matrix);

  // Trace of ρ²
  double pur = 0.0;
  for (int i = 0; i < rows; i++) {
    pur += rho_squared.get_data()[i * rho_squared.stride + i].real;
  }

  return pur;
}

double DensityMatrix::entropy() const {
  std::vector<double> eigs = eigenvalues();

  double S = 0.0;
  for (double lambda : eigs) {
    if (lambda > 1e-14) { // Avoid log(0)
      S -= lambda * std::log2(lambda);
    }
  }

  return S;
}

bool DensityMatrix::is_valid(double tol) const {
  // Check 1: Hermitian
  if (!is_hermitian(tol)) {
    return false;
  }

  // Check 2: Trace = 1
  QGD_Complex16 tr = trace();
  if (std::abs(tr.real - 1.0) > tol || std::abs(tr.imag) > tol) {
    return false;
  }

  // Check 3: Positive semi-definite (all eigenvalues ≥ 0)
  std::vector<double> eigs = eigenvalues();
  for (double eig : eigs) {
    if (eig < -tol) {
      return false;
    }
  }

  return true;
}

bool DensityMatrix::is_hermitian(double tol) const {
  // Check ρ(i,j) = conj(ρ(j,i)) for all i,j
  for (int i = 0; i < rows; i++) {
    for (int j = i; j < cols; j++) {
      QGD_Complex16 rho_ij = data[i * stride + j];
      QGD_Complex16 rho_ji = data[j * stride + i];

      // Check real part: Re(ρ(i,j)) = Re(ρ(j,i))
      if (std::abs(rho_ij.real - rho_ji.real) > tol) {
        return false;
      }

      // Check imaginary part: Im(ρ(i,j)) = -Im(ρ(j,i))
      if (std::abs(rho_ij.imag + rho_ji.imag) > tol) {
        return false;
      }
    }
  }
  return true;
}

std::vector<double> DensityMatrix::eigenvalues() const {
  // Copy matrix (LAPACK modifies input)
  DensityMatrix rho_copy = clone();

  std::vector<double> eigs(rows);

  // LAPACK workspace query
  char jobz = 'N'; // Eigenvalues only (not eigenvectors)
  char uplo = 'U'; // Upper triangle
  int n = rows;
  int lda = stride;
  int lwork = 2 * n; // Workspace size
  std::vector<QGD_Complex16> work(lwork);
  std::vector<double> rwork(3 * n);
  int info;

  // Call LAPACK Hermitian eigenvalue solver
  zheev_(&jobz, &uplo, &n, rho_copy.data, &lda, eigs.data(), work.data(),
         &lwork, rwork.data(), &info);

  if (info != 0) {
    throw std::runtime_error(
        "DensityMatrix::eigenvalues: LAPACK zheev failed with info=" +
        std::to_string(info));
  }

  // Sort descending (LAPACK returns ascending)
  std::sort(eigs.rbegin(), eigs.rend());

  return eigs;
}

// ================================================================
// Operations
// ================================================================

void DensityMatrix::apply_unitary(const matrix_base<QGD_Complex16> &U) {
  if (U.rows != rows || U.cols != cols) {
    throw std::runtime_error(
        "DensityMatrix::apply_unitary: dimension mismatch. " +
        std::string("Expected ") + std::to_string(rows) + "x" +
        std::to_string(cols) + ", got " + std::to_string(U.rows) + "x" +
        std::to_string(U.cols));
  }

  // Wrap as Matrix for BLAS operations
  Matrix rho_matrix(data, rows, cols, stride);
  Matrix U_matrix(const_cast<QGD_Complex16 *>(U.data), U.rows, U.cols,
                  U.stride);

  // Step 1: Compute U * ρ
  Matrix temp = dot(U_matrix, rho_matrix);

  // Step 2: Compute (U * ρ) * U†
  // Create U† (conjugate transpose)
  Matrix U_dagger = U_matrix.copy();
  U_dagger.conjugate();
  U_dagger.transpose();

  // Compute result and copy back to original data
  Matrix result = dot(temp, U_dagger);
  memcpy(data, result.get_data(), rows * cols * sizeof(QGD_Complex16));
}

DensityMatrix
DensityMatrix::partial_trace(const std::vector<int> &trace_out) const {
  // Validate inputs
  if (trace_out.empty()) {
    return clone(); // No qubits to trace out
  }

  for (int q : trace_out) {
    if (q < 0 || q >= qbit_num_) {
      throw std::invalid_argument(
          "DensityMatrix::partial_trace: qubit index out of range");
    }
  }

  // Calculate dimensions
  int trace_out_num = trace_out.size();
  int keep_num = qbit_num_ - trace_out_num;

  if (keep_num < 1) {
    throw std::invalid_argument(
        "DensityMatrix::partial_trace: cannot trace out all qubits");
  }

  int dim_reduced = 1 << keep_num;
  int dim_traced = 1 << trace_out_num;

  // Determine which qubits to keep
  std::vector<int> keep_qubits;
  for (int q = 0; q < qbit_num_; q++) {
    bool should_trace = false;
    for (int t : trace_out) {
      if (q == t) {
        should_trace = true;
        break;
      }
    }
    if (!should_trace) {
      keep_qubits.push_back(q);
    }
  }

  // Create reduced density matrix
  DensityMatrix rho_reduced(keep_num);
  memset(rho_reduced.data, 0,
         dim_reduced * dim_reduced * sizeof(QGD_Complex16));

  // Compute partial trace
  // ρ_A(i,j) = Σ_k ⟨i,k|ρ|j,k⟩ where k runs over traced qubits

  for (int i = 0; i < dim_reduced; i++) {
    for (int j = 0; j < dim_reduced; j++) {

      QGD_Complex16 sum;
      sum.real = 0.0;
      sum.imag = 0.0;

      // Sum over all basis states of traced-out qubits
      for (int k = 0; k < dim_traced; k++) {
        // Construct full basis state indices
        int full_i = 0;
        int full_j = 0;

        for (int q_idx = 0; q_idx < keep_num; q_idx++) {
          int q = keep_qubits[q_idx];
          if ((i >> q_idx) & 1) {
            full_i |= (1 << q);
          }
          if ((j >> q_idx) & 1) {
            full_j |= (1 << q);
          }
        }

        for (int t_idx = 0; t_idx < trace_out_num; t_idx++) {
          int q = trace_out[t_idx];
          if ((k >> t_idx) & 1) {
            full_i |= (1 << q);
            full_j |= (1 << q);
          }
        }

        // Add ρ(full_i, full_j) to sum
        sum.real += data[full_i * stride + full_j].real;
        sum.imag += data[full_i * stride + full_j].imag;
      }

      rho_reduced.data[i * dim_reduced + j] = sum;
    }
  }

  return rho_reduced;
}

DensityMatrix DensityMatrix::clone() const {
  DensityMatrix copy(qbit_num_);
  memcpy(copy.data, data, rows * stride * sizeof(QGD_Complex16));
  return copy;
}

// ================================================================
// Static Factory Methods
// ================================================================

DensityMatrix DensityMatrix::maximally_mixed(int qbit_num) {
  DensityMatrix rho(qbit_num);

  int dim = 1 << qbit_num;
  double prob = 1.0 / dim;

  // Set diagonal to 1/dim (identity matrix scaled)
  memset(rho.data, 0, dim * dim * sizeof(QGD_Complex16));
  for (int i = 0; i < dim; i++) {
    rho.data[i * rho.stride + i].real = prob;
    rho.data[i * rho.stride + i].imag = 0.0;
  }

  return rho;
}

// ================================================================
// Utilities
// ================================================================

void DensityMatrix::print() const {
  std::cout << "\n=== Density Matrix ===" << std::endl;
  std::cout << "Qubits: " << qbit_num_ << std::endl;
  std::cout << "Dimension: " << rows << " × " << cols << std::endl;

  QGD_Complex16 tr = trace();
  std::cout << "Trace: " << tr.real;
  if (std::abs(tr.imag) > 1e-10) {
    std::cout << " + " << tr.imag << "i";
  }
  std::cout << std::endl;

  std::cout << "Purity: " << purity() << std::endl;
  std::cout << "Entropy: " << entropy() << std::endl;
  std::cout << "Valid: " << (is_valid() ? "Yes" : "No") << std::endl;

  std::cout << "\nMatrix elements:" << std::endl;
  for (int i = 0; i < std::min(rows, 8);
       i++) { // Limit output for large matrices
    for (int j = 0; j < std::min(cols, 8); j++) {
      QGD_Complex16 val = data[i * stride + j];
      std::cout << "(" << val.real << "," << val.imag << ") ";
    }
    if (cols > 8)
      std::cout << "...";
    std::cout << std::endl;
  }
  if (rows > 8)
    std::cout << "..." << std::endl;
  std::cout << std::endl;
}

void DensityMatrix::validate_dimensions() const {
  if (rows != cols) {
    throw std::runtime_error("DensityMatrix: matrix must be square");
  }

  int expected_dim = 1 << qbit_num_;
  if (rows != expected_dim) {
    throw std::runtime_error("DensityMatrix: dimension mismatch. Expected " +
                             std::to_string(expected_dim) + ", got " +
                             std::to_string(rows));
  }
}

} // namespace density
} // namespace squander
