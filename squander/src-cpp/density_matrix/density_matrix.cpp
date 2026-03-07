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

// ================================================================
// Optimized Local Unitary Application
// ================================================================

void DensityMatrix::apply_single_qubit_unitary(
    const matrix_base<QGD_Complex16> &u_2x2, int target_qbit) {
  if (target_qbit < 0 || target_qbit >= qbit_num_) {
    throw std::runtime_error(
        "DensityMatrix::apply_single_qubit_unitary: target_qbit out of range");
  }

  if (u_2x2.rows != 2 || u_2x2.cols != 2) {
    throw std::runtime_error(
        "DensityMatrix::apply_single_qubit_unitary: kernel must be 2x2");
  }

  int dim = rows;
  int target_step = 1 << target_qbit;

  // Temporary storage for the result
  std::vector<QGD_Complex16> temp(dim * dim);

  // The 2×2 kernel elements
  QGD_Complex16 u00 = u_2x2.data[0];
  QGD_Complex16 u01 = u_2x2.data[1];
  QGD_Complex16 u10 = u_2x2.data[2];
  QGD_Complex16 u11 = u_2x2.data[3];

  // Conjugates for U†
  QGD_Complex16 u00_conj = {u00.real, -u00.imag};
  QGD_Complex16 u01_conj = {u01.real, -u01.imag};
  QGD_Complex16 u10_conj = {u10.real, -u10.imag};
  QGD_Complex16 u11_conj = {u11.real, -u11.imag};

  // Apply ρ' = U ρ U†
  // For each element ρ'(i,j), we compute the transformation
  for (int i = 0; i < dim; i++) {
    int i_bit = (i >> target_qbit) & 1;
    int i_partner = i ^ target_step;

    for (int j = 0; j < dim; j++) {
      int j_bit = (j >> target_qbit) & 1;
      int j_partner = j ^ target_step;

      // Get the 2×2 block of ρ
      QGD_Complex16 r00, r01, r10, r11;

      if (i_bit == 0 && j_bit == 0) {
        r00 = data[i * stride + j];
        r01 = data[i * stride + j_partner];
        r10 = data[i_partner * stride + j];
        r11 = data[i_partner * stride + j_partner];
      } else if (i_bit == 0 && j_bit == 1) {
        r00 = data[i * stride + j_partner];
        r01 = data[i * stride + j];
        r10 = data[i_partner * stride + j_partner];
        r11 = data[i_partner * stride + j];
      } else if (i_bit == 1 && j_bit == 0) {
        r00 = data[i_partner * stride + j];
        r01 = data[i_partner * stride + j_partner];
        r10 = data[i * stride + j];
        r11 = data[i * stride + j_partner];
      } else {
        r00 = data[i_partner * stride + j_partner];
        r01 = data[i_partner * stride + j];
        r10 = data[i * stride + j_partner];
        r11 = data[i * stride + j];
      }

      // Compute U * block
      QGD_Complex16 t00, t01, t10, t11;

      // t = U * r
      t00.real = u00.real * r00.real - u00.imag * r00.imag +
                 u01.real * r10.real - u01.imag * r10.imag;
      t00.imag = u00.real * r00.imag + u00.imag * r00.real +
                 u01.real * r10.imag + u01.imag * r10.real;

      t01.real = u00.real * r01.real - u00.imag * r01.imag +
                 u01.real * r11.real - u01.imag * r11.imag;
      t01.imag = u00.real * r01.imag + u00.imag * r01.real +
                 u01.real * r11.imag + u01.imag * r11.real;

      t10.real = u10.real * r00.real - u10.imag * r00.imag +
                 u11.real * r10.real - u11.imag * r10.imag;
      t10.imag = u10.real * r00.imag + u10.imag * r00.real +
                 u11.real * r10.imag + u11.imag * r10.real;

      t11.real = u10.real * r01.real - u10.imag * r01.imag +
                 u11.real * r11.real - u11.imag * r11.imag;
      t11.imag = u10.real * r01.imag + u10.imag * r01.real +
                 u11.real * r11.imag + u11.imag * r11.real;

      // Compute result = t * U† (right multiply by conjugate transpose)
      // Select which element of the result we need based on i_bit, j_bit
      QGD_Complex16 result_elem;

      if (i_bit == 0 && j_bit == 0) {
        // result[0][0] = t00 * u00† + t01 * u01†
        result_elem.real = t00.real * u00_conj.real - t00.imag * u00_conj.imag +
                           t01.real * u01_conj.real - t01.imag * u01_conj.imag;
        result_elem.imag = t00.real * u00_conj.imag + t00.imag * u00_conj.real +
                           t01.real * u01_conj.imag + t01.imag * u01_conj.real;
      } else if (i_bit == 0 && j_bit == 1) {
        // result[0][1] = t00 * u10† + t01 * u11†
        result_elem.real = t00.real * u10_conj.real - t00.imag * u10_conj.imag +
                           t01.real * u11_conj.real - t01.imag * u11_conj.imag;
        result_elem.imag = t00.real * u10_conj.imag + t00.imag * u10_conj.real +
                           t01.real * u11_conj.imag + t01.imag * u11_conj.real;
      } else if (i_bit == 1 && j_bit == 0) {
        // result[1][0] = t10 * u00† + t11 * u01†
        result_elem.real = t10.real * u00_conj.real - t10.imag * u00_conj.imag +
                           t11.real * u01_conj.real - t11.imag * u01_conj.imag;
        result_elem.imag = t10.real * u00_conj.imag + t10.imag * u00_conj.real +
                           t11.real * u01_conj.imag + t11.imag * u01_conj.real;
      } else {
        // result[1][1] = t10 * u10† + t11 * u11†
        result_elem.real = t10.real * u10_conj.real - t10.imag * u10_conj.imag +
                           t11.real * u11_conj.real - t11.imag * u11_conj.imag;
        result_elem.imag = t10.real * u10_conj.imag + t10.imag * u10_conj.real +
                           t11.real * u11_conj.imag + t11.imag * u11_conj.real;
      }

      temp[i * dim + j] = result_elem;
    }
  }

  // Copy result back
  memcpy(data, temp.data(), dim * dim * sizeof(QGD_Complex16));
}

void DensityMatrix::apply_two_qubit_unitary(
    const matrix_base<QGD_Complex16> &u_2x2, int target_qbit, int control_qbit) {
  if (target_qbit < 0 || target_qbit >= qbit_num_) {
    throw std::runtime_error(
        "DensityMatrix::apply_two_qubit_unitary: target_qbit out of range");
  }
  if (control_qbit < 0 || control_qbit >= qbit_num_) {
    throw std::runtime_error(
        "DensityMatrix::apply_two_qubit_unitary: control_qbit out of range");
  }
  if (target_qbit == control_qbit) {
    throw std::runtime_error(
        "DensityMatrix::apply_two_qubit_unitary: target and control must differ");
  }

  if (u_2x2.rows != 2 || u_2x2.cols != 2) {
    throw std::runtime_error(
        "DensityMatrix::apply_two_qubit_unitary: kernel must be 2x2");
  }

  int dim = rows;
  int target_step = 1 << target_qbit;
  int control_step = 1 << control_qbit;

  // Temporary storage
  std::vector<QGD_Complex16> temp(dim * dim);
  memcpy(temp.data(), data, dim * dim * sizeof(QGD_Complex16));

  // The 2×2 kernel elements
  QGD_Complex16 u00 = u_2x2.data[0];
  QGD_Complex16 u01 = u_2x2.data[1];
  QGD_Complex16 u10 = u_2x2.data[2];
  QGD_Complex16 u11 = u_2x2.data[3];

  // For controlled gates, we only apply the 2x2 kernel when control=1
  // This is equivalent to applying identity when control=0

  for (int i = 0; i < dim; i++) {
    int i_control = (i >> control_qbit) & 1;

    for (int j = 0; j < dim; j++) {
      int j_control = (j >> control_qbit) & 1;

      // Only modify elements where control conditions are met
      // For ρ' = U ρ U†, we need to consider all combinations
      
      QGD_Complex16 result = {0.0, 0.0};

      // Sum over internal indices
      for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
          // U acts on target when control=1, identity otherwise
          QGD_Complex16 u_ia, u_jb_conj;

          int i_target = (i >> target_qbit) & 1;
          int j_target = (j >> target_qbit) & 1;

          // Get U element for row transformation
          if (i_control == 1) {
            if (i_target == 0 && a == 0)
              u_ia = u00;
            else if (i_target == 0 && a == 1)
              u_ia = u01;
            else if (i_target == 1 && a == 0)
              u_ia = u10;
            else
              u_ia = u11;
          } else {
            // Identity: only diagonal
            u_ia = (i_target == a) ? QGD_Complex16{1.0, 0.0}
                                   : QGD_Complex16{0.0, 0.0};
          }

          // Get U† element for column transformation
          if (j_control == 1) {
            if (j_target == 0 && b == 0)
              u_jb_conj = {u00.real, -u00.imag};
            else if (j_target == 0 && b == 1)
              u_jb_conj = {u10.real, -u10.imag};
            else if (j_target == 1 && b == 0)
              u_jb_conj = {u01.real, -u01.imag};
            else
              u_jb_conj = {u11.real, -u11.imag};
          } else {
            u_jb_conj = (j_target == b) ? QGD_Complex16{1.0, 0.0}
                                        : QGD_Complex16{0.0, 0.0};
          }

          // Compute index for ρ element
          int i_prime = (i & ~target_step) | (a << target_qbit);
          int j_prime = (j & ~target_step) | (b << target_qbit);

          QGD_Complex16 rho_ab = data[i_prime * stride + j_prime];

          // result += u_ia * rho_ab * u_jb_conj
          QGD_Complex16 prod1;
          prod1.real = u_ia.real * rho_ab.real - u_ia.imag * rho_ab.imag;
          prod1.imag = u_ia.real * rho_ab.imag + u_ia.imag * rho_ab.real;

          result.real +=
              prod1.real * u_jb_conj.real - prod1.imag * u_jb_conj.imag;
          result.imag +=
              prod1.real * u_jb_conj.imag + prod1.imag * u_jb_conj.real;
        }
      }

      temp[i * dim + j] = result;
    }
  }

  memcpy(data, temp.data(), dim * dim * sizeof(QGD_Complex16));
}

void DensityMatrix::apply_local_unitary(
    const matrix_base<QGD_Complex16> &u_kernel,
    const std::vector<int> &target_qbits) {
  
  int k = target_qbits.size();
  
  if (k == 0) {
    return;  // Nothing to do
  }
  
  if (k == 1) {
    apply_single_qubit_unitary(u_kernel, target_qbits[0]);
    return;
  }
  
  // Validate
  int kernel_dim = 1 << k;
  if (u_kernel.rows != kernel_dim || u_kernel.cols != kernel_dim) {
    throw std::runtime_error(
        "DensityMatrix::apply_local_unitary: kernel size mismatch");
  }
  
  for (int q : target_qbits) {
    if (q < 0 || q >= qbit_num_) {
      throw std::runtime_error(
          "DensityMatrix::apply_local_unitary: qubit index out of range");
    }
  }

  int dim = rows;
  std::vector<QGD_Complex16> temp(dim * dim);

  // General k-qubit application: ρ'(i,j) = Σ_{a,b} U(i_t,a) ρ(i',j') U†(b,j_t)
  // where i_t = target bits of i, i' = i with target bits replaced by a

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      QGD_Complex16 result = {0.0, 0.0};

      // Extract target bits from i and j
      int i_target = 0, j_target = 0;
      for (int ki = 0; ki < k; ki++) {
        if ((i >> target_qbits[ki]) & 1) {
          i_target |= (1 << ki);
        }
        if ((j >> target_qbits[ki]) & 1) {
          j_target |= (1 << ki);
        }
      }

      // Sum over all internal indices
      for (int a = 0; a < kernel_dim; a++) {
        for (int b = 0; b < kernel_dim; b++) {
          // Compute indices with replaced target bits
          int i_prime = i;
          int j_prime = j;
          for (int ki = 0; ki < k; ki++) {
            // Clear target bit
            i_prime &= ~(1 << target_qbits[ki]);
            j_prime &= ~(1 << target_qbits[ki]);
            // Set from a, b
            if ((a >> ki) & 1) {
              i_prime |= (1 << target_qbits[ki]);
            }
            if ((b >> ki) & 1) {
              j_prime |= (1 << target_qbits[ki]);
            }
          }

          // U[i_target, a]
          QGD_Complex16 u_ia = u_kernel.data[i_target * kernel_dim + a];
          // U†[b, j_target] = conj(U[j_target, b])
          QGD_Complex16 u_jb = u_kernel.data[j_target * kernel_dim + b];
          QGD_Complex16 u_jb_conj = {u_jb.real, -u_jb.imag};

          // ρ[i', j']
          QGD_Complex16 rho_ab = data[i_prime * stride + j_prime];

          // result += u_ia * rho_ab * u_jb_conj
          QGD_Complex16 prod1;
          prod1.real = u_ia.real * rho_ab.real - u_ia.imag * rho_ab.imag;
          prod1.imag = u_ia.real * rho_ab.imag + u_ia.imag * rho_ab.real;

          result.real +=
              prod1.real * u_jb_conj.real - prod1.imag * u_jb_conj.imag;
          result.imag +=
              prod1.real * u_jb_conj.imag + prod1.imag * u_jb_conj.real;
        }
      }

      temp[i * dim + j] = result;
    }
  }

  memcpy(data, temp.data(), dim * dim * sizeof(QGD_Complex16));
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
