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

#pragma once

#include "matrix.h"        // For Matrix wrapper
#include "matrix_base.hpp" // From existing SQUANDER
#include <memory>
#include <stdexcept>
#include <vector>

namespace squander {
namespace density {

/**
 * @brief Quantum density matrix ρ for mixed-state representation
 *
 * Inherits from matrix_base<QGD_Complex16> to reuse:
 * - Automatic memory management (reference counting)
 * - BLAS integration via dot()
 * - Cache-line aligned memory
 * - Thread-safe operations with TBB spin_mutex
 *
 * Adds density matrix specific:
 * - Validation (Hermitian, Tr(ρ)=1, ρ≥0)
 * - Quantum properties (purity, entropy)
 * - Unitary evolution (ρ → UρU†)
 * - Partial trace operations
 *
 * Mathematical properties:
 * - ρ = ρ† (Hermitian)
 * - Tr(ρ) = 1 (normalized)
 * - ρ ≥ 0 (positive semi-definite)
 * - Tr(ρ²) ∈ [1/2^n, 1] (purity)
 *
 * Example:
 * @code
 *   using namespace squander::density;
 *   DensityMatrix rho(2);  // 2 qubits, initialized to |00⟩⟨00|
 *   std::cout << "Purity: " << rho.purity() << std::endl;  // 1.0
 * @endcode
 */
class DensityMatrix : public matrix_base<QGD_Complex16> {
public:
  // ================================================================
  // Constructors & Destructor
  // ================================================================

  /**
   * @brief Create density matrix for n qubits
   * @param qbit_num Number of qubits (must be >= 1)
   * @throws std::invalid_argument if qbit_num < 1
   *
   * Initializes to pure ground state: ρ = |0...0⟩⟨0...0|
   * Matrix dimension is 2^qbit_num × 2^qbit_num
   */
  explicit DensityMatrix(int qbit_num);

  /**
   * @brief Create from state vector: ρ = |ψ⟩⟨ψ|
   * @param state_vector Pure state (column vector, inherits from matrix_base)
   * @throws std::invalid_argument if state_vector is not column vector or dim
   * not power of 2
   *
   * Computes outer product to create density matrix.
   * Resulting matrix represents a pure state with Tr(ρ²) = 1.
   */
  explicit DensityMatrix(const matrix_base<QGD_Complex16> &state_vector);

  /**
   * @brief Wrap existing data (non-owning wrapper)
   * @param data Pointer to matrix data (row-major, size dim × dim)
   * @param dim Matrix dimension (must be power of 2)
   * @throws std::invalid_argument if dim is not power of 2
   *
   * Creates non-owning view of existing memory.
   * Caller is responsible for lifetime management of data.
   */
  DensityMatrix(QGD_Complex16 *data, int dim);

  /**
   * @brief Copy constructor
   * @param other Density matrix to copy
   *
   * Uses matrix_base reference counting - shares memory until modified.
   */
  DensityMatrix(const DensityMatrix &other);

  /**
   * @brief Move constructor
   * @param other Density matrix to move from
   *
   * Takes ownership of other's data, leaving other in valid but empty state.
   */
  DensityMatrix(DensityMatrix &&other) noexcept;

  /**
   * @brief Destructor (uses base class default)
   */
  ~DensityMatrix() = default;

  // ================================================================
  // Assignment Operators
  // ================================================================

  DensityMatrix &operator=(const DensityMatrix &other);
  DensityMatrix &operator=(DensityMatrix &&other) noexcept;

  /**
   * @brief Element access: ρ(i,j)
   * @param i Row index (0 to dim-1)
   * @param j Column index (0 to dim-1)
   * @return Reference to matrix element
   * @throws std::out_of_range if indices invalid
   */
  QGD_Complex16 &operator()(int i, int j);
  const QGD_Complex16 &operator()(int i, int j) const;

  // ================================================================
  // Basic Properties
  // ================================================================

  /**
   * @brief Get number of qubits
   * @return Number of qubits n (matrix dimension is 2^n)
   */
  int get_qbit_num() const { return qbit_num_; }

  /**
   * @brief Get matrix dimension (2^qbit_num)
   * @return Matrix dimension
   */
  int get_dim() const { return rows; }

  // ================================================================
  // Quantum Properties
  // ================================================================

  /**
   * @brief Calculate trace: Tr(ρ)
   * @return Complex number (should be 1+0i for valid density matrix)
   *
   * For valid density matrix: Tr(ρ) = 1
   * Computational complexity: O(2^n)
   */
  QGD_Complex16 trace() const;

  /**
   * @brief Calculate purity: Tr(ρ²)
   * @return Real number in [1/2^n, 1]
   *         - 1 = pure state
   *         - 1/2^n = maximally mixed state
   *
   * Computational complexity: O(2^(3n)) via matrix multiplication
   */
  double purity() const;

  /**
   * @brief von Neumann entropy: S(ρ) = -Tr(ρ log₂ ρ)
   * @return Real number ≥ 0
   *         - 0 = pure state
   *         - log₂(2^n) = n bits = maximally mixed state
   *
   * Computational complexity: O(2^(3n)) via eigenvalue decomposition
   */
  double entropy() const;

  /**
   * @brief Check if valid density matrix
   * @param tol Numerical tolerance (default: 1e-10)
   * @return true if: ρ=ρ†, Tr(ρ)=1, ρ≥0
   *
   * Performs three checks:
   * 1. Hermitian: ρ(i,j) = conj(ρ(j,i))
   * 2. Normalized: Tr(ρ) = 1
   * 3. Positive semi-definite: all eigenvalues ≥ 0
   */
  bool is_valid(double tol = 1e-10) const;

  /**
   * @brief Get eigenvalues (sorted descending)
   * @return Vector of eigenvalues (all should be ≥ 0 for valid density matrix)
   *
   * Uses LAPACK zheev for Hermitian eigenvalue decomposition.
   * Computational complexity: O(2^(3n))
   */
  std::vector<double> eigenvalues() const;

  // ================================================================
  // Operations
  // ================================================================

  /**
   * @brief Apply unitary transformation: ρ → UρU†
   * @param U Unitary matrix (must be matrix_base<QGD_Complex16>)
   * @throws std::runtime_error if dimension mismatch
   *
   * Modifies density matrix in-place.
   * Uses BLAS for efficient matrix multiplication:
   * - temp = U * ρ
   * - ρ = temp * U†
   *
   * Computational complexity: O(2^(3n))
   */
  void apply_unitary(const matrix_base<QGD_Complex16> &U);

  /**
   * @brief Compute partial trace over specified qubits
   * @param trace_out List of qubit indices to trace out
   * @return Reduced density matrix
   *
   * Example: For 3 qubits, trace_out={2} gives 2-qubit reduced density matrix.
   * ρ_A = Tr_B(ρ_AB) where B are the traced-out qubits.
   *
   * Computational complexity: O(2^(2n))
   */
  DensityMatrix partial_trace(const std::vector<int> &trace_out) const;

  /**
   * @brief Create deep copy
   * @return New density matrix with copied data
   *
   * Unlike copy constructor (which shares memory), this creates independent
   * copy.
   */
  DensityMatrix clone() const;

  // ================================================================
  // Static Factory Methods
  // ================================================================

  /**
   * @brief Create maximally mixed state: ρ = I/2^n
   * @param qbit_num Number of qubits
   * @return Maximally mixed density matrix
   *
   * Properties:
   * - Purity: 1/2^n (minimal)
   * - Entropy: n bits (maximal)
   * - All eigenvalues equal: 1/2^n
   */
  static DensityMatrix maximally_mixed(int qbit_num);

  // ================================================================
  // Utilities
  // ================================================================

  /**
   * @brief Print matrix with properties
   *
   * Displays:
   * - Number of qubits
   * - Matrix dimension
   * - Trace
   * - Purity
   * - Validity
   * - Matrix elements
   */
  void print() const;

private:
  int qbit_num_; ///< Number of qubits

  // Helper methods
  void validate_dimensions() const;
  bool is_hermitian(double tol) const;
};

} // namespace density
} // namespace squander
