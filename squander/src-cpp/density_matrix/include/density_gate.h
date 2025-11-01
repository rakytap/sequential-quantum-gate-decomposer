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

#include "Gate.h" // Existing SQUANDER gate
#include "density_matrix.h"
#include "matrix_real.h" // For parameters

namespace squander {
namespace density {

/**
 * @brief Wrapper for existing Gate to apply to density matrices
 *
 * Reuses existing gate matrices but applies ρ → UρU† instead of |ψ⟩ → U|ψ⟩.
 * Does NOT take ownership of the wrapped gate.
 *
 * Example:
 * @code
 *   Gate* h_gate = new H(qbit_num, target_qbit);
 *   DensityGate density_h(h_gate);
 *
 *   Matrix_real params;  // Empty for non-parametric gates
 *   DensityMatrix rho(qbit_num);
 *   density_h.apply_to(params, rho);  // Applies Hadamard to density matrix
 * @endcode
 */
class DensityGate {
public:
  /**
   * @brief Wrap existing gate
   * @param gate Pointer to existing gate (NOT owned by this wrapper)
   * @throws std::invalid_argument if gate is nullptr
   *
   * The wrapped gate must remain valid for the lifetime of this wrapper.
   */
  explicit DensityGate(Gate *gate);

  /**
   * @brief Apply gate to density matrix: ρ → UρU†
   * @param params Gate parameters (empty for non-parametric gates)
   * @param rho Density matrix to transform (modified in-place)
   * @throws std::runtime_error if dimensions don't match
   *
   * Steps:
   * 1. Get gate matrix U from wrapped gate
   * 2. Apply ρ → UρU† using density matrix method
   */
  void apply_to(const matrix_base<double> &params, DensityMatrix &rho);

  /**
   * @brief Get underlying gate pointer
   * @return Pointer to wrapped gate (not owned)
   */
  Gate *get_gate() const { return gate_; }

  /**
   * @brief Get gate type
   */
  gate_type get_type() const;

  /**
   * @brief Get target qubit
   */
  int get_target_qbit() const;

  /**
   * @brief Get control qubit (-1 if none)
   */
  int get_control_qbit() const;

  /**
   * @brief Get number of parameters
   */
  int get_parameter_num() const;

  /**
   * @brief Get total number of qubits in system
   */
  int get_qbit_num() const;

private:
  Gate *gate_; ///< Pointer to existing gate (NOT owned)
};

} // namespace density
} // namespace squander
