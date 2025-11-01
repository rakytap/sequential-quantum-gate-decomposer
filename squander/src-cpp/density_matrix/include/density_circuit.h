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

#include "Gates_block.h" // From existing SQUANDER
#include "density_gate.h"
#include "density_matrix.h"
#include <memory>
#include <vector>

namespace squander {
namespace density {

/**
 * @brief Circuit for density matrix evolution
 *
 * Wraps existing Gates_block but applies operations to density matrices.
 * Provides same gate addition interface as qgd_Circuit.
 *
 * Example:
 * @code
 *   using namespace squander::density;
 *
 *   DensityCircuit circuit(2);  // 2-qubit circuit
 *   circuit.add_H(0);           // Hadamard on qubit 0
 *   circuit.add_CNOT(1, 0);     // CNOT
 *
 *   DensityMatrix rho(2);       // |00⟩⟨00|
 *   Matrix_real params;
 *   circuit.apply_to(params, rho);  // Apply circuit
 * @endcode
 */
class DensityCircuit {
public:
  /**
   * @brief Create empty circuit
   * @param qbit_num Number of qubits
   */
  explicit DensityCircuit(int qbit_num);

  /**
   * @brief Wrap existing Gates_block
   * @param circuit Pointer to existing circuit (takes ownership if
   * owns_circuit=true)
   * @param owns_circuit Whether to take ownership
   */
  explicit DensityCircuit(Gates_block *circuit, bool owns_circuit = false);

  /**
   * @brief Destructor
   */
  ~DensityCircuit();

  // ================================================================
  // Gate Addition (delegates to underlying Gates_block)
  // ================================================================

  void add_H(int target);
  void add_X(int target);
  void add_Y(int target);
  void add_Z(int target);
  void add_S(int target);
  void add_Sdg(int target);
  void add_T(int target);
  void add_Tdg(int target);
  void add_SX(int target);

  void add_RX(int target);
  void add_RY(int target);
  void add_RZ(int target);
  void add_U1(int target);
  void add_U2(int target);
  void add_U3(int target);

  void add_CNOT(int target, int control);
  void add_CZ(int target, int control);
  void add_CH(int target, int control);
  void add_CRY(int target, int control);
  void add_CRZ(int target, int control);
  void add_CRX(int target, int control);
  void add_CP(int target, int control);

  /**
   * @brief Apply circuit to density matrix
   * @param params Gate parameters
   * @param rho Initial density matrix (modified in-place)
   *
   * Applies each gate in sequence: ρ → U₁...U_n ρ U_n†...U₁†
   */
  void apply_to(const matrix_base<double> &params, DensityMatrix &rho);

  /**
   * @brief Get circuit unitary matrix
   * @param params Gate parameters
   * @return Full circuit unitary U = U_n...U₂U₁
   */
  Matrix get_unitary(const matrix_base<double> &params);

  /**
   * @brief Get number of qubits
   */
  int get_qbit_num() const { return qbit_num_; }

  /**
   * @brief Get number of parameters
   */
  int get_parameter_num() const;

  /**
   * @brief Get underlying Gates_block (if wrapped)
   * @return Pointer to underlying circuit (may be nullptr if not wrapping)
   */
  Gates_block *get_gates_block() const { return gates_block_; }

private:
  int qbit_num_;             ///< Number of qubits
  Gates_block *gates_block_; ///< Pointer to existing circuit
  bool owns_circuit_;        ///< Whether we own the circuit

  void sync_gates(); ///< Sync with gates_block if needed
};

} // namespace density
} // namespace squander
