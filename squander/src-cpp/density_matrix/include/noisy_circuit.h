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

#include "density_matrix.h"
#include "density_operation.h"
#include <memory>
#include <vector>

// Forward declarations for gate types
class Gate;
class Gates_block;

namespace squander {
namespace density {

/**
 * @brief Quantum circuit with noise for density matrix simulation.
 *
 * NoisyCircuit manages a sequence of unitary gates and noise channels,
 * providing a unified interface to build and execute noisy quantum circuits.
 *
 * Key features:
 * - Unified API for adding gates and noise channels
 * - Optimized local kernel application (O(4^N) per gate instead of O(8^N))
 * - Support for both fixed and parametric noise
 * - Sequential execution (each operation applied in order)
 *
 * Example:
 * @code
 *   NoisyCircuit circuit(2);
 *   circuit.add_H(0);
 *   circuit.add_CNOT(1, 0);
 *   circuit.add_depolarizing(2, 0.01);  // Fixed 1% noise
 *   circuit.add_RZ(0);                   // Parametric gate
 *   circuit.add_phase_damping(0);        // Parametric noise
 *
 *   DensityMatrix rho(2);
 *   std::vector<double> params = {0.5, 0.02};  // RZ angle, phase damping λ
 *   circuit.apply_to(params.data(), params.size(), rho);
 * @endcode
 */
class NoisyCircuit {
public:
  // ================================================================
  // Constructors & Destructor
  // ================================================================

  /**
   * @brief Create empty circuit for n qubits
   * @param qbit_num Number of qubits (must be >= 1)
   */
  explicit NoisyCircuit(int qbit_num);

  /**
   * @brief Destructor
   */
  ~NoisyCircuit();

  // Disable copy (operations own their resources)
  NoisyCircuit(const NoisyCircuit &) = delete;
  NoisyCircuit &operator=(const NoisyCircuit &) = delete;

  // Move semantics
  NoisyCircuit(NoisyCircuit &&other) noexcept;
  NoisyCircuit &operator=(NoisyCircuit &&other) noexcept;

  // ================================================================
  // Single-Qubit Gates (constant - no parameters)
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

  // ================================================================
  // Single-Qubit Parametric Gates
  // ================================================================

  void add_RX(int target); ///< 1 parameter
  void add_RY(int target); ///< 1 parameter
  void add_RZ(int target); ///< 1 parameter
  void add_U1(int target); ///< 1 parameter
  void add_U2(int target); ///< 2 parameters
  void add_U3(int target); ///< 3 parameters

  // ================================================================
  // Two-Qubit Gates (constant)
  // ================================================================

  void add_CNOT(int target, int control);
  void add_CZ(int target, int control);
  void add_CH(int target, int control);

  // ================================================================
  // Two-Qubit Parametric Gates
  // ================================================================

  void add_CRY(int target, int control); ///< 1 parameter
  void add_CRZ(int target, int control); ///< 1 parameter
  void add_CRX(int target, int control); ///< 1 parameter
  void add_CP(int target, int control);  ///< 1 parameter

  // ================================================================
  // Noise Channel Addition
  // ================================================================

  /**
   * @brief Add parametric depolarizing noise channel (1 parameter)
   * @param qbit_num Number of qubits the noise acts on
   */
  void add_depolarizing(int qbit_num);

  /**
   * @brief Add fixed depolarizing noise channel (0 parameters)
   * @param qbit_num Number of qubits the noise acts on
   * @param error_rate Fixed error rate p ∈ [0,1]
   */
  void add_depolarizing(int qbit_num, double error_rate);

  /**
   * @brief Add parametric amplitude damping noise (1 parameter)
   * @param target Target qubit index
   */
  void add_amplitude_damping(int target);

  /**
   * @brief Add fixed amplitude damping noise (T1 relaxation, 0 parameters)
   * @param target Target qubit index
   * @param gamma Fixed damping γ ∈ [0,1]
   */
  void add_amplitude_damping(int target, double gamma);

  /**
   * @brief Add parametric phase damping noise (1 parameter)
   * @param target Target qubit index
   */
  void add_phase_damping(int target);

  /**
   * @brief Add fixed phase damping noise (T2 dephasing, 0 parameters)
   * @param target Target qubit index
   * @param lambda Fixed dephasing λ ∈ [0,1]
   */
  void add_phase_damping(int target, double lambda);

  // ================================================================
  // Circuit Execution
  // ================================================================

  /**
   * @brief Apply entire circuit to density matrix
   * @param params Parameter array (gate params followed by noise params)
   * @param param_count Total number of parameters
   * @param rho Density matrix to modify in-place
   *
   * Parameters are consumed in the order operations were added.
   * Each parametric operation extracts its parameters from the array.
   */
  void apply_to(const double *params, int param_count, DensityMatrix &rho);

  /**
   * @brief Apply circuit using Matrix_real (backward compatibility)
   */
  void apply_to(const matrix_base<double> &params, DensityMatrix &rho);

  // ================================================================
  // Properties
  // ================================================================

  /**
   * @brief Get number of qubits
   */
  int get_qbit_num() const { return qbit_num_; }

  /**
   * @brief Get total number of parameters needed
   */
  int get_parameter_num() const { return total_params_; }

  /**
   * @brief Get number of operations in the circuit
   */
  size_t get_operation_count() const { return operations_.size(); }

  // ================================================================
  // Inspection
  // ================================================================

  /**
   * @brief Information about a circuit operation
   */
  struct OperationInfo {
    std::string name;
    bool is_unitary;
    int param_count;
    int param_start;
  };

  /**
   * @brief Get information about all operations
   */
  std::vector<OperationInfo> get_operation_info() const;

private:
  int qbit_num_;
  std::vector<std::unique_ptr<IDensityOperation>> operations_;
  std::vector<int> param_starts_;
  int total_params_ = 0;

  // Helper to add an operation and update parameter tracking
  void add_operation(std::unique_ptr<IDensityOperation> op);

  // Helper to add a gate by type
  void add_gate_internal(Gate *gate);
};

} // namespace density
} // namespace squander
