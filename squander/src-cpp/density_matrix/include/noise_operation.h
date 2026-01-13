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

#include "density_operation.h"

namespace squander {
namespace density {

/**
 * @brief Base class for noise channel operations (CPTP maps).
 *
 * Noise channels are non-unitary quantum operations that model
 * decoherence and errors. They are represented via Kraus operators:
 *   ρ → Σᵢ KᵢρKᵢ†
 *
 * Unlike gates, noise channels:
 * - Cannot be represented as unitary matrices
 * - Generally decrease purity (mix the state)
 * - Preserve trace: Tr(ρ') = Tr(ρ) = 1
 */
class NoiseOperation : public IDensityOperation {
public:
  bool is_unitary() const override { return false; }
};

/**
 * @brief Depolarizing noise channel: ρ → (1-p)ρ + p·I/2^n
 *
 * Represents uniform noise that replaces the quantum state with the
 * maximally mixed state with probability p. Physically models
 * interaction with a thermal environment.
 *
 * Properties:
 * - Purity decreases: Tr(ρ'²) < Tr(ρ²) for p > 0
 * - p = 0: No noise (identity channel)
 * - p = 1: Complete depolarization (maximally mixed output)
 *
 * Can be parametric (error rate from parameter array) or fixed.
 *
 * Example:
 * @code
 *   // Fixed 1% error rate
 *   auto noise = std::make_unique<DepolarizingOp>(2, 0.01);
 *
 *   // Parametric (error rate from optimizer)
 *   auto noise = std::make_unique<DepolarizingOp>(2);
 * @endcode
 */
class DepolarizingOp : public NoiseOperation {
public:
  /**
   * @brief Create depolarizing noise with fixed error rate
   * @param qbit_num Number of qubits the noise acts on
   * @param error_rate Fixed depolarizing probability p ∈ [0, 1]
   */
  DepolarizingOp(int qbit_num, double error_rate);

  /**
   * @brief Create parametric depolarizing noise (1 parameter)
   * @param qbit_num Number of qubits the noise acts on
   */
  explicit DepolarizingOp(int qbit_num);

  void apply_to_density(const double *params, int param_count,
                        DensityMatrix &rho) override;

  int get_parameter_num() const override { return is_parametric_ ? 1 : 0; }
  std::string get_name() const override { return "Depolarizing"; }

  std::unique_ptr<IDensityOperation> clone() const override;

  // Accessors
  int get_qbit_num() const { return qbit_num_; }
  double get_error_rate() const { return fixed_error_rate_; }
  bool is_parametric() const { return is_parametric_; }

private:
  int qbit_num_;
  double fixed_error_rate_;
  bool is_parametric_;

  void apply_depolarizing(DensityMatrix &rho, double error_rate);
};

/**
 * @brief Amplitude damping channel (T1 relaxation): |1⟩ → |0⟩ decay
 *
 * Models energy relaxation from the excited state to the ground state.
 * Represents spontaneous emission in real quantum systems.
 *
 * Kraus operators:
 * - K₀ = [[1, 0], [0, √(1-γ)]]
 * - K₁ = [[0, √γ], [0, 0]]
 *
 * where γ = 1 - exp(-t/T1) is the damping parameter.
 *
 * Properties:
 * - γ = 0: No damping (identity channel)
 * - γ = 1: Complete decay (|1⟩ → |0⟩)
 * - Asymptotic state: |0⟩⟨0| (ground state)
 *
 * Can be parametric (γ from parameter array) or fixed.
 */
class AmplitudeDampingOp : public NoiseOperation {
public:
  /**
   * @brief Create amplitude damping with fixed gamma
   * @param target_qbit Target qubit index
   * @param gamma Damping parameter γ = 1 - exp(-t/T1) ∈ [0, 1]
   */
  AmplitudeDampingOp(int target_qbit, double gamma);

  /**
   * @brief Create parametric amplitude damping (1 parameter)
   * @param target_qbit Target qubit index
   */
  explicit AmplitudeDampingOp(int target_qbit);

  void apply_to_density(const double *params, int param_count,
                        DensityMatrix &rho) override;

  int get_parameter_num() const override { return is_parametric_ ? 1 : 0; }
  std::string get_name() const override { return "AmplitudeDamping"; }

  std::unique_ptr<IDensityOperation> clone() const override;

  // Accessors
  int get_target_qbit() const { return target_qbit_; }
  double get_gamma() const { return fixed_gamma_; }
  bool is_parametric() const { return is_parametric_; }

private:
  int target_qbit_;
  double fixed_gamma_;
  bool is_parametric_;

  void apply_amplitude_damping(DensityMatrix &rho, double gamma);
};

/**
 * @brief Phase damping channel (T2 dephasing): loss of coherence
 *
 * Models phase randomization without energy loss. Represents
 * decoherence in real quantum systems where the environment
 * measures the qubit in the computational basis.
 *
 * Kraus operators:
 * - K₀ = [[1, 0], [0, √(1-λ)]]
 * - K₁ = [[0, 0], [0, √λ]]
 *
 * where λ = 1 - exp(-t/T2) is the dephasing parameter.
 *
 * Properties:
 * - λ = 0: No dephasing (identity channel)
 * - λ = 1: Complete dephasing (diagonal state)
 * - Only affects off-diagonal elements (coherences)
 * - Diagonal elements unchanged
 *
 * Can be parametric (λ from parameter array) or fixed.
 */
class PhaseDampingOp : public NoiseOperation {
public:
  /**
   * @brief Create phase damping with fixed lambda
   * @param target_qbit Target qubit index
   * @param lambda Dephasing parameter λ = 1 - exp(-t/T2) ∈ [0, 1]
   */
  PhaseDampingOp(int target_qbit, double lambda);

  /**
   * @brief Create parametric phase damping (1 parameter)
   * @param target_qbit Target qubit index
   */
  explicit PhaseDampingOp(int target_qbit);

  void apply_to_density(const double *params, int param_count,
                        DensityMatrix &rho) override;

  int get_parameter_num() const override { return is_parametric_ ? 1 : 0; }
  std::string get_name() const override { return "PhaseDamping"; }

  std::unique_ptr<IDensityOperation> clone() const override;

  // Accessors
  int get_target_qbit() const { return target_qbit_; }
  double get_lambda() const { return fixed_lambda_; }
  bool is_parametric() const { return is_parametric_; }

private:
  int target_qbit_;
  double fixed_lambda_;
  bool is_parametric_;

  void apply_phase_damping(DensityMatrix &rho, double lambda);
};

} // namespace density
} // namespace squander

