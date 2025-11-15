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
#include <memory>
#include <string>

namespace squander {
namespace density {

/**
 * @brief Base class for quantum noise channels
 *
 * Implements Kraus operator formalism: ρ → Σᵢ KᵢρKᵢ†
 * where Kᵢ are Kraus operators satisfying Σᵢ Kᵢ†Kᵢ = I (trace preservation)
 */
class NoiseChannel {
public:
  virtual ~NoiseChannel() = default;

  /**
   * @brief Apply noise channel to density matrix
   * @param rho Density matrix (modified in-place)
   */
  virtual void apply(DensityMatrix &rho) = 0;

  /**
   * @brief Get channel name
   */
  virtual std::string get_name() const = 0;
};

/**
 * @brief Depolarizing channel: ρ → (1-p)ρ + p·I/2^n
 *
 * Represents uniform noise that replaces quantum state with maximally mixed
 * state. Equivalent to applying X, Y, Z with equal probability p/3 each.
 *
 * Kraus operators:
 * - K₀ = √(1-p) I
 * - K₁ = √(p/3) X
 * - K₂ = √(p/3) Y
 * - K₃ = √(p/3) Z
 *
 * Example:
 * @code
 *   DepolarizingChannel noise(2, 0.01);  // 1% depolarizing noise
 *   DensityMatrix rho(2);
 *   noise.apply(rho);  // ρ → 0.99ρ + 0.01·I/4
 * @endcode
 */
class DepolarizingChannel : public NoiseChannel {
public:
  /**
   * @brief Constructor
   * @param qbit_num Number of qubits (applies to full system)
   * @param error_rate Error rate p ∈ [0, 1]
   * @throws std::invalid_argument if error_rate not in [0,1]
   */
  DepolarizingChannel(int qbit_num, double error_rate);

  void apply(DensityMatrix &rho) override;
  std::string get_name() const override { return "Depolarizing"; }

  double get_error_rate() const { return error_rate_; }
  int get_qbit_num() const { return qbit_num_; }

private:
  int qbit_num_;
  double error_rate_;
};

/**
 * @brief Amplitude damping (T1 relaxation): |1⟩ → |0⟩ decay
 *
 * Models energy relaxation from excited state to ground state.
 * Represents spontaneous emission in real quantum systems.
 *
 * Kraus operators:
 * - K₀ = [[1, 0], [0, √(1-γ)]]
 * - K₁ = [[0, √γ], [0, 0]]
 *
 * where γ = 1 - exp(-t/T1)
 *
 * Example:
 * @code
 *   AmplitudeDampingChannel t1_noise(0, 0.05);  // 5% amplitude damping on
 * qubit 0 DensityMatrix rho(2); t1_noise.apply(rho);
 * @endcode
 */
class AmplitudeDampingChannel : public NoiseChannel {
public:
  /**
   * @brief Constructor
   * @param target_qbit Target qubit index
   * @param gamma Damping parameter γ = 1 - exp(-t/T1) ∈ [0, 1]
   * @throws std::invalid_argument if gamma not in [0,1]
   */
  AmplitudeDampingChannel(int target_qbit, double gamma);

  void apply(DensityMatrix &rho) override;
  std::string get_name() const override { return "AmplitudeDamping"; }

  double get_gamma() const { return gamma_; }
  int get_target_qbit() const { return target_qbit_; }

private:
  int target_qbit_;
  double gamma_; ///< γ = 1 - exp(-t/T1)
};

/**
 * @brief Phase damping (T2 dephasing): loss of coherence
 *
 * Models phase randomization without energy loss.
 * Represents decoherence in real quantum systems.
 *
 * Kraus operators:
 * - K₀ = [[1, 0], [0, √(1-λ)]]
 * - K₁ = [[0, 0], [0, √λ]]
 *
 * where λ = 1 - exp(-t/T2)
 *
 * Example:
 * @code
 *   PhaseDampingChannel t2_noise(0, 0.03);  // 3% phase damping on qubit 0
 *   DensityMatrix rho(2);
 *   t2_noise.apply(rho);
 * @endcode
 */
class PhaseDampingChannel : public NoiseChannel {
public:
  /**
   * @brief Constructor
   * @param target_qbit Target qubit index
   * @param lambda Dephasing parameter λ = 1 - exp(-t/T2) ∈ [0, 1]
   * @throws std::invalid_argument if lambda not in [0,1]
   */
  PhaseDampingChannel(int target_qbit, double lambda);

  void apply(DensityMatrix &rho) override;
  std::string get_name() const override { return "PhaseDamping"; }

  double get_lambda() const { return lambda_; }
  int get_target_qbit() const { return target_qbit_; }

private:
  int target_qbit_;
  double lambda_; ///< λ = 1 - exp(-t/T2)
};

} // namespace density
} // namespace squander
