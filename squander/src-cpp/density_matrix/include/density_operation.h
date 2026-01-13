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

#include <memory>
#include <string>

namespace squander {
namespace density {

// Forward declaration
class DensityMatrix;

/**
 * @brief Interface for any operation that can be applied to a density matrix.
 *
 * This is the core abstraction that unifies unitary gates and noise channels.
 * Each operation knows how to apply itself efficiently to a density matrix
 * using optimized local kernel application (O(4^N) instead of O(8^N)).
 *
 * Example implementations:
 * - GateOperation: Wraps existing Gate objects
 * - DepolarizingOp: Depolarizing noise channel
 * - AmplitudeDampingOp: T1 relaxation
 * - PhaseDampingOp: T2 dephasing
 */
class IDensityOperation {
public:
  virtual ~IDensityOperation() = default;

  /**
   * @brief Apply this operation to the density matrix
   * @param params Pointer to this operation's parameters (may be nullptr if
   * param_count=0)
   * @param param_count Number of parameters for this operation
   * @param rho The density matrix to modify in-place
   */
  virtual void apply_to_density(const double *params, int param_count,
                                DensityMatrix &rho) = 0;

  /**
   * @brief Get number of free parameters this operation requires
   * @return Number of parameters (0 for constant gates/fixed noise)
   */
  virtual int get_parameter_num() const = 0;

  /**
   * @brief Get operation name for debugging/display
   * @return Human-readable name (e.g., "H", "CNOT", "Depolarizing")
   */
  virtual std::string get_name() const = 0;

  /**
   * @brief Check if this is a unitary operation (gate) or non-unitary (noise)
   * @return true for unitary gates, false for noise channels
   */
  virtual bool is_unitary() const = 0;

  /**
   * @brief Create a deep copy of this operation
   * @return Unique pointer to the cloned operation
   */
  virtual std::unique_ptr<IDensityOperation> clone() const = 0;
};

} // namespace density
} // namespace squander

