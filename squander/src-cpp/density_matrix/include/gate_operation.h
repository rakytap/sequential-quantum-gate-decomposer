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

#include "Gate.h"
#include "density_operation.h"
#include "matrix_real.h"

namespace squander {
namespace density {

/**
 * @brief Wraps an existing Gate for density matrix operations.
 *
 * This adapter allows existing SQUANDER gates (H, CNOT, RZ, etc.) to be
 * used in noisy circuits operating on density matrices.
 *
 * Uses optimized local kernel application:
 * - Single-qubit gates: O(2^{2N}) instead of O(2^{3N})
 * - Two-qubit gates: O(2^{2N+2}) instead of O(2^{3N})
 *
 * Does NOT modify the original Gate class - clean separation of concerns.
 *
 * Example:
 * @code
 *   Gate* h = new H(2, 0);
 *   auto op = std::make_unique<GateOperation>(h, true);
 *   op->apply_to_density(nullptr, 0, rho);  // H on qubit 0
 * @endcode
 */
class GateOperation : public IDensityOperation {
public:
  /**
   * @brief Wrap an existing gate
   * @param gate The gate to wrap
   * @param owns_gate If true, this wrapper will delete the gate on destruction
   */
  GateOperation(Gate *gate, bool owns_gate = true);

  /**
   * @brief Destructor - deletes gate if owns_gate was true
   */
  ~GateOperation() override;

  // Disable copy (use clone() instead)
  GateOperation(const GateOperation &) = delete;
  GateOperation &operator=(const GateOperation &) = delete;

  // Move semantics
  GateOperation(GateOperation &&other) noexcept;
  GateOperation &operator=(GateOperation &&other) noexcept;

  /**
   * @brief Apply gate to density matrix using optimized local kernel
   *
   * For single-qubit gates, applies the 2x2 kernel directly.
   * For two-qubit gates, applies the 4x4 kernel directly.
   * This is O(2^{2N+k}) instead of O(2^{3N}) for k-qubit gates.
   */
  void apply_to_density(const double *params, int param_count,
                        DensityMatrix &rho) override;

  int get_parameter_num() const override;
  std::string get_name() const override;
  bool is_unitary() const override { return true; }

  std::unique_ptr<IDensityOperation> clone() const override;

  /**
   * @brief Get the underlying gate (for inspection)
   */
  Gate *get_gate() const { return gate_; }

  /**
   * @brief Get target qubit index
   */
  int get_target_qbit() const;

  /**
   * @brief Get control qubit index (-1 if not a controlled gate)
   */
  int get_control_qbit() const;

private:
  Gate *gate_;
  bool owns_gate_;
};

} // namespace density
} // namespace squander

