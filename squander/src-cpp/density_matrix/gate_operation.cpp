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

#include "gate_operation.h"
#include "density_matrix.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace squander {
namespace density {

// ================================================================
// Constructor & Destructor
// ================================================================

GateOperation::GateOperation(Gate *gate, bool owns_gate)
    : gate_(gate), owns_gate_(owns_gate) {
  if (!gate) {
    throw std::invalid_argument("GateOperation: gate cannot be null");
  }
}

GateOperation::~GateOperation() {
  if (owns_gate_ && gate_) {
    delete gate_;
    gate_ = nullptr;
  }
}

// ================================================================
// Move Semantics
// ================================================================

GateOperation::GateOperation(GateOperation &&other) noexcept
    : gate_(other.gate_), owns_gate_(other.owns_gate_) {
  other.gate_ = nullptr;
  other.owns_gate_ = false;
}

GateOperation &GateOperation::operator=(GateOperation &&other) noexcept {
  if (this != &other) {
    if (owns_gate_ && gate_) {
      delete gate_;
    }
    gate_ = other.gate_;
    owns_gate_ = other.owns_gate_;
    other.gate_ = nullptr;
    other.owns_gate_ = false;
  }
  return *this;
}

// ================================================================
// Core Operation
// ================================================================

void GateOperation::apply_to_density(const double *params, int param_count,
                                     DensityMatrix &rho) {
  // Get the small unitary kernel (2x2 for single-qubit, 4x4 for general
  // two-qubit)
  Matrix kernel;
  int num_params = gate_->get_parameter_num();

  if (num_params == 0) {
    // Constant gate (H, X, CNOT, etc.) - use no-parameter version
    kernel = gate_->calc_one_qubit_u3();
  } else {
    // Parametric gate (RZ, U3, etc.) - compute kernel with parameters
    if (param_count < num_params || params == nullptr) {
      throw std::runtime_error(
          "GateOperation::apply_to_density: not enough parameters");
    }

    // Get the angles for this gate type
    double theta_over_2 = 0, phi = 0, lambda = 0;

    // Extract parameters based on gate type
    Matrix_real params_mat(const_cast<double *>(params), 1, param_count);

    // Different gates have different parameter mappings
    // Most parametric single-qubit gates use theta/2, phi, lambda
    if (num_params == 1) {
      // Single parameter gates (RX, RY, RZ, U1)
      // The parameter typically maps to theta/2 or lambda depending on gate
      // type Use the gate's own parameter extraction
      theta_over_2 = params[0] / 2.0;
      phi = 0;
      lambda = 0;

      // For RZ gate, the parameter goes to lambda
      gate_type type = gate_->get_type();
      if (type == RZ_OPERATION || type == U1_OPERATION) {
        theta_over_2 = 0;
        phi = 0;
        lambda = params[0];
      } else if (type == RX_OPERATION) {
        theta_over_2 = params[0] / 2.0;
        phi = -M_PI / 2.0;
        lambda = M_PI / 2.0;
      } else if (type == RY_OPERATION) {
        theta_over_2 = params[0] / 2.0;
        phi = 0;
        lambda = 0;
      }
    } else if (num_params == 2) {
      // U2 gate: phi, lambda (theta fixed at pi/2)
      theta_over_2 = M_PI / 4.0;
      phi = params[0];
      lambda = params[1];
    } else if (num_params == 3) {
      // U3 gate: theta, phi, lambda
      theta_over_2 = params[0] / 2.0;
      phi = params[1];
      lambda = params[2];
    }

    kernel = gate_->calc_one_qubit_u3(theta_over_2 * 2, phi, lambda);
  }

  // Get target and control qubits
  int target = gate_->get_target_qbit();
  int control = gate_->get_control_qbit();

  // Apply using optimized local method
  if (control < 0) {
    // Single-qubit gate
    rho.apply_single_qubit_unitary(kernel, target);
  } else {
    // Two-qubit controlled gate
    rho.apply_two_qubit_unitary(kernel, target, control);
  }
}

// ================================================================
// Property Accessors
// ================================================================

int GateOperation::get_parameter_num() const {
  return gate_ ? gate_->get_parameter_num() : 0;
}

std::string GateOperation::get_name() const {
  return gate_ ? gate_->get_name() : "Unknown";
}

int GateOperation::get_target_qbit() const {
  return gate_ ? gate_->get_target_qbit() : -1;
}

int GateOperation::get_control_qbit() const {
  return gate_ ? gate_->get_control_qbit() : -1;
}

// ================================================================
// Clone
// ================================================================

std::unique_ptr<IDensityOperation> GateOperation::clone() const {
  if (!gate_) {
    throw std::runtime_error("GateOperation::clone: gate is null");
  }
  return std::make_unique<GateOperation>(gate_->clone(), true);
}

} // namespace density
} // namespace squander
