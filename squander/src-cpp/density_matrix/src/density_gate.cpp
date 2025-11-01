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

#include "density_gate.h"

namespace squander {
namespace density {

DensityGate::DensityGate(Gate *gate) : gate_(gate) {
  if (!gate) {
    throw std::invalid_argument("DensityGate: cannot wrap null gate");
  }
}

void DensityGate::apply_to(const matrix_base<double> &params,
                           DensityMatrix &rho) {
  if (!gate_) {
    throw std::runtime_error("DensityGate: wrapped gate is null");
  }

  // Verify dimensions match
  if (gate_->get_qbit_num() != rho.get_qbit_num()) {
    throw std::runtime_error(
        "DensityGate::apply_to: qubit number mismatch. " +
        std::string("Gate has ") + std::to_string(gate_->get_qbit_num()) +
        " qubits, density matrix has " + std::to_string(rho.get_qbit_num()));
  }

  // Get gate unitary matrix from existing gate
  // Wrap params as Matrix_real for gate interface
  Matrix_real params_matrix(const_cast<double *>(params.data), params.rows,
                            params.cols);

  Matrix U = gate_->get_matrix(params_matrix);

  // Apply unitary to density matrix: ρ → UρU†
  rho.apply_unitary(U);
}

gate_type DensityGate::get_type() const {
  if (!gate_) {
    throw std::runtime_error("DensityGate: wrapped gate is null");
  }
  return gate_->get_type();
}

int DensityGate::get_target_qbit() const {
  if (!gate_) {
    throw std::runtime_error("DensityGate: wrapped gate is null");
  }
  return gate_->get_target_qbit();
}

int DensityGate::get_control_qbit() const {
  if (!gate_) {
    throw std::runtime_error("DensityGate: wrapped gate is null");
  }
  return gate_->get_control_qbit();
}

int DensityGate::get_parameter_num() const {
  if (!gate_) {
    throw std::runtime_error("DensityGate: wrapped gate is null");
  }
  return gate_->get_parameter_num();
}

int DensityGate::get_qbit_num() const {
  if (!gate_) {
    throw std::runtime_error("DensityGate: wrapped gate is null");
  }
  return gate_->get_qbit_num();
}

} // namespace density
} // namespace squander
