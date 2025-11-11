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

#include "noisy_circuit.h"

namespace squander {
namespace density {

// ================================================================
// Constructors & Destructor
// ================================================================

NoisyCircuit::NoisyCircuit(int qbit_num)
    : qbit_num_(qbit_num), gates_block_(nullptr), owns_circuit_(true) {
  if (qbit_num < 1) {
    throw std::invalid_argument("NoisyCircuit: qbit_num must be >= 1");
  }

  // Create new Gates_block
  gates_block_ = new Gates_block(qbit_num);
}

NoisyCircuit::NoisyCircuit(Gates_block *circuit, bool owns_circuit)
    : gates_block_(circuit), owns_circuit_(owns_circuit) {
  if (!circuit) {
    throw std::invalid_argument("NoisyCircuit: cannot wrap null circuit");
  }

  qbit_num_ = circuit->get_qbit_num();
}

NoisyCircuit::~NoisyCircuit() {
  if (owns_circuit_ && gates_block_) {
    delete gates_block_;
    gates_block_ = nullptr;
  }
}

// ================================================================
// Gate Addition (delegates to Gates_block)
// ================================================================

void NoisyCircuit::add_H(int target) { gates_block_->add_h(target); }

void NoisyCircuit::add_X(int target) { gates_block_->add_x(target); }

void NoisyCircuit::add_Y(int target) { gates_block_->add_y(target); }

void NoisyCircuit::add_Z(int target) { gates_block_->add_z(target); }

void NoisyCircuit::add_S(int target) { gates_block_->add_s(target); }

void NoisyCircuit::add_Sdg(int target) { gates_block_->add_sdg(target); }

void NoisyCircuit::add_T(int target) { gates_block_->add_t(target); }

void NoisyCircuit::add_Tdg(int target) { gates_block_->add_tdg(target); }

void NoisyCircuit::add_SX(int target) { gates_block_->add_sx(target); }

void NoisyCircuit::add_RX(int target) { gates_block_->add_rx(target); }

void NoisyCircuit::add_RY(int target) { gates_block_->add_ry(target); }

void NoisyCircuit::add_RZ(int target) { gates_block_->add_rz(target); }

void NoisyCircuit::add_U1(int target) { gates_block_->add_u1(target); }

void NoisyCircuit::add_U2(int target) { gates_block_->add_u2(target); }

void NoisyCircuit::add_U3(int target) { gates_block_->add_u3(target); }

void NoisyCircuit::add_CNOT(int target, int control) {
  gates_block_->add_cnot(target, control);
}

void NoisyCircuit::add_CZ(int target, int control) {
  gates_block_->add_cz(target, control);
}

void NoisyCircuit::add_CH(int target, int control) {
  gates_block_->add_ch(target, control);
}

void NoisyCircuit::add_CRY(int target, int control) {
  gates_block_->add_cry(target, control);
}

void NoisyCircuit::add_CRZ(int target, int control) {
  gates_block_->add_crz(target, control);
}

void NoisyCircuit::add_CRX(int target, int control) {
  gates_block_->add_crx(target, control);
}

void NoisyCircuit::add_CP(int target, int control) {
  gates_block_->add_cp(target, control);
}

// ================================================================
// Circuit Application
// ================================================================

void NoisyCircuit::apply_to(const matrix_base<double> &params,
                              DensityMatrix &rho) {
  if (!gates_block_) {
    throw std::runtime_error("NoisyCircuit: gates_block is null");
  }

  // Verify dimensions
  if (rho.get_qbit_num() != qbit_num_) {
    throw std::runtime_error("NoisyCircuit::apply_to: qubit number mismatch");
  }

  // Get full circuit unitary
  Matrix_real params_matrix(const_cast<double *>(params.data), params.rows,
                            params.cols);

  Matrix U = gates_block_->get_matrix(params_matrix);

  // Apply to density matrix: ρ → UρU†
  rho.apply_unitary(U);
}

Matrix NoisyCircuit::get_unitary(const matrix_base<double> &params) {
  if (!gates_block_) {
    throw std::runtime_error("NoisyCircuit: gates_block is null");
  }

  Matrix_real params_matrix(const_cast<double *>(params.data), params.rows,
                            params.cols);

  return gates_block_->get_matrix(params_matrix);
}

int NoisyCircuit::get_parameter_num() const {
  if (!gates_block_) {
    return 0;
  }
  return gates_block_->get_parameter_num();
}

void NoisyCircuit::sync_gates() {
  // For now, we use gates_block directly via get_Matrix()
}

} // namespace density
} // namespace squander
