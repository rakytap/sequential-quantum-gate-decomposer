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
#include "gate_operation.h"
#include "noise_operation.h"

// Include gate headers
#include "CH.h"
#include "CNOT.h"
#include "CP.h"
#include "CRX.h"
#include "CRY.h"
#include "CRZ.h"
#include "CZ.h"
#include "H.h"
#include "RX.h"
#include "RY.h"
#include "RZ.h"
#include "S.h"
#include "SDG.h"
#include "SX.h"
#include "T.h"
#include "Tdg.h"
#include "U1.h"
#include "U2.h"
#include "U3.h"
#include "X.h"
#include "Y.h"
#include "Z.h"

namespace squander {
namespace density {

// ================================================================
// Constructors & Destructor
// ================================================================

NoisyCircuit::NoisyCircuit(int qbit_num) : qbit_num_(qbit_num) {
  if (qbit_num < 1) {
    throw std::invalid_argument("NoisyCircuit: qbit_num must be >= 1");
  }
}

NoisyCircuit::~NoisyCircuit() = default;

// ================================================================
// Move Semantics
// ================================================================

NoisyCircuit::NoisyCircuit(NoisyCircuit &&other) noexcept
    : qbit_num_(other.qbit_num_), operations_(std::move(other.operations_)),
      param_starts_(std::move(other.param_starts_)),
      total_params_(other.total_params_) {
  other.qbit_num_ = 0;
  other.total_params_ = 0;
}

NoisyCircuit &NoisyCircuit::operator=(NoisyCircuit &&other) noexcept {
  if (this != &other) {
    qbit_num_ = other.qbit_num_;
    operations_ = std::move(other.operations_);
    param_starts_ = std::move(other.param_starts_);
    total_params_ = other.total_params_;
    other.qbit_num_ = 0;
    other.total_params_ = 0;
  }
  return *this;
}

// ================================================================
// Internal Helpers
// ================================================================

void NoisyCircuit::add_operation(std::unique_ptr<IDensityOperation> op) {
  param_starts_.push_back(total_params_);
  total_params_ += op->get_parameter_num();
  operations_.push_back(std::move(op));
}

void NoisyCircuit::add_gate_internal(Gate *gate) {
  add_operation(std::unique_ptr<GateOperation>(new GateOperation(gate, true)));
}

// ================================================================
// Single-Qubit Constant Gates
// ================================================================

void NoisyCircuit::add_H(int target) {
  add_gate_internal(new H(qbit_num_, target));
}

void NoisyCircuit::add_X(int target) {
  add_gate_internal(new X(qbit_num_, target));
}

void NoisyCircuit::add_Y(int target) {
  add_gate_internal(new Y(qbit_num_, target));
}

void NoisyCircuit::add_Z(int target) {
  add_gate_internal(new Z(qbit_num_, target));
}

void NoisyCircuit::add_S(int target) {
  add_gate_internal(new S(qbit_num_, target));
}

void NoisyCircuit::add_Sdg(int target) {
  add_gate_internal(new SDG(qbit_num_, target));
}

void NoisyCircuit::add_T(int target) {
  add_gate_internal(new T(qbit_num_, target));
}

void NoisyCircuit::add_Tdg(int target) {
  add_gate_internal(new Tdg(qbit_num_, target));
}

void NoisyCircuit::add_SX(int target) {
  add_gate_internal(new SX(qbit_num_, target));
}

// ================================================================
// Single-Qubit Parametric Gates
// ================================================================

void NoisyCircuit::add_RX(int target) {
  add_gate_internal(new RX(qbit_num_, target));
}

void NoisyCircuit::add_RY(int target) {
  add_gate_internal(new RY(qbit_num_, target));
}

void NoisyCircuit::add_RZ(int target) {
  add_gate_internal(new RZ(qbit_num_, target));
}

void NoisyCircuit::add_U1(int target) {
  add_gate_internal(new U1(qbit_num_, target));
}

void NoisyCircuit::add_U2(int target) {
  add_gate_internal(new U2(qbit_num_, target));
}

void NoisyCircuit::add_U3(int target) {
  add_gate_internal(new U3(qbit_num_, target));
}

// ================================================================
// Two-Qubit Constant Gates
// ================================================================

void NoisyCircuit::add_CNOT(int target, int control) {
  add_gate_internal(new CNOT(qbit_num_, target, control));
}

void NoisyCircuit::add_CZ(int target, int control) {
  add_gate_internal(new CZ(qbit_num_, target, control));
}

void NoisyCircuit::add_CH(int target, int control) {
  add_gate_internal(new CH(qbit_num_, target, control));
}

// ================================================================
// Two-Qubit Parametric Gates
// ================================================================

void NoisyCircuit::add_CRY(int target, int control) {
  add_gate_internal(new CRY(qbit_num_, target, control));
}

void NoisyCircuit::add_CRZ(int target, int control) {
  add_gate_internal(new CRZ(qbit_num_, target, control));
}

void NoisyCircuit::add_CRX(int target, int control) {
  add_gate_internal(new CRX(qbit_num_, target, control));
}

void NoisyCircuit::add_CP(int target, int control) {
  add_gate_internal(new CP(qbit_num_, target, control));
}

// ================================================================
// Noise Channels
// ================================================================

void NoisyCircuit::add_depolarizing(int qbit_num) {
  add_operation(std::unique_ptr<DepolarizingOp>(new DepolarizingOp(qbit_num)));
}

void NoisyCircuit::add_depolarizing(int qbit_num, double error_rate) {
  add_operation(std::unique_ptr<DepolarizingOp>(
      new DepolarizingOp(qbit_num, error_rate)));
}

void NoisyCircuit::add_amplitude_damping(int target) {
  add_operation(
      std::unique_ptr<AmplitudeDampingOp>(new AmplitudeDampingOp(target)));
}

void NoisyCircuit::add_amplitude_damping(int target, double gamma) {
  add_operation(std::unique_ptr<AmplitudeDampingOp>(
      new AmplitudeDampingOp(target, gamma)));
}

void NoisyCircuit::add_phase_damping(int target) {
  add_operation(std::unique_ptr<PhaseDampingOp>(new PhaseDampingOp(target)));
}

void NoisyCircuit::add_phase_damping(int target, double lambda) {
  add_operation(
      std::unique_ptr<PhaseDampingOp>(new PhaseDampingOp(target, lambda)));
}

// ================================================================
// Circuit Execution
// ================================================================

void NoisyCircuit::apply_to(const double *params, int param_count,
                            DensityMatrix &rho) {
  if (rho.get_qbit_num() != qbit_num_) {
    throw std::runtime_error("NoisyCircuit::apply_to: qubit number mismatch");
  }

  if (param_count < total_params_) {
    throw std::runtime_error(
        "NoisyCircuit::apply_to: not enough parameters. Expected " +
        std::to_string(total_params_) + ", got " + std::to_string(param_count));
  }

  // Apply each operation in sequence
  for (size_t i = 0; i < operations_.size(); i++) {
    auto &op = operations_[i];
    int start = param_starts_[i];
    int count = op->get_parameter_num();

    const double *op_params = (count > 0) ? (params + start) : nullptr;
    op->apply_to_density(op_params, count, rho);
  }
}

void NoisyCircuit::apply_to(const matrix_base<double> &params,
                            DensityMatrix &rho) {
  apply_to(params.data, params.rows * params.cols, rho);
}

// ================================================================
// Inspection
// ================================================================

std::vector<NoisyCircuit::OperationInfo>
NoisyCircuit::get_operation_info() const {
  std::vector<OperationInfo> info;
  info.reserve(operations_.size());

  for (size_t i = 0; i < operations_.size(); i++) {
    OperationInfo oi;
    oi.name = operations_[i]->get_name();
    oi.is_unitary = operations_[i]->is_unitary();
    oi.param_count = operations_[i]->get_parameter_num();
    oi.param_start = param_starts_[i];
    info.push_back(oi);
  }

  return info;
}

} // namespace density
} // namespace squander
