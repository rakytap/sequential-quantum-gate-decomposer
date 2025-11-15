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

#include "noise_channel.h"
#include <cmath>

namespace squander {
namespace density {

// ================================================================
// Depolarizing Channel
// ================================================================

DepolarizingChannel::DepolarizingChannel(int qbit_num, double error_rate)
    : qbit_num_(qbit_num), error_rate_(error_rate) {
  if (error_rate < 0.0 || error_rate > 1.0) {
    throw std::invalid_argument(
        "DepolarizingChannel: error_rate must be in [0,1]");
  }
}

void DepolarizingChannel::apply(DensityMatrix &rho) {
  int dim = rho.get_dim();

  if (rho.get_qbit_num() != qbit_num_) {
    throw std::runtime_error(
        "DepolarizingChannel::apply: qubit number mismatch");
  }

  // Compute trace (for normalization check)
  QGD_Complex16 tr = rho.trace();
  double trace_val = tr.real;

  // Apply: ρ → (1-p)ρ + (p/dim)Tr(ρ)·I
  double depol_prob = error_rate_ / dim;
  double identity_factor = depol_prob * trace_val;

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      QGD_Complex16 &element = rho(i, j);

      if (i == j) {
        // Diagonal: (1-p)ρ(i,i) + p/dim
        element.real = (1.0 - error_rate_) * element.real + identity_factor;
        element.imag = (1.0 - error_rate_) * element.imag;
      } else {
        // Off-diagonal: (1-p)ρ(i,j)
        element.real *= (1.0 - error_rate_);
        element.imag *= (1.0 - error_rate_);
      }
    }
  }
}

// ================================================================
// Amplitude Damping Channel
// ================================================================

AmplitudeDampingChannel::AmplitudeDampingChannel(int target_qbit, double gamma)
    : target_qbit_(target_qbit), gamma_(gamma) {
  if (gamma < 0.0 || gamma > 1.0) {
    throw std::invalid_argument(
        "AmplitudeDampingChannel: gamma must be in [0,1]");
  }
}

void AmplitudeDampingChannel::apply(DensityMatrix &rho) {
  int dim = rho.get_dim();
  int qbit_num = rho.get_qbit_num();

  if (target_qbit_ < 0 || target_qbit_ >= qbit_num) {
    throw std::runtime_error(
        "AmplitudeDampingChannel::apply: target_qbit out of range");
  }

  // Kraus operators for amplitude damping:
  // K₀ = [[1, 0], [0, √(1-γ)]]
  // K₁ = [[0, √γ], [0, 0]]

  double sqrt_gamma = std::sqrt(gamma_);
  double sqrt_1_minus_gamma = std::sqrt(1.0 - gamma_);

  // Create temporary storage for new density matrix
  DensityMatrix rho_new(qbit_num);
  memset(rho_new.data, 0, dim * dim * sizeof(QGD_Complex16));

  int target_step = 1 << target_qbit_;

  // Apply Kraus operators
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {

      // Determine states of target qubit
      bool i_target = (i & target_step) != 0;
      bool j_target = (j & target_step) != 0;

      // Apply K₀ρK₀†
      if (i_target == j_target) {
        double factor = i_target ? (1.0 - gamma_) : 1.0;
        rho_new(i, j).real += factor * rho(i, j).real;
        rho_new(i, j).imag += factor * rho(i, j).imag;
      }

      // Apply K₁ρK₁†: [[0, √γ], [0, 0]] ρ [[0, 0], [√γ, 0]]
      if (!i_target && !j_target) {
        int i_flip = i | target_step;
        int j_flip = j | target_step;
        rho_new(i, j).real += gamma_ * rho(i_flip, j_flip).real;
        rho_new(i, j).imag += gamma_ * rho(i_flip, j_flip).imag;
      }
    }
  }

  // Copy result back
  memcpy(rho.data, rho_new.data, dim * dim * sizeof(QGD_Complex16));
}

// ================================================================
// Phase Damping Channel
// ================================================================

PhaseDampingChannel::PhaseDampingChannel(int target_qbit, double lambda)
    : target_qbit_(target_qbit), lambda_(lambda) {
  if (lambda < 0.0 || lambda > 1.0) {
    throw std::invalid_argument("PhaseDampingChannel: lambda must be in [0,1]");
  }
}

void PhaseDampingChannel::apply(DensityMatrix &rho) {
  int dim = rho.get_dim();
  int qbit_num = rho.get_qbit_num();

  if (target_qbit_ < 0 || target_qbit_ >= qbit_num) {
    throw std::runtime_error(
        "PhaseDampingChannel::apply: target_qbit out of range");
  }

  // Kraus operators for phase damping:
  // K₀ = [[1, 0], [0, √(1-λ)]]
  // K₁ = [[0, 0], [0, √λ]]

  int target_step = 1 << target_qbit_;

  // Phase damping only affects off-diagonal elements between |0⟩ and |1⟩
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      bool i_target = (i & target_step) != 0;
      bool j_target = (j & target_step) != 0;

      // If target qubit states differ, reduce coherence
      if (i_target != j_target) {
        rho(i, j).real *= std::sqrt(1.0 - lambda_);
        rho(i, j).imag *= std::sqrt(1.0 - lambda_);
      }
      // Diagonal elements unchanged
    }
  }
}

} // namespace density
} // namespace squander
