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

#include "noise_operation.h"
#include "density_matrix.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace squander {
namespace density {

// ================================================================
// DepolarizingOp
// ================================================================

DepolarizingOp::DepolarizingOp(int qbit_num, double error_rate)
    : qbit_num_(qbit_num), fixed_error_rate_(error_rate), is_parametric_(false) {
  if (qbit_num < 1) {
    throw std::invalid_argument("DepolarizingOp: qbit_num must be >= 1");
  }
  if (error_rate < 0.0 || error_rate > 1.0) {
    throw std::invalid_argument("DepolarizingOp: error_rate must be in [0, 1]");
  }
}

DepolarizingOp::DepolarizingOp(int qbit_num)
    : qbit_num_(qbit_num), fixed_error_rate_(0.0), is_parametric_(true) {
  if (qbit_num < 1) {
    throw std::invalid_argument("DepolarizingOp: qbit_num must be >= 1");
  }
}

void DepolarizingOp::apply_to_density(const double *params, int param_count,
                                      DensityMatrix &rho) {
  double error_rate;
  if (is_parametric_) {
    if (param_count < 1 || params == nullptr) {
      throw std::runtime_error(
          "DepolarizingOp: parametric mode requires 1 parameter");
    }
    error_rate = params[0];
    // Clamp to valid range
    if (error_rate < 0.0)
      error_rate = 0.0;
    if (error_rate > 1.0)
      error_rate = 1.0;
  } else {
    error_rate = fixed_error_rate_;
  }

  apply_depolarizing(rho, error_rate);
}

void DepolarizingOp::apply_depolarizing(DensityMatrix &rho, double p) {
  if (rho.get_qbit_num() != qbit_num_) {
    throw std::runtime_error(
        "DepolarizingOp: qubit number mismatch with density matrix");
  }

  int dim = rho.get_dim();

  // Compute trace for normalization
  QGD_Complex16 tr = rho.trace();
  double trace_val = tr.real;

  // Apply: ρ → (1-p)ρ + (p/dim)·Tr(ρ)·I
  double identity_factor = (p / dim) * trace_val;

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      QGD_Complex16 &element = rho(i, j);

      if (i == j) {
        // Diagonal: (1-p)ρ(i,i) + p/dim
        element.real = (1.0 - p) * element.real + identity_factor;
        element.imag = (1.0 - p) * element.imag;
      } else {
        // Off-diagonal: (1-p)ρ(i,j)
        element.real *= (1.0 - p);
        element.imag *= (1.0 - p);
      }
    }
  }
}

std::unique_ptr<IDensityOperation> DepolarizingOp::clone() const {
  if (is_parametric_) {
    return std::make_unique<DepolarizingOp>(qbit_num_);
  } else {
    return std::make_unique<DepolarizingOp>(qbit_num_, fixed_error_rate_);
  }
}

// ================================================================
// AmplitudeDampingOp
// ================================================================

AmplitudeDampingOp::AmplitudeDampingOp(int target_qbit, double gamma)
    : target_qbit_(target_qbit), fixed_gamma_(gamma), is_parametric_(false) {
  if (target_qbit < 0) {
    throw std::invalid_argument(
        "AmplitudeDampingOp: target_qbit must be >= 0");
  }
  if (gamma < 0.0 || gamma > 1.0) {
    throw std::invalid_argument("AmplitudeDampingOp: gamma must be in [0, 1]");
  }
}

AmplitudeDampingOp::AmplitudeDampingOp(int target_qbit)
    : target_qbit_(target_qbit), fixed_gamma_(0.0), is_parametric_(true) {
  if (target_qbit < 0) {
    throw std::invalid_argument(
        "AmplitudeDampingOp: target_qbit must be >= 0");
  }
}

void AmplitudeDampingOp::apply_to_density(const double *params, int param_count,
                                          DensityMatrix &rho) {
  double gamma;
  if (is_parametric_) {
    if (param_count < 1 || params == nullptr) {
      throw std::runtime_error(
          "AmplitudeDampingOp: parametric mode requires 1 parameter");
    }
    gamma = params[0];
    // Clamp to valid range
    if (gamma < 0.0)
      gamma = 0.0;
    if (gamma > 1.0)
      gamma = 1.0;
  } else {
    gamma = fixed_gamma_;
  }

  apply_amplitude_damping(rho, gamma);
}

void AmplitudeDampingOp::apply_amplitude_damping(DensityMatrix &rho,
                                                  double gamma) {
  int dim = rho.get_dim();
  int qbit_num = rho.get_qbit_num();

  if (target_qbit_ >= qbit_num) {
    throw std::runtime_error(
        "AmplitudeDampingOp: target_qbit out of range for density matrix");
  }

  // Kraus operators for amplitude damping on target qubit:
  // K₀ = [[1, 0], [0, √(1-γ)]]
  // K₁ = [[0, √γ], [0, 0]]
  //
  // ρ' = K₀ρK₀† + K₁ρK₁†

  // Create temporary storage
  DensityMatrix rho_new(qbit_num);
  memset(rho_new.data, 0, dim * dim * sizeof(QGD_Complex16));

  int target_step = 1 << target_qbit_;
  double sqrt_one_minus_gamma = std::sqrt(1.0 - gamma);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      bool i_target = (i & target_step) != 0;
      bool j_target = (j & target_step) != 0;

      // Apply K₀ρK₀†
      if (i_target == j_target) {
        double factor = i_target ? (1.0 - gamma) : 1.0;
        rho_new(i, j).real += factor * rho(i, j).real;
        rho_new(i, j).imag += factor * rho(i, j).imag;
      } else {
        // Off-diagonal between |0⟩ and |1⟩ states: multiply by √(1-γ)
        double factor = sqrt_one_minus_gamma;
        rho_new(i, j).real += factor * rho(i, j).real;
        rho_new(i, j).imag += factor * rho(i, j).imag;
      }

      // Apply K₁ρK₁†: maps |1⟩⟨1| → γ|0⟩⟨0|
      if (!i_target && !j_target) {
        int i_flip = i | target_step;
        int j_flip = j | target_step;
        rho_new(i, j).real += gamma * rho(i_flip, j_flip).real;
        rho_new(i, j).imag += gamma * rho(i_flip, j_flip).imag;
      }
    }
  }

  // Copy result back
  memcpy(rho.data, rho_new.data, dim * dim * sizeof(QGD_Complex16));
}

std::unique_ptr<IDensityOperation> AmplitudeDampingOp::clone() const {
  if (is_parametric_) {
    return std::make_unique<AmplitudeDampingOp>(target_qbit_);
  } else {
    return std::make_unique<AmplitudeDampingOp>(target_qbit_, fixed_gamma_);
  }
}

// ================================================================
// PhaseDampingOp
// ================================================================

PhaseDampingOp::PhaseDampingOp(int target_qbit, double lambda)
    : target_qbit_(target_qbit), fixed_lambda_(lambda), is_parametric_(false) {
  if (target_qbit < 0) {
    throw std::invalid_argument("PhaseDampingOp: target_qbit must be >= 0");
  }
  if (lambda < 0.0 || lambda > 1.0) {
    throw std::invalid_argument("PhaseDampingOp: lambda must be in [0, 1]");
  }
}

PhaseDampingOp::PhaseDampingOp(int target_qbit)
    : target_qbit_(target_qbit), fixed_lambda_(0.0), is_parametric_(true) {
  if (target_qbit < 0) {
    throw std::invalid_argument("PhaseDampingOp: target_qbit must be >= 0");
  }
}

void PhaseDampingOp::apply_to_density(const double *params, int param_count,
                                      DensityMatrix &rho) {
  double lambda;
  if (is_parametric_) {
    if (param_count < 1 || params == nullptr) {
      throw std::runtime_error(
          "PhaseDampingOp: parametric mode requires 1 parameter");
    }
    lambda = params[0];
    // Clamp to valid range
    if (lambda < 0.0)
      lambda = 0.0;
    if (lambda > 1.0)
      lambda = 1.0;
  } else {
    lambda = fixed_lambda_;
  }

  apply_phase_damping(rho, lambda);
}

void PhaseDampingOp::apply_phase_damping(DensityMatrix &rho, double lambda) {
  int dim = rho.get_dim();
  int qbit_num = rho.get_qbit_num();

  if (target_qbit_ >= qbit_num) {
    throw std::runtime_error(
        "PhaseDampingOp: target_qbit out of range for density matrix");
  }

  // Phase damping only affects off-diagonal elements between |0⟩ and |1⟩
  // ρ(i,j) → √(1-λ) · ρ(i,j) when target qubit states differ

  int target_step = 1 << target_qbit_;
  double sqrt_one_minus_lambda = std::sqrt(1.0 - lambda);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      bool i_target = (i & target_step) != 0;
      bool j_target = (j & target_step) != 0;

      // If target qubit states differ, reduce coherence
      if (i_target != j_target) {
        rho(i, j).real *= sqrt_one_minus_lambda;
        rho(i, j).imag *= sqrt_one_minus_lambda;
      }
      // Diagonal elements unchanged
    }
  }
}

std::unique_ptr<IDensityOperation> PhaseDampingOp::clone() const {
  if (is_parametric_) {
    return std::make_unique<PhaseDampingOp>(target_qbit_);
  } else {
    return std::make_unique<PhaseDampingOp>(target_qbit_, fixed_lambda_);
  }
}

} // namespace density
} // namespace squander

