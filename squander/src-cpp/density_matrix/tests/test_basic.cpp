/*
Copyright 2025 SQUANDER Contributors

C++ Unit Tests for Density Matrix Module - Approach B Implementation
*/

#include "CNOT.h"
#include "Gate.h"
#include "H.h"
#include "RZ.h"
#include "X.h"
#include "density_matrix.h"
#include "gate_operation.h"
#include "noise_channel.h"
#include "noise_operation.h"
#include "noisy_circuit.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace squander::density;

// Test utilities
#define ASSERT_NEAR(a, b, tol)                                                 \
  if (std::abs((a) - (b)) > (tol)) {                                           \
    std::cerr << "FAILED: Expected " << (a) << " ≈ " << (b)                    \
              << ", difference: " << std::abs((a) - (b)) << std::endl;         \
    return 1;                                                                  \
  }

#define ASSERT_TRUE(cond)                                                      \
  if (!(cond)) {                                                               \
    std::cerr << "FAILED: Condition false: " #cond << std::endl;               \
    return 1;                                                                  \
  }

// ================================================================
// Test 1: Basic Construction
// ================================================================

int test_construction() {
  std::cout << "Test 1: Construction..." << std::flush;

  // 2-qubit density matrix
  DensityMatrix rho(2);

  ASSERT_TRUE(rho.get_qbit_num() == 2);
  ASSERT_TRUE(rho.get_dim() == 4);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 2: Ground State
// ================================================================

int test_ground_state() {
  std::cout << "Test 2: Ground state initialization..." << std::flush;

  DensityMatrix rho(2);

  // Should be |00⟩⟨00| = diag(1, 0, 0, 0)
  ASSERT_NEAR(rho(0, 0).real, 1.0, 1e-10);
  ASSERT_NEAR(rho(0, 0).imag, 0.0, 1e-10);

  for (int i = 1; i < 4; i++) {
    ASSERT_NEAR(rho(i, i).real, 0.0, 1e-10);
    ASSERT_NEAR(rho(i, i).imag, 0.0, 1e-10);
  }

  // Check trace = 1
  QGD_Complex16 tr = rho.trace();
  ASSERT_NEAR(tr.real, 1.0, 1e-10);
  ASSERT_NEAR(tr.imag, 0.0, 1e-10);

  // Check purity = 1 (pure state)
  double pur = rho.purity();
  ASSERT_NEAR(pur, 1.0, 1e-10);

  // Check entropy = 0 (pure state)
  double S = rho.entropy();
  ASSERT_NEAR(S, 0.0, 1e-10);

  // Check validity
  ASSERT_TRUE(rho.is_valid());

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 3: State Vector Construction
// ================================================================

int test_state_vector_construction() {
  std::cout << "Test 3: State vector construction..." << std::flush;

  // Create |+⟩ = (|0⟩ + |1⟩)/√2
  Matrix psi(2, 1);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  psi.get_data()[0].real = inv_sqrt2;
  psi.get_data()[0].imag = 0.0;
  psi.get_data()[1].real = inv_sqrt2;
  psi.get_data()[1].imag = 0.0;

  // Convert to density matrix
  DensityMatrix rho(psi);

  // Check: ρ = |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 1).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 1).real, 0.5, 1e-10);

  // Check purity = 1 (still pure)
  double pur = rho.purity();
  ASSERT_NEAR(pur, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 4: Maximally Mixed State
// ================================================================

int test_maximally_mixed() {
  std::cout << "Test 4: Maximally mixed state..." << std::flush;

  DensityMatrix rho = DensityMatrix::maximally_mixed(2);

  // Check trace = 1
  QGD_Complex16 tr = rho.trace();
  ASSERT_NEAR(tr.real, 1.0, 1e-10);

  // Check purity = 1/4 (maximally mixed for 2 qubits)
  double pur = rho.purity();
  ASSERT_NEAR(pur, 0.25, 1e-10);

  // Check entropy = 2 bits (maximal for 2 qubits)
  double S = rho.entropy();
  ASSERT_NEAR(S, 2.0, 1e-10);

  // Check validity
  ASSERT_TRUE(rho.is_valid());

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 5: Unitary Evolution (Full Matrix)
// ================================================================

int test_unitary_evolution() {
  std::cout << "Test 5: Unitary evolution..." << std::flush;

  // Start with |0⟩
  Matrix psi(2, 1);
  psi.get_data()[0].real = 1.0;
  psi.get_data()[0].imag = 0.0;
  psi.get_data()[1].real = 0.0;
  psi.get_data()[1].imag = 0.0;

  DensityMatrix rho(psi);

  // Apply Hadamard: H = [[1, 1], [1, -1]] / √2
  Matrix H_matrix(2, 2);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  H_matrix.get_data()[0].real = inv_sqrt2;
  H_matrix.get_data()[0].imag = 0.0;
  H_matrix.get_data()[1].real = inv_sqrt2;
  H_matrix.get_data()[1].imag = 0.0;
  H_matrix.get_data()[2].real = inv_sqrt2;
  H_matrix.get_data()[2].imag = 0.0;
  H_matrix.get_data()[3].real = -inv_sqrt2;
  H_matrix.get_data()[3].imag = 0.0;

  rho.apply_unitary(H_matrix);

  // Check: ρ should be |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 1).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 1).real, 0.5, 1e-10);

  // Still pure
  double pur = rho.purity();
  ASSERT_NEAR(pur, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 6: apply_single_qubit_unitary (Local Kernel)
// ================================================================

int test_single_qubit_local() {
  std::cout << "Test 6: Single-qubit local kernel..." << std::flush;

  // Start with |00⟩
  DensityMatrix rho(2);

  // Create Hadamard 2x2 kernel
  Matrix H_kernel(2, 2);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  H_kernel.get_data()[0] = {inv_sqrt2, 0.0};
  H_kernel.get_data()[1] = {inv_sqrt2, 0.0};
  H_kernel.get_data()[2] = {inv_sqrt2, 0.0};
  H_kernel.get_data()[3] = {-inv_sqrt2, 0.0};

  // Apply H on qubit 0
  rho.apply_single_qubit_unitary(H_kernel, 0);

  // Should create |+0⟩ = (|00⟩ + |01⟩)/√2 in this qubit convention
  // Qubit 0 is most significant bit: |q0 q1⟩
  // ρ should have entries at (0,0), (0,1), (1,0), (1,1) = 0.5
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 1).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 1).real, 0.5, 1e-10);

  // Still pure
  ASSERT_NEAR(rho.purity(), 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 7: apply_two_qubit_unitary (Controlled Gate)
// ================================================================

int test_two_qubit_controlled() {
  std::cout << "Test 7: Two-qubit controlled gate..." << std::flush;

  // Create |+0⟩ state: H on qubit 0 first
  DensityMatrix rho(2);

  Matrix H_kernel(2, 2);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  H_kernel.get_data()[0] = {inv_sqrt2, 0.0};
  H_kernel.get_data()[1] = {inv_sqrt2, 0.0};
  H_kernel.get_data()[2] = {inv_sqrt2, 0.0};
  H_kernel.get_data()[3] = {-inv_sqrt2, 0.0};

  rho.apply_single_qubit_unitary(H_kernel, 0);

  // Apply X kernel controlled by qubit 0 on qubit 1 (CNOT-like)
  Matrix X_kernel(2, 2);
  X_kernel.get_data()[0] = {0.0, 0.0};
  X_kernel.get_data()[1] = {1.0, 0.0};
  X_kernel.get_data()[2] = {1.0, 0.0};
  X_kernel.get_data()[3] = {0.0, 0.0};

  rho.apply_two_qubit_unitary(X_kernel, 1, 0);

  // Should create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
  // ρ should have entries at (0,0), (0,3), (3,0), (3,3) = 0.5
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 3).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(3, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(3, 3).real, 0.5, 1e-10);

  // Still pure
  ASSERT_NEAR(rho.purity(), 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 8: GateOperation Wrapper
// ================================================================

int test_gate_operation() {
  std::cout << "Test 8: GateOperation wrapper..." << std::flush;

  DensityMatrix rho(2);

  // Create H gate and wrap it
  Gate *h_gate = new H(2, 0);
  GateOperation h_op(h_gate, true);

  ASSERT_TRUE(h_op.is_unitary());
  ASSERT_TRUE(h_op.get_name() == "H");
  ASSERT_TRUE(h_op.get_parameter_num() == 0);

  // Apply via GateOperation
  h_op.apply_to_density(nullptr, 0, rho);

  // Check result
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 2).real, 0.5, 1e-10);
  ASSERT_NEAR(rho.purity(), 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 9: Depolarizing Noise Operation (Fixed)
// ================================================================

int test_depolarizing_fixed() {
  std::cout << "Test 9: Depolarizing noise (fixed)..." << std::flush;

  DensityMatrix rho(2);

  double initial_purity = rho.purity();
  ASSERT_NEAR(initial_purity, 1.0, 1e-10);

  // Create fixed depolarizing noise
  DepolarizingOp noise(2, 0.5); // 50% noise

  ASSERT_TRUE(!noise.is_unitary());
  ASSERT_TRUE(noise.get_name() == "Depolarizing");
  ASSERT_TRUE(noise.get_parameter_num() == 0);

  noise.apply_to_density(nullptr, 0, rho);

  // Purity should decrease
  double final_purity = rho.purity();
  ASSERT_TRUE(final_purity < initial_purity);

  // Trace should still be 1
  QGD_Complex16 tr = rho.trace();
  ASSERT_NEAR(tr.real, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 10: Depolarizing Noise Operation (Parametric)
// ================================================================

int test_depolarizing_parametric() {
  std::cout << "Test 10: Depolarizing noise (parametric)..." << std::flush;

  DensityMatrix rho(2);

  // Create parametric depolarizing noise
  DepolarizingOp noise(2); // Parametric

  ASSERT_TRUE(noise.is_parametric());
  ASSERT_TRUE(noise.get_parameter_num() == 1);

  double p = 0.3;
  noise.apply_to_density(&p, 1, rho);

  // Purity should decrease
  double final_purity = rho.purity();
  ASSERT_TRUE(final_purity < 1.0);

  // Trace should still be 1
  QGD_Complex16 tr = rho.trace();
  ASSERT_NEAR(tr.real, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 11: Amplitude Damping
// ================================================================

int test_amplitude_damping() {
  std::cout << "Test 11: Amplitude damping..." << std::flush;

  // Start with |1⟩ state
  Matrix psi(2, 1);
  psi.get_data()[0] = {0.0, 0.0};
  psi.get_data()[1] = {1.0, 0.0};
  DensityMatrix rho(psi);

  ASSERT_NEAR(rho(1, 1).real, 1.0, 1e-10);

  // Apply amplitude damping with γ = 0.5
  AmplitudeDampingOp noise(0, 0.5);

  ASSERT_TRUE(!noise.is_unitary());
  ASSERT_TRUE(noise.get_name() == "AmplitudeDamping");

  noise.apply_to_density(nullptr, 0, rho);

  // Population should transfer from |1⟩ to |0⟩
  // ρ(0,0) should increase, ρ(1,1) should decrease
  ASSERT_TRUE(rho(0, 0).real > 0.0);
  ASSERT_TRUE(rho(1, 1).real < 1.0);

  // Trace should still be 1
  QGD_Complex16 tr = rho.trace();
  ASSERT_NEAR(tr.real, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 12: Phase Damping
// ================================================================

int test_phase_damping() {
  std::cout << "Test 12: Phase damping..." << std::flush;

  // Start with |+⟩ state (has off-diagonal coherence)
  Matrix psi(2, 1);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  psi.get_data()[0] = {inv_sqrt2, 0.0};
  psi.get_data()[1] = {inv_sqrt2, 0.0};
  DensityMatrix rho(psi);

  double initial_coherence = std::abs(rho(0, 1).real);
  ASSERT_NEAR(initial_coherence, 0.5, 1e-10);

  // Apply phase damping with λ = 0.5
  PhaseDampingOp noise(0, 0.5);

  ASSERT_TRUE(!noise.is_unitary());
  ASSERT_TRUE(noise.get_name() == "PhaseDamping");

  noise.apply_to_density(nullptr, 0, rho);

  // Off-diagonal elements should decay
  double final_coherence = std::abs(rho(0, 1).real);
  ASSERT_TRUE(final_coherence < initial_coherence);

  // Diagonal elements should be unchanged
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(1, 1).real, 0.5, 1e-10);

  // Trace should still be 1
  QGD_Complex16 tr = rho.trace();
  ASSERT_NEAR(tr.real, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 13: NoisyCircuit Bell State
// ================================================================

int test_noisy_circuit_bell() {
  std::cout << "Test 13: NoisyCircuit Bell state..." << std::flush;

  NoisyCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);

  ASSERT_TRUE(circuit.get_qbit_num() == 2);
  ASSERT_TRUE(circuit.get_operation_count() == 2);
  ASSERT_TRUE(circuit.get_parameter_num() == 0);

  DensityMatrix rho(2);
  circuit.apply_to(nullptr, 0, rho);

  // Should create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 3).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(3, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(3, 3).real, 0.5, 1e-10);

  // Still pure
  ASSERT_NEAR(rho.purity(), 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 14: NoisyCircuit with Fixed Noise
// ================================================================

int test_noisy_circuit_with_noise() {
  std::cout << "Test 14: NoisyCircuit with fixed noise..." << std::flush;

  NoisyCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);
  circuit.add_depolarizing(2, 0.1); // Fixed 10% depolarizing

  ASSERT_TRUE(circuit.get_operation_count() == 3);
  ASSERT_TRUE(circuit.get_parameter_num() == 0);

  DensityMatrix rho(2);
  circuit.apply_to(nullptr, 0, rho);

  // Should be mixed state (purity < 1)
  ASSERT_TRUE(rho.purity() < 1.0);
  ASSERT_TRUE(rho.is_valid());

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 15: NoisyCircuit with Parametric Noise
// ================================================================

int test_noisy_circuit_parametric_noise() {
  std::cout << "Test 15: NoisyCircuit with parametric noise..." << std::flush;

  NoisyCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);
  circuit.add_depolarizing(2); // Parametric (1 param)

  ASSERT_TRUE(circuit.get_operation_count() == 3);
  ASSERT_TRUE(circuit.get_parameter_num() == 1);

  DensityMatrix rho(2);
  double params[1] = {0.2}; // 20% depolarizing
  circuit.apply_to(params, 1, rho);

  // Should be mixed state
  ASSERT_TRUE(rho.purity() < 1.0);
  ASSERT_TRUE(rho.is_valid());

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 16: Mixed Gates and Noise
// ================================================================

int test_mixed_gates_and_noise() {
  std::cout << "Test 16: Mixed gates and noise..." << std::flush;

  NoisyCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_amplitude_damping(0, 0.1);
  circuit.add_CNOT(1, 0);
  circuit.add_phase_damping(1, 0.05);
  circuit.add_depolarizing(2, 0.02);

  ASSERT_TRUE(circuit.get_operation_count() == 5);
  ASSERT_TRUE(circuit.get_parameter_num() == 0);

  DensityMatrix rho(2);
  circuit.apply_to(nullptr, 0, rho);

  // Should be valid mixed state
  ASSERT_TRUE(rho.is_valid());
  ASSERT_TRUE(rho.purity() < 1.0);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 17: Operation Info
// ================================================================

int test_operation_info() {
  std::cout << "Test 17: Operation info..." << std::flush;

  NoisyCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);
  circuit.add_depolarizing(2, 0.1);
  circuit.add_phase_damping(0);

  auto info = circuit.get_operation_info();

  ASSERT_TRUE(info.size() == 4);
  ASSERT_TRUE(info[0].name == "H");
  ASSERT_TRUE(info[0].is_unitary == true);
  ASSERT_TRUE(info[0].param_count == 0);

  ASSERT_TRUE(info[1].name == "CNOT");
  ASSERT_TRUE(info[1].is_unitary == true);

  ASSERT_TRUE(info[2].name == "Depolarizing");
  ASSERT_TRUE(info[2].is_unitary == false);
  ASSERT_TRUE(info[2].param_count == 0);

  ASSERT_TRUE(info[3].name == "PhaseDamping");
  ASSERT_TRUE(info[3].is_unitary == false);
  ASSERT_TRUE(info[3].param_count == 1);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 18: Partial Trace
// ================================================================

int test_partial_trace() {
  std::cout << "Test 18: Partial trace..." << std::flush;

  // Create Bell state
  NoisyCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);

  DensityMatrix rho_full(2);
  circuit.apply_to(nullptr, 0, rho_full);

  // Trace out qubit 1
  std::vector<int> trace_out = {1};
  DensityMatrix rho_reduced = rho_full.partial_trace(trace_out);

  // For Bell state, reduced density matrix should be maximally mixed: I/2
  ASSERT_NEAR(rho_reduced(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho_reduced(1, 1).real, 0.5, 1e-10);
  ASSERT_NEAR(rho_reduced(0, 1).real, 0.0, 1e-10);
  ASSERT_NEAR(rho_reduced(1, 0).real, 0.0, 1e-10);

  // Purity should be 0.5 (maximally mixed single qubit)
  double pur = rho_reduced.purity();
  ASSERT_NEAR(pur, 0.5, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 19: Clone Operations
// ================================================================

int test_clone_operations() {
  std::cout << "Test 19: Clone operations..." << std::flush;

  // Test GateOperation clone
  Gate *h_gate = new H(2, 0);
  auto h_op = std::make_unique<GateOperation>(h_gate, true);
  auto h_clone = h_op->clone();

  ASSERT_TRUE(h_clone->get_name() == "H");
  ASSERT_TRUE(h_clone->is_unitary());

  // Test noise operation clones
  auto dep_op = std::make_unique<DepolarizingOp>(2, 0.1);
  auto dep_clone = dep_op->clone();

  ASSERT_TRUE(dep_clone->get_name() == "Depolarizing");
  ASSERT_TRUE(!dep_clone->is_unitary());

  auto amp_op = std::make_unique<AmplitudeDampingOp>(0, 0.2);
  auto amp_clone = amp_op->clone();

  ASSERT_TRUE(amp_clone->get_name() == "AmplitudeDamping");

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 20: Legacy NoiseChannel Compatibility
// ================================================================

int test_legacy_noise_channel() {
  std::cout << "Test 20: Legacy NoiseChannel compatibility..." << std::flush;

  DensityMatrix rho(2);

  // Use legacy interface
  DepolarizingChannel noise(2, 0.5);
  noise.apply(rho);

  // Should work the same
  ASSERT_TRUE(rho.purity() < 1.0);
  ASSERT_TRUE(rho.is_valid());

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Main Test Runner
// ================================================================

int main() {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Density Matrix C++ Unit Tests" << std::endl;
  std::cout << "  Approach B: Interface Segregation" << std::endl;
  std::cout << "========================================\n" << std::endl;

  int result = 0;

  // Basic tests
  result |= test_construction();
  result |= test_ground_state();
  result |= test_state_vector_construction();
  result |= test_maximally_mixed();
  result |= test_unitary_evolution();

  // Local kernel application tests
  result |= test_single_qubit_local();
  result |= test_two_qubit_controlled();

  // IDensityOperation interface tests
  result |= test_gate_operation();
  result |= test_depolarizing_fixed();
  result |= test_depolarizing_parametric();
  result |= test_amplitude_damping();
  result |= test_phase_damping();

  // NoisyCircuit tests
  result |= test_noisy_circuit_bell();
  result |= test_noisy_circuit_with_noise();
  result |= test_noisy_circuit_parametric_noise();
  result |= test_mixed_gates_and_noise();
  result |= test_operation_info();

  // Additional tests
  result |= test_partial_trace();
  result |= test_clone_operations();
  result |= test_legacy_noise_channel();

  if (result == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  All 20 tests PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
  } else {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Some tests FAILED ✗" << std::endl;
    std::cout << "========================================\n" << std::endl;
  }

  return result;
}
