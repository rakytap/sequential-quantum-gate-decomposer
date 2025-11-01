/*
Copyright 2025 SQUANDER Contributors

C++ Unit Tests for Density Matrix Module
*/

#include "CNOT.h"
#include "Gate.h"
#include "H.h"
#include "X.h"
#include "density_circuit.h"
#include "density_gate.h"
#include "density_matrix.h"
#include "noise_channel.h"
#include <cassert>
#include <cmath>
#include <iostream>
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
// Test 5: Unitary Evolution
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
// Test 6: Depolarizing Noise
// ================================================================

int test_depolarizing_noise() {
  std::cout << "Test 6: Depolarizing noise..." << std::flush;

  // Start with pure state
  DensityMatrix rho(2);

  double initial_purity = rho.purity();
  ASSERT_NEAR(initial_purity, 1.0, 1e-10);

  // Apply depolarizing noise
  DepolarizingChannel noise(2, 0.5); // 50% noise
  noise.apply(rho);

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
// Test 7: Circuit Application
// ================================================================

int test_circuit_application() {
  std::cout << "Test 7: Circuit application..." << std::flush;

  // Create Bell state circuit: H(0) - CNOT(1,0)
  DensityCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);

  // Start with |00⟩
  DensityMatrix rho(2);

  // Apply circuit
  Matrix_real params; // Empty parameters
  circuit.apply_to(params, rho);

  // Check: should create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
  // ρ = |Φ+⟩⟨Φ+| has non-zero elements at (0,0), (0,3), (3,0), (3,3)
  ASSERT_NEAR(rho(0, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(0, 3).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(3, 0).real, 0.5, 1e-10);
  ASSERT_NEAR(rho(3, 3).real, 0.5, 1e-10);

  // Still pure
  double pur = rho.purity();
  ASSERT_NEAR(pur, 1.0, 1e-10);

  std::cout << " PASSED" << std::endl;
  return 0;
}

// ================================================================
// Test 8: Partial Trace
// ================================================================

int test_partial_trace() {
  std::cout << "Test 8: Partial trace..." << std::flush;

  // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
  DensityCircuit circuit(2);
  circuit.add_H(0);
  circuit.add_CNOT(1, 0);

  DensityMatrix rho_full(2);
  Matrix_real params;
  circuit.apply_to(params, rho_full);

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
// Main Test Runner
// ================================================================

int main() {
  std::cout << "\n========================================" << std::endl;
  std::cout << "  Density Matrix C++ Unit Tests" << std::endl;
  std::cout << "========================================\n" << std::endl;

  int result = 0;

  result |= test_construction();
  result |= test_ground_state();
  result |= test_state_vector_construction();
  result |= test_maximally_mixed();
  result |= test_unitary_evolution();
  result |= test_depolarizing_noise();
  result |= test_circuit_application();
  result |= test_partial_trace();

  if (result == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests PASSED ✓" << std::endl;
    std::cout << "========================================\n" << std::endl;
  } else {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Some tests FAILED ✗" << std::endl;
    std::cout << "========================================\n" << std::endl;
  }

  return result;
}
