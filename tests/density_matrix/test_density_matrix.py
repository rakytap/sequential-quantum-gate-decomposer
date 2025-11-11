"""
Python Unit Tests for Density Matrix Module
"""

import pytest
import numpy as np

try:
    from squander.density_matrix import (
        DensityMatrix,
        NoisyCircuit,
        DepolarizingChannel,
        AmplitudeDampingChannel,
        PhaseDampingChannel,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False


@pytest.mark.skipif(not HAS_MODULE, reason="Density matrix module not built")
class TestDensityMatrixBasics:
    """Test basic density matrix operations."""

    def test_construction(self):
        """Test density matrix construction."""
        rho = DensityMatrix(qbit_num=2)

        assert rho.qbit_num == 2
        assert rho.dim == 4

    def test_ground_state(self):
        """Test ground state initialization."""
        rho = DensityMatrix(qbit_num=2)

        # Convert to NumPy for easier checking
        rho_np = rho.to_numpy()

        # Should be |00⟩⟨00| = diag(1, 0, 0, 0)
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0

        np.testing.assert_allclose(rho_np, expected, atol=1e-10)

    def test_trace(self):
        """Test trace calculation."""
        rho = DensityMatrix(qbit_num=2)

        tr = rho.trace()
        assert np.isclose(tr.real, 1.0, atol=1e-10)
        assert np.isclose(tr.imag, 0.0, atol=1e-10)

    def test_purity_pure_state(self):
        """Test that pure states have purity = 1."""
        rho = DensityMatrix(qbit_num=2)

        purity = rho.purity()
        assert np.isclose(purity, 1.0, atol=1e-10), \
            f"Pure state should have purity=1, got {purity}"

    def test_purity_mixed_state(self):
        """Test that mixed states have purity < 1."""
        rho = DensityMatrix.maximally_mixed(qbit_num=2)

        purity = rho.purity()
        expected_purity = 0.25  # 1/2^2

        assert np.isclose(purity, expected_purity, atol=1e-10), \
            f"Maximally mixed 2-qubit state should have purity=0.25, got {purity}"

    def test_entropy_pure_state(self):
        """Test that pure states have zero entropy."""
        rho = DensityMatrix(qbit_num=2)

        entropy = rho.entropy()
        assert np.isclose(entropy, 0.0, atol=1e-10), \
            f"Pure state should have entropy=0, got {entropy}"

    def test_entropy_maximally_mixed(self):
        """Test entropy of maximally mixed state."""
        rho = DensityMatrix.maximally_mixed(qbit_num=2)

        entropy = rho.entropy()
        expected_entropy = 2.0  # log2(2^2) = 2 bits

        assert np.isclose(entropy, expected_entropy, atol=1e-10), \
            f"Maximally mixed 2-qubit state should have entropy=2, got {entropy}"

    def test_is_valid(self):
        """Test density matrix validation."""
        # Pure state should be valid
        rho_pure = DensityMatrix(qbit_num=2)
        assert rho_pure.is_valid()

        # Maximally mixed should be valid
        rho_mixed = DensityMatrix.maximally_mixed(qbit_num=2)
        assert rho_mixed.is_valid()

    def test_from_state_vector(self):
        """Test construction from state vector."""
        # Create |+⟩ = (|0⟩ + |1⟩)/√2
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)

        rho = DensityMatrix(psi)

        # Check: ρ = |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
        expected = np.outer(psi, psi.conj())
        rho_np = rho.to_numpy()

        np.testing.assert_allclose(rho_np, expected, atol=1e-10)

        # Should still be pure
        assert np.isclose(rho.purity(), 1.0, atol=1e-10)

    def test_eigenvalues(self):
        """Test eigenvalue computation."""
        # Pure state: one eigenvalue = 1, rest = 0
        rho = DensityMatrix(qbit_num=2)
        eigs = rho.eigenvalues()

        assert len(eigs) == 4
        assert np.isclose(eigs[0], 1.0, atol=1e-10)
        assert np.all(np.abs(eigs[1:]) < 1e-10)

        # Maximally mixed: all eigenvalues = 0.25
        rho_mixed = DensityMatrix.maximally_mixed(qbit_num=2)
        eigs_mixed = rho_mixed.eigenvalues()

        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(eigs_mixed, expected, atol=1e-10)


@pytest.mark.skipif(not HAS_MODULE, reason="Density matrix module not built")
class TestNoisyCircuit:
    """Test noisy circuit operations."""

    def test_circuit_construction(self):
        """Test circuit construction."""
        circuit = NoisyCircuit(qbit_num=2)

        assert circuit.qbit_num == 2
        assert circuit.parameter_num == 0  # No parametric gates yet

    def test_add_gates(self):
        """Test adding gates to circuit."""
        circuit = NoisyCircuit(2)

        circuit.add_H(0)
        circuit.add_X(1)
        circuit.add_CNOT(1, 0)

        # Should not raise

    def test_hadamard_application(self):
        """Test Hadamard gate application."""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)

        # Start with |00⟩
        rho = DensityMatrix(qbit_num=2)

        # Apply circuit
        circuit.apply_to(np.array([]), rho)

        # Qubit 0 should now be in |+⟩ state
        # ρ should have structure reflecting H ⊗ I applied to |00⟩
        purity = rho.purity()
        assert np.isclose(purity, 1.0, atol=1e-10), \
            "Hadamard should preserve purity"

    def test_bell_state_creation(self):
        """Test creation of Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)

        # Start with |00⟩
        rho = DensityMatrix(qbit_num=2)

        # Apply circuit
        circuit.apply_to(np.array([]), rho)

        # Check purity = 1 (should still be pure)
        purity = rho.purity()
        assert np.isclose(purity, 1.0, atol=1e-10)

        # Check specific elements
        rho_np = rho.to_numpy()

        # |Φ+⟩⟨Φ+| has elements at (0,0), (0,3), (3,0), (3,3)
        assert np.isclose(rho_np[0, 0], 0.5, atol=1e-10)
        assert np.isclose(rho_np[0, 3], 0.5, atol=1e-10)
        assert np.isclose(rho_np[3, 0], 0.5, atol=1e-10)
        assert np.isclose(rho_np[3, 3], 0.5, atol=1e-10)

    def test_parametric_gate(self):
        """Test parametric gate (RZ)."""
        circuit = NoisyCircuit(2)
        circuit.add_RZ(0)

        rho = DensityMatrix(qbit_num=2)

        # Apply with angle π/4
        params = np.array([np.pi / 4])
        circuit.apply_to(params, rho)

        # Should preserve purity (unitary evolution)
        purity = rho.purity()
        assert np.isclose(purity, 1.0, atol=1e-10)


@pytest.mark.skipif(not HAS_MODULE, reason="Density matrix module not built")
class TestNoiseChannels:
    """Test noise channel implementations."""

    def test_depolarizing_channel(self):
        """Test depolarizing noise reduces purity."""
        rho = DensityMatrix(qbit_num=2)

        initial_purity = rho.purity()
        assert np.isclose(initial_purity, 1.0)

        # Apply 10% depolarizing noise
        noise = DepolarizingChannel(qbit_num=2, error_rate=0.1)
        noise.apply(rho)

        final_purity = rho.purity()

        # Purity should decrease
        assert final_purity < initial_purity, \
            f"Noise should reduce purity: {initial_purity} → {final_purity}"

        # Trace should still be 1
        tr = rho.trace()
        assert np.isclose(tr.real, 1.0, atol=1e-10)

    def test_depolarizing_convergence(self):
        """Test depolarizing noise converges to maximally mixed."""
        rho = DensityMatrix(qbit_num=2)

        # Apply very strong noise (99%)
        noise = DepolarizingChannel(qbit_num=2, error_rate=0.99)
        noise.apply(rho)

        # Should be close to maximally mixed (purity ≈ 0.25)
        purity = rho.purity()
        assert np.isclose(purity, 0.25, atol=0.01), \
            f"Strong depolarizing should give purity ≈ 0.25, got {purity}"

    def test_amplitude_damping(self):
        """Test amplitude damping channel."""
        # Start with |1⟩ (excited state)
        psi = np.array([0, 1], dtype=complex)
        rho = DensityMatrix(psi)

        # Apply amplitude damping (30% probability of decay)
        noise = AmplitudeDampingChannel(target_qbit=0, gamma=0.3)
        noise.apply(rho)

        # Population in |1⟩ should decrease
        rho_np = rho.to_numpy()
        pop_1 = rho_np[1, 1].real

        # After damping: ρ₁₁ ≈ (1-γ) = 0.7
        assert pop_1 < 1.0, "Amplitude damping should reduce |1⟩ population"
        assert pop_1 > 0.6, "Population should not decay completely with γ=0.3"

    def test_phase_damping(self):
        """Test phase damping channel."""
        # Start with |+⟩ = (|0⟩ + |1⟩)/√2 (has coherence)
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = DensityMatrix(psi)

        rho_np_before = rho.to_numpy()
        coherence_before = abs(rho_np_before[0, 1])

        # Apply phase damping
        noise = PhaseDampingChannel(target_qbit=0, **{'lambda': 0.5})
        noise.apply(rho)

        rho_np_after = rho.to_numpy()
        coherence_after = abs(rho_np_after[0, 1])

        # Coherence should decrease
        assert coherence_after < coherence_before, \
            "Phase damping should reduce coherence"


@pytest.mark.skipif(not HAS_MODULE, reason="Density matrix module not built")
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_circuit_with_noise(self):
        """Test circuit application followed by noise."""
        # Create GHZ-like circuit
        circuit = NoisyCircuit(3)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)
        circuit.add_CNOT(2, 1)

        rho = DensityMatrix(qbit_num=3)

        # Apply circuit
        circuit.apply_to(np.array([]), rho)

        # Should still be pure
        purity_before = rho.purity()
        assert np.isclose(purity_before, 1.0, atol=1e-10)

        # Add noise
        noise = DepolarizingChannel(qbit_num=3, error_rate=0.05)
        noise.apply(rho)

        # Should now be mixed
        purity_after = rho.purity()
        assert purity_after < 1.0

        # But trace should still be 1
        tr = rho.trace()
        assert np.isclose(tr.real, 1.0, atol=1e-10)

    def test_partial_trace(self):
        """Test partial trace operation."""
        # Create Bell state
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)

        rho_full = DensityMatrix(qbit_num=2)
        circuit.apply_to(np.array([]), rho_full)

        # Trace out qubit 1
        rho_reduced = rho_full.partial_trace([1])

        # For Bell state, reduced should be maximally mixed
        purity_reduced = rho_reduced.purity()
        assert np.isclose(purity_reduced, 0.5, atol=1e-10), \
            f"Bell state reduced density matrix should have purity=0.5, got {purity_reduced}"

    def test_comparison_with_state_vector(self):
        """Test that density matrix evolution matches state vector for pure states."""
        from squander.gates.qgd_Circuit import qgd_Circuit

        # Create same circuit in both frameworks
        sv_circuit = qgd_Circuit(2)
        sv_circuit.add_H(0)
        sv_circuit.add_CNOT(1, 0)

        dm_circuit = NoisyCircuit(2)
        dm_circuit.add_H(0)
        dm_circuit.add_CNOT(1, 0)

        # Apply to state vector
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0
        U = sv_circuit.get_Matrix(np.array([]))
        sv_final = U @ sv

        # Apply to density matrix
        rho = DensityMatrix(qbit_num=2)
        dm_circuit.apply_to(np.array([]), rho)

        # Density matrix from state vector
        rho_expected = np.outer(sv_final, sv_final.conj())
        rho_actual = rho.to_numpy()

        np.testing.assert_allclose(rho_actual, rho_expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
