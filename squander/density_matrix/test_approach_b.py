"""
Python Unit Tests for Density Matrix Module - Approach B Implementation

Run with: pytest test_approach_b.py -v
"""

import numpy as np
import pytest


# Import the C++ bindings
try:
    from squander.density_matrix import (
        DensityMatrix,
        NoisyCircuit,
        DepolarizingChannel,
        AmplitudeDampingChannel,
        PhaseDampingChannel,
    )
except ImportError:
    pytest.skip("density_matrix module not built", allow_module_level=True)


class TestDensityMatrix:
    """Test DensityMatrix class"""

    def test_construction(self):
        """Test basic construction"""
        rho = DensityMatrix(2)
        assert rho.qbit_num == 2
        assert rho.dim == 4

    def test_ground_state(self):
        """Test ground state initialization"""
        rho = DensityMatrix(2)
        arr = rho.to_numpy()

        # Should be |00⟩⟨00| = diag(1, 0, 0, 0)
        assert np.isclose(arr[0, 0], 1.0)
        for i in range(1, 4):
            assert np.isclose(arr[i, i], 0.0)

        # Properties
        assert np.isclose(rho.trace(), 1.0)
        assert np.isclose(rho.purity(), 1.0)
        assert np.isclose(rho.entropy(), 0.0)
        assert rho.is_valid()

    def test_state_vector_construction(self):
        """Test construction from state vector"""
        # Create |+⟩ state
        psi = np.array([1, 1]) / np.sqrt(2)
        rho = DensityMatrix(psi)

        arr = rho.to_numpy()

        # Should be |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(arr, expected)
        assert np.isclose(rho.purity(), 1.0)

    def test_maximally_mixed(self):
        """Test maximally mixed state"""
        rho = DensityMatrix.maximally_mixed(2)

        assert np.isclose(rho.trace(), 1.0)
        assert np.isclose(rho.purity(), 0.25)
        assert np.isclose(rho.entropy(), 2.0)
        assert rho.is_valid()

    def test_apply_unitary(self):
        """Test unitary application"""
        rho = DensityMatrix(1)

        # Apply Hadamard
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        rho.apply_unitary(H)

        arr = rho.to_numpy()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(arr, expected, atol=1e-10)

    def test_apply_single_qubit_local(self):
        """Test optimized single-qubit unitary"""
        rho = DensityMatrix(2)

        # Apply Hadamard on qubit 0
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        rho.apply_single_qubit_unitary(H, 0)

        arr = rho.to_numpy()

        # Should create |+0⟩ state
        # Qubit 0 is most significant bit in this convention
        # |+0⟩ = (|00⟩ + |01⟩)/√2 corresponds to indices 0 and 1
        assert np.isclose(arr[0, 0], 0.5)
        assert np.isclose(arr[0, 1], 0.5)
        assert np.isclose(arr[1, 0], 0.5)
        assert np.isclose(arr[1, 1], 0.5)

    def test_partial_trace(self):
        """Test partial trace"""
        # Create Bell state manually
        bell = np.array([[0.5, 0, 0, 0.5],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0.5, 0, 0, 0.5]])
        rho = DensityMatrix.from_numpy(bell)

        # Trace out qubit 1
        rho_reduced = rho.partial_trace([1])

        arr = rho_reduced.to_numpy()
        # Should be I/2
        expected = np.array([[0.5, 0], [0, 0.5]])
        assert np.allclose(arr, expected)

    def test_clone(self):
        """Test cloning"""
        rho1 = DensityMatrix(2)
        rho2 = rho1.clone()

        # Modify rho1
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        rho1.apply_single_qubit_unitary(H, 0)

        # rho2 should be unchanged
        arr2 = rho2.to_numpy()
        assert np.isclose(arr2[0, 0], 1.0)


class TestNoisyCircuit:
    """Test NoisyCircuit class"""

    def test_construction(self):
        """Test circuit construction"""
        circuit = NoisyCircuit(2)
        assert circuit.qbit_num == 2
        assert circuit.parameter_num == 0
        assert len(circuit) == 0

    def test_bell_state(self):
        """Test Bell state creation"""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)

        assert len(circuit) == 2
        assert circuit.parameter_num == 0

        rho = DensityMatrix(2)
        circuit.apply_to(np.array([]), rho)

        arr = rho.to_numpy()
        assert np.isclose(arr[0, 0], 0.5)
        assert np.isclose(arr[0, 3], 0.5)
        assert np.isclose(arr[3, 0], 0.5)
        assert np.isclose(arr[3, 3], 0.5)
        assert np.isclose(rho.purity(), 1.0)

    def test_fixed_depolarizing(self):
        """Test fixed depolarizing noise"""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)
        circuit.add_depolarizing(2, error_rate=0.1)

        assert len(circuit) == 3
        assert circuit.parameter_num == 0

        rho = DensityMatrix(2)
        circuit.apply_to(np.array([]), rho)

        assert rho.purity() < 1.0
        assert rho.is_valid()

    def test_parametric_depolarizing(self):
        """Test parametric depolarizing noise"""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)
        circuit.add_depolarizing(2)  # Parametric

        assert len(circuit) == 3
        assert circuit.parameter_num == 1

        rho = DensityMatrix(2)
        params = np.array([0.2])  # 20% error rate
        circuit.apply_to(params, rho)

        assert rho.purity() < 1.0
        assert rho.is_valid()

    def test_amplitude_damping(self):
        """Test amplitude damping"""
        circuit = NoisyCircuit(1)
        circuit.add_X(0)  # Start in |1⟩
        circuit.add_amplitude_damping(0, gamma=0.5)

        rho = DensityMatrix(1)
        circuit.apply_to(np.array([]), rho)

        arr = rho.to_numpy()
        # Should have some population in |0⟩
        assert arr[0, 0].real > 0
        assert arr[1, 1].real < 1
        assert np.isclose(rho.trace(), 1.0)

    def test_phase_damping(self):
        """Test phase damping"""
        circuit = NoisyCircuit(1)
        circuit.add_H(0)  # Create |+⟩ state with coherence
        circuit.add_phase_damping(0, lambda_param=0.5)

        rho = DensityMatrix(1)
        circuit.apply_to(np.array([]), rho)

        arr = rho.to_numpy()
        # Off-diagonal elements should be reduced
        assert np.abs(arr[0, 1]) < 0.5
        # Diagonal should be unchanged
        assert np.isclose(arr[0, 0].real, 0.5)
        assert np.isclose(arr[1, 1].real, 0.5)

    def test_mixed_operations(self):
        """Test mixed gates and noise"""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_amplitude_damping(0, gamma=0.1)
        circuit.add_CNOT(1, 0)
        circuit.add_phase_damping(1, lambda_param=0.05)
        circuit.add_depolarizing(2, error_rate=0.02)

        assert len(circuit) == 5
        assert circuit.parameter_num == 0

        rho = DensityMatrix(2)
        circuit.apply_to(np.array([]), rho)

        assert rho.is_valid()
        assert rho.purity() < 1.0

    def test_operation_info(self):
        """Test operation info"""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)
        circuit.add_depolarizing(2, error_rate=0.1)
        circuit.add_phase_damping(0)  # Parametric

        info = circuit.get_operation_info()

        assert len(info) == 4
        assert info[0].name == "H"
        assert info[0].is_unitary == True
        assert info[0].param_count == 0

        assert info[1].name == "CNOT"
        assert info[1].is_unitary == True

        assert info[2].name == "Depolarizing"
        assert info[2].is_unitary == False
        assert info[2].param_count == 0

        assert info[3].name == "PhaseDamping"
        assert info[3].is_unitary == False
        assert info[3].param_count == 1

    def test_all_gates(self):
        """Test all gate types work"""
        circuit = NoisyCircuit(2)

        # Single-qubit constant gates
        circuit.add_H(0)
        circuit.add_X(0)
        circuit.add_Y(0)
        circuit.add_Z(0)
        circuit.add_S(0)
        circuit.add_Sdg(0)
        circuit.add_T(0)
        circuit.add_Tdg(0)
        circuit.add_SX(0)

        # Two-qubit gates
        circuit.add_CNOT(1, 0)
        circuit.add_CZ(1, 0)
        circuit.add_CH(1, 0)

        rho = DensityMatrix(2)
        circuit.apply_to(np.array([]), rho)

        # Should complete without error and produce valid state
        assert rho.is_valid()


class TestLegacyNoiseChannels:
    """Test legacy noise channel interface"""

    def test_depolarizing_channel(self):
        """Test legacy depolarizing channel"""
        rho = DensityMatrix(2)
        noise = DepolarizingChannel(2, 0.5)
        noise.apply(rho)

        assert rho.purity() < 1.0
        assert rho.is_valid()

    def test_amplitude_damping_channel(self):
        """Test legacy amplitude damping channel"""
        # Start with |1⟩
        psi = np.array([0, 1], dtype=complex)
        rho = DensityMatrix(psi)

        noise = AmplitudeDampingChannel(0, 0.5)
        noise.apply(rho)

        arr = rho.to_numpy()
        assert arr[0, 0].real > 0  # Some population transferred to |0⟩
        assert rho.is_valid()

    def test_phase_damping_channel(self):
        """Test legacy phase damping channel"""
        # Create |+⟩ state
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = DensityMatrix(psi)

        noise = PhaseDampingChannel(0, 0.5)
        noise.apply(rho)

        arr = rho.to_numpy()
        assert np.abs(arr[0, 1]) < 0.5  # Coherence reduced
        assert rho.is_valid()


class TestPerformance:
    """Performance-related tests"""

    def test_large_circuit(self):
        """Test moderately large circuit"""
        n_qubits = 4
        circuit = NoisyCircuit(n_qubits)

        # Add 20 operations
        for _ in range(5):
            circuit.add_H(0)
            circuit.add_CNOT(1, 0)
            circuit.add_CNOT(2, 1)
            circuit.add_CNOT(3, 2)

        rho = DensityMatrix(n_qubits)
        circuit.apply_to(np.array([]), rho)

        assert rho.is_valid()

    def test_repeated_noise(self):
        """Test repeated noise application"""
        circuit = NoisyCircuit(2)

        # Interleave gates with noise (10% per step for more noticeable effect)
        for _ in range(10):
            circuit.add_H(0)
            circuit.add_depolarizing(2, error_rate=0.1)

        rho = DensityMatrix(2)
        circuit.apply_to(np.array([]), rho)

        # Purity should decrease (10 applications of 10% noise)
        # Not necessarily below 0.5, but significantly reduced
        assert rho.purity() < 0.9
        assert rho.is_valid()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

