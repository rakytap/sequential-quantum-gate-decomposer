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


HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _assert_purity_reduced(rho):
    assert rho.purity() < 1.0
    assert rho.is_valid()


def _assert_population_transferred(rho):
    rho_np = rho.to_numpy()
    assert rho_np[0, 0].real > 0
    assert rho.is_valid()


def _assert_coherence_reduced(rho):
    rho_np = rho.to_numpy()
    assert np.abs(rho_np[0, 1]) < 0.5
    assert rho.is_valid()


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

    def test_apply_unitary(self):
        """Test unitary application."""
        rho = DensityMatrix(qbit_num=1)

        # Apply Hadamard
        rho.apply_unitary(HADAMARD)

        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(rho.to_numpy(), expected, atol=1e-10)

    def test_apply_single_qubit_unitary(self):
        """Test optimized single-qubit unitary."""
        rho = DensityMatrix(qbit_num=2)

        # Apply Hadamard on qubit 0
        rho.apply_single_qubit_unitary(HADAMARD, 0)

        rho_np = rho.to_numpy()

        # Should create |+0⟩ state
        # Qubit 0 is most significant bit in this convention
        # |+0⟩ = (|00⟩ + |01⟩)/√2 corresponds to indices 0 and 1
        assert np.isclose(rho_np[0, 0], 0.5)
        assert np.isclose(rho_np[0, 1], 0.5)
        assert np.isclose(rho_np[1, 0], 0.5)
        assert np.isclose(rho_np[1, 1], 0.5)

    def test_clone(self):
        """Test cloning."""
        rho1 = DensityMatrix(qbit_num=2)
        rho2 = rho1.clone()

        # Modify rho1
        rho1.apply_single_qubit_unitary(HADAMARD, 0)

        # rho2 should be unchanged
        rho2_np = rho2.to_numpy()
        assert np.isclose(rho2_np[0, 0], 1.0)

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
class TestNoisyCircuitUnitary:
    """Test unitary-only circuit operations."""

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
class TestNoisyCircuitNoise:
    """Test noise operations in circuits."""

    @pytest.mark.parametrize(
        "error_rate, params, expected_param_num",
        [
            pytest.param(0.1, np.array([]), 0, id="fixed"),
            pytest.param(None, np.array([0.2]), 1, id="parametric"),
        ],
    )
    def test_depolarizing(self, error_rate, params, expected_param_num):
        """Test depolarizing noise in circuit."""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)
        if error_rate is None:
            circuit.add_depolarizing(2)
        else:
            circuit.add_depolarizing(2, error_rate=error_rate)

        assert circuit.parameter_num == expected_param_num

        rho = DensityMatrix(qbit_num=2)
        circuit.apply_to(params, rho)

        assert rho.purity() < 1.0
        assert rho.is_valid()

    def test_local_depolarizing(self):
        """Test local single-qubit depolarizing noise in circuit."""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_local_depolarizing(0, error_rate=0.2)

        rho = DensityMatrix(qbit_num=2)
        circuit.apply_to(np.array([]), rho)

        assert rho.purity() < 1.0
        assert rho.is_valid()

    def test_local_depolarizing_invalid_target_raises(self):
        """Test local single-qubit depolarizing rejects out-of-range targets."""
        circuit = NoisyCircuit(1)
        circuit.add_local_depolarizing(1, error_rate=0.2)

        rho = DensityMatrix(qbit_num=1)
        with pytest.raises(RuntimeError, match="target_qbit out of range"):
            circuit.apply_to(np.array([]), rho)

    def test_amplitude_damping(self):
        """Test amplitude damping in circuit."""
        circuit = NoisyCircuit(1)
        circuit.add_X(0)  # Start in |1⟩
        circuit.add_amplitude_damping(0, gamma=0.5)

        rho = DensityMatrix(qbit_num=1)
        circuit.apply_to(np.array([]), rho)

        rho_np = rho.to_numpy()
        # Should have some population in |0⟩
        assert rho_np[0, 0].real > 0
        assert rho_np[1, 1].real < 1
        assert np.isclose(rho.trace().real, 1.0)

    def test_phase_damping(self):
        """Test phase damping in circuit."""
        circuit = NoisyCircuit(1)
        circuit.add_H(0)  # Create |+⟩ state with coherence
        circuit.add_phase_damping(0, lambda_param=0.5)

        rho = DensityMatrix(qbit_num=1)
        circuit.apply_to(np.array([]), rho)

        rho_np = rho.to_numpy()
        # Off-diagonal elements should be reduced
        assert np.abs(rho_np[0, 1]) < 0.5
        # Diagonal should be unchanged
        assert np.isclose(rho_np[0, 0].real, 0.5)
        assert np.isclose(rho_np[1, 1].real, 0.5)


@pytest.mark.skipif(not HAS_MODULE, reason="Density matrix module not built")
class TestNoisyCircuitMixed:
    """Test combined circuit operations."""

    def test_mixed_operations(self):
        """Test mixed gates and noise in circuit."""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_amplitude_damping(0, gamma=0.1)
        circuit.add_CNOT(1, 0)
        circuit.add_phase_damping(1, lambda_param=0.05)
        circuit.add_depolarizing(2, error_rate=0.02)

        assert circuit.parameter_num == 0

        rho = DensityMatrix(qbit_num=2)
        circuit.apply_to(np.array([]), rho)

        assert rho.is_valid()
        assert rho.purity() < 1.0

    def test_operation_info(self):
        """Test operation info from circuit."""
        circuit = NoisyCircuit(2)
        circuit.add_H(0)
        circuit.add_CNOT(1, 0)
        circuit.add_depolarizing(2, error_rate=0.1)
        circuit.add_phase_damping(0)  # Parametric

        info = circuit.get_operation_info()

        assert len(info) == 4
        assert info[0].name == "H"
        assert info[0].is_unitary is True
        assert info[0].param_count == 0

        assert info[1].name == "CNOT"
        assert info[1].is_unitary is True

        assert info[2].name == "Depolarizing"
        assert info[2].is_unitary is False
        assert info[2].param_count == 0

        assert info[3].name == "PhaseDamping"
        assert info[3].is_unitary is False
        assert info[3].param_count == 1

    def test_u3_cnot_operation_info(self):
        """Test HEA-relevant U3/CNOT lowering metadata."""
        circuit = NoisyCircuit(2)
        circuit.add_U3(0)
        circuit.add_CNOT(1, 0)
        circuit.add_local_depolarizing(0, error_rate=0.05)
        circuit.add_phase_damping(1, lambda_param=0.02)

        info = circuit.get_operation_info()

        assert len(info) == 4
        assert info[0].name == "U3"
        assert info[0].param_count == 3
        assert info[0].param_start == 0

        assert info[1].name == "CNOT"
        assert info[1].param_count == 0
        assert info[1].param_start == 3

        assert info[2].name == "LocalDepolarizing"
        assert info[2].param_count == 0
        assert info[2].param_start == 3

        assert info[3].name == "PhaseDamping"
        assert info[3].param_count == 0
        assert info[3].param_start == 3

    def test_story2_density_energy_helper(self):
        """Test Story 2 exact-energy helper on a deterministic tiny fixture."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validate_squander_vs_qiskit import density_energy

        hamiltonian = np.array(
            [[0.6, 0.2 + 0.1j], [0.2 - 0.1j, -0.4]],
            dtype=complex,
        )
        density_matrix = np.array(
            [[0.7, 0.1 - 0.05j], [0.1 + 0.05j, 0.3]],
            dtype=complex,
        )

        energy_real, energy_imag = density_energy(hamiltonian, density_matrix)
        expected = np.trace(hamiltonian @ density_matrix)

        assert np.isclose(energy_real, np.real(expected), atol=1e-12)
        assert np.isclose(energy_imag, np.imag(expected), atol=1e-12)

    def test_story2_representative_microcase_passes(self):
        """Test one mandatory Story 2 microcase end-to-end."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.circuits import STORY2_MANDATORY_MICROCASES
        from benchmarks.density_matrix.validate_squander_vs_qiskit import (
            validate_story2_case,
        )

        case = next(
            case
            for case in STORY2_MANDATORY_MICROCASES
            if case["case_name"] == "story2_2q_u3_cnot_local_depolarizing"
        )
        result = validate_story2_case(case, verbose=False)

        assert result["status"] == "pass"
        assert result["energy_pass"]
        assert result["density_valid_pass"]
        assert result["trace_pass"]
        assert result["observable_pass"]
        assert result["qbit_num"] == 2
        assert len(result["parameter_vector"]) == 6

    def test_story2_mixed_required_noise_case_passes(self):
        """Test the mandatory mixed required-noise Story 2 microcase."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.circuits import STORY2_MANDATORY_MICROCASES
        from benchmarks.density_matrix.validate_squander_vs_qiskit import (
            validate_story2_case,
        )

        case = next(
            case
            for case in STORY2_MANDATORY_MICROCASES
            if case["case_name"] == "story2_3q_u3_cnot_mixed_local_noise"
        )
        result = validate_story2_case(case, verbose=False)

        assert result["status"] == "pass"
        assert result["energy_pass"]
        assert result["density_valid_pass"]
        assert result["trace_pass"]
        assert result["observable_pass"]
        assert result["required_gate_coverage_pass"]
        assert result["required_noise_model_coverage_pass"]
        assert result["noise_sequence_match_pass"]
        assert result["mixed_sequence_order_pass"]
        assert result["operation_audit_pass"]
        assert result["noise_operation_sequence"] == [
            "local_depolarizing",
            "amplitude_damping",
            "phase_damping",
        ]
        assert result["noise_operation_targets"] == [0, 1, 2]

    def test_task4_story2_required_local_noise_bundle_schema(self):
        """Test the Task 4 Story 2 exact micro-validation bundle schema."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.task4_story2_required_local_noise_micro_validation import (
            REQUIRED_LOCAL_NOISE_MODELS,
            build_artifact_bundle,
            run_story2_validation,
        )

        results = run_story2_validation(verbose=False)
        bundle = build_artifact_bundle(results)

        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["total_cases"] == 7
        assert bundle["summary"]["passed_cases"] == 7
        assert bundle["summary"]["pass_rate"] == 1.0
        assert bundle["summary"]["required_cases"] == 7
        assert bundle["summary"]["required_passed_cases"] == 7
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["optional_cases"] == 0
        assert bundle["summary"]["optional_cases_count_toward_mandatory_baseline"] == 0
        assert bundle["summary"]["mandatory_baseline_completed"]
        assert bundle["summary"]["exact_threshold_passed_cases"] == 7
        assert bundle["summary"]["operation_audit_passed_cases"] == 7
        assert bundle["summary"]["mixed_sequence_case_count"] == 1
        assert bundle["summary"]["mixed_sequence_passed_cases"] == 1
        assert set(bundle["requirements"]["required_local_noise_models"]) == set(
            REQUIRED_LOCAL_NOISE_MODELS
        )
        assert "required" in bundle["requirements"]["support_tier_vocabulary"]
        assert set(bundle["summary"]["required_noise_models_covered"]) == set(
            REQUIRED_LOCAL_NOISE_MODELS
        )
        assert all(case["task4_story2_case_pass"] for case in bundle["cases"])
        assert all(case["support_tier"] == "required" for case in bundle["cases"])
        assert all(case["case_purpose"] == "mandatory_baseline" for case in bundle["cases"])
        assert all(case["counts_toward_mandatory_baseline"] for case in bundle["cases"])

    def test_task4_story3_optional_noise_classification_bundle_schema(self):
        """Test the Task 4 Story 3 optional classification bundle schema."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.task4_story3_optional_noise_classification_validation import (
            OPTIONAL_WHOLE_REGISTER_CASES,
            build_artifact_bundle,
            run_validation,
        )

        story1_bundle, story2_bundle, optional_results = run_validation(verbose=False)
        bundle = build_artifact_bundle(story1_bundle, story2_bundle, optional_results)

        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["required_cases"] == 10
        assert bundle["summary"]["required_passed_cases"] == 10
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["optional_cases"] == len(OPTIONAL_WHOLE_REGISTER_CASES)
        assert bundle["summary"]["optional_passed_cases"] == len(OPTIONAL_WHOLE_REGISTER_CASES)
        assert bundle["summary"]["optional_pass_rate"] == 1.0
        assert bundle["summary"]["optional_cases_count_toward_mandatory_baseline"] == 0
        assert bundle["summary"]["mandatory_baseline_completed"]
        assert set(bundle["summary"]["support_tiers_present"]) == {"optional", "required"}
        assert bundle["required_artifacts"]["story1"]["status"] == "pass"
        assert bundle["required_artifacts"]["story2"]["status"] == "pass"
        assert "optional" in bundle["requirements"]["support_tier_vocabulary"]
        assert all(case["support_tier"] == "optional" for case in bundle["cases"])
        assert all(not case["counts_toward_mandatory_baseline"] for case in bundle["cases"])
        assert all(case["whole_register_baseline_classification_pass"] for case in bundle["cases"])
        assert all(case["task4_story3_case_pass"] for case in bundle["cases"])
        assert all(case["noise_operation_sequence"] == ["depolarizing"] for case in bundle["cases"])

    def test_all_gates(self):
        """Test all supported gate types."""
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

        rho = DensityMatrix(qbit_num=2)
        circuit.apply_to(np.array([]), rho)

        # Should complete without error and produce valid state
        assert rho.is_valid()


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

    @pytest.mark.parametrize(
        "channel_factory, rho_factory, validate",
        [
            pytest.param(
                lambda: DepolarizingChannel(2, 0.5),
                lambda: DensityMatrix(qbit_num=2),
                _assert_purity_reduced,
                id="depolarizing",
            ),
            pytest.param(
                lambda: AmplitudeDampingChannel(0, 0.5),
                lambda: DensityMatrix(np.array([0, 1], dtype=complex)),
                _assert_population_transferred,
                id="amplitude_damping",
            ),
            pytest.param(
                lambda: PhaseDampingChannel(0, 0.5),
                lambda: DensityMatrix(
                    np.array([1, 1], dtype=complex) / np.sqrt(2)
                ),
                _assert_coherence_reduced,
                id="phase_damping",
            ),
        ],
    )
    def test_legacy_channel_signatures(
        self, channel_factory, rho_factory, validate
    ):
        """Test legacy channel constructor signatures."""
        rho = rho_factory()
        noise = channel_factory()
        noise.apply(rho)
        validate(rho)


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

    def test_partial_trace_from_numpy(self):
        """Test partial trace from NumPy-constructed density matrix."""
        # Create Bell state manually
        bell = np.array([[0.5, 0, 0, 0.5],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0.5, 0, 0, 0.5]], dtype=complex)
        rho = DensityMatrix.from_numpy(bell)

        # Trace out qubit 1
        rho_reduced = rho.partial_trace([1])

        expected = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        np.testing.assert_allclose(rho_reduced.to_numpy(), expected, atol=1e-10)

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

    def test_u3_cnot_comparison_with_state_vector(self):
        """Test HEA-relevant U3 plus CNOT evolution against state vector."""
        from squander.gates.qgd_Circuit import qgd_Circuit

        sv_circuit = qgd_Circuit(2)
        sv_circuit.add_U3(0)
        sv_circuit.add_CNOT(1, 0)

        dm_circuit = NoisyCircuit(2)
        dm_circuit.add_U3(0)
        dm_circuit.add_CNOT(1, 0)

        params = np.array([0.21, -0.13, 0.37], dtype=np.float64)

        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0
        U = sv_circuit.get_Matrix(params)
        sv_final = U @ sv

        rho = DensityMatrix(qbit_num=2)
        dm_circuit.apply_to(params, rho)

        rho_expected = np.outer(sv_final, sv_final.conj())
        rho_actual = rho.to_numpy()

        np.testing.assert_allclose(rho_actual, rho_expected, atol=1e-10)


@pytest.mark.skipif(not HAS_MODULE, reason="Density matrix module not built")
@pytest.mark.slow
class TestPerformance:
    """Performance-related tests."""

    def test_large_circuit(self):
        """Test moderately large circuit."""
        n_qubits = 4
        circuit = NoisyCircuit(n_qubits)

        # Add 20 operations
        for _ in range(5):
            circuit.add_H(0)
            circuit.add_CNOT(1, 0)
            circuit.add_CNOT(2, 1)
            circuit.add_CNOT(3, 2)

        rho = DensityMatrix(qbit_num=n_qubits)
        circuit.apply_to(np.array([]), rho)

        assert rho.is_valid()

    def test_repeated_noise(self):
        """Test repeated noise application."""
        circuit = NoisyCircuit(2)

        # Interleave gates with noise (10% per step for more noticeable effect)
        for _ in range(10):
            circuit.add_H(0)
            circuit.add_depolarizing(2, error_rate=0.1)

        rho = DensityMatrix(qbit_num=2)
        circuit.apply_to(np.array([]), rho)

        # Purity should decrease (10 applications of 10% noise)
        # Not necessarily below 0.5, but significantly reduced
        assert rho.purity() < 0.9
        assert rho.is_valid()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
