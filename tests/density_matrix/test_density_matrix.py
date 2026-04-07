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

    def test_density_energy_helper_matches_trace_estimate(self):
        """Test the exact-energy helper on a deterministic tiny fixture."""
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

    def test_representative_microcase_passes(self):
        """Test one mandatory micro-validation case end-to-end."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.circuits import MANDATORY_MICROCASES
        from benchmarks.density_matrix.validate_squander_vs_qiskit import (
            validate_microcase,
        )

        case = next(
            case
            for case in MANDATORY_MICROCASES
            if case["case_name"] == "micro_validation_2q_u3_cnot_local_depolarizing"
        )
        result = validate_microcase(case, verbose=False)

        assert result["status"] == "pass"
        assert result["energy_pass"]
        assert result["density_valid_pass"]
        assert result["trace_pass"]
        assert result["observable_pass"]
        assert result["qbit_num"] == 2
        assert len(result["parameter_vector"]) == 6

    def test_mixed_required_noise_microcase_passes(self):
        """Test the mandatory mixed required-noise microcase."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.circuits import MANDATORY_MICROCASES
        from benchmarks.density_matrix.validate_squander_vs_qiskit import (
            validate_microcase,
        )

        case = next(
            case
            for case in MANDATORY_MICROCASES
            if case["case_name"] == "micro_validation_3q_u3_cnot_mixed_local_noise"
        )
        result = validate_microcase(case, verbose=False)

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

    def test_required_local_noise_micro_bundle_schema(self):
        """Test the required local-noise micro-validation bundle schema."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.noise_support.required_local_noise_micro_validation import (
            REQUIRED_LOCAL_NOISE_MODELS,
            build_artifact_bundle,
            run_required_local_noise_micro_validation,
        )

        results = run_required_local_noise_micro_validation(verbose=False)
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
        assert all(case["required_local_noise_microcase_pass"] for case in bundle["cases"])
        assert all(case["support_tier"] == "required" for case in bundle["cases"])
        assert all(case["case_purpose"] == "mandatory_baseline" for case in bundle["cases"])
        assert all(case["counts_toward_mandatory_baseline"] for case in bundle["cases"])

    def test_local_correctness_bundle_schema(self):
        """Test the local-correctness bundle schema."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.validation_evidence.local_correctness_validation import (
            MANDATORY_CASE_NAMES,
            run_validation,
        )

        (
            micro_validation_bundle,
            required_local_noise_micro_validation_bundle,
            bundle,
        ) = run_validation(verbose=False)

        assert micro_validation_bundle["status"] == "pass"
        assert required_local_noise_micro_validation_bundle["status"] == "pass"
        assert bundle["status"] == "pass"
        assert bundle["backend"] == "density_matrix"
        assert bundle["summary"]["total_cases"] == len(MANDATORY_CASE_NAMES)
        assert bundle["summary"]["passed_cases"] == len(MANDATORY_CASE_NAMES)
        assert bundle["summary"]["required_cases"] == len(MANDATORY_CASE_NAMES)
        assert bundle["summary"]["required_passed_cases"] == len(MANDATORY_CASE_NAMES)
        assert bundle["summary"]["required_pass_rate"] == 1.0
        assert bundle["summary"]["mandatory_baseline_completed"] is True
        assert bundle["summary"]["exact_threshold_passed_cases"] == len(
            MANDATORY_CASE_NAMES
        )
        assert bundle["summary"]["operation_audit_passed_cases"] == len(
            MANDATORY_CASE_NAMES
        )
        assert bundle["summary"]["stable_case_ids_present"] is True
        assert bundle["summary"]["missing_mandatory_case_names"] == []
        assert bundle["summary"]["duplicate_case_names"] == []
        assert bundle["summary"]["unexpected_case_names"] == []
        assert bundle["summary"]["all_cases_required"] is True
        assert bundle["summary"]["all_cases_count_toward_mandatory_baseline"] is True
        assert bundle["summary"]["local_correctness_gate_completed"] is True
        assert set(bundle["requirements"]["mandatory_case_names"]) == set(
            MANDATORY_CASE_NAMES
        )
        assert bundle["required_artifacts"]["micro_validation_reference"]["status"] == "pass"
        assert bundle["required_artifacts"]["required_local_noise_micro_validation"]["status"] == "pass"
        assert all(case["support_tier"] == "required" for case in bundle["cases"])
        assert all(case["counts_toward_mandatory_baseline"] for case in bundle["cases"])

    def test_missing_case_blocks_local_correctness_closure(self):
        """Test that missing mandatory microcases fail the local-correctness gate."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        import copy

        from benchmarks.density_matrix.validation_evidence.local_correctness_validation import (
            build_artifact_bundle,
            run_validation,
        )

        (
            micro_validation_bundle,
            required_local_noise_micro_validation_bundle,
            _,
        ) = run_validation(verbose=False)
        broken_required_local_noise_micro_validation_bundle = copy.deepcopy(
            required_local_noise_micro_validation_bundle
        )
        omitted_case = broken_required_local_noise_micro_validation_bundle["cases"].pop()

        bundle = build_artifact_bundle(
            micro_validation_bundle,
            broken_required_local_noise_micro_validation_bundle,
        )

        assert bundle["status"] == "fail"
        assert bundle["summary"]["stable_case_ids_present"] is False
        assert bundle["summary"]["local_correctness_gate_completed"] is False
        assert omitted_case["case_name"] in bundle["summary"]["missing_mandatory_case_names"]

    def test_optional_noise_classification_bundle_schema(self):
        """Test the optional-noise classification bundle schema."""
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

        from benchmarks.density_matrix.noise_support.optional_noise_classification_validation import (
            OPTIONAL_WHOLE_REGISTER_CASES,
            build_artifact_bundle,
            run_validation,
        )

        (
            required_local_noise_bundle,
            required_local_noise_micro_bundle,
            optional_results,
        ) = run_validation(verbose=False)
        bundle = build_artifact_bundle(
            required_local_noise_bundle,
            required_local_noise_micro_bundle,
            optional_results,
        )

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
        assert bundle["required_artifacts"]["required_local_noise_bundle"]["status"] == "pass"
        assert (
            bundle["required_artifacts"]["required_local_noise_micro_bundle"]["status"]
            == "pass"
        )
        assert "optional" in bundle["requirements"]["support_tier_vocabulary"]
        assert all(case["support_tier"] == "optional" for case in bundle["cases"])
        assert all(not case["counts_toward_mandatory_baseline"] for case in bundle["cases"])
        assert all(case["whole_register_baseline_classification_pass"] for case in bundle["cases"])
        assert all(case["optional_noise_case_pass"] for case in bundle["cases"])
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


def test_workflow_contract_validation_schema_module_level():
    """Test the workflow-contract bundle schema."""
    from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
        CONTRACT_VERSION,
        STATUS_VOCABULARY,
        THRESHOLD_CORE_FIELDS,
        WORKFLOW_ID,
        run_validation,
    )

    reference_bundle, artifact = run_validation(verbose=False)

    assert reference_bundle["status"] == "pass"
    assert artifact["status"] == "pass"
    assert artifact["workflow_id"] == WORKFLOW_ID
    assert artifact["contract_version"] == CONTRACT_VERSION
    assert artifact["backend"] == "density_matrix"
    assert artifact["reference_backend"] == "qiskit_aer_density_matrix"
    assert artifact["requirements"]["workflow_id"] == WORKFLOW_ID
    assert artifact["requirements"]["contract_version"] == CONTRACT_VERSION
    assert artifact["requirements"]["end_to_end_qubits"] == [4, 6]
    assert artifact["requirements"]["fixed_parameter_matrix_qubits"] == [4, 6, 8, 10]
    assert artifact["requirements"]["documented_anchor_qubit"] == 10
    assert artifact["requirements"]["fixed_parameter_sets_per_size"] == 10
    assert artifact["requirements"]["required_threshold_fields"] == list(
        THRESHOLD_CORE_FIELDS
    )
    assert artifact["input_contract"]["workflow_family"] == "noisy_vqe_ground_state_estimation"
    assert artifact["input_contract"]["ansatz"]["family"] == "HEA"
    assert artifact["input_contract"]["backend_selection"]["selected_backend"] == "density_matrix"
    assert artifact["input_contract"]["backend_selection"]["silent_fallback_allowed"] is False
    assert (
        artifact["input_contract"]["seed_policy"]["bounded_optimization_trace"][
            "random_seed_required"
        ]
        is False
    )
    assert artifact["output_contract"]["case_status_vocabulary"] == list(STATUS_VOCABULARY)
    assert set(artifact["output_contract"]["aggregate_status_semantics"].keys()) == {
        "pass",
        "fail",
    }
    assert "required_unsupported_case_fields" in artifact["output_contract"]
    assert artifact["thresholds"]["absolute_energy_error"] == pytest.approx(1e-8)
    assert artifact["thresholds"]["required_end_to_end_qubits"] == [4, 6]
    assert artifact["thresholds"]["required_workflow_qubits"] == [4, 6, 8, 10]
    assert set(artifact["boundary_classification"].keys()) == {
        "supported",
        "optional",
        "deferred",
        "unsupported",
    }
    assert artifact["summary"]["mandatory_reference_artifact_count"] == 6
    assert artifact["summary"]["reference_artifact_count"] == 6
    assert artifact["summary"]["contract_sections_complete"] is True
    assert artifact["summary"]["required_threshold_field_count"] == len(
        THRESHOLD_CORE_FIELDS
    )
    assert any(
        entry["artifact_id"] == "workflow_baseline_bundle"
        for entry in artifact["reference_artifacts"]
    )


def test_workflow_contract_missing_boundary_classification_fails_validation_module_level():
    """Test that missing boundary classes fail workflow-contract validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    _, artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["boundary_classification"].pop("unsupported")

    with pytest.raises(ValueError, match="boundary classes mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_workflow_contract_missing_workflow_id_fails_validation_module_level():
    """Test that missing workflow identity fails workflow-contract validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    _, artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["workflow_id"] = ""

    with pytest.raises(ValueError, match="workflow_id must be non-empty"):
        validate_artifact_bundle(broken_artifact)


def test_workflow_contract_missing_threshold_field_fails_validation_module_level():
    """Test that missing threshold fields fail workflow-contract validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    _, artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["thresholds"].pop("required_workflow_qubits")

    with pytest.raises(ValueError, match="thresholds are missing required field"):
        validate_artifact_bundle(broken_artifact)


def test_end_to_end_trace_bundle_schema_module_level():
    """Test the end-to-end trace bundle schema."""
    from benchmarks.density_matrix.workflow_evidence.end_to_end_trace_validation import (
        MANDATORY_END_TO_END_CASE_NAMES,
        TRACE_CASE_NAME,
        WORKFLOW_ID,
        run_validation,
    )

    workflow_contract, workflow_bundle, trace_artifact, bundle = run_validation(
        verbose=False
    )

    assert workflow_contract["status"] == "pass"
    assert workflow_bundle["status"] == "pass"
    assert trace_artifact["case_name"] == TRACE_CASE_NAME
    assert bundle["status"] == "pass"
    assert bundle["workflow_id"] == WORKFLOW_ID
    assert bundle["summary"]["total_end_to_end_cases"] == len(MANDATORY_END_TO_END_CASE_NAMES)
    assert bundle["summary"]["passed_end_to_end_cases"] == len(MANDATORY_END_TO_END_CASE_NAMES)
    assert bundle["summary"]["stable_case_ids_present"] is True
    assert bundle["summary"]["required_trace_present"] is True
    assert bundle["summary"]["required_trace_completed"] is True
    assert bundle["summary"]["required_trace_bridge_supported"] is True
    assert bundle["summary"]["all_cases_match_contract"] is True
    assert bundle["summary"]["end_to_end_qubits_match_contract"] is True
    assert bundle["summary"]["trace_case_name_matches_contract"] is True
    assert bundle["summary"]["trace_matches_contract"] is True
    assert bundle["summary"]["workflow_thresholds_match_contract"] is True
    assert bundle["summary"]["end_to_end_gate_completed"] is True
    assert bundle["requirements"]["required_trace_case_name"] == TRACE_CASE_NAME
    assert bundle["requirements"]["mandatory_end_to_end_qubits"] == [4, 6]
    assert bundle["requirements"]["workflow_id"] == WORKFLOW_ID
    assert bundle["thresholds"]["required_end_to_end_qubits"] == [4, 6]
    assert bundle["thresholds"]["required_trace_case_name"] == TRACE_CASE_NAME
    assert [case["case_name"] for case in bundle["cases"]] == list(
        MANDATORY_END_TO_END_CASE_NAMES
    )
    assert all(case["required_workflow_case"] for case in bundle["cases"])
    assert bundle["trace_artifact"]["required_workflow_trace"] is True


def test_end_to_end_trace_missing_end_to_end_case_blocks_closure_module_level():
    """Test that missing mandatory end-to-end cases fail the trace gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.end_to_end_trace_validation import (
        VALIDATION_TRACE_ARTIFACT_PATH,
        VALIDATION_WORKFLOW_BASELINE_PATH,
        build_artifact_bundle,
        _load_json,
        _load_workflow_contract,
    )

    workflow_contract = _load_workflow_contract()
    workflow_bundle = copy.deepcopy(_load_json(VALIDATION_WORKFLOW_BASELINE_PATH))
    trace_artifact = _load_json(VALIDATION_TRACE_ARTIFACT_PATH)
    workflow_bundle["cases"] = [
        case
        for case in workflow_bundle["cases"]
        if case["case_name"] != "exact_regime_6q_set_00"
    ]

    bundle = build_artifact_bundle(workflow_contract, workflow_bundle, trace_artifact)

    assert bundle["status"] == "fail"
    assert bundle["summary"]["stable_case_ids_present"] is False
    assert bundle["summary"]["end_to_end_gate_completed"] is False
    assert "exact_regime_6q_set_00" in bundle["summary"]["missing_mandatory_case_names"]


def test_end_to_end_trace_contract_mismatch_blocks_closure_module_level():
    """Test that workflow-contract trace-name drift blocks the trace gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.end_to_end_trace_validation import (
        VALIDATION_TRACE_ARTIFACT_PATH,
        VALIDATION_WORKFLOW_BASELINE_PATH,
        build_artifact_bundle,
        _load_json,
        _load_workflow_contract,
    )

    workflow_contract = copy.deepcopy(_load_workflow_contract())
    workflow_bundle = _load_json(VALIDATION_WORKFLOW_BASELINE_PATH)
    trace_artifact = _load_json(VALIDATION_TRACE_ARTIFACT_PATH)
    workflow_contract["input_contract"]["execution_modes"]["bounded_optimization_trace"][
        "canonical_trace_case_name"
    ] = "unexpected_trace_6q"

    bundle = build_artifact_bundle(workflow_contract, workflow_bundle, trace_artifact)

    assert bundle["status"] == "fail"
    assert bundle["summary"]["required_trace_present"] is False
    assert bundle["summary"]["trace_case_name_matches_contract"] is False
    assert bundle["summary"]["end_to_end_gate_completed"] is False


def test_end_to_end_trace_incomplete_trace_blocks_closure_module_level():
    """Test that incomplete trace evidence fails the trace gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.end_to_end_trace_validation import (
        VALIDATION_TRACE_ARTIFACT_PATH,
        VALIDATION_WORKFLOW_BASELINE_PATH,
        build_artifact_bundle,
        _load_json,
        _load_workflow_contract,
    )

    workflow_contract = _load_workflow_contract()
    workflow_bundle = _load_json(VALIDATION_WORKFLOW_BASELINE_PATH)
    trace_artifact = copy.deepcopy(_load_json(VALIDATION_TRACE_ARTIFACT_PATH))
    trace_artifact["workflow_completed"] = False

    bundle = build_artifact_bundle(workflow_contract, workflow_bundle, trace_artifact)

    assert bundle["status"] == "fail"
    assert bundle["summary"]["required_trace_completed"] is False
    assert bundle["summary"]["end_to_end_gate_completed"] is False


def test_matrix_baseline_bundle_schema_module_level():
    """Test the matrix-baseline bundle schema."""
    from benchmarks.density_matrix.workflow_evidence.matrix_baseline_validation import (
        WORKFLOW_ID,
        run_validation,
    )

    workflow_contract, end_to_end_trace_bundle, workflow_baseline_bundle, bundle = run_validation(
        verbose=False
    )

    assert workflow_contract["status"] == "pass"
    assert end_to_end_trace_bundle["status"] == "pass"
    assert workflow_baseline_bundle["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["workflow_id"] == WORKFLOW_ID
    assert bundle["summary"]["required_cases"] == 40
    assert bundle["summary"]["required_passed_cases"] == 40
    assert bundle["summary"]["required_pass_rate"] == 1.0
    assert bundle["summary"]["stable_case_ids_present"] is True
    assert bundle["summary"]["stable_parameter_set_ids_present"] is True
    assert bundle["summary"]["documented_10q_anchor_present"] is True
    assert bundle["summary"]["workflow_inventory_matches_contract"] is True
    assert bundle["summary"]["workflow_thresholds_match_contract"] is True
    assert bundle["summary"]["all_cases_match_contract"] is True
    assert bundle["summary"]["matrix_gate_completed"] is True
    assert bundle["requirements"]["mandatory_workflow_qubits"] == [4, 6, 8, 10]
    assert bundle["requirements"]["fixed_parameter_sets_per_size"] == 10
    assert bundle["requirements"]["documented_anchor_qubit"] == 10
    assert bundle["thresholds"]["required_workflow_qubits"] == [4, 6, 8, 10]
    assert bundle["thresholds"]["fixed_parameter_sets_per_size"] == 10
    assert bundle["thresholds"]["documented_anchor_qubit"] == 10
    assert bundle["required_artifacts"]["end_to_end_trace_reference"]["status"] == "pass"
    assert len(bundle["cases"]) == 40
    assert all(case["required_matrix_case"] for case in bundle["cases"])


def test_matrix_baseline_missing_matrix_case_blocks_closure_module_level():
    """Test that missing matrix cases fail the matrix baseline gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.matrix_baseline_validation import (
        VALIDATION_WORKFLOW_BASELINE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        build_artifact_bundle,
    )

    workflow_contract = _load_workflow_contract()
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    workflow_baseline_bundle = copy.deepcopy(
        _load_json(VALIDATION_WORKFLOW_BASELINE_PATH)
    )
    workflow_baseline_bundle["cases"] = [
        case
        for case in workflow_baseline_bundle["cases"]
        if case["case_name"] != "exact_regime_10q_set_09"
    ]

    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        workflow_baseline_bundle,
    )

    assert bundle["status"] == "fail"
    assert bundle["summary"]["stable_case_ids_present"] is False
    assert bundle["summary"]["matrix_gate_completed"] is False
    assert "exact_regime_10q_set_09" in bundle["summary"]["missing_mandatory_case_names"]


def test_matrix_baseline_threshold_contract_mismatch_blocks_closure_module_level():
    """Test that workflow-contract threshold drift blocks the matrix gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.matrix_baseline_validation import (
        VALIDATION_WORKFLOW_BASELINE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        build_artifact_bundle,
    )

    workflow_contract = copy.deepcopy(_load_workflow_contract())
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    workflow_baseline_bundle = _load_json(VALIDATION_WORKFLOW_BASELINE_PATH)
    workflow_contract["thresholds"]["absolute_energy_error"] = 1e-7

    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        workflow_baseline_bundle,
    )

    assert bundle["status"] == "fail"
    assert bundle["summary"]["workflow_thresholds_match_contract"] is False
    assert bundle["summary"]["matrix_gate_completed"] is False


def test_matrix_baseline_case_contract_mismatch_fails_validation_module_level():
    """Test that case-level contract mismatch fails matrix-baseline validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.matrix_baseline_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    _, _, _, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["cases"][0]["workflow_id"] = "wrong_workflow_id"

    with pytest.raises(ValueError, match="does not match bundle workflow_id"):
        validate_artifact_bundle(broken_bundle)


def test_unsupported_workflow_bundle_schema_module_level():
    """Test the unsupported-workflow bundle schema."""
    from benchmarks.density_matrix.workflow_evidence.unsupported_workflow_validation import (
        BACKEND_MISMATCH_CASE_NAME,
        run_validation,
    )

    (
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_noise_bundle,
        backend_mismatch_case,
        bundle,
    ) = run_validation(verbose=False)

    assert workflow_contract["status"] == "pass"
    assert end_to_end_trace_bundle["status"] == "pass"
    assert matrix_baseline_bundle["status"] == "pass"
    assert unsupported_noise_bundle["status"] == "pass"
    assert backend_mismatch_case["case_name"] == BACKEND_MISMATCH_CASE_NAME
    assert bundle["status"] == "pass"
    assert bundle["summary"]["unsupported_status_cases"] == bundle["summary"]["total_cases"]
    assert bundle["summary"]["mandatory_baseline_case_count"] == 0
    assert bundle["summary"]["backend_incompatible_case_present"] is True
    assert bundle["summary"]["all_cases_match_contract"] is True
    assert bundle["summary"]["unsupported_gate_completed"] is True
    assert (
        bundle["requirements"]["required_case_fields"]
        == workflow_contract["output_contract"]["required_unsupported_case_fields"]
    )
    assert bundle["thresholds"]["silent_fallback_allowed"] is False
    assert "backend_incompatible_request" in bundle["summary"]["categories_present"]
    assert all(case["required_unsupported_case"] for case in bundle["cases"])


def test_unsupported_workflow_missing_backend_mismatch_case_blocks_closure_module_level():
    """Test that missing backend-mismatch evidence fails the unsupported gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.unsupported_workflow_validation import (
        BACKEND_MISMATCH_CASE_PATH,
        UNSUPPORTED_NOISE_BUNDLE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        _load_matrix_baseline_bundle,
        build_artifact_bundle,
    )

    workflow_contract = _load_workflow_contract()
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    matrix_baseline_bundle = _load_matrix_baseline_bundle()
    unsupported_noise_bundle = _load_json(UNSUPPORTED_NOISE_BUNDLE_PATH)
    backend_mismatch_case = copy.deepcopy(_load_json(BACKEND_MISMATCH_CASE_PATH))
    backend_mismatch_case["case_name"] = "missing_backend_case"

    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_noise_bundle,
        backend_mismatch_case,
    )

    assert bundle["status"] == "fail"
    assert bundle["summary"]["backend_incompatible_case_present"] is False
    assert bundle["summary"]["unsupported_gate_completed"] is False


def test_unsupported_workflow_missing_first_condition_fails_validation_module_level():
    """Test that missing first-unsupported-condition fields fail validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.unsupported_workflow_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["cases"][0].pop("first_unsupported_condition")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact_bundle(broken_bundle)


def test_unsupported_workflow_required_field_mismatch_raises_module_level():
    """Test that workflow-contract field drift is enforced by the unsupported gate."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.unsupported_workflow_validation import (
        BACKEND_MISMATCH_CASE_PATH,
        UNSUPPORTED_NOISE_BUNDLE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        _load_matrix_baseline_bundle,
        build_artifact_bundle,
    )

    workflow_contract = copy.deepcopy(_load_workflow_contract())
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    matrix_baseline_bundle = _load_matrix_baseline_bundle()
    unsupported_noise_bundle = _load_json(UNSUPPORTED_NOISE_BUNDLE_PATH)
    backend_mismatch_case = _load_json(BACKEND_MISMATCH_CASE_PATH)
    workflow_contract["output_contract"]["required_unsupported_case_fields"].append(
        "workflow_contract_only_required_field"
    )

    with pytest.raises(ValueError, match="missing required fields"):
        build_artifact_bundle(
            workflow_contract,
            end_to_end_trace_bundle,
            matrix_baseline_bundle,
            unsupported_noise_bundle,
            backend_mismatch_case,
        )


def test_workflow_interpretation_bundle_schema_module_level():
    """Test the workflow-interpretation bundle schema."""
    from benchmarks.density_matrix.workflow_evidence.workflow_interpretation_validation import (
        WORKFLOW_ID,
        run_validation,
    )

    (
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        optional_noise_bundle,
        bundle,
    ) = run_validation(verbose=False)

    assert workflow_contract["status"] == "pass"
    assert end_to_end_trace_bundle["status"] == "pass"
    assert matrix_baseline_bundle["status"] == "pass"
    assert unsupported_workflow_bundle["status"] == "pass"
    assert optional_noise_bundle["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["workflow_id"] == WORKFLOW_ID
    assert bundle["summary"]["workflow_contract_complete"] is True
    assert bundle["summary"]["end_to_end_gate_complete"] is True
    assert bundle["summary"]["matrix_baseline_gate_complete"] is True
    assert bundle["summary"]["unsupported_workflow_gate_complete"] is True
    assert bundle["summary"]["unsupported_case_field_alignment"] is True
    assert bundle["summary"]["mandatory_artifacts_complete"] is True
    assert bundle["summary"]["optional_evidence_supplemental"] is True
    assert bundle["summary"]["unsupported_evidence_negative_only"] is True
    assert bundle["summary"]["main_workflow_claim_completed"] is True
    assert bundle["summary"]["optional_cases_count_toward_mandatory_baseline"] == 0
    assert bundle["required_artifacts"]["unsupported_workflow_reference"]["status"] == "pass"


def test_workflow_interpretation_optional_evidence_counting_toward_mandatory_blocks_closure_module_level():
    """Test that optional evidence cannot count toward the main workflow claim."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_interpretation_validation import (
        OPTIONAL_NOISE_BUNDLE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        _load_matrix_baseline_bundle,
        _load_unsupported_workflow_bundle,
        build_artifact_bundle,
    )

    workflow_contract = _load_workflow_contract()
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    matrix_baseline_bundle = _load_matrix_baseline_bundle()
    unsupported_workflow_bundle = _load_unsupported_workflow_bundle()
    optional_noise_bundle = copy.deepcopy(_load_json(OPTIONAL_NOISE_BUNDLE_PATH))
    optional_noise_bundle["summary"]["optional_cases_count_toward_mandatory_baseline"] = 1

    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        optional_noise_bundle,
    )

    assert bundle["status"] == "fail"
    assert bundle["summary"]["optional_evidence_supplemental"] is False
    assert bundle["summary"]["main_workflow_claim_completed"] is False


def test_workflow_interpretation_incomplete_mandatory_artifact_blocks_closure_module_level():
    """Test that incomplete mandatory artifacts block claim closure."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_interpretation_validation import (
        OPTIONAL_NOISE_BUNDLE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        _load_matrix_baseline_bundle,
        _load_unsupported_workflow_bundle,
        build_artifact_bundle,
    )

    workflow_contract = _load_workflow_contract()
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    matrix_baseline_bundle = copy.deepcopy(_load_matrix_baseline_bundle())
    unsupported_workflow_bundle = _load_unsupported_workflow_bundle()
    optional_noise_bundle = _load_json(OPTIONAL_NOISE_BUNDLE_PATH)
    matrix_baseline_bundle["status"] = "fail"

    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        optional_noise_bundle,
    )

    assert bundle["status"] == "fail"
    assert "matrix_baseline_reference" in bundle["summary"]["incomplete_mandatory_artifacts"]
    assert bundle["summary"]["mandatory_artifacts_complete"] is False
    assert bundle["summary"]["main_workflow_claim_completed"] is False


def test_workflow_interpretation_field_alignment_required_for_main_claim_module_level():
    """Test that unsupported-field drift blocks the main workflow claim."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_interpretation_validation import (
        OPTIONAL_NOISE_BUNDLE_PATH,
        _load_json,
        _load_workflow_contract,
        _load_end_to_end_trace_bundle,
        _load_matrix_baseline_bundle,
        _load_unsupported_workflow_bundle,
        build_artifact_bundle,
    )

    workflow_contract = _load_workflow_contract()
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle()
    matrix_baseline_bundle = _load_matrix_baseline_bundle()
    unsupported_workflow_bundle = copy.deepcopy(_load_unsupported_workflow_bundle())
    optional_noise_bundle = _load_json(OPTIONAL_NOISE_BUNDLE_PATH)
    unsupported_workflow_bundle["requirements"]["required_case_fields"].append(
        "exact_regime_only_field_drift"
    )

    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        optional_noise_bundle,
    )

    assert bundle["status"] == "fail"
    assert bundle["summary"]["unsupported_case_field_alignment"] is False
    assert bundle["summary"]["unsupported_evidence_negative_only"] is False
    assert bundle["summary"]["main_workflow_claim_completed"] is False


def test_workflow_publication_bundle_schema_module_level():
    """Test the workflow-publication bundle schema."""
    from benchmarks.density_matrix.workflow_evidence.workflow_publication_bundle import (
        WORKFLOW_ID,
        run_validation,
    )

    (
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        workflow_interpretation_bundle,
        bundle,
    ) = run_validation(verbose=False)

    assert workflow_contract["status"] == "pass"
    assert end_to_end_trace_bundle["status"] == "pass"
    assert matrix_baseline_bundle["status"] == "pass"
    assert unsupported_workflow_bundle["status"] == "pass"
    assert workflow_interpretation_bundle["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["workflow_id"] == WORKFLOW_ID
    assert bundle["summary"]["mandatory_artifact_count"] == 5
    assert bundle["summary"]["present_artifact_count"] == 5
    assert bundle["summary"]["status_match_count"] == 5
    assert bundle["summary"]["workflow_identity_match_count"] == 5
    assert bundle["summary"]["semantic_match_count"] == 5
    assert len(bundle["artifacts"]) == 5
    assert all(artifact["mandatory"] for artifact in bundle["artifacts"])


def test_workflow_publication_missing_artifact_entry_fails_validation_module_level():
    """Test that missing mandatory artifact entries fail publication validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_publication_bundle import (
        DEFAULT_OUTPUT_DIR,
        run_validation,
        validate_workflow_publication_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["artifacts"] = broken_bundle["artifacts"][:-1]

    with pytest.raises(ValueError, match="missing required artifact IDs"):
        validate_workflow_publication_bundle(broken_bundle, DEFAULT_OUTPUT_DIR)


def test_workflow_publication_mismatched_contract_version_fails_validation_module_level():
    """Test that mismatched contract versions fail publication validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_publication_bundle import (
        DEFAULT_OUTPUT_DIR,
        run_validation,
        validate_workflow_publication_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["artifacts"][0]["summary"]["contract_version"] = "v2"

    with pytest.raises(ValueError, match="mismatched contract_version"):
        validate_workflow_publication_bundle(broken_bundle, DEFAULT_OUTPUT_DIR)


def test_workflow_publication_missing_semantic_flag_fails_validation_module_level():
    """Test that missing semantic closure flags fail publication validation."""
    import copy

    from benchmarks.density_matrix.workflow_evidence.workflow_publication_bundle import (
        DEFAULT_OUTPUT_DIR,
        run_validation,
        validate_workflow_publication_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["artifacts"][1]["summary"]["end_to_end_gate_completed"] = False

    with pytest.raises(ValueError, match="missing required semantic closure flags"):
        validate_workflow_publication_bundle(broken_bundle, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
