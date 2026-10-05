from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
    build_phase3_continuity_partition_descriptor_set,
)
from squander.partitioning import noisy_runtime as noisy_runtime_mod
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    build_runtime_audit_record,
    execute_partitioned_density,
)
from tests.partitioning.fixtures.continuity import build_phase2_continuity_vqe
from tests.partitioning.fixtures.runtime import (
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
    build_initial_parameters,
    density_energy,
    execute_partitioned_with_reference,
)
from tests.partitioning.fixtures.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_microcase_descriptor_set,
    build_structured_descriptor_set,
    iter_microcase_descriptor_sets,
    _noise,
    _noise_value,
    _u3,
)
from benchmarks.density_matrix.correctness_evidence.mf1a_q4_baseline_validation import (
    MF1A_QA001_LAMBDA_MIN_FLOOR,
    MF1A_QA001_MATRIX_TOL,
    evaluate_mf1a_qa001,
)
from squander.density_matrix import DensityMatrix


def test_mf1a_q4_baseline_matches_sequential_under_exact_qa001():
    vqe, _, _ = build_phase2_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(
        vqe, max_partition_qubits=2
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(
        descriptor_set, parameters, allow_fusion=False
    )
    reference = noisy_runtime_mod.execute_sequential_density_reference(
        descriptor_set, parameters
    )
    qa001 = evaluate_mf1a_qa001(result.density_matrix, reference)

    assert result.workload_id == "phase2_xxz_hea_q4_continuity"
    assert result.requested_runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.partition_count > 1
    assert result.actual_fused_execution is False
    assert result.exact_output_present is True
    assert qa001["qa001_pass"] is True
    assert qa001["first_failure"] is None


def test_mf1a_qa001_rejects_nonfinite_entry_before_eigensolver():
    candidate = np.eye(2, dtype=np.complex128) / 2
    candidate[0, 0] = np.nan
    called = False

    def eigenvalues(_candidate):
        nonlocal called
        called = True
        return [0.5, 0.5]

    qa001 = evaluate_mf1a_qa001(
        candidate, np.eye(2) / 2, eigenvalues_fn=eigenvalues
    )
    assert called is False
    assert qa001["first_failure"] == "finite_entries"
    assert qa001["qa001_pass"] is False


def test_mf1a_qa001_rejects_frobenius_or_max_abs_excess():
    candidate = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    reference = candidate.copy()
    reference[0, 1] = 2e-10
    qa001 = evaluate_mf1a_qa001(
        candidate, reference, eigenvalues_fn=lambda _: [1.0, 0.0]
    )
    assert qa001["frobenius_norm_diff_pass"] is False
    assert qa001["max_abs_diff_pass"] is False
    assert qa001["first_failure"] == "frobenius_norm_diff"


def test_mf1a_qa001_rejects_trace_excess():
    candidate = np.diag([1.0 + 2e-10, 0.0]).astype(np.complex128)
    qa001 = evaluate_mf1a_qa001(
        candidate, candidate, eigenvalues_fn=lambda _: [1.0, 0.0]
    )
    assert qa001["trace_abs_deviation_pass"] is False
    assert qa001["first_failure"] == "trace_abs_deviation"


def test_mf1a_qa001_rejects_lambda_below_floor():
    candidate = np.diag([1.0, 0.0]).astype(np.complex128)
    qa001 = evaluate_mf1a_qa001(
        candidate,
        candidate,
        eigenvalues_fn=lambda _: [1.0, MF1A_QA001_LAMBDA_MIN_FLOOR - 1e-15],
    )
    assert qa001["lambda_min_pass"] is False
    assert qa001["first_failure"] == "lambda_min"


def test_mf1a_qa001_rejects_eigensolver_failure():
    candidate = np.diag([1.0, 0.0]).astype(np.complex128)

    def fail(_candidate):
        raise RuntimeError("zheev failed")

    qa001 = evaluate_mf1a_qa001(candidate, candidate, eigenvalues_fn=fail)
    assert qa001["eigensolver_pass"] is False
    assert qa001["first_failure"] == "eigensolver"


def test_mf1a_qa001_rejects_nonfinite_eigenvalue_or_reported_residual():
    candidate = np.diag([1.0, 0.0]).astype(np.complex128)
    qa001 = evaluate_mf1a_qa001(
        candidate, candidate, eigenvalues_fn=lambda _: [1.0, np.nan]
    )
    assert qa001["finite_eigenvalues_pass"] is False
    assert qa001["first_failure"] == "finite_eigenvalues"

    qa001 = evaluate_mf1a_qa001(
        candidate,
        np.full((2, 2), np.inf, dtype=np.complex128),
        eigenvalues_fn=lambda _: [1.0, 0.0],
    )
    assert qa001["finite_residuals_pass"] is False


def test_mf1a_qa001_accepts_values_exactly_on_each_inclusive_threshold():
    candidate = np.diag(
        [1.0 + MF1A_QA001_MATRIX_TOL, -MF1A_QA001_MATRIX_TOL]
    ).astype(np.complex128)
    reference = candidate.copy()
    reference[0, 1] = MF1A_QA001_MATRIX_TOL
    qa001 = evaluate_mf1a_qa001(
        candidate,
        reference,
        eigenvalues_fn=lambda _: [1.0, MF1A_QA001_LAMBDA_MIN_FLOOR],
    )
    assert qa001["qa001_pass"] is True


def test_mf1a_qa001_reports_first_failure_in_frozen_evaluation_order():
    candidate = np.diag([2.0, 0.0]).astype(np.complex128)
    reference = np.zeros((2, 2), dtype=np.complex128)
    qa001 = evaluate_mf1a_qa001(
        candidate, reference, eigenvalues_fn=lambda _: [2.0, -1.0]
    )
    assert qa001["first_failure"] == "frobenius_norm_diff"


def test_mf1a_qa001_uses_existing_eigenvalues_contract_on_asymmetric_input():
    candidate = np.array([[0.5, 0.4], [0.0, 0.5]], dtype=np.complex128)
    qa001 = evaluate_mf1a_qa001(candidate, candidate)
    expected = min(DensityMatrix.from_numpy(candidate).eigenvalues())
    assert qa001["lambda_min"] == pytest.approx(expected)


@pytest.mark.parametrize("qbit_num", [4, 6])
def test_phase3_partitioned_runtime_continuity_runtime_executes_supported_anchor(qbit_num):
    vqe, _, _ = build_phase2_continuity_vqe(qbit_num)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    result = execute_partitioned_density(descriptor_set, parameters)

    assert result.requested_mode == PARTITIONED_DENSITY_MODE
    assert result.source_type == "generated_hea"
    assert result.workload_id == f"phase2_xxz_hea_q{qbit_num}_continuity"
    assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.requested_runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.partition_count > 0
    assert result.exact_output_present is True
    assert result.rho_is_valid is True


@pytest.mark.parametrize("qbit_num", [4, 6])
def test_partitioned_runtime_continuity_energy_matches_existing_density_backend(qbit_num):
    vqe, hamiltonian, _ = build_phase2_continuity_vqe(qbit_num)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    result = execute_partitioned_density(descriptor_set, parameters)

    partitioned_energy_real, partitioned_energy_imag = density_energy(
        hamiltonian, result.density_matrix_numpy()
    )
    continuity_energy = float(vqe.Optimization_Problem(parameters))

    assert abs(partitioned_energy_real - continuity_energy) <= PHASE3_RUNTIME_ENERGY_TOL
    assert abs(partitioned_energy_imag) <= PHASE3_RUNTIME_DENSITY_TOL


def test_partitioned_runtime_mandatory_microcases_execute_through_shared_runtime_surface():
    for metadata, descriptor_set in iter_microcase_descriptor_sets():
        parameters = build_initial_parameters(descriptor_set.parameter_count)
        result = execute_partitioned_density(descriptor_set, parameters)

        assert result.requested_mode == PARTITIONED_DENSITY_MODE
        assert result.source_type == "microcase_builder"
        assert result.workload_id == metadata["case_name"]
        assert result.partition_count > 0
        assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
        assert result.requested_runtime_path == PHASE3_RUNTIME_PATH_BASELINE
        assert result.exact_output_present is True


def test_partitioned_runtime_mandatory_structured_case_executes_through_shared_runtime_surface():
    descriptor_set = build_structured_descriptor_set(
        STRUCTURED_FAMILY_NAMES[0],
        qbit_num=STRUCTURED_QUBITS[0],
        noise_pattern=MANDATORY_NOISE_PATTERNS[0],
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(descriptor_set, parameters)

    assert result.source_type == "structured_family_builder"
    assert result.partition_count > 0
    assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.requested_runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.rho_is_valid is True


def test_partitioned_runtime_continuity_runtime_audit_record_tracks_provenance():
    vqe, _, _ = build_phase2_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(descriptor_set, parameters)
    audit = build_runtime_audit_record(result, metadata={"case_kind": "continuity"})

    assert audit["provenance"]["source_type"] == "generated_hea"
    assert audit["summary"]["partition_count"] == result.partition_count
    assert audit["summary"]["descriptor_member_count"] == result.descriptor_member_count
    assert audit["metadata"]["case_kind"] == "continuity"
    assert audit["requested_runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
    assert audit["qbit_num"] == result.qbit_num
    assert audit["parameter_count"] == result.parameter_count


def test_partitioned_runtime_semantics_boundary_microcase_matches_sequential_reference():
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result, _, density_metrics = execute_partitioned_with_reference(
        descriptor_set, parameters
    )

    assert result.partition_count > 0
    assert result.remapped_partition_count > 0
    assert result.parameter_routing_segment_count > 0
    assert density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_partitioned_runtime_semantics_structured_case_matches_sequential_reference():
    descriptor_set = build_structured_descriptor_set(
        STRUCTURED_FAMILY_NAMES[0],
        qbit_num=STRUCTURED_QUBITS[0],
        noise_pattern=MANDATORY_NOISE_PATTERNS[0],
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result, _, density_metrics = execute_partitioned_with_reference(
        descriptor_set, parameters
    )

    assert result.partition_count > 0
    assert result.parameter_routing_segment_count > 0
    assert density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_partitioned_runtime_fusion_requested_path_without_actual_fusion_downgrades_runtime_path():
    """allow_fusion upgrades the request to fused path, but singleton unitary segments never fuse."""
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=2,
        source_type="test",
        workload_id="story3_singleton_unitary_segments",
        operation_specs=[
            _u3(0),
            _noise(
                "local_depolarizing",
                0,
                0,
                _noise_value("local_depolarizing"),
            ),
            _u3(1),
        ],
    )
    descriptor_set = build_partition_descriptor_set(surface)
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(descriptor_set, parameters, allow_fusion=True)

    assert result.requested_runtime_path == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
    assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert not result.actual_fused_execution


def test_runtime_operation_alignment_descriptor_and_segment_policies():
    descriptor_set = build_microcase_descriptor_set("microcase_2q_entangler_local_depolarizing")
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    validated, _ = noisy_runtime_mod.validate_runtime_request(
        descriptor_set, parameters, runtime_path=noisy_runtime_mod.PHASE3_RUNTIME_PATH_BASELINE
    )
    partition = validated.partitions[0]
    rp = noisy_runtime_mod.PHASE3_RUNTIME_PATH_BASELINE

    circuit_full, ordered_full = noisy_runtime_mod._build_runtime_circuit(
        validated,
        partition.members,
        qbit_num=validated.qbit_num,
        runtime_path=rp,
    )
    noisy_runtime_mod._validate_runtime_operation_alignment(
        validated,
        circuit_full,
        ordered_full,
        runtime_path=rp,
        member_sequence_kind="descriptor",
        param_start_policy="from_member_attr",
        param_start_attr="local_param_start",
    )

    segment = partition.members[:3]
    assert all(validated.canonical_operation_for(m).is_unitary for m in segment)
    circuit_seg, ordered_seg = noisy_runtime_mod._build_runtime_circuit(
        validated,
        segment,
        qbit_num=validated.qbit_num,
        runtime_path=rp,
    )
    noisy_runtime_mod._validate_runtime_operation_alignment(
        validated,
        circuit_seg,
        ordered_seg,
        runtime_path=rp,
        member_sequence_kind="segment",
        param_start_policy="segment_accumulated",
    )
