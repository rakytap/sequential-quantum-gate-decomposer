from pathlib import Path
import sys

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
    assert result.fallback_used is False
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
        assert result.fallback_used is False


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
