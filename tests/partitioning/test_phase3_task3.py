from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
    build_initial_parameters,
    density_energy,
    execute_partitioned_with_reference,
)
from benchmarks.density_matrix.partitioned_runtime.runtime_audit_validation import (
    build_cases as build_story6_cases,
)
from benchmarks.density_matrix.partitioned_runtime.runtime_handoff_validation import (
    build_cases as build_story3_cases,
)
from benchmarks.density_matrix.partitioned_runtime.runtime_output_validation import (
    build_cases as build_story5_cases,
)
from benchmarks.density_matrix.partitioned_runtime.unsupported_runtime_validation import (
    build_cases as build_story7_cases,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_phase3_story1_continuity_vqe,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_story2_microcase_descriptor_set,
    build_story2_structured_descriptor_set,
    iter_story2_microcase_descriptor_sets,
)
from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY,
    PHASE3_ENTRY_ROUTE_MICROCASE,
    PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY,
    PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY,
    PHASE3_WORKLOAD_FAMILY_MICROCASE,
    PHASE3_WORKLOAD_FAMILY_STRUCTURED,
    build_phase3_continuity_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_SCHEMA_VERSION,
    build_runtime_audit_record,
    execute_partitioned_density,
)


@pytest.mark.parametrize("qbit_num", [4, 6])
def test_phase3_task3_story1_continuity_runtime_executes_supported_anchor(qbit_num):
    vqe, _, _ = build_phase3_story1_continuity_vqe(qbit_num)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    result = execute_partitioned_density(descriptor_set, parameters)

    assert result.runtime_schema_version == PHASE3_RUNTIME_SCHEMA_VERSION
    assert result.requested_mode == PARTITIONED_DENSITY_MODE
    assert result.source_type == "generated_hea"
    assert result.entry_route == PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    assert result.workload_family == PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    assert result.workload_id == f"phase2_xxz_hea_q{qbit_num}_continuity"
    assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.partition_count > 0
    assert result.exact_output_present is True
    assert result.fallback_used is False
    assert result.rho_is_valid is True


@pytest.mark.parametrize("qbit_num", [4, 6])
def test_phase3_task3_story1_continuity_energy_matches_existing_density_backend(qbit_num):
    vqe, hamiltonian, _ = build_phase3_story1_continuity_vqe(qbit_num)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    result = execute_partitioned_density(descriptor_set, parameters)

    partitioned_energy_real, partitioned_energy_imag = density_energy(
        hamiltonian, result.density_matrix_numpy()
    )
    continuity_energy = float(vqe.Optimization_Problem(parameters))

    assert abs(partitioned_energy_real - continuity_energy) <= PHASE3_RUNTIME_ENERGY_TOL
    assert abs(partitioned_energy_imag) <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase3_task3_story2_microcases_execute_through_shared_runtime_surface():
    for metadata, descriptor_set in iter_story2_microcase_descriptor_sets():
        parameters = build_initial_parameters(descriptor_set.parameter_count)
        result = execute_partitioned_density(descriptor_set, parameters)

        assert result.runtime_schema_version == PHASE3_RUNTIME_SCHEMA_VERSION
        assert result.requested_mode == PARTITIONED_DENSITY_MODE
        assert result.source_type == "microcase_builder"
        assert result.entry_route == PHASE3_ENTRY_ROUTE_MICROCASE
        assert result.workload_family == PHASE3_WORKLOAD_FAMILY_MICROCASE
        assert result.workload_id == metadata["case_name"]
        assert result.partition_count > 0
        assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
        assert result.exact_output_present is True
        assert result.fallback_used is False


def test_phase3_task3_story2_structured_case_executes_through_shared_runtime_surface():
    descriptor_set = build_story2_structured_descriptor_set(
        STRUCTURED_FAMILY_NAMES[0],
        qbit_num=STRUCTURED_QUBITS[0],
        noise_pattern=MANDATORY_NOISE_PATTERNS[0],
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(descriptor_set, parameters)

    assert result.runtime_schema_version == PHASE3_RUNTIME_SCHEMA_VERSION
    assert result.source_type == "structured_family_builder"
    assert result.entry_route == PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY
    assert result.workload_family == PHASE3_WORKLOAD_FAMILY_STRUCTURED
    assert result.partition_count > 0
    assert result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert result.rho_is_valid is True


def test_phase3_task3_story3_runtime_audit_record_tracks_continuity_provenance():
    vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    result = execute_partitioned_density(descriptor_set, parameters)
    audit = build_runtime_audit_record(result, metadata={"case_kind": "continuity"})

    assert audit["runtime_schema_version"] == PHASE3_RUNTIME_SCHEMA_VERSION
    assert audit["provenance"]["source_type"] == "generated_hea"
    assert audit["provenance"]["entry_route"] == PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    assert audit["provenance"]["workload_family"] == PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    assert audit["summary"]["partition_count"] == result.partition_count
    assert audit["summary"]["descriptor_member_count"] == result.descriptor_member_count
    assert audit["metadata"]["case_kind"] == "continuity"


def test_phase3_task3_story3_runtime_handoff_cases_share_shape():
    cases = build_story3_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(partition_key_sets) == 1


def test_phase3_task3_story4_boundary_microcase_matches_sequential_reference():
    descriptor_set = build_story2_microcase_descriptor_set(
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


def test_phase3_task3_story4_structured_case_matches_sequential_reference():
    descriptor_set = build_story2_structured_descriptor_set(
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


def test_phase3_task3_story5_output_cases_share_exact_output_shape():
    cases = build_story5_cases()
    exact_output_key_sets = {frozenset(case["exact_output"].keys()) for case in cases}

    assert len(cases) == 3
    assert len(exact_output_key_sets) == 1
    for case in cases:
        assert case["result_output_pass"] is True
        assert case["exact_output_present"] is True
        assert "shape" in case["exact_output"]
        assert "trace_real" in case["exact_output"]
        assert "density_real" in case["exact_output"]
        assert "density_imag" in case["exact_output"]


def test_phase3_task3_story6_runtime_audit_cases_share_shape():
    cases = build_story6_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(partition_key_sets) == 1
    assert {case["runtime_path"] for case in cases} == {PHASE3_RUNTIME_PATH_BASELINE}


def test_phase3_task3_story7_negative_matrix_has_expected_categories_and_no_fallback():
    cases = build_story7_cases()
    expected_categories = {
        "wrong_requested_mode": "runtime_request",
        "parameter_count_mismatch": "runtime_request",
        "unsupported_gate_name": "unsupported_runtime_operation",
        "unsupported_noise_name": "unsupported_runtime_operation",
        "gate_fixed_value": "descriptor_to_runtime_mismatch",
    }

    assert {case["case_name"] for case in cases} == set(expected_categories)
    for case in cases:
        assert case["status"] == "unsupported"
        assert case["unsupported_category"] == expected_categories[case["case_name"]]
        assert case["fallback_used"] is False
        assert case["supported_runtime_case_recorded"] is False
