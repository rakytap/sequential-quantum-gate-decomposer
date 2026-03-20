from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

from benchmarks.density_matrix.correctness_evidence.common import (
    TASK6_CASE_SCHEMA_VERSION,
    TASK6_NEGATIVE_RECORD_SCHEMA_VERSION,
    TASK6_REFERENCE_BACKEND_EXTERNAL,
    TASK6_REFERENCE_BACKEND_INTERNAL,
    TASK6_RUNTIME_CLASS_BASELINE,
    build_task6_selected_candidate,
    build_validation_slice,
)
from benchmarks.density_matrix.correctness_evidence.task6_case_selection import (
    build_task6_case_contexts,
)
from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
    build_density_comparison_metrics,
    density_energy,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.planner_calibration.calibration_records import (
    execute_qiskit_density_reference,
)
from benchmarks.density_matrix.planner_surface.unsupported_descriptor_validation import (
    build_cases as build_task2_unsupported_cases,
)
from benchmarks.density_matrix.planner_surface.unsupported_planner_validation import (
    build_cases as build_task1_unsupported_cases,
)
from benchmarks.density_matrix.planner_calibration.claim_selection import (
    TASK5_CLAIM_STATUS_SUPPORTED,
)
from benchmarks.density_matrix.partitioned_runtime.unsupported_runtime_validation import (
    build_cases as build_task3_unsupported_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_FUSED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
)


def _base_case_record(metadata: dict[str, Any], descriptor_set) -> dict[str, Any]:
    selected_candidate = build_task6_selected_candidate()
    external_reference_required = bool(metadata["external_reference_required"])
    return {
        "record_schema_version": TASK6_CASE_SCHEMA_VERSION,
        "candidate_schema_version": selected_candidate["candidate_schema_version"],
        "claim_status": TASK5_CLAIM_STATUS_SUPPORTED,
        "case_name": metadata["case_name"],
        "case_kind": metadata["case_kind"],
        "candidate_id": metadata["candidate_id"],
        "planner_family": metadata["planner_family"],
        "planner_variant": metadata["planner_variant"],
        "planner_settings": dict(metadata["planner_settings"]),
        "max_partition_qubits": metadata["max_partition_qubits"],
        "task5_selected_candidate_id": metadata["task5_selected_candidate_id"],
        "task5_claim_selection_schema_version": metadata[
            "task5_claim_selection_schema_version"
        ],
        "task5_claim_selection_rule": metadata["task5_claim_selection_rule"],
        "requested_mode": descriptor_set.requested_mode,
        "source_type": descriptor_set.source_type,
        "entry_route": descriptor_set.entry_route,
        "workload_family": descriptor_set.workload_family,
        "workload_id": descriptor_set.workload_id,
        "qbit_num": descriptor_set.qbit_num,
        "parameter_count": descriptor_set.parameter_count,
        "family_name": metadata["family_name"],
        "noise_pattern": metadata["noise_pattern"],
        "seed": metadata["seed"],
        "topology": metadata["topology"],
        "validation_slice": build_validation_slice(
            external_reference_required=external_reference_required
        ),
        "external_reference_required": external_reference_required,
        "reference_backend_internal": TASK6_REFERENCE_BACKEND_INTERNAL,
        "reference_backend_external": (
            TASK6_REFERENCE_BACKEND_EXTERNAL if external_reference_required else None
        ),
        "story1_matrix_pass": True,
    }


def _runtime_classification(runtime_result) -> str:
    if runtime_result.actual_fused_execution:
        return PHASE3_FUSION_CLASS_FUSED
    if runtime_result.supported_unfused_region_count > 0:
        return PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
    if runtime_result.deferred_region_count > 0:
        return PHASE3_FUSION_CLASS_DEFERRED
    return TASK6_RUNTIME_CLASS_BASELINE


def build_task6_positive_record(case_context) -> dict[str, Any]:
    record = _base_case_record(case_context.metadata, case_context.descriptor_set)

    runtime_result, reference_density, density_metrics = execute_fused_with_reference(
        case_context.descriptor_set, case_context.parameters
    )
    runtime_payload = runtime_result.to_dict(include_density_matrix=False)

    internal_reference_pass = (
        density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.rho_is_valid
    )
    output_integrity_pass = (
        runtime_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.rho_is_valid
    )

    continuity_energy_required = case_context.hamiltonian is not None
    continuity_energy_error = None
    continuity_energy_pass = True
    if continuity_energy_required:
        runtime_energy_real, _ = density_energy(
            case_context.hamiltonian, runtime_result.density_matrix_numpy()
        )
        reference_energy_real, _ = density_energy(
            case_context.hamiltonian, reference_density.to_numpy()
        )
        continuity_energy_error = float(abs(runtime_energy_real - reference_energy_real))
        continuity_energy_pass = continuity_energy_error <= PHASE3_RUNTIME_ENERGY_TOL

    external_reference_required = bool(record["external_reference_required"])
    external_metrics = None
    external_reference_pass = True
    if external_reference_required:
        aer_density = execute_qiskit_density_reference(
            case_context.descriptor_set, case_context.parameters
        )
        external_metrics = build_density_comparison_metrics(
            runtime_result.density_matrix, aer_density
        )
        external_reference_pass = (
            external_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and external_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        )

    record.update(
        {
            "runtime_schema_version": runtime_payload["runtime_schema_version"],
            "planner_schema_version": runtime_payload["planner_schema_version"],
            "descriptor_schema_version": runtime_payload["descriptor_schema_version"],
            "runtime_path": runtime_payload["runtime_path"],
            "runtime_path_classification": _runtime_classification(runtime_result),
            "supported_runtime_case": (
                runtime_payload["summary"]["exact_output_present"]
                and not runtime_payload["summary"]["fallback_used"]
            ),
            "fallback_used": runtime_payload["summary"]["fallback_used"],
            "exact_output_present": runtime_payload["summary"]["exact_output_present"],
            "runtime_ms": runtime_payload["summary"]["runtime_ms"],
            "peak_rss_kb": runtime_payload["summary"]["peak_rss_kb"],
            "peak_rss_mb": float(runtime_payload["summary"]["peak_rss_kb"]) / 1024.0,
            "partition_count": runtime_payload["summary"]["partition_count"],
            "descriptor_member_count": runtime_payload["summary"][
                "descriptor_member_count"
            ],
            "gate_count": runtime_payload["summary"]["gate_count"],
            "noise_count": runtime_payload["summary"]["noise_count"],
            "max_partition_span": runtime_payload["summary"]["max_partition_span"],
            "partition_member_counts": runtime_payload["summary"][
                "partition_member_counts"
            ],
            "remapped_partition_count": runtime_payload["summary"][
                "remapped_partition_count"
            ],
            "parameter_routing_segment_count": runtime_payload["summary"][
                "parameter_routing_segment_count"
            ],
            "fused_region_count": runtime_payload["summary"]["fused_region_count"],
            "supported_unfused_region_count": runtime_payload["summary"][
                "supported_unfused_region_count"
            ],
            "deferred_region_count": runtime_payload["summary"]["deferred_region_count"],
            "fused_gate_count": runtime_payload["summary"]["fused_gate_count"],
            "supported_unfused_gate_count": runtime_payload["summary"][
                "supported_unfused_gate_count"
            ],
            "actual_fused_execution": runtime_payload["summary"][
                "actual_fused_execution"
            ],
            "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
            "max_abs_diff": density_metrics["max_abs_diff"],
            "internal_reference_pass": internal_reference_pass,
            "trace_deviation": runtime_result.trace_deviation,
            "rho_is_valid": runtime_result.rho_is_valid,
            "rho_is_valid_tol": runtime_payload["summary"]["rho_is_valid_tol"],
            "output_integrity_pass": output_integrity_pass,
            "continuity_energy_required": continuity_energy_required,
            "continuity_energy_error": continuity_energy_error,
            "continuity_energy_pass": continuity_energy_pass,
            "external_frobenius_norm_diff": (
                None if external_metrics is None else external_metrics["frobenius_norm_diff"]
            ),
            "external_max_abs_diff": (
                None if external_metrics is None else external_metrics["max_abs_diff"]
            ),
            "external_reference_pass": external_reference_pass,
            "partitions": runtime_payload["partitions"],
            "fused_regions": runtime_payload["fused_regions"],
            "exact_output": runtime_payload["exact_output"],
        }
    )
    return record


@lru_cache(maxsize=1)
def _build_task6_positive_records_cached() -> tuple[dict[str, Any], ...]:
    return tuple(
        build_task6_positive_record(case_context)
        for case_context in build_task6_case_contexts()
    )


def build_task6_positive_records() -> list[dict[str, Any]]:
    return deepcopy(list(_build_task6_positive_records_cached()))


def _normalize_negative_case(
    case: dict[str, Any], *, boundary_stage: str, origin_suite_name: str
) -> dict[str, Any]:
    return {
        "negative_record_schema_version": TASK6_NEGATIVE_RECORD_SCHEMA_VERSION,
        "origin_suite_name": origin_suite_name,
        "boundary_stage": boundary_stage,
        "case_name": case.get("case_name"),
        "status": case.get("status"),
        "unsupported_category": case.get("unsupported_category"),
        "first_unsupported_condition": case.get("first_unsupported_condition"),
        "failure_stage": case.get("failure_stage"),
        "source_type": case.get("source_type"),
        "requested_mode": case.get("requested_mode"),
        "workload_id": case.get("workload_id"),
        "descriptor_schema_version": case.get("descriptor_schema_version"),
        "runtime_path": case.get("runtime_path"),
        "fallback_used": bool(case.get("fallback_used", False)),
        "supported_case_recorded": bool(
            case.get("supported_partitioned_case_recorded")
            or case.get("supported_descriptor_case_recorded")
            or case.get("supported_runtime_case_recorded")
        ),
        "reason": case.get("reason"),
        "task5_selected_candidate_id": build_task6_selected_candidate()["candidate_id"],
    }


@lru_cache(maxsize=1)
def _build_task6_negative_records_cached() -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    records.extend(
        _normalize_negative_case(
            case,
            boundary_stage="planner_entry",
            origin_suite_name="phase3_task1_story5_unsupported_planner_boundary",
        )
        for case in build_task1_unsupported_cases()
    )
    records.extend(
        _normalize_negative_case(
            case,
            boundary_stage="descriptor_generation",
            origin_suite_name="phase3_task2_story6_unsupported_descriptors",
        )
        for case in build_task2_unsupported_cases()
    )
    records.extend(
        _normalize_negative_case(
            case,
            boundary_stage="runtime_stage",
            origin_suite_name="phase3_task3_story7_unsupported_runtime",
        )
        for case in build_task3_unsupported_cases()
    )
    return tuple(records)


def build_task6_negative_records() -> list[dict[str, Any]]:
    return deepcopy(list(_build_task6_negative_records_cached()))


def task6_counted_supported_case(record: dict[str, Any]) -> bool:
    if not record["supported_runtime_case"]:
        return False
    if not record["internal_reference_pass"]:
        return False
    if not record["output_integrity_pass"]:
        return False
    if record["continuity_energy_required"] and not record["continuity_energy_pass"]:
        return False
    if record["external_reference_required"] and not record["external_reference_pass"]:
        return False
    return True
