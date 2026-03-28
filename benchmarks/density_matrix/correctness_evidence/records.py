from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

from benchmarks.density_matrix.evidence_core import (
    build_runtime_correctness_bridge_fields,
    counted_supported_case,
)
from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_EXTERNAL,
    CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
    CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
    build_selected_candidate,
    build_validation_slice,
)
from benchmarks.density_matrix.correctness_evidence.case_selection import (
    build_correctness_evidence_case_contexts,
)
from benchmarks.density_matrix.partitioned_runtime.common import execute_fused_with_reference
from benchmarks.density_matrix.planner_surface.unsupported_descriptor_validation import (
    SUITE_NAME as UNSUPPORTED_DESCRIPTOR_ORIGIN_SUITE_NAME,
    build_unsupported_descriptor_cases as build_descriptor_unsupported_cases,
)
from benchmarks.density_matrix.planner_surface.unsupported_planner_validation import (
    SUITE_NAME as UNSUPPORTED_PLANNER_ORIGIN_SUITE_NAME,
    build_cases as build_planner_unsupported_cases,
)
from benchmarks.density_matrix.planner_calibration.claim_selection import (
    PLANNER_CALIBRATION_CLAIM_STATUS_SUPPORTED,
)
from benchmarks.density_matrix.partitioned_runtime.unsupported_runtime_validation import (
    SUITE_NAME as UNSUPPORTED_RUNTIME_ORIGIN_SUITE_NAME,
    build_cases as build_runtime_unsupported_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_FUSED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
)


def _base_case_record(metadata: dict[str, Any], descriptor_set) -> dict[str, Any]:
    selected_candidate = build_selected_candidate()
    external_reference_required = bool(metadata["external_reference_required"])
    return {
        "record_schema_version": CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
        "candidate_schema_version": selected_candidate["candidate_schema_version"],
        "claim_status": PLANNER_CALIBRATION_CLAIM_STATUS_SUPPORTED,
        "case_name": metadata["case_name"],
        "case_kind": metadata["case_kind"],
        "candidate_id": metadata["candidate_id"],
        "planner_family": metadata["planner_family"],
        "planner_variant": metadata["planner_variant"],
        "planner_settings": dict(metadata["planner_settings"]),
        "max_partition_qubits": metadata["max_partition_qubits"],
        "planner_calibration_selected_candidate_id": metadata["planner_calibration_selected_candidate_id"],
        "planner_calibration_claim_selection_schema_version": metadata[
            "planner_calibration_claim_selection_schema_version"
        ],
        "planner_calibration_claim_selection_rule": metadata["planner_calibration_claim_selection_rule"],
        "requested_mode": descriptor_set.requested_mode,
        "source_type": descriptor_set.source_type,
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
        "reference_backend_internal": CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
        "reference_backend_external": (
            CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_EXTERNAL if external_reference_required else None
        ),
        "correctness_matrix_pass": True,
    }


def _runtime_classification(runtime_result) -> str:
    if runtime_result.actual_fused_execution:
        return PHASE3_FUSION_CLASS_FUSED
    if runtime_result.supported_unfused_region_count > 0:
        return PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
    if runtime_result.deferred_region_count > 0:
        return PHASE3_FUSION_CLASS_DEFERRED
    return CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE


def build_correctness_evidence_positive_record(case_context) -> dict[str, Any]:
    record = _base_case_record(case_context.metadata, case_context.descriptor_set)

    runtime_result, reference_density, density_metrics = execute_fused_with_reference(
        case_context.descriptor_set, case_context.parameters
    )
    runtime_payload = runtime_result.to_dict(include_density_matrix=False)

    external_reference_required = bool(record["external_reference_required"])
    record.update(
        build_runtime_correctness_bridge_fields(
            case_context,
            runtime_result,
            reference_density,
            density_metrics,
            external_reference_required=external_reference_required,
            runtime_payload=runtime_payload,
        )
    )
    record.update(
        {
            "runtime_path_classification": _runtime_classification(runtime_result),
            "partitions": runtime_payload["partitions"],
            "fused_regions": runtime_payload["fused_regions"],
            "exact_output": runtime_payload["exact_output"],
        }
    )
    return record


@lru_cache(maxsize=1)
def _build_correctness_evidence_positive_records_cached() -> tuple[dict[str, Any], ...]:
    return tuple(
        build_correctness_evidence_positive_record(case_context)
        for case_context in build_correctness_evidence_case_contexts()
    )


def build_correctness_evidence_positive_records() -> list[dict[str, Any]]:
    return deepcopy(list(_build_correctness_evidence_positive_records_cached()))


def _normalize_negative_case(
    case: dict[str, Any], *, boundary_stage: str, origin_suite_name: str
) -> dict[str, Any]:
    return {
        "negative_record_schema_version": CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
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
        "runtime_path": case.get("runtime_path"),
        "supported_case_recorded": bool(
            case.get("supported_partitioned_case_recorded")
            or case.get("supported_descriptor_case_recorded")
            or case.get("supported_runtime_case_recorded")
        ),
        "reason": case.get("reason"),
        "planner_calibration_selected_candidate_id": build_selected_candidate()["candidate_id"],
    }


@lru_cache(maxsize=1)
def _build_correctness_evidence_negative_records_cached() -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    records.extend(
        _normalize_negative_case(
            case,
            boundary_stage="planner_entry",
            origin_suite_name=UNSUPPORTED_PLANNER_ORIGIN_SUITE_NAME,
        )
        for case in build_planner_unsupported_cases()
    )
    records.extend(
        _normalize_negative_case(
            case,
            boundary_stage="descriptor_generation",
            origin_suite_name=UNSUPPORTED_DESCRIPTOR_ORIGIN_SUITE_NAME,
        )
        for case in build_descriptor_unsupported_cases()
    )
    records.extend(
        _normalize_negative_case(
            case,
            boundary_stage="runtime_stage",
            origin_suite_name=UNSUPPORTED_RUNTIME_ORIGIN_SUITE_NAME,
        )
        for case in build_runtime_unsupported_cases()
    )
    return tuple(records)


def build_correctness_evidence_negative_records() -> list[dict[str, Any]]:
    return deepcopy(list(_build_correctness_evidence_negative_records_cached()))


def build_positive_records() -> list[dict[str, Any]]:
    return build_correctness_evidence_positive_records()


def build_negative_records() -> list[dict[str, Any]]:
    return build_correctness_evidence_negative_records()


def correctness_evidence_counted_supported_case(record: dict[str, Any]) -> bool:
    return counted_supported_case(record)
