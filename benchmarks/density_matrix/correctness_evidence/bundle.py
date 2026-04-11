from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_PHASE31_PACKAGE_SCHEMA_VERSION,
    CORRECTNESS_PACKAGE_SCHEMA_VERSION,
    build_selected_candidate,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_negative_records,
    build_phase31_correctness_evidence_positive_records,
    build_positive_records,
    counted_supported_case,
)


@lru_cache(maxsize=1)
def _build_correctness_package_payload_cached() -> dict:
    positive_cases = build_positive_records()
    negative_cases = build_negative_records()
    counted_supported_cases = [
        case for case in positive_cases if counted_supported_case(case)
    ]
    package = {
        "schema_version": CORRECTNESS_PACKAGE_SCHEMA_VERSION,
        "selected_candidate": build_selected_candidate(),
        "cases": positive_cases,
        "negative_cases": negative_cases,
        "summary": {
            "total_cases": len(positive_cases),
            "counted_supported_cases": len(counted_supported_cases),
            "excluded_supported_cases": len(positive_cases) - len(counted_supported_cases),
            "external_reference_cases": sum(
                case["external_reference_required"] for case in positive_cases
            ),
            "actual_fused_cases": sum(
                case["actual_fused_execution"] for case in positive_cases
            ),
            "unsupported_boundary_cases": len(negative_cases),
            "planner_entry_boundary_cases": sum(
                case["boundary_stage"] == "planner_entry" for case in negative_cases
            ),
            "descriptor_generation_boundary_cases": sum(
                case["boundary_stage"] == "descriptor_generation"
                for case in negative_cases
            ),
            "runtime_stage_boundary_cases": sum(
                case["boundary_stage"] == "runtime_stage" for case in negative_cases
            ),
        },
    }
    return package


def build_correctness_package_payload() -> dict:
    return deepcopy(_build_correctness_package_payload_cached())


def build_correctness_evidence_correctness_package_payload() -> dict:
    return build_correctness_package_payload()


@lru_cache(maxsize=1)
def _build_phase31_correctness_package_payload_cached() -> dict:
    positive_cases = build_phase31_correctness_evidence_positive_records()
    negative_cases = build_negative_records()
    counted_supported_cases = [
        case for case in positive_cases if counted_supported_case(case)
    ]
    return {
        "schema_version": CORRECTNESS_EVIDENCE_PHASE31_PACKAGE_SCHEMA_VERSION,
        "selected_candidate": build_selected_candidate(),
        "cases": positive_cases,
        "negative_cases": negative_cases,
        "summary": {
            "total_cases": len(positive_cases),
            "counted_supported_cases": len(counted_supported_cases),
            "excluded_supported_cases": len(positive_cases) - len(counted_supported_cases),
            "external_reference_cases": sum(
                case["external_reference_required"] for case in positive_cases
            ),
            "actual_fused_cases": sum(
                case["actual_fused_execution"] for case in positive_cases
            ),
            "unsupported_boundary_cases": len(negative_cases),
            "planner_entry_boundary_cases": sum(
                case["boundary_stage"] == "planner_entry" for case in negative_cases
            ),
            "descriptor_generation_boundary_cases": sum(
                case["boundary_stage"] == "descriptor_generation"
                for case in negative_cases
            ),
            "runtime_stage_boundary_cases": sum(
                case["boundary_stage"] == "runtime_stage" for case in negative_cases
            ),
            "continuity_cases": sum(
                case["case_kind"] == "continuity" for case in positive_cases
            ),
            "microcases": sum(case["case_kind"] == "microcase" for case in positive_cases),
            "structured_cases": sum(
                case["case_kind"] == "structured_family" for case in positive_cases
            ),
        },
    }


def build_phase31_correctness_package_payload() -> dict:
    """Stage-A bounded Phase 3.1 package (six positive cases + shared negative boundary records)."""
    return deepcopy(_build_phase31_correctness_package_payload_cached())
