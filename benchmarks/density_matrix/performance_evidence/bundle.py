from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from benchmarks.density_matrix.correctness_evidence.bundle import (
    build_correctness_evidence_correctness_package_payload,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION,
    build_performance_evidence_boundary_evidence,
    build_performance_evidence_selected_candidate,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_benchmark_records,
)
from benchmarks.density_matrix.planner_calibration.bundle import (
    build_planner_calibration_calibration_bundle_payload,
)


@lru_cache(maxsize=1)
def _build_performance_evidence_benchmark_package_payload_cached() -> dict:
    planner_calibration_bundle = build_planner_calibration_calibration_bundle_payload()
    correctness_evidence_bundle = build_correctness_evidence_correctness_package_payload()
    cases = build_performance_evidence_benchmark_records()
    negative_cases = list(build_performance_evidence_boundary_evidence())
    return {
        "schema_version": PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION,
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "cases": cases,
        "negative_cases": negative_cases,
        "required_artifacts": [
            {
                "artifact_id": "planner_calibration_calibration_bundle",
                "schema_version": planner_calibration_bundle["schema_version"],
                "selected_candidate_id": planner_calibration_bundle["selected_candidate"]["candidate_id"],
            },
            {
                "artifact_id": "correctness_evidence_correctness_package",
                "schema_version": correctness_evidence_bundle["schema_version"],
                "counted_supported_cases": correctness_evidence_bundle["summary"]["counted_supported_cases"],
            },
        ],
        "summary": {
            "total_cases": len(cases),
            "counted_supported_cases": sum(
                case["counted_supported_benchmark_case"] for case in cases
            ),
            "diagnosis_only_cases": sum(case["diagnosis_only_case"] for case in cases),
            "excluded_cases": sum(case["benchmark_status"] == "excluded" for case in cases),
            "representative_review_cases": sum(
                case["representative_review_case"] for case in cases
            ),
            "positive_threshold_pass_cases": sum(
                case["positive_threshold_pass"] for case in cases
            ),
            "correctness_evidence_reference_available": sum(
                case["correctness_evidence_reference_available"] for case in cases
            ),
            "correctness_evidence_boundary_cases": len(negative_cases),
        },
    }


def build_performance_evidence_benchmark_package_payload() -> dict:
    return deepcopy(_build_performance_evidence_benchmark_package_payload_cached())
