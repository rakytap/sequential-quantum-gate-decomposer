from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from benchmarks.density_matrix.planner_calibration.claim_selection import (
    TASK5_CLAIM_STATUS_COMPARISON,
    TASK5_CLAIM_STATUS_SUPPORTED,
    build_task5_claim_selection_payload,
)

TASK5_CALIBRATION_BUNDLE_SCHEMA_VERSION = "phase3_task5_calibration_bundle_v1"


@lru_cache(maxsize=1)
def _build_task5_calibration_bundle_payload_cached() -> dict:
    claim_payload = build_task5_claim_selection_payload()
    cases = claim_payload["cases"]
    return {
        "schema_version": TASK5_CALIBRATION_BUNDLE_SCHEMA_VERSION,
        "claim_selection_schema_version": claim_payload["schema_version"],
        "claim_selection_rule": claim_payload["claim_selection_rule"],
        "selected_candidate": claim_payload["selected_candidate"],
        "comparison_candidate_ids": [
            summary["candidate_id"]
            for summary in claim_payload["candidate_summaries"]
            if summary["candidate_id"] != claim_payload["selected_candidate"]["candidate_id"]
        ],
        "candidate_summaries": claim_payload["candidate_summaries"],
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "supported_claim_cases": sum(
                case["claim_status"] == TASK5_CLAIM_STATUS_SUPPORTED for case in cases
            ),
            "comparison_cases": sum(
                case["claim_status"] == TASK5_CLAIM_STATUS_COMPARISON for case in cases
            ),
            "counted_calibration_cases": sum(
                case["counted_calibration_case"] for case in cases
            ),
            "external_reference_cases": sum(
                case["external_reference_required"] for case in cases
            ),
            "continuity_cases": sum(case["case_kind"] == "continuity" for case in cases),
            "microcases": sum(case["case_kind"] == "microcase" for case in cases),
            "structured_cases": sum(
                case["case_kind"] == "structured_family" for case in cases
            ),
        },
    }


def build_task5_calibration_bundle_payload() -> dict:
    return deepcopy(_build_task5_calibration_bundle_payload_cached())
