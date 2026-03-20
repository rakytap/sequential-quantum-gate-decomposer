from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import statistics
from collections import defaultdict

from benchmarks.density_matrix.planner_calibration.calibration_records import (
    build_task5_calibration_records,
)

TASK5_CLAIM_SELECTION_SCHEMA_VERSION = "phase3_task5_claim_selection_v1"
TASK5_CLAIM_SELECTION_RULE = (
    "min_median_density_aware_score_then_min_p90_then_smaller_span_budget"
)
TASK5_CLAIM_STATUS_SUPPORTED = "supported_claim"
TASK5_CLAIM_STATUS_COMPARISON = "comparison_baseline"


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    index = int(0.9 * (len(ordered) - 1))
    return float(ordered[index])


def _build_candidate_summary(candidate_id: str, records: list[dict]) -> dict:
    density_scores = [record["density_aware_score"] for record in records]
    runtime_values = [record["runtime_ms"] for record in records]
    planning_values = [record["planning_time_ms"] for record in records]
    peak_rss_values = [record["peak_rss_mb"] for record in records]
    return {
        "candidate_id": candidate_id,
        "planner_family": records[0]["planner_family"],
        "planner_variant": records[0]["planner_variant"],
        "max_partition_qubits": records[0]["max_partition_qubits"],
        "total_cases": len(records),
        "counted_cases": sum(record["counted_calibration_case"] for record in records),
        "median_density_aware_score": float(statistics.median(density_scores)),
        "p90_density_aware_score": _p90(density_scores),
        "max_density_aware_score": float(max(density_scores)),
        "mean_runtime_ms": float(statistics.mean(runtime_values)),
        "mean_planning_time_ms": float(statistics.mean(planning_values)),
        "mean_peak_rss_mb": float(statistics.mean(peak_rss_values)),
    }


def select_supported_candidate(candidate_summaries: list[dict]) -> dict:
    return min(
        candidate_summaries,
        key=lambda summary: (
            summary["median_density_aware_score"],
            summary["p90_density_aware_score"],
            summary["max_partition_qubits"],
            summary["candidate_id"],
        ),
    )


@lru_cache(maxsize=1)
def _build_task5_claim_selection_payload_cached() -> dict:
    records = build_task5_calibration_records()
    grouped_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[record["candidate_id"]].append(record)

    candidate_summaries = [
        _build_candidate_summary(candidate_id, grouped_records[candidate_id])
        for candidate_id in sorted(grouped_records)
    ]
    selected_summary = select_supported_candidate(candidate_summaries)
    selected_candidate_id = selected_summary["candidate_id"]

    annotated_records = []
    for record in records:
        annotated_record = dict(record)
        annotated_record["claim_selection_schema_version"] = (
            TASK5_CLAIM_SELECTION_SCHEMA_VERSION
        )
        annotated_record["claim_selection_rule"] = TASK5_CLAIM_SELECTION_RULE
        annotated_record["selected_candidate_id"] = selected_candidate_id
        annotated_record["claim_status"] = (
            TASK5_CLAIM_STATUS_SUPPORTED
            if record["candidate_id"] == selected_candidate_id
            else TASK5_CLAIM_STATUS_COMPARISON
        )
        annotated_records.append(annotated_record)

    return {
        "schema_version": TASK5_CLAIM_SELECTION_SCHEMA_VERSION,
        "claim_selection_rule": TASK5_CLAIM_SELECTION_RULE,
        "selected_candidate": selected_summary,
        "candidate_summaries": candidate_summaries,
        "cases": annotated_records,
    }


def build_task5_claim_selection_payload() -> dict:
    return deepcopy(_build_task5_claim_selection_payload_cached())
