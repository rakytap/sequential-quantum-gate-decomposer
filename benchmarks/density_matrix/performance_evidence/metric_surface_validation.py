#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 5 comparable metric surface.

Verifies that counted and diagnosis-only Task 7 benchmark cases share one
comparable metric vocabulary, including auditable repeated-timing fields for the
representative review set.

Run with:
    python benchmarks/density_matrix/performance_evidence/metric_surface_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    build_performance_evidence_selected_candidate,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_benchmark_records,
)

SUITE_NAME = "phase3_performance_evidence_metric_surface"
ARTIFACT_FILENAME = "metric_surface_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("metric_surface")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    return build_performance_evidence_benchmark_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    metric_surface_pass = all(
        case["runtime_ms"] is not None
        and case["peak_rss_kb"] is not None
        and case["planning_time_ms"] is not None
        and case["partition_count"] is not None
        and case["max_partition_span"] is not None
        and case["runtime_path"] is not None
        for case in cases
    ) and all(
        (
            case["representative_review_case"]
            and case["timing_mode"] == "median_3"
            and len(case["sequential_runtime_ms_samples"]) == 3
            and len(case["fused_runtime_ms_samples"]) == 3
        )
        or (not case["representative_review_case"] and case["timing_mode"] == "single_run")
        for case in cases
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if metric_surface_pass else "fail",
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "representative_review_cases": sum(
                case["representative_review_case"] for case in cases
            ),
            "counted_supported_cases": sum(
                case["counted_supported_benchmark_case"] for case in cases
            ),
            "diagnosis_only_cases": sum(case["diagnosis_only_case"] for case in cases),
            "single_run_cases": sum(case["timing_mode"] == "single_run" for case in cases),
            "median_timed_cases": sum(case["timing_mode"] == "median_3" for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 5 bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the Task 7 Story 5 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "median_timed_cases={median_timed_cases}, single_run_cases={single_run_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
