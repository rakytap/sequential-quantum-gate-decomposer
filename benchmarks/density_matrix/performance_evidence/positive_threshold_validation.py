#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 3 positive-threshold review surface.

Evaluates the bounded representative structured review set against the
sequential baseline and records per-case positive-threshold verdicts without
turning the result into a universal acceleration claim.

Run with:
    python benchmarks/density_matrix/performance_evidence/positive_threshold_validation.py
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

SUITE_NAME = "phase3_performance_evidence_positive_threshold"
ARTIFACT_FILENAME = "positive_threshold_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("positive_threshold")
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
    return [
        case
        for case in build_performance_evidence_benchmark_records()
        if case["representative_review_case"]
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    review_surface_pass = all(
        case["sequential_median_runtime_ms"] is not None
        and case["fused_median_runtime_ms"] is not None
        and case["sequential_median_peak_rss_kb"] is not None
        and case["fused_median_peak_rss_kb"] is not None
        for case in cases
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if review_surface_pass and len(cases) > 0 else "fail",
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "actual_fused_review_cases": sum(case["actual_fused_execution"] for case in cases),
            "positive_threshold_pass_cases": sum(
                case["positive_threshold_pass"] for case in cases
            ),
            "diagnosis_candidate_cases": sum(
                (not case["positive_threshold_pass"]) and case["counted_supported_benchmark_case"]
                for case in cases
            ),
            "review_groups": sorted(
                {case["review_group_id"] for case in cases if case["review_group_id"] is not None}
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 3 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 7 Story 3 bundle into.",
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
            "positive_threshold_pass_cases={positive_threshold_pass_cases}, diagnosis_candidate_cases={diagnosis_candidate_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
