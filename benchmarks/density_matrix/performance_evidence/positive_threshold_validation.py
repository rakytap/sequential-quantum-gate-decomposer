#!/usr/bin/env python3
"""Positive-threshold review surface validation for performance evidence.

Evaluates the bounded representative structured review set against the sequential
baseline and records per-case positive-threshold verdicts without turning the
result into a universal acceleration claim.

Run with:
    python benchmarks/density_matrix/performance_evidence/positive_threshold_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    performance_evidence_output_dir,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_benchmark_records,
)
from benchmarks.density_matrix.performance_evidence.validation_support import (
    assemble_record_schema_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "performance_evidence_positive_threshold"
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


def build_positive_threshold_cases() -> list[dict]:
    return [
        case
        for case in build_performance_evidence_benchmark_records()
        if case["representative_review_case"]
    ]


def build_positive_threshold_bundle(cases: list[dict]) -> dict:
    review_surface_pass = all(
        case["sequential_median_runtime_ms"] is not None
        and case["fused_median_runtime_ms"] is not None
        and case["sequential_median_peak_rss_kb"] is not None
        and case["fused_median_peak_rss_kb"] is not None
        for case in cases
    )
    status = "pass" if review_surface_pass and len(cases) > 0 else "fail"
    summary = {
        "total_cases": len(cases),
        "actual_fused_review_cases": sum(case["actual_fused_execution"] for case in cases),
        "positive_threshold_pass_cases": sum(case["positive_threshold_pass"] for case in cases),
        "diagnosis_candidate_cases": sum(
            (not case["positive_threshold_pass"]) and case["counted_supported_benchmark_case"]
            for case in cases
        ),
        "review_groups": sorted(
            {case["review_group_id"] for case in cases if case["review_group_id"] is not None}
        ),
    }
    bundle = assemble_record_schema_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Positive-threshold bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_positive_threshold_cases,
        build_artifact_bundle=build_positive_threshold_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the positive-threshold bundle into.",
        quiet_report=lambda b: print(
            "positive_threshold_pass_cases={positive_threshold_pass_cases}, diagnosis_candidate_cases={diagnosis_candidate_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
