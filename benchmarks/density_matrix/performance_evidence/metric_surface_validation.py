#!/usr/bin/env python3
"""Comparable metric surface validation for performance evidence.

Verifies that counted and diagnosis-only benchmark cases share one comparable
metric vocabulary, including auditable repeated-timing fields for the
representative review set.

Run with:
    python benchmarks/density_matrix/performance_evidence/metric_surface_validation.py
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

SUITE_NAME = "performance_evidence_metric_surface"
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


def build_metric_surface_cases() -> list[dict]:
    return build_performance_evidence_benchmark_records()


def build_metric_surface_bundle(cases: list[dict]) -> dict:
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
    status = "pass" if metric_surface_pass else "fail"
    summary = {
        "total_cases": len(cases),
        "representative_review_cases": sum(case["representative_review_case"] for case in cases),
        "counted_supported_cases": sum(case["counted_supported_benchmark_case"] for case in cases),
        "diagnosis_only_cases": sum(case["diagnosis_only_case"] for case in cases),
        "single_run_cases": sum(case["timing_mode"] == "single_run" for case in cases),
        "median_timed_cases": sum(case["timing_mode"] == "median_3" for case in cases),
    }
    bundle = assemble_record_schema_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Metric surface bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_metric_surface_cases,
        build_artifact_bundle=build_metric_surface_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the metric surface bundle into.",
        quiet_report=lambda b: print(
            "median_timed_cases={median_timed_cases}, single_run_cases={single_run_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
