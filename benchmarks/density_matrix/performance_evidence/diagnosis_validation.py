#!/usr/bin/env python3
"""Diagnosis-path validation for performance evidence.

Verifies that representative cases which do not satisfy the measurable-benefit path
remain benchmark-grounded, carry explicit bottleneck reasons, and preserve follow-on
branch visibility.

Run with:
    python benchmarks/density_matrix/performance_evidence/diagnosis_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    build_performance_evidence_boundary_evidence,
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

SUITE_NAME = "performance_evidence_diagnosis"
ARTIFACT_FILENAME = "diagnosis_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("diagnosis")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_diagnosis_cases() -> list[dict]:
    return [
        case
        for case in build_performance_evidence_benchmark_records()
        if case["diagnosis_only_case"]
    ]


def build_diagnosis_bundle(cases: list[dict]) -> dict:
    diagnosis_surface_pass = all(case["diagnosis_reasons"] for case in cases)
    status = "pass" if diagnosis_surface_pass else "fail"
    summary = {
        "total_cases": len(cases),
        "correctness_evidence_boundary_cases": len(build_performance_evidence_boundary_evidence()),
        "no_real_fused_coverage_cases": sum(
            "no_real_fused_coverage" in case["diagnosis_reasons"] for case in cases
        ),
        "limited_fused_coverage_cases": sum(
            "limited_fused_coverage_due_to_noise_boundaries" in case["diagnosis_reasons"]
            for case in cases
        ),
        "runtime_slower_cases": sum(
            "fused_runtime_slower_than_sequential_reference" in case["diagnosis_reasons"]
            for case in cases
        ),
    }
    bundle = assemble_record_schema_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Diagnosis bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_diagnosis_cases,
        build_artifact_bundle=build_diagnosis_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the diagnosis bundle into.",
        quiet_report=lambda b: print(
            "total_cases={total_cases}, limited_fused_coverage_cases={limited_fused_coverage_cases}, runtime_slower_cases={runtime_slower_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
