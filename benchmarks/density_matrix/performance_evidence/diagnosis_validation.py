#!/usr/bin/env python3
"""Diagnosis-path validation for performance evidence.

Verifies that representative cases which do not satisfy the measurable-benefit path
remain benchmark-grounded, carry explicit bottleneck reasons, and preserve follow-on
branch visibility.

Run with:
    python benchmarks/density_matrix/performance_evidence/diagnosis_validation.py
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
    build_performance_evidence_boundary_evidence,
    build_performance_evidence_selected_candidate,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_benchmark_records,
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
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if diagnosis_surface_pass else "fail",
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "correctness_evidence_boundary_cases": len(build_performance_evidence_boundary_evidence()),
            "no_real_fused_coverage_cases": sum(
                "no_real_fused_coverage" in case["diagnosis_reasons"] for case in cases
            ),
            "limited_fused_coverage_cases": sum(
                "limited_fused_coverage_due_to_noise_boundaries"
                in case["diagnosis_reasons"]
                for case in cases
            ),
            "runtime_slower_cases": sum(
                "fused_runtime_slower_than_sequential_reference"
                in case["diagnosis_reasons"]
                for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Diagnosis bundle missing required fields: {}".format(
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
        help="Directory to write the diagnosis bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    cases = build_diagnosis_cases()
    bundle = build_diagnosis_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "total_cases={total_cases}, limited_fused_coverage_cases={limited_fused_coverage_cases}, runtime_slower_cases={runtime_slower_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
