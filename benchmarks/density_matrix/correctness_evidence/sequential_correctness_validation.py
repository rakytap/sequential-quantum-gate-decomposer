#!/usr/bin/env python3
"""Validation: Phase 3 Task 6 Story 2 sequential-baseline correctness gate.

Runs the selected Task 5 supported candidate through the fused-capable runtime
surface and checks every mandatory Task 6 case against the sequential density
reference.

Run with:
    python benchmarks/density_matrix/correctness_evidence/sequential_correctness_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
    build_correctness_evidence_selected_candidate,
    build_correctness_evidence_software_metadata,
    correctness_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_correctness_evidence_positive_records,
)

SUITE_NAME = "phase3_correctness_evidence_sequential_correctness"
ARTIFACT_FILENAME = "sequential_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("sequential_correctness")
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
    return build_correctness_evidence_positive_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    internal_passes = sum(case["internal_reference_pass"] for case in cases)
    supported_runtime_cases = sum(case["supported_runtime_case"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if internal_passes == len(cases) and supported_runtime_cases == len(cases)
        else "fail",
        "record_schema_version": CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_correctness_evidence_software_metadata(),
        "selected_candidate": build_correctness_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "supported_runtime_cases": supported_runtime_cases,
            "internal_reference_passes": internal_passes,
            "actual_fused_cases": sum(case["actual_fused_execution"] for case in cases),
            "baseline_path_cases": sum(
                case["runtime_path_classification"] == "plain_partitioned_baseline"
                for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 6 Story 2 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 6 Story 2 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "supported_runtime_cases={supported_runtime_cases}, internal_reference_passes={internal_reference_passes}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
