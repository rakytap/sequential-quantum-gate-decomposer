#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 2 counted-supported benchmark gate.

Verifies that positive Task 7 benchmark evidence closes only from
correctness-preserving supported cases with stable provenance and explicit
runtime-path identity.

Run with:
    python benchmarks/density_matrix/performance_evidence/counted_supported_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    TASK7_CASE_SCHEMA_VERSION,
    build_task7_selected_candidate,
    build_task7_software_metadata,
    task7_story_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_task7_core_benchmark_records,
    task7_counted_supported_case,
)

SUITE_NAME = "phase3_task7_story2_counted_supported"
ARTIFACT_FILENAME = "counted_supported_bundle.json"
DEFAULT_OUTPUT_DIR = task7_story_output_dir("story2_counted_supported")
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
    return build_task7_core_benchmark_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    counted_supported_cases = [case for case in cases if task7_counted_supported_case(case)]
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if len(counted_supported_cases) == len(cases)
        and all(case["benchmark_status"] != "excluded" for case in counted_supported_cases)
        else "fail",
        "record_schema_version": TASK7_CASE_SCHEMA_VERSION,
        "software": build_task7_software_metadata(),
        "selected_candidate": build_task7_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "counted_supported_cases": len(counted_supported_cases),
            "excluded_cases": len(cases) - len(counted_supported_cases),
            "task6_reference_available": sum(
                case["task6_reference_available"] for case in cases
            ),
            "task6_counted_reference_available": sum(
                case["task6_counted_reference_available"] for case in cases
            ),
            "supported_runtime_cases": sum(case["supported_runtime_case"] for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 2 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 7 Story 2 bundle into.",
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
            "counted_supported_cases={counted_supported_cases}, excluded_cases={excluded_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
