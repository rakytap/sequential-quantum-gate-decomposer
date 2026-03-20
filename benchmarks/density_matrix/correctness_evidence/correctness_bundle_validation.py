#!/usr/bin/env python3
"""Validation: Phase 3 Task 6 Story 7 correctness package.

Builds the shared Task 6 correctness package by joining positive supported
records and stage-separated unsupported-boundary evidence through one stable
machine-reviewable surface.

Run with:
    python benchmarks/density_matrix/correctness_evidence/correctness_bundle_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.bundle import (
    build_task6_correctness_package_payload,
)
from benchmarks.density_matrix.correctness_evidence.common import (
    TASK6_CORRECTNESS_PACKAGE_SCHEMA_VERSION,
    build_task6_software_metadata,
    task6_story_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    task6_counted_supported_case,
)

SUITE_NAME = "phase3_task6_story7_correctness_package"
ARTIFACT_FILENAME = "correctness_package_bundle.json"
DEFAULT_OUTPUT_DIR = task6_story_output_dir("story7_correctness_package")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
    "negative_cases",
)


def build_cases() -> list[dict]:
    return build_task6_correctness_package_payload()["cases"]


def build_artifact_bundle() -> dict:
    payload = build_task6_correctness_package_payload()
    cases = payload["cases"]
    negative_cases = payload["negative_cases"]
    counted_supported_cases = sum(task6_counted_supported_case(case) for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if counted_supported_cases == len(cases)
        and len(negative_cases) > 0
        and all(case["status"] == "unsupported" for case in negative_cases)
        else "fail",
        "schema_version": TASK6_CORRECTNESS_PACKAGE_SCHEMA_VERSION,
        "software": build_task6_software_metadata(),
        "selected_candidate": payload["selected_candidate"],
        "summary": payload["summary"],
        "cases": cases,
        "negative_cases": negative_cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 6 Story 7 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 6 Story 7 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    bundle = build_artifact_bundle()
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "counted_supported_cases={counted_supported_cases}, unsupported_boundary_cases={unsupported_boundary_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
