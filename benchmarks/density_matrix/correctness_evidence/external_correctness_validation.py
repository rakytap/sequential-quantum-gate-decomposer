#!/usr/bin/env python3
"""Validation: Phase 3 Task 6 Story 3 bounded Qiskit Aer slice.

Checks the required external Task 6 slice against Qiskit Aer while keeping the
slice explicitly bounded to mandatory microcases plus the selected small
continuity subset.

Run with:
    python benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    TASK6_CASE_SCHEMA_VERSION,
    build_task6_selected_candidate,
    build_task6_software_metadata,
    task6_story_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_task6_positive_records,
)

SUITE_NAME = "phase3_task6_story3_external_slice"
ARTIFACT_FILENAME = "external_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = task6_story_output_dir("story3_external_slice")
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
    return [case for case in build_task6_positive_records() if case["external_reference_required"]]


def build_artifact_bundle(cases: list[dict]) -> dict:
    external_passes = sum(case["external_reference_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if len(cases) == 4 and external_passes == len(cases) else "fail",
        "record_schema_version": TASK6_CASE_SCHEMA_VERSION,
        "software": build_task6_software_metadata(),
        "selected_candidate": build_task6_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "external_reference_passes": external_passes,
            "microcases": sum(case["case_kind"] == "microcase" for case in cases),
            "continuity_cases": sum(case["case_kind"] == "continuity" for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 6 Story 3 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 6 Story 3 bundle into.",
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
            "external_reference_passes={external_reference_passes}, total_cases={total_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
