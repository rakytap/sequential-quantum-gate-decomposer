#!/usr/bin/env python3
"""Validation: Phase 3 Task 6 Story 4 output integrity and continuity agreement.

Ensures trace validity, density validity, and required continuity-anchor energy
agreement remain first-class parts of the Task 6 correctness surface.

Run with:
    python benchmarks/density_matrix/correctness_evidence/output_integrity_validation.py
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

SUITE_NAME = "phase3_task6_story4_output_integrity"
ARTIFACT_FILENAME = "output_integrity_bundle.json"
DEFAULT_OUTPUT_DIR = task6_story_output_dir("story4_output_integrity")
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
    return build_task6_positive_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    output_integrity_passes = sum(case["output_integrity_pass"] for case in cases)
    continuity_cases = [case for case in cases if case["continuity_energy_required"]]
    continuity_energy_passes = sum(case["continuity_energy_pass"] for case in continuity_cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if output_integrity_passes == len(cases)
        and continuity_energy_passes == len(continuity_cases)
        else "fail",
        "record_schema_version": TASK6_CASE_SCHEMA_VERSION,
        "software": build_task6_software_metadata(),
        "selected_candidate": build_task6_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "output_integrity_passes": output_integrity_passes,
            "continuity_cases": len(continuity_cases),
            "continuity_energy_passes": continuity_energy_passes,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 6 Story 4 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 6 Story 4 bundle into.",
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
            "output_integrity_passes={output_integrity_passes}, continuity_energy_passes={continuity_energy_passes}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
