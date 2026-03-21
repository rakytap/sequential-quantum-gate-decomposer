#!/usr/bin/env python3
"""Validation: Phase 3 Task 6 Story 6 unsupported-boundary surface.

Builds a stage-separated negative-evidence layer for planner-entry,
descriptor-generation, and runtime-stage unsupported or deferred behavior.

Run with:
    python benchmarks/density_matrix/correctness_evidence/unsupported_boundary_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
    build_correctness_evidence_selected_candidate,
    build_correctness_evidence_software_metadata,
    correctness_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_correctness_evidence_negative_records,
)

SUITE_NAME = "phase3_correctness_evidence_unsupported_boundary"
ARTIFACT_FILENAME = "unsupported_boundary_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("unsupported_boundary")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "negative_record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    return build_correctness_evidence_negative_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    stage_counts = {
        "planner_entry_cases": sum(case["boundary_stage"] == "planner_entry" for case in cases),
        "descriptor_generation_cases": sum(
            case["boundary_stage"] == "descriptor_generation" for case in cases
        ),
        "runtime_stage_cases": sum(case["boundary_stage"] == "runtime_stage" for case in cases),
    }
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if all(case["status"] == "unsupported" for case in cases)
        and all(not case["fallback_used"] for case in cases)
        else "fail",
        "negative_record_schema_version": CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
        "software": build_correctness_evidence_software_metadata(),
        "selected_candidate": build_correctness_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            **stage_counts,
            "unsupported_cases": sum(case["status"] == "unsupported" for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 6 Story 6 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 6 Story 6 bundle into.",
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
            "planner_entry_cases={planner_entry_cases}, descriptor_generation_cases={descriptor_generation_cases}, runtime_stage_cases={runtime_stage_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
