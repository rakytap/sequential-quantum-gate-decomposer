#!/usr/bin/env python3
"""Validation: Phase 3 Task 6 Story 1 correctness matrix.

Builds the mandatory Task 6 case inventory for the selected Task 5 supported
candidate and records stable case identity plus validation-slice membership.

Run with:
    python benchmarks/density_matrix/correctness_evidence/correctness_matrix_validation.py
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    TASK6_CASE_SCHEMA_VERSION,
    build_task6_selected_candidate,
    build_task6_software_metadata,
    task6_story_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.task6_case_selection import (
    TASK6_CASE_KIND_CONTINUITY,
    TASK6_CASE_KIND_MICROCASE,
    TASK6_CASE_KIND_STRUCTURED,
    build_task6_case_contexts,
)

SUITE_NAME = "phase3_task6_story1_correctness_matrix"
ARTIFACT_FILENAME = "correctness_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = task6_story_output_dir("story1_correctness_matrix")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


@lru_cache(maxsize=1)
def _build_story1_cases_cached() -> tuple[dict, ...]:
    return tuple(dict(case_context.metadata) for case_context in build_task6_case_contexts())


def build_cases() -> list[dict]:
    return deepcopy(list(_build_story1_cases_cached()))


def build_artifact_bundle(cases: list[dict]) -> dict:
    continuity_cases = sum(case["case_kind"] == TASK6_CASE_KIND_CONTINUITY for case in cases)
    microcases = sum(case["case_kind"] == TASK6_CASE_KIND_MICROCASE for case in cases)
    structured_cases = sum(case["case_kind"] == TASK6_CASE_KIND_STRUCTURED for case in cases)
    external_slice_cases = sum(case["external_reference_required"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if continuity_cases == 4
        and microcases == 3
        and structured_cases == 18
        and external_slice_cases == 4
        else "fail",
        "record_schema_version": TASK6_CASE_SCHEMA_VERSION,
        "software": build_task6_software_metadata(),
        "selected_candidate": build_task6_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            "continuity_cases": continuity_cases,
            "microcases": microcases,
            "structured_cases": structured_cases,
            "external_slice_cases": external_slice_cases,
            "internal_only_cases": sum(
                case["validation_slice"] == "internal_only" for case in cases
            ),
            "internal_plus_external_cases": sum(
                case["validation_slice"] == "internal_plus_external" for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 6 Story 1 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 6 Story 1 bundle into.",
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
            "selected_candidate={candidate_id}, total_cases={total_cases}".format(
                candidate_id=bundle["selected_candidate"]["candidate_id"],
                total_cases=bundle["summary"]["total_cases"],
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
