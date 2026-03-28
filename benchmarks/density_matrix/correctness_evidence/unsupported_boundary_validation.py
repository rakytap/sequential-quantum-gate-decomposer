#!/usr/bin/env python3
"""Unsupported-boundary (negative evidence) validation.

Builds a stage-separated negative-evidence layer for planner-entry,
descriptor-generation, and runtime-stage unsupported or deferred behavior.

Run with:
    python benchmarks/density_matrix/correctness_evidence/unsupported_boundary_validation.py
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
    correctness_evidence_output_dir,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_correctness_evidence_negative_records,
)
from benchmarks.density_matrix.correctness_evidence.validation_support import (
    assemble_negative_boundary_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "correctness_evidence_unsupported_boundary"
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
    status = "pass" if all(case["status"] == "unsupported" for case in cases) else "fail"
    summary = {
        "total_cases": len(cases),
        **stage_counts,
        "unsupported_cases": sum(case["status"] == "unsupported" for case in cases),
    }
    bundle = assemble_negative_boundary_bundle(
        SUITE_NAME,
        status,
        CORRECTNESS_EVIDENCE_NEGATIVE_RECORD_SCHEMA_VERSION,
        summary,
        cases,
    )
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Unsupported boundary bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the unsupported boundary bundle into.",
        quiet_report=lambda b: print(
            "planner_entry_cases={planner_entry_cases}, descriptor_generation_cases={descriptor_generation_cases}, runtime_stage_cases={runtime_stage_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
