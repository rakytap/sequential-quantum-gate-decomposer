#!/usr/bin/env python3
"""Correctness matrix validation for density-matrix correctness evidence.

Builds the mandatory case inventory for the selected supported calibration
candidate and records stable case identity plus validation-slice membership.

Run with:
    python benchmarks/density_matrix/correctness_evidence/correctness_matrix_validation.py
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.case_selection import (
    CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY,
    CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE,
    CORRECTNESS_EVIDENCE_CASE_KIND_STRUCTURED,
    build_correctness_evidence_case_contexts,
)
from benchmarks.density_matrix.correctness_evidence.common import (
    correctness_evidence_output_dir,
)
from benchmarks.density_matrix.correctness_evidence.validation_support import (
    assemble_positive_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "correctness_evidence_correctness_matrix"
ARTIFACT_FILENAME = "correctness_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("correctness_matrix")
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
def _build_correctness_matrix_cases_cached() -> tuple[dict, ...]:
    return tuple(dict(case_context.metadata) for case_context in build_correctness_evidence_case_contexts())


def build_cases() -> list[dict]:
    return deepcopy(list(_build_correctness_matrix_cases_cached()))


def build_artifact_bundle(cases: list[dict]) -> dict:
    continuity_cases = sum(case["case_kind"] == CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY for case in cases)
    microcases = sum(case["case_kind"] == CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE for case in cases)
    structured_cases = sum(case["case_kind"] == CORRECTNESS_EVIDENCE_CASE_KIND_STRUCTURED for case in cases)
    external_slice_cases = sum(case["external_reference_required"] for case in cases)
    status = (
        "pass"
        if continuity_cases == 4
        and microcases == 3
        and structured_cases == 18
        and external_slice_cases == 4
        else "fail"
    )
    summary = {
        "total_cases": len(cases),
        "continuity_cases": continuity_cases,
        "microcases": microcases,
        "structured_cases": structured_cases,
        "external_slice_cases": external_slice_cases,
        "internal_only_cases": sum(case["validation_slice"] == "internal_only" for case in cases),
        "internal_plus_external_cases": sum(
            case["validation_slice"] == "internal_plus_external" for case in cases
        ),
    }
    bundle = assemble_positive_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Correctness matrix bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the correctness matrix bundle into.",
        quiet_report=lambda b: print(
            "selected_candidate={candidate_id}, total_cases={total_cases}".format(
                candidate_id=b["selected_candidate"]["candidate_id"],
                total_cases=b["summary"]["total_cases"],
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
