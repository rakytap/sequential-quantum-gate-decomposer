#!/usr/bin/env python3
"""External correctness validation (bounded Qiskit Aer reference slice).

Checks the required external-reference slice against Qiskit Aer while keeping
the slice explicitly bounded to mandatory microcases plus the selected small
continuity subset.

Run with:
    python benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py
"""

from __future__ import annotations

from benchmarks.density_matrix.correctness_evidence.common import (
    correctness_evidence_output_dir,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_correctness_evidence_positive_records,
)
from benchmarks.density_matrix.correctness_evidence.validation_support import (
    assemble_positive_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "correctness_evidence_external_correctness"
ARTIFACT_FILENAME = "external_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("external_correctness")
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
    return [
        case
        for case in build_correctness_evidence_positive_records()
        if case["external_reference_required"]
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    external_passes = sum(case["external_reference_pass"] for case in cases)
    status = "pass" if len(cases) == 4 and external_passes == len(cases) else "fail"
    summary = {
        "total_cases": len(cases),
        "external_reference_passes": external_passes,
        "microcases": sum(case["case_kind"] == "microcase" for case in cases),
        "continuity_cases": sum(case["case_kind"] == "continuity" for case in cases),
    }
    bundle = assemble_positive_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "External correctness bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the external correctness bundle into.",
        quiet_report=lambda b: print(
            "external_reference_passes={external_reference_passes}, total_cases={total_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
