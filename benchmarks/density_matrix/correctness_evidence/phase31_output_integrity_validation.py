#!/usr/bin/env python3
"""Stage-A Phase 3.1 output integrity and continuity energy (six cases).

Run with:
    python benchmarks/density_matrix/correctness_evidence/phase31_output_integrity_validation.py
"""

from __future__ import annotations

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
    phase31_correctness_evidence_output_dir,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_phase31_correctness_evidence_positive_records,
)
from benchmarks.density_matrix.correctness_evidence.validation_support import (
    assemble_positive_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "correctness_evidence_phase31_output_integrity"
ARTIFACT_FILENAME = "phase31_output_integrity_bundle.json"
DEFAULT_OUTPUT_DIR = phase31_correctness_evidence_output_dir("output_integrity")
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
    return build_phase31_correctness_evidence_positive_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    output_integrity_passes = sum(case["output_integrity_pass"] for case in cases)
    continuity_cases = [case for case in cases if case["continuity_energy_required"]]
    continuity_energy_passes = sum(case["continuity_energy_pass"] for case in continuity_cases)
    status = (
        "pass"
        if len(cases) == 6
        and output_integrity_passes == len(cases)
        and continuity_energy_passes == len(continuity_cases)
        else "fail"
    )
    summary = {
        "package_kind": "phase31_stage_a_bounded",
        "total_cases": len(cases),
        "output_integrity_passes": output_integrity_passes,
        "continuity_cases": len(continuity_cases),
        "continuity_energy_passes": continuity_energy_passes,
    }
    bundle = assemble_positive_case_bundle(
        SUITE_NAME,
        status,
        summary,
        cases,
        record_schema_version=CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
    )
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Phase 3.1 output integrity bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the Phase 3.1 output integrity bundle into.",
        quiet_report=lambda b: print(
            "output_integrity_passes={output_integrity_passes}, total_cases={total_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
