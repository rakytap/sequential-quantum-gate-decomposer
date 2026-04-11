#!/usr/bin/env python3
"""Stage-A Phase 3.1 sequential / internal reference gate (six cases).

Run with:
    python benchmarks/density_matrix/correctness_evidence/phase31_sequential_correctness_validation.py
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

SUITE_NAME = "correctness_evidence_phase31_sequential_correctness"
ARTIFACT_FILENAME = "phase31_sequential_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = phase31_correctness_evidence_output_dir("sequential_correctness")
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
    internal_passes = sum(case["internal_reference_pass"] for case in cases)
    supported_runtime_cases = sum(case["supported_runtime_case"] for case in cases)
    status = (
        "pass"
        if len(cases) == 6
        and internal_passes == len(cases)
        and supported_runtime_cases == len(cases)
        else "fail"
    )
    summary = {
        "package_kind": "phase31_stage_a_bounded",
        "total_cases": len(cases),
        "supported_runtime_cases": supported_runtime_cases,
        "internal_reference_passes": internal_passes,
        "actual_fused_cases": sum(case["actual_fused_execution"] for case in cases),
    }
    bundle = assemble_positive_case_bundle(
        SUITE_NAME,
        status,
        summary,
        cases,
        record_schema_version=CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION,
    )
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Phase 3.1 sequential correctness bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the Phase 3.1 sequential correctness bundle into.",
        quiet_report=lambda b: print(
            "supported_runtime_cases={supported_runtime_cases}, internal_reference_passes={internal_reference_passes}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
