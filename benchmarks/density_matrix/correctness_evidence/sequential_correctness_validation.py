#!/usr/bin/env python3
"""Sequential baseline correctness validation (internal reference gate).

Runs the selected supported calibration candidate through the fused-capable
runtime surface and checks every mandatory matrix case against the sequential
density reference.

Run with:
    python benchmarks/density_matrix/correctness_evidence/sequential_correctness_validation.py
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

SUITE_NAME = "correctness_evidence_sequential_correctness"
ARTIFACT_FILENAME = "sequential_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("sequential_correctness")
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
    return build_correctness_evidence_positive_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    internal_passes = sum(case["internal_reference_pass"] for case in cases)
    supported_runtime_cases = sum(case["supported_runtime_case"] for case in cases)
    status = (
        "pass"
        if internal_passes == len(cases) and supported_runtime_cases == len(cases)
        else "fail"
    )
    summary = {
        "total_cases": len(cases),
        "supported_runtime_cases": supported_runtime_cases,
        "internal_reference_passes": internal_passes,
        "actual_fused_cases": sum(case["actual_fused_execution"] for case in cases),
        "baseline_path_cases": sum(
            case["runtime_path_classification"] == "plain_partitioned_baseline"
            for case in cases
        ),
    }
    bundle = assemble_positive_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Sequential correctness bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the sequential correctness bundle into.",
        quiet_report=lambda b: print(
            "supported_runtime_cases={supported_runtime_cases}, internal_reference_passes={internal_reference_passes}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
