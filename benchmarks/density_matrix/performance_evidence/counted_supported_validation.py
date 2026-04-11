#!/usr/bin/env python3
"""Counted-supported gate validation for performance evidence.

Verifies that positive benchmark evidence closes only from correctness-preserving
supported cases with stable provenance and explicit runtime-path identity.

Run with:
    python benchmarks/density_matrix/performance_evidence/counted_supported_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    performance_evidence_output_dir,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_core_benchmark_records,
    performance_evidence_counted_supported_case,
)
from benchmarks.density_matrix.performance_evidence.validation_support import (
    assemble_record_schema_case_bundle,
)
from benchmarks.density_matrix.validation_scaffold import (
    require_bundle_fields,
    run_case_slice_cli,
)

SUITE_NAME = "performance_evidence_counted_supported"
ARTIFACT_FILENAME = "counted_supported_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("counted_supported")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
)


def build_counted_supported_cases() -> list[dict]:
    return build_performance_evidence_core_benchmark_records()


def build_counted_supported_bundle(cases: list[dict]) -> dict:
    counted_supported_cases = [
        case for case in cases if performance_evidence_counted_supported_case(case)
    ]
    status = (
        "pass"
        if len(counted_supported_cases) == len(cases)
        and all(case["benchmark_status"] != "excluded" for case in counted_supported_cases)
        else "fail"
    )
    summary = {
        "total_cases": len(cases),
        "counted_supported_cases": len(counted_supported_cases),
        "excluded_cases": len(cases) - len(counted_supported_cases),
        "correctness_evidence_reference_available": sum(
            case["correctness_evidence_reference_available"] for case in cases
        ),
        "correctness_evidence_counted_reference_available": sum(
            case["correctness_evidence_counted_reference_available"] for case in cases
        ),
        "supported_runtime_cases": sum(case["supported_runtime_case"] for case in cases),
    }
    bundle = assemble_record_schema_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Counted-supported bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_counted_supported_cases,
        build_artifact_bundle=build_counted_supported_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the counted-supported bundle into.",
        quiet_report=lambda b: print(
            "counted_supported_cases={counted_supported_cases}, excluded_cases={excluded_cases}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
