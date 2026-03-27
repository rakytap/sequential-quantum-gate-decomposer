#!/usr/bin/env python3
"""Runtime and fusion path classification validation.

Records how the fused-capable runtime classifies supported matrix cases while
keeping those classifications directly comparable to the same correctness
thresholds used elsewhere in the evidence pipeline.

Run with:
    python benchmarks/density_matrix/correctness_evidence/runtime_classification_validation.py
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
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

SUITE_NAME = "correctness_evidence_runtime_classification"
ARTIFACT_FILENAME = "runtime_classification_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("runtime_classification")
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
    classification_counts = {
        "actually_fused": sum(
            case["runtime_path_classification"] == "actually_fused" for case in cases
        ),
        "supported_but_unfused": sum(
            case["runtime_path_classification"] == "supported_but_unfused"
            for case in cases
        ),
        "deferred_or_unsupported_candidate": sum(
            case["runtime_path_classification"] == "deferred_or_unsupported_candidate"
            for case in cases
        ),
        CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE: sum(
            case["runtime_path_classification"] == CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE
            for case in cases
        ),
    }
    status = (
        "pass"
        if sum(classification_counts.values()) == len(cases)
        and all(case["supported_runtime_case"] for case in cases)
        else "fail"
    )
    summary = {
        "total_cases": len(cases),
        **classification_counts,
        "actual_fused_cases": sum(case["actual_fused_execution"] for case in cases),
    }
    bundle = assemble_positive_case_bundle(SUITE_NAME, status, summary, cases)
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Runtime classification bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return run_case_slice_cli(
        argv,
        build_cases=build_cases,
        build_artifact_bundle=build_artifact_bundle,
        artifact_filename=ARTIFACT_FILENAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        description=__doc__ or "",
        output_dir_help="Directory to write the runtime classification bundle into.",
        quiet_report=lambda b: print(
            "actually_fused={actually_fused}, supported_but_unfused={supported_but_unfused}, deferred_or_unsupported_candidate={deferred_or_unsupported_candidate}".format(
                **b["summary"]
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
