#!/usr/bin/env python3
"""Runtime and fusion path classification validation.

Records how the fused-capable runtime classifies supported matrix cases while
keeping those classifications directly comparable to the same correctness
thresholds used elsewhere in the evidence pipeline.

Run with:
    python benchmarks/density_matrix/correctness_evidence/runtime_classification_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
    CORRECTNESS_EVIDENCE_RUNTIME_CLASS_BASELINE,
    build_correctness_evidence_selected_candidate,
    build_correctness_evidence_software_metadata,
    correctness_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    build_correctness_evidence_positive_records,
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
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if sum(classification_counts.values()) == len(cases)
        and all(case["supported_runtime_case"] for case in cases)
        else "fail",
        "record_schema_version": CORRECTNESS_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_correctness_evidence_software_metadata(),
        "selected_candidate": build_correctness_evidence_selected_candidate(),
        "summary": {
            "total_cases": len(cases),
            **classification_counts,
            "actual_fused_cases": sum(case["actual_fused_execution"] for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Runtime classification bundle missing required fields: {}".format(
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
        help="Directory to write the runtime classification bundle into.",
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
            "actually_fused={actually_fused}, supported_but_unfused={supported_but_unfused}, deferred_or_unsupported_candidate={deferred_or_unsupported_candidate}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
