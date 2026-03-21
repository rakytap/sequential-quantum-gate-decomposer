#!/usr/bin/env python3
"""Counted-supported gate validation for performance evidence.

Verifies that positive benchmark evidence closes only from correctness-preserving
supported cases with stable provenance and explicit runtime-path identity.

Run with:
    python benchmarks/density_matrix/performance_evidence/counted_supported_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    build_performance_evidence_selected_candidate,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.performance_evidence.records import (
    build_performance_evidence_core_benchmark_records,
    performance_evidence_counted_supported_case,
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
    counted_supported_cases = [case for case in cases if performance_evidence_counted_supported_case(case)]
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if len(counted_supported_cases) == len(cases)
        and all(case["benchmark_status"] != "excluded" for case in counted_supported_cases)
        else "fail",
        "record_schema_version": PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "summary": {
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
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Counted-supported bundle missing required fields: {}".format(
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
        help="Directory to write the counted-supported bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    cases = build_counted_supported_cases()
    bundle = build_counted_supported_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "counted_supported_cases={counted_supported_cases}, excluded_cases={excluded_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
