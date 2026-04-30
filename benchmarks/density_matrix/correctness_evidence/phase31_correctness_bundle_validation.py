#!/usr/bin/env python3
"""Stage-A Phase 3.1 correctness package bundle (six positives + negative boundary).

Run with:
    python benchmarks/density_matrix/correctness_evidence/phase31_correctness_bundle_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.bundle import (
    build_phase31_correctness_package_payload,
)
from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_PHASE31_PACKAGE_SCHEMA_VERSION,
    build_package_software_metadata,
    phase31_correctness_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import counted_supported_case
from benchmarks.density_matrix.validation_scaffold import require_bundle_fields

SUITE_NAME = "correctness_evidence_phase31_correctness_package"
ARTIFACT_FILENAME = "phase31_correctness_package_bundle.json"
DEFAULT_OUTPUT_DIR = phase31_correctness_evidence_output_dir("correctness_package")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "selected_candidate",
    "summary",
    "cases",
    "negative_cases",
)


def build_cases() -> list[dict]:
    return build_phase31_correctness_package_payload()["cases"]


def build_artifact_bundle() -> dict:
    payload = build_phase31_correctness_package_payload()
    cases = payload["cases"]
    negative_cases = payload["negative_cases"]
    counted_supported_cases = sum(counted_supported_case(case) for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if len(cases) == 6
        and counted_supported_cases == len(cases)
        and len(negative_cases) > 0
        and all(case["status"] == "unsupported" for case in negative_cases)
        else "fail",
        "schema_version": CORRECTNESS_EVIDENCE_PHASE31_PACKAGE_SCHEMA_VERSION,
        "software": build_package_software_metadata(),
        "selected_candidate": payload["selected_candidate"],
        "summary": payload["summary"],
        "cases": cases,
        "negative_cases": negative_cases,
    }
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Phase 3.1 correctness package bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the Phase 3.1 correctness package bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    bundle = build_artifact_bundle()
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "status={status}, total_cases={total_cases}, counted_supported_cases={counted_supported_cases}".format(
                status=bundle["status"],
                total_cases=bundle["summary"]["total_cases"],
                counted_supported_cases=bundle["summary"]["counted_supported_cases"],
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
