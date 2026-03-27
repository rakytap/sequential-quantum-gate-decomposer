#!/usr/bin/env python3
"""Summary consistency validation for the correctness-evidence package.

Interprets the shared correctness package bundle and verifies that downstream
rollups and closure flags stay consistent with the underlying per-case evidence.

Run with:
    python benchmarks/density_matrix/correctness_evidence/summary_consistency_validation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.density_matrix.correctness_evidence.correctness_bundle_validation import (
    build_artifact_bundle as build_correctness_package_bundle,
)
from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION,
    build_correctness_evidence_software_metadata,
    correctness_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.correctness_evidence.records import (
    correctness_evidence_counted_supported_case,
)
from benchmarks.density_matrix.validation_scaffold import require_bundle_fields

SUITE_NAME = "correctness_evidence_summary_consistency"
ARTIFACT_FILENAME = "summary_consistency_bundle.json"
DEFAULT_OUTPUT_DIR = correctness_evidence_output_dir("summary_consistency")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "selected_candidate",
    "requirements",
    "summary",
    "required_artifacts",
)


def build_artifact_bundle() -> dict:
    correctness_package = build_correctness_package_bundle()
    cases = correctness_package["cases"]
    negative_cases = correctness_package["negative_cases"]
    counted_supported_cases = sum(correctness_evidence_counted_supported_case(case) for case in cases)
    summary_consistency_pass = (
        correctness_package["summary"]["total_cases"] == len(cases)
        and correctness_package["summary"]["counted_supported_cases"] == counted_supported_cases
        and correctness_package["summary"]["unsupported_boundary_cases"] == len(negative_cases)
    )
    main_correctness_claim_completed = (
        counted_supported_cases == len(cases)
        and len(negative_cases) > 0
        and all(case["status"] == "unsupported" for case in negative_cases)
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if summary_consistency_pass else "fail",
        "schema_version": CORRECTNESS_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "software": build_correctness_evidence_software_metadata(),
        "selected_candidate": correctness_package["selected_candidate"],
        "requirements": {
            "main_claim_rule": (
                "Only mandatory, complete, supported evidence may close the main "
                "correctness claim for this package."
            ),
            "boundary_visibility_rule": (
                "Excluded, unsupported, or deferred evidence must remain visible "
                "as claim-boundary evidence in downstream summaries."
            ),
        },
        "summary": {
            "summary_consistency_pass": summary_consistency_pass,
            "main_correctness_claim_completed": main_correctness_claim_completed,
            "counted_supported_cases": counted_supported_cases,
            "excluded_supported_cases": len(cases) - counted_supported_cases,
            "unsupported_boundary_cases": len(negative_cases),
            "planner_entry_boundary_cases": sum(
                case["boundary_stage"] == "planner_entry" for case in negative_cases
            ),
            "descriptor_generation_boundary_cases": sum(
                case["boundary_stage"] == "descriptor_generation"
                for case in negative_cases
            ),
            "runtime_stage_boundary_cases": sum(
                case["boundary_stage"] == "runtime_stage" for case in negative_cases
            ),
        },
        "required_artifacts": [
            {
                "artifact_id": "correctness_package",
                "suite_name": correctness_package["suite_name"],
                "status": correctness_package["status"],
                "schema_version": correctness_package["schema_version"],
            }
        ],
    }
    require_bundle_fields(bundle, ARTIFACT_CORE_FIELDS, "Summary consistency bundle")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the summary consistency bundle into.",
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
            "summary_consistency_pass={summary_consistency_pass}, main_correctness_claim_completed={main_correctness_claim_completed}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
