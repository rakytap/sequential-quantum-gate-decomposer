#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 8 summary-consistency guardrails.

Interprets the shared Task 7 benchmark package and verifies that downstream
rollups, limitation carry-forward, and claim-closure flags stay consistent with
the underlying per-case evidence.

Run with:
    python benchmarks/density_matrix/performance_evidence/summary_consistency_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.benchmark_bundle_validation import (
    build_artifact_bundle as build_story7_bundle,
)
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_SUMMARY_SCHEMA_VERSION,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)

SUITE_NAME = "phase3_performance_evidence_summary_consistency"
ARTIFACT_FILENAME = "summary_consistency_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("summary_consistency")
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
    story7_bundle = build_story7_bundle()
    cases = story7_bundle["cases"]
    negative_cases = story7_bundle["negative_cases"]
    counted_supported_cases = sum(case["counted_supported_benchmark_case"] for case in cases)
    positive_threshold_pass_cases = sum(case["positive_threshold_pass"] for case in cases)
    diagnosis_only_cases = sum(case["diagnosis_only_case"] for case in cases)
    excluded_cases = sum(case["benchmark_status"] == "excluded" for case in cases)
    representative_review_cases = sum(case["representative_review_case"] for case in cases)

    summary_consistency_pass = (
        story7_bundle["summary"]["total_cases"] == len(cases)
        and story7_bundle["summary"]["counted_supported_cases"] == counted_supported_cases
        and story7_bundle["summary"]["positive_threshold_pass_cases"]
        == positive_threshold_pass_cases
        and story7_bundle["summary"]["diagnosis_only_cases"] == diagnosis_only_cases
        and story7_bundle["summary"]["excluded_cases"] == excluded_cases
        and story7_bundle["summary"]["representative_review_cases"]
        == representative_review_cases
        and story7_bundle["summary"]["correctness_evidence_boundary_cases"] == len(negative_cases)
    )

    positive_benchmark_claim_completed = positive_threshold_pass_cases >= 1
    diagnosis_grounded_closure_completed = (
        positive_threshold_pass_cases == 0
        and diagnosis_only_cases >= 1
        and all(case["diagnosis_reasons"] for case in cases if case["diagnosis_only_case"])
        and len(negative_cases) > 0
    )
    main_benchmark_claim_completed = (
        positive_benchmark_claim_completed or diagnosis_grounded_closure_completed
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if summary_consistency_pass and main_benchmark_claim_completed else "fail",
        "schema_version": PERFORMANCE_EVIDENCE_SUMMARY_SCHEMA_VERSION,
        "software": build_performance_evidence_software_metadata(),
        "selected_candidate": story7_bundle["selected_candidate"],
        "requirements": {
            "positive_claim_rule": (
                "Only counted representative Task 7 evidence may close a positive "
                "benchmark claim."
            ),
            "diagnosis_rule": (
                "If no representative case meets the positive threshold, diagnosis-"
                "only cases must remain benchmark-grounded and visible as explicit "
                "limitation evidence."
            ),
            "boundary_visibility_rule": (
                "Excluded, unsupported, or deferred evidence must remain visible "
                "as claim-boundary evidence in downstream summaries."
            ),
        },
        "summary": {
            "summary_consistency_pass": summary_consistency_pass,
            "positive_benchmark_claim_completed": positive_benchmark_claim_completed,
            "diagnosis_grounded_closure_completed": diagnosis_grounded_closure_completed,
            "main_benchmark_claim_completed": main_benchmark_claim_completed,
            "counted_supported_cases": counted_supported_cases,
            "positive_threshold_pass_cases": positive_threshold_pass_cases,
            "diagnosis_only_cases": diagnosis_only_cases,
            "excluded_cases": excluded_cases,
            "representative_review_cases": representative_review_cases,
            "correctness_evidence_boundary_cases": len(negative_cases),
        },
        "required_artifacts": [
            {
                "artifact_id": "benchmark_package",
                "suite_name": story7_bundle["suite_name"],
                "status": story7_bundle["status"],
                "schema_version": story7_bundle["schema_version"],
            }
        ],
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 8 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 7 Story 8 bundle into.",
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
            "summary_consistency_pass={summary_consistency_pass}, main_benchmark_claim_completed={main_benchmark_claim_completed}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
