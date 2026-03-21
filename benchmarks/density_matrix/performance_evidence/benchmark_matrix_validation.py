#!/usr/bin/env python3
"""Validation: Phase 3 Task 7 Story 1 benchmark-matrix surface.

Freezes the dual-anchor Task 7 benchmark matrix and representative review-set
membership through one stable machine-reviewable surface.

Run with:
    python benchmarks/density_matrix/performance_evidence/benchmark_matrix_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY,
    PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED,
    build_performance_evidence_selected_candidate,
    build_performance_evidence_software_metadata,
    performance_evidence_output_dir,
    write_artifact_bundle,
)
from benchmarks.density_matrix.performance_evidence.case_selection import (
    build_performance_evidence_inventory_cases,
)

SUITE_NAME = "phase3_performance_evidence_benchmark_matrix"
ARTIFACT_FILENAME = "benchmark_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = performance_evidence_output_dir("benchmark_matrix")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "selected_candidate",
    "software",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    return build_performance_evidence_inventory_cases()


def build_artifact_bundle(cases: list[dict]) -> dict:
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass",
        "selected_candidate": build_performance_evidence_selected_candidate(),
        "software": build_performance_evidence_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "continuity_cases": sum(
                case["benchmark_slice"] == PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY
                for case in cases
            ),
            "structured_cases": sum(
                case["benchmark_slice"] == PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED
                for case in cases
            ),
            "representative_review_cases": sum(
                case["representative_review_case"] for case in cases
            ),
            "continuity_qbits": sorted(
                {case["qbit_num"] for case in cases if case["case_kind"] == "continuity"}
            ),
            "structured_qbits": sorted(
                {
                    case["qbit_num"]
                    for case in cases
                    if case["case_kind"] == "structured_family"
                }
            ),
            "structured_seed_count": len(
                {
                    case["seed"]
                    for case in cases
                    if case["case_kind"] == "structured_family"
                }
            ),
            "noise_patterns": sorted(
                {
                    case["noise_pattern"]
                    for case in cases
                    if case["noise_pattern"] is not None
                }
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 7 Story 1 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 7 Story 1 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, ARTIFACT_FILENAME)

    if not args.quiet:
        print(
            "total_cases={total_cases}, continuity_cases={continuity_cases}, structured_cases={structured_cases}, representative_review_cases={representative_review_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
