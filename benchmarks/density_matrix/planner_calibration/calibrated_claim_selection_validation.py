#!/usr/bin/env python3
"""Validation: Phase 3 Task 5 Story 5 supported-claim selection surface.

Selects one supported Task 5 planner claim from the correctness-gated
calibration matrix using an explicit benchmark-grounded rule and keeps the other
candidate settings visible as comparison baselines.

Run with:
    python benchmarks/density_matrix/planner_calibration/calibrated_claim_selection_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.claim_selection import (
    PLANNER_CALIBRATION_CLAIM_SELECTION_SCHEMA_VERSION,
    PLANNER_CALIBRATION_CLAIM_SELECTION_RULE,
    PLANNER_CALIBRATION_CLAIM_STATUS_COMPARISON,
    PLANNER_CALIBRATION_CLAIM_STATUS_SUPPORTED,
    build_planner_calibration_claim_selection_payload,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata

SUITE_NAME = "phase3_planner_calibration_claim_selection"
ARTIFACT_FILENAME = "claim_selection_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "claim_selection"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "claim_selection_schema_version",
    "software",
    "summary",
    "candidate_summaries",
    "cases",
)


def build_cases() -> list[dict]:
    return build_planner_calibration_claim_selection_payload()["cases"]


def build_artifact_bundle() -> dict:
    payload = build_planner_calibration_claim_selection_payload()
    cases = payload["cases"]
    candidate_summaries = payload["candidate_summaries"]
    selected_candidate = payload["selected_candidate"]
    supported_claim_cases = sum(
        case["claim_status"] == PLANNER_CALIBRATION_CLAIM_STATUS_SUPPORTED for case in cases
    )
    comparison_cases = sum(
        case["claim_status"] == PLANNER_CALIBRATION_CLAIM_STATUS_COMPARISON for case in cases
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if supported_claim_cases > 0
        and comparison_cases > 0
        and len(candidate_summaries) == 3
        else "fail",
        "claim_selection_schema_version": PLANNER_CALIBRATION_CLAIM_SELECTION_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "supported_claim_cases": supported_claim_cases,
            "comparison_cases": comparison_cases,
            "selected_candidate_id": selected_candidate["candidate_id"],
            "claim_selection_rule": PLANNER_CALIBRATION_CLAIM_SELECTION_RULE,
        },
        "selected_candidate": selected_candidate,
        "candidate_summaries": candidate_summaries,
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 5 Story 5 bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the Task 5 Story 5 bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    bundle = build_artifact_bundle()
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        print(
            "selected_candidate={selected_candidate_id}, rule={claim_selection_rule}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
