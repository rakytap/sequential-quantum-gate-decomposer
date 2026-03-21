#!/usr/bin/env python3
"""Validation: Phase 3 planner-calibration integrated bundle (shared artifact).

Packages the selected supported claim, comparison baselines, and all calibrated
case records into one stable machine-reviewable bundle for later validation,
benchmark, and publication consumers.

Run with:
    python benchmarks/density_matrix/planner_calibration/calibration_bundle_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.bundle import (
    PLANNER_CALIBRATION_CALIBRATION_BUNDLE_SCHEMA_VERSION,
    build_planner_calibration_calibration_bundle_payload,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata

SUITE_NAME = "phase3_planner_calibration_calibration_bundle"
ARTIFACT_FILENAME = "calibration_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "calibration_bundle"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "calibration_bundle_schema_version",
    "software",
    "summary",
    "candidate_summaries",
    "cases",
)


def build_artifact_bundle() -> dict:
    payload = build_planner_calibration_calibration_bundle_payload()
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if payload["summary"]["supported_claim_cases"] > 0
        and payload["summary"]["comparison_cases"] > 0
        and payload["summary"]["counted_calibration_cases"] == payload["summary"]["total_cases"]
        else "fail",
        "calibration_bundle_schema_version": PLANNER_CALIBRATION_CALIBRATION_BUNDLE_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": payload["summary"],
        "claim_selection_rule": payload["claim_selection_rule"],
        "selected_candidate": payload["selected_candidate"],
        "comparison_candidate_ids": payload["comparison_candidate_ids"],
        "candidate_summaries": payload["candidate_summaries"],
        "cases": payload["cases"],
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Integrated calibration bundle missing required fields: {}".format(
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
        help="Directory to write the integrated calibration bundle into.",
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
            "selected_candidate={candidate_id}, cases={total_cases}".format(
                candidate_id=bundle["selected_candidate"]["candidate_id"],
                total_cases=bundle["summary"]["total_cases"],
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
