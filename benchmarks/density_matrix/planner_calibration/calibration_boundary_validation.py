#!/usr/bin/env python3
"""Validation: Phase 3 planner-calibration explicit claim-boundary surface.

Packages the supported calibration claim together with explicit approximation
areas, comparison-baseline visibility, and deferred follow-on branches.

Run with:
    python benchmarks/density_matrix/planner_calibration/calibration_boundary_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.boundary import (
    PLANNER_CALIBRATION_BOUNDARY_SCHEMA_VERSION,
    build_planner_calibration_boundary_payload,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata

SUITE_NAME = "phase3_planner_calibration_claim_boundary"
ARTIFACT_FILENAME = "claim_boundary_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "claim_boundary"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "boundary_schema_version",
    "software",
    "summary",
)


def build_artifact_bundle() -> dict:
    payload = build_planner_calibration_boundary_payload()
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if payload["summary"]["comparison_baseline_case_count"] > 0
        and payload["summary"]["approximation_area_count"] > 0
        and payload["summary"]["deferred_follow_on_branch_count"] > 0
        else "fail",
        "boundary_schema_version": PLANNER_CALIBRATION_BOUNDARY_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": payload["summary"],
        "supported_claim": payload["supported_claim"],
        "comparison_baselines": payload["comparison_baselines"],
        "comparison_baseline_cases": payload["comparison_baseline_cases"],
        "diagnosis_only_cases": payload["diagnosis_only_cases"],
        "approximation_areas": payload["approximation_areas"],
        "deferred_follow_on_branches": payload["deferred_follow_on_branches"],
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Claim-boundary validation bundle missing required fields: {}".format(
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
        help="Directory to write the claim-boundary validation bundle into.",
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
            "selected_candidate={selected_candidate_id}, comparison_cases={comparison_baseline_case_count}, deferred={deferred_follow_on_branch_count}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
