#!/usr/bin/env python3
"""Validation: Phase 3 planner-calibration correctness-gated surface.

Builds the full calibration matrix, applies the internal exactness gate to every
candidate-workload case, and applies the external Aer gate on the required
microcase and representative small continuity slice.

Run with:
    python benchmarks/density_matrix/planner_calibration/calibration_correctness_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.calibration_records import (
    PLANNER_CALIBRATION_CALIBRATION_RECORD_SCHEMA_VERSION,
    PLANNER_CALIBRATION_REFERENCE_BACKEND,
    build_planner_calibration_calibration_records,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata

SUITE_NAME = "phase3_planner_calibration_correctness_gate"
ARTIFACT_FILENAME = "calibration_correctness_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "calibration_correctness"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "record_schema_version",
    "software",
    "summary",
    "cases",
)


def build_cases() -> list[dict]:
    return build_planner_calibration_calibration_records()


def build_artifact_bundle(cases: list[dict]) -> dict:
    internal_correctness_passes = sum(case["internal_correctness_pass"] for case in cases)
    external_reference_cases = sum(case["external_reference_required"] for case in cases)
    external_reference_passes = sum(
        case["external_reference_required"] and case["external_reference_pass"]
        for case in cases
    )
    counted_calibration_cases = sum(case["counted_calibration_case"] for case in cases)
    continuity_energy_cases = sum(
        case["continuity_energy_error"] is not None for case in cases
    )
    continuity_energy_passes = sum(
        (case["continuity_energy_error"] is not None) and case["continuity_energy_pass"]
        for case in cases
    )
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if internal_correctness_passes == len(cases)
        and external_reference_passes == external_reference_cases
        and counted_calibration_cases == len(cases)
        and continuity_energy_passes == continuity_energy_cases
        else "fail",
        "record_schema_version": PLANNER_CALIBRATION_CALIBRATION_RECORD_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "internal_correctness_passes": internal_correctness_passes,
            "external_reference_cases": external_reference_cases,
            "external_reference_passes": external_reference_passes,
            "counted_calibration_cases": counted_calibration_cases,
            "continuity_energy_cases": continuity_energy_cases,
            "continuity_energy_passes": continuity_energy_passes,
            "reference_backend": PLANNER_CALIBRATION_REFERENCE_BACKEND,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Calibration correctness bundle missing required fields: {}".format(
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
        help="Directory to write the calibration correctness bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        print(
            "cases={total_cases}, internal={internal_correctness_passes}, external={external_reference_passes}/{external_reference_cases}, counted={counted_calibration_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
