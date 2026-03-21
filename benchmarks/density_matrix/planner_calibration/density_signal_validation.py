#!/usr/bin/env python3
"""Validation: Phase 3 Task 5 Story 3 density-aware signal surface.

Builds a representative contrast matrix across the supported Task 5 candidate
settings, records state-vector-style proxy scores alongside benchmark-grounded
density-aware scores, and shows that explicit noise placement changes the
recorded density-aware score on real noisy mixed-state workloads.

Run with:
    python benchmarks/density_matrix/planner_calibration/density_signal_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    execute_partitioned_with_reference,
)
from benchmarks.density_matrix.planner_calibration.signals import (
    PLANNER_CALIBRATION_DENSITY_AWARE_OBJECTIVE_NAME,
    PLANNER_CALIBRATION_DENSITY_SIGNAL_SCHEMA_VERSION,
    apply_density_scores_and_rankings,
    build_density_signal_record,
)
from benchmarks.density_matrix.planner_calibration.case_selection import (
    iter_density_signal_cases,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from squander.partitioning.noisy_runtime import PHASE3_RUNTIME_PATH_BASELINE

SUITE_NAME = "phase3_planner_calibration_density_signal"
ARTIFACT_FILENAME = "density_signal_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "density_signal"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "signal_schema_version",
    "software",
    "summary",
    "cases",
)


def _signal_case(metadata: dict, descriptor_set, parameters) -> dict:
    runtime_result, _, density_metrics = execute_partitioned_with_reference(
        descriptor_set, parameters, allow_fusion=False
    )
    return build_density_signal_record(
        metadata,
        descriptor_set,
        runtime_result,
        density_metrics,
    )


def _finalize_signal_records(records: list[dict]) -> list[dict]:
    apply_density_scores_and_rankings(
        records, objective_name=PLANNER_CALIBRATION_DENSITY_AWARE_OBJECTIVE_NAME
    )
    for record in records:
        record["density_signal_pass"] = (
            record["signal_schema_version"] == PLANNER_CALIBRATION_DENSITY_SIGNAL_SCHEMA_VERSION
            and record["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
            and record["partition_count"] > 0
            and record["state_vector_proxy_score"] > 0.0
            and record["density_aware_score"] > 0.0
            and record["density_aware_rank"] >= 1
            and record["state_vector_proxy_rank"] >= 1
        )
    return records


def _noise_sensitive_slice_count(records: list[dict]) -> int:
    grouped_scores: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["noise_pattern"] is None:
            continue
        key = (
            record["candidate_id"],
            record["family_name"] or record["workload_family"],
            record["qbit_num"],
        )
        grouped_scores[key][record["noise_pattern"]] = record["density_aware_score"]

    return sum(
        len(score_by_pattern) > 1
        and len({round(score, 12) for score in score_by_pattern.values()}) > 1
        for score_by_pattern in grouped_scores.values()
    )


def build_cases() -> list[dict]:
    records = [
        _signal_case(metadata, descriptor_set, parameters)
        for metadata, descriptor_set, parameters, _ in iter_density_signal_cases()
    ]
    return _finalize_signal_records(records)


def build_artifact_bundle(cases: list[dict]) -> dict:
    density_signal_passes = sum(case["density_signal_pass"] for case in cases)
    noise_sensitive_slices = _noise_sensitive_slice_count(cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if density_signal_passes == len(cases) and noise_sensitive_slices >= 1
        else "fail",
        "signal_schema_version": PLANNER_CALIBRATION_DENSITY_SIGNAL_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "density_signal_passes": density_signal_passes,
            "candidate_ids": sorted({case["candidate_id"] for case in cases}),
            "workload_ids": sorted({case["workload_id"] for case in cases}),
            "noise_patterns": sorted(
                {case["noise_pattern"] for case in cases if case["noise_pattern"] is not None}
            ),
            "noise_sensitive_slices": noise_sensitive_slices,
            "calibrated_memory_weight": cases[0]["calibrated_memory_weight"]
            if cases
            else 0.0,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 5 Story 3 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 5 Story 3 bundle into.",
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
        for case in cases:
            print(
                "{workload_id} {candidate_id}: density_score={density_aware_score:.3f}, proxy_score={state_vector_proxy_score:.3f}, rank={density_aware_rank}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
