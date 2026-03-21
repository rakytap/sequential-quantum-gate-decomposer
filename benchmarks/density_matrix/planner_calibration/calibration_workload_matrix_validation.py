#!/usr/bin/env python3
"""Validation: Phase 3 Task 5 Story 2 workload-matrix surface.

Builds the full Task 5 candidate-plus-workload inventory and records stable
workload provenance for the mandatory continuity, microcase, and structured
workload classes.

Run with:
    python benchmarks/density_matrix/planner_calibration/calibration_workload_matrix_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_calibration.common import (
    PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION,
    PLANNER_CALIBRATION_SUPPORTED_CANDIDATE_PARTITION_QUBITS,
)
from benchmarks.density_matrix.planner_calibration.case_selection import (
    PLANNER_CALIBRATION_CASE_KIND_CONTINUITY,
    PLANNER_CALIBRATION_CASE_KIND_MICROCASE,
    PLANNER_CALIBRATION_CASE_KIND_STRUCTURED,
    PLANNER_CALIBRATION_CONTINUITY_QUBITS,
    iter_planner_calibration_candidate_cases,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from benchmarks.density_matrix.planner_surface.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    mandatory_microcase_definitions,
)

SUITE_NAME = "phase3_planner_calibration_workload_matrix"
ARTIFACT_FILENAME = "calibration_workload_matrix_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_calibration"
    / "calibration_workload_matrix"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "candidate_schema_version",
    "software",
    "summary",
    "cases",
)


def _case_from_descriptor(metadata: dict, descriptor_set) -> dict:
    payload = descriptor_set.to_dict()
    return {
        "candidate_id": metadata["candidate_id"],
        "planner_family": metadata["planner_family"],
        "planner_variant": metadata["planner_variant"],
        "max_partition_qubits": metadata["max_partition_qubits"],
        "case_kind": metadata["case_kind"],
        "source_type": payload["source_type"],
        "entry_route": payload["entry_route"],
        "workload_family": payload["workload_family"],
        "workload_id": payload["workload_id"],
        "qbit_num": payload["qbit_num"],
        "family_name": metadata["family_name"],
        "noise_pattern": metadata["noise_pattern"],
        "seed": metadata["seed"],
        "planning_time_ms": metadata["planning_time_ms"],
        "partition_count": payload["partition_count"],
        "descriptor_member_count": payload["descriptor_member_count"],
        "max_partition_span": payload["max_partition_span"],
        "partition_member_counts": payload["partition_member_counts"],
        "workload_matrix_pass": (
            metadata["max_partition_qubits"] in PLANNER_CALIBRATION_SUPPORTED_CANDIDATE_PARTITION_QUBITS
            and payload["partition_count"] > 0
            and payload["workload_id"] == metadata["workload_id"]
        ),
    }


def build_cases() -> list[dict]:
    return [
        _case_from_descriptor(metadata, descriptor_set)
        for metadata, descriptor_set, _, _ in iter_planner_calibration_candidate_cases()
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    candidate_ids = sorted({case["candidate_id"] for case in cases})
    continuity_cases = [
        case for case in cases if case["case_kind"] == PLANNER_CALIBRATION_CASE_KIND_CONTINUITY
    ]
    microcase_cases = [case for case in cases if case["case_kind"] == PLANNER_CALIBRATION_CASE_KIND_MICROCASE]
    structured_cases = [
        case for case in cases if case["case_kind"] == PLANNER_CALIBRATION_CASE_KIND_STRUCTURED
    ]
    microcase_workload_ids = {
        definition["case_name"] for definition in mandatory_microcase_definitions()
    }
    expected_structured_combinations = {
        (family_name, qbit_num, noise_pattern)
        for family_name in STRUCTURED_FAMILY_NAMES
        for qbit_num in STRUCTURED_QUBITS
        for noise_pattern in MANDATORY_NOISE_PATTERNS
    }
    observed_structured_combinations = {
        (case["family_name"], case["qbit_num"], case["noise_pattern"])
        for case in structured_cases
    }
    workload_matrix_passes = sum(case["workload_matrix_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if workload_matrix_passes == len(cases)
        and {case["qbit_num"] for case in continuity_cases} == set(PLANNER_CALIBRATION_CONTINUITY_QUBITS)
        and {case["workload_id"] for case in microcase_cases} == microcase_workload_ids
        and observed_structured_combinations == expected_structured_combinations
        else "fail",
        "candidate_schema_version": PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "workload_matrix_passes": workload_matrix_passes,
            "candidate_ids": candidate_ids,
            "continuity_cases": len(continuity_cases),
            "microcases": len(microcase_cases),
            "structured_cases": len(structured_cases),
            "continuity_qubits": sorted({case["qbit_num"] for case in continuity_cases}),
            "microcase_workload_ids": sorted({case["workload_id"] for case in microcase_cases}),
            "structured_family_names": sorted(
                {case["family_name"] for case in structured_cases}
            ),
            "structured_noise_patterns": sorted(
                {case["noise_pattern"] for case in structured_cases}
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 5 Story 2 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 5 Story 2 bundle into.",
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
            "cases={total_cases}, continuity={continuity_cases}, microcases={microcases}, structured={structured_cases}".format(
                **bundle["summary"]
            )
        )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
