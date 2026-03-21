#!/usr/bin/env python3
"""Validation: Phase 3 Task 2 Story 2 mandatory workload descriptors.

Builds one continuity-reference descriptor case plus the required 2 to 4 qubit
microcases and structured noisy `U3` / `CNOT` workload families, then verifies
that all of them emit the same Task 2 descriptor contract.

Run with:
    python benchmarks/density_matrix/planner_surface/mandatory_workload_descriptor_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    iter_microcase_descriptor_sets,
    iter_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import (
    PHASE3_DESCRIPTOR_SCHEMA_VERSION,
    build_phase3_continuity_partition_descriptor_set,
)

SUITE_NAME = "phase3_planner_surface_workload_descriptors"
ARTIFACT_FILENAME = "mandatory_workload_descriptor_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_surface"
    / "mandatory_workload"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "requested_mode",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def _descriptor_case(case_kind: str, metadata: dict, descriptor_set) -> dict:
    payload = descriptor_set.to_dict()
    return {
        "case_kind": case_kind,
        "case_name": payload["workload_id"],
        "status": "pass",
        "requested_mode": payload["requested_mode"],
        "schema_version": payload["schema_version"],
        "planner_schema_version": payload["planner_schema_version"],
        "source_type": payload["source_type"],
        "entry_route": payload["entry_route"],
        "workload_family": payload["workload_family"],
        "workload_id": payload["workload_id"],
        "qbit_num": payload["qbit_num"],
        "partition_count": payload["partition_count"],
        "descriptor_member_count": payload["descriptor_member_count"],
        "gate_count": payload["gate_count"],
        "noise_count": payload["noise_count"],
        "max_partition_qubits": payload["max_partition_qubits"],
        "max_partition_span": payload["max_partition_span"],
        "partition_member_counts": payload["partition_member_counts"],
        "descriptor_partitions": payload["partitions"],
        "metadata": metadata,
    }


def run_validation(verbose: bool = True) -> list[dict]:
    cases: list[dict] = []

    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    continuity_descriptor_set = build_phase3_continuity_partition_descriptor_set(
        continuity_vqe
    )
    cases.append(
        _descriptor_case(
            "continuity_reference",
            {"qbit_num": 4, "workload_id": continuity_descriptor_set.workload_id},
            continuity_descriptor_set,
        )
    )

    for metadata, descriptor_set in iter_microcase_descriptor_sets():
        cases.append(_descriptor_case("microcase", metadata, descriptor_set))
    for metadata, descriptor_set in iter_structured_descriptor_sets():
        cases.append(_descriptor_case("structured_family", metadata, descriptor_set))

    if verbose:
        print("Phase 3 Task 2 Story 2 mandatory workload descriptors:")
        for case in cases:
            print(
                "  {case_name}: q={qbit_num}, partitions={partition_count}, members={descriptor_member_count}".format(
                    **case
                )
            )
    return cases


def build_artifact_bundle(cases: list[dict]) -> dict:
    payload_key_sets = {frozenset(case.keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["descriptor_partitions"][0].keys())
        for case in cases
        if case["descriptor_partitions"]
    }
    member_key_sets = {
        frozenset(case["descriptor_partitions"][0]["members"][0].keys())
        for case in cases
        if case["descriptor_partitions"] and case["descriptor_partitions"][0]["members"]
    }
    passed_cases = sum(case["status"] == "pass" for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if passed_cases == len(cases)
        and len(payload_key_sets) == 1
        and len(partition_key_sets) == 1
        and len(member_key_sets) == 1
        else "fail",
        "requested_mode": "partitioned_density",
        "schema_version": PHASE3_DESCRIPTOR_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "continuity_reference_count": sum(
                case["case_kind"] == "continuity_reference" for case in cases
            ),
            "microcase_count": sum(case["case_kind"] == "microcase" for case in cases),
            "structured_family_count": sum(
                case["case_kind"] == "structured_family" for case in cases
            ),
            "required_structured_families": list(STRUCTURED_FAMILY_NAMES),
            "required_structured_qubits": list(STRUCTURED_QUBITS),
            "required_noise_patterns": list(MANDATORY_NOISE_PATTERNS),
            "payload_schema_count": len(payload_key_sets),
            "partition_schema_count": len(partition_key_sets),
            "member_schema_count": len(member_key_sets),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Mandatory workload descriptor bundle missing required fields: {}".format(
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
        help="Directory to write the mandatory workload descriptor bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    results = run_validation(verbose=not args.quiet)
    bundle = build_artifact_bundle(results)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
