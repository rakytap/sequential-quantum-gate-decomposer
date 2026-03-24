#!/usr/bin/env python3
"""Partition-descriptor ordering and noise placement validation.

Emits order-and-noise audits for one continuity case, the boundary-heavy 4-qubit
microcase, and one structured-family case. Verifies that partition descriptors
preserve exact within-partition order and keep noise placement explicit as
first-class descriptor content.

Run with:
    python benchmarks/density_matrix/planner_surface/descriptor_ordering_validation.py
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
    build_microcase_descriptor_set,
    build_microcase_surface,
    build_structured_descriptor_set,
    build_structured_surface,
    iter_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import (
    DESCRIPTOR_SCHEMA_VERSION,
    PLANNER_OP_KIND_NOISE,
    build_phase3_continuity_partition_descriptor_set,
    build_phase3_continuity_planner_surface,
)

SUITE_NAME = "phase3_planner_surface_descriptor_ordering"
ARTIFACT_FILENAME = "descriptor_ordering_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_surface"
    / "descriptor_ordering"
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


def _flatten_canonical_indices(payload: dict) -> list[int]:
    return [
        member["canonical_operation_index"]
        for partition in payload["partitions"]
        for member in partition["members"]
    ]


def _flatten_member_names(payload: dict) -> list[str]:
    return [
        member["name"]
        for partition in payload["partitions"]
        for member in partition["members"]
    ]


def _case_from_surface(case_kind: str, metadata: dict, surface, descriptor_set) -> dict:
    surface_payload = surface.to_dict()
    descriptor_payload = descriptor_set.to_dict()
    expected_names = [operation["name"] for operation in surface_payload["operations"]]
    actual_names = _flatten_member_names(descriptor_payload)
    ordering_pass = (
        descriptor_payload["schema_version"] == DESCRIPTOR_SCHEMA_VERSION
        and _flatten_canonical_indices(descriptor_payload)
        == list(range(surface_payload["operation_count"]))
        and actual_names == expected_names
        and descriptor_payload["noise_count"]
        == sum(
            member["kind"] == PLANNER_OP_KIND_NOISE
            for partition in descriptor_payload["partitions"]
            for member in partition["members"]
        )
    )

    return {
        "case_kind": case_kind,
        "case_name": descriptor_payload["workload_id"],
        "status": "pass",
        "ordering_pass": ordering_pass,
        "requested_mode": descriptor_payload["requested_mode"],
        "schema_version": descriptor_payload["schema_version"],
        "planner_schema_version": descriptor_payload["planner_schema_version"],
        "source_type": descriptor_payload["source_type"],
        "workload_id": descriptor_payload["workload_id"],
        "qbit_num": descriptor_payload["qbit_num"],
        "partition_count": descriptor_payload["partition_count"],
        "descriptor_member_count": descriptor_payload["descriptor_member_count"],
        "gate_count": descriptor_payload["gate_count"],
        "noise_count": descriptor_payload["noise_count"],
        "boundary_mixed_partition_count": sum(
            partition["gate_count"] > 0 and partition["noise_count"] > 0
            for partition in descriptor_payload["partitions"]
        ),
        "expected_name_sequence": expected_names,
        "actual_name_sequence": actual_names,
        "descriptor_partitions": descriptor_payload["partitions"],
        "metadata": metadata,
    }


def run_validation(verbose: bool = True) -> list[dict]:
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    continuity_case = _case_from_surface(
        "continuity",
        {"qbit_num": 4, "workload_id": "phase2_xxz_hea_q4_continuity"},
        build_phase3_continuity_planner_surface(continuity_vqe),
        build_phase3_continuity_partition_descriptor_set(continuity_vqe),
    )

    boundary_surface = build_microcase_surface(
        "microcase_4q_partition_boundary_triplet"
    )
    boundary_descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    boundary_case = _case_from_surface(
        "boundary_microcase",
        {
            "case_name": "microcase_4q_partition_boundary_triplet",
            "noise_pattern": "dense",
            "qbit_num": 4,
        },
        boundary_surface,
        boundary_descriptor_set,
    )

    structured_metadata, _ = next(iter(iter_structured_descriptor_sets()))
    structured_surface = build_structured_surface(
        structured_metadata["family_name"],
        qbit_num=structured_metadata["qbit_num"],
        noise_pattern=structured_metadata["noise_pattern"],
        seed=structured_metadata["seed"],
    )
    structured_descriptor_set = build_structured_descriptor_set(
        structured_metadata["family_name"],
        qbit_num=structured_metadata["qbit_num"],
        noise_pattern=structured_metadata["noise_pattern"],
        seed=structured_metadata["seed"],
    )
    structured_case = _case_from_surface(
        "structured_family",
        structured_metadata,
        structured_surface,
        structured_descriptor_set,
    )

    cases = [continuity_case, boundary_case, structured_case]
    if verbose:
        print("Descriptor ordering and noise placement:")
        for case in cases:
            print(
                "  {case_name}: members={descriptor_member_count}, mixed_partitions={boundary_mixed_partition_count}, ordering_pass={ordering_pass}".format(
                    **case
                )
            )
    return cases


def build_artifact_bundle(cases: list[dict]) -> dict:
    ordering_passes = sum(case["ordering_pass"] for case in cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if passed_cases == len(cases) and ordering_passes == len(cases)
        else "fail",
        "requested_mode": "partitioned_density",
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "ordering_passes": ordering_passes,
            "continuity_cases": sum(
                case["case_kind"] == "continuity" for case in cases
            ),
            "boundary_microcases": sum(
                case["case_kind"] == "boundary_microcase" for case in cases
            ),
            "structured_family_cases": sum(
                case["case_kind"] == "structured_family" for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Descriptor ordering bundle missing required fields: {}".format(
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
        help="Directory to write the descriptor ordering bundle into.",
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
