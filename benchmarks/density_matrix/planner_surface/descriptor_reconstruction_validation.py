#!/usr/bin/env python3
"""Partition-descriptor reconstruction validation.

Emits reconstruction-oriented audits for one continuity case, the boundary-heavy
4-qubit microcase, and one structured-family case. Verifies that partition
descriptors expose reviewable qubit-support, remapping, and parameter-routing
metadata.

Run with:
    python benchmarks/density_matrix/planner_surface/descriptor_reconstruction_validation.py
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
    build_structured_descriptor_set,
    iter_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import (
    build_phase3_continuity_partition_descriptor_set,
    validate_partition_descriptor_set,
)

SUITE_NAME = "phase3_planner_surface_descriptor_reconstruction"
ARTIFACT_FILENAME = "descriptor_reconstruction_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_surface"
    / "descriptor_reconstruction"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "requested_mode",
    "software",
    "summary",
    "cases",
)


def _reconstruction_case(case_kind: str, metadata: dict, descriptor_set) -> dict:
    validated = validate_partition_descriptor_set(descriptor_set)
    payload = validated.to_dict()
    return {
        "case_kind": case_kind,
        "case_name": payload["workload_id"],
        "status": "pass",
        "reconstruction_pass": True,
        "requested_mode": payload["requested_mode"],
        "source_type": payload["source_type"],
        "workload_id": payload["workload_id"],
        "qbit_num": payload["qbit_num"],
        "partition_count": payload["partition_count"],
        "descriptor_member_count": payload["descriptor_member_count"],
        "remapped_partition_count": sum(
            partition["requires_remap"] for partition in payload["partitions"]
        ),
        "parameter_routing_segment_count": sum(
            len(partition["parameter_routing"]) for partition in payload["partitions"]
        ),
        "descriptor_partitions": payload["partitions"],
        "metadata": metadata,
    }


def run_validation(verbose: bool = True) -> list[dict]:
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    continuity_case = _reconstruction_case(
        "continuity",
        {"qbit_num": 4},
        build_phase3_continuity_partition_descriptor_set(continuity_vqe),
    )

    boundary_case = _reconstruction_case(
        "boundary_microcase",
        {"case_name": "microcase_4q_partition_boundary_triplet", "qbit_num": 4},
        build_microcase_descriptor_set("microcase_4q_partition_boundary_triplet"),
    )

    structured_metadata, _ = next(iter(iter_structured_descriptor_sets()))
    structured_case = _reconstruction_case(
        "structured_family",
        structured_metadata,
        build_structured_descriptor_set(
            structured_metadata["family_name"],
            qbit_num=structured_metadata["qbit_num"],
            noise_pattern=structured_metadata["noise_pattern"],
            seed=structured_metadata["seed"],
        ),
    )

    cases = [continuity_case, boundary_case, structured_case]
    if verbose:
        print("Partition-descriptor reconstruction:")
        for case in cases:
            print(
                "  {case_name}: remapped_partitions={remapped_partition_count}, parameter_segments={parameter_routing_segment_count}, reconstruction_pass={reconstruction_pass}".format(
                    **case
                )
            )
    return cases


def build_artifact_bundle(cases: list[dict]) -> dict:
    reconstruction_passes = sum(case["reconstruction_pass"] for case in cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if passed_cases == len(cases) and reconstruction_passes == len(cases)
        else "fail",
        "requested_mode": "partitioned_density",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "reconstruction_passes": reconstruction_passes,
            "continuity_cases": sum(
                case["case_kind"] == "continuity" for case in cases
            ),
            "boundary_microcases": sum(
                case["case_kind"] == "boundary_microcase" for case in cases
            ),
            "structured_family_cases": sum(
                case["case_kind"] == "structured_family" for case in cases
            ),
            "remapped_partition_count": sum(
                case["remapped_partition_count"] for case in cases
            ),
            "parameter_routing_segment_count": sum(
                case["parameter_routing_segment_count"] for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Descriptor reconstruction bundle missing required fields: {}".format(
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
        help="Directory to write the descriptor reconstruction bundle into.",
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
