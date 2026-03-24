#!/usr/bin/env python3
"""Unsupported partition-descriptor boundary validation.

Exercises representative unsupported or lossy descriptor-generation situations
and records the stable no-fallback outcome for each one. Unsupported descriptor
requests fail before runtime and are not silently relabeled as supported
descriptor behavior.

Run with:
    python benchmarks/density_matrix/planner_surface/unsupported_descriptor_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
)
from squander.partitioning.noisy_planner import (
    DESCRIPTOR_SCHEMA_VERSION,
    PLANNER_OP_KIND_GATE,
    PLANNER_OP_KIND_NOISE,
    preflight_descriptor_request,
    validate_partition_descriptor_set_against_surface,
)

SUITE_NAME = "phase3_planner_surface_unsupported_descriptors"
ARTIFACT_FILENAME = "unsupported_descriptor_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "planner_surface"
    / "unsupported_descriptor"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def _unsupported_case(case_name: str, runner) -> dict:
    try:
        runner()
    except Exception as err:
        payload = err.to_dict() if hasattr(err, "to_dict") else {"reason": str(err)}
        payload.update(
            {
                "case_name": case_name,
                "status": "unsupported",
                "fallback_used": False,
                "supported_descriptor_case_recorded": False,
            }
        )
        return payload
    return {
        "case_name": case_name,
        "status": "unexpected_pass",
        "fallback_used": False,
        "supported_descriptor_case_recorded": False,
        "reason": "unsupported descriptor case unexpectedly passed",
    }


def _build_continuity_bridge_request():
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    bridge = continuity_vqe.describe_density_bridge()
    return continuity_vqe, bridge


def _drop_last_descriptor_member_runner():
    surface = build_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    partitions = list(descriptor_set.partitions)
    last_partition = partitions[-1]
    partitions[-1] = replace(
        last_partition,
        canonical_operation_indices=last_partition.canonical_operation_indices[:-1],
        members=last_partition.members[:-1],
    )
    tampered = replace(descriptor_set, partitions=tuple(partitions))
    return lambda: validate_partition_descriptor_set_against_surface(surface, tampered)


def _hide_noise_member_runner():
    surface = build_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    partitions = list(descriptor_set.partitions)
    for partition_index, partition in enumerate(partitions):
        for member_index, member in enumerate(partition.members):
            if member.kind == PLANNER_OP_KIND_NOISE:
                members = list(partition.members)
                members[member_index] = replace(
                    member,
                    kind=PLANNER_OP_KIND_GATE,
                    is_unitary=True,
                )
                partitions[partition_index] = replace(partition, members=tuple(members))
                tampered = replace(descriptor_set, partitions=tuple(partitions))
                return lambda: validate_partition_descriptor_set_against_surface(
                    surface, tampered
                )
    raise ValueError("No noise member available for hidden-noise negative case")


def _incomplete_remap_runner():
    surface = build_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    partitions = list(descriptor_set.partitions)
    for partition_index, partition in enumerate(partitions):
        if not partition.requires_remap or len(partition.local_to_global_qbits) < 2:
            continue
        for member_index, member in enumerate(partition.members):
            if member.local_target_qbit is None:
                continue
            wrong_local_target = (member.local_target_qbit + 1) % len(
                partition.local_to_global_qbits
            )
            if wrong_local_target == member.local_target_qbit:
                continue
            members = list(partition.members)
            members[member_index] = replace(
                member, local_target_qbit=wrong_local_target
            )
            partitions[partition_index] = replace(partition, members=tuple(members))
            tampered = replace(descriptor_set, partitions=tuple(partitions))
            return lambda: validate_partition_descriptor_set_against_surface(
                surface, tampered
            )
    raise ValueError("No remapped partition available for incomplete-remap case")


def _ambiguous_parameter_routing_runner():
    surface = build_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    partitions = list(descriptor_set.partitions)
    for partition_index, partition in enumerate(partitions):
        for member_index, member in enumerate(partition.members):
            if member.param_count <= 0:
                continue
            members = list(partition.members)
            members[member_index] = replace(
                member, local_param_start=member.local_param_start + 1
            )
            partitions[partition_index] = replace(partition, members=tuple(members))
            tampered = replace(descriptor_set, partitions=tuple(partitions))
            return lambda: validate_partition_descriptor_set_against_surface(
                surface, tampered
            )
    raise ValueError("No parameterized member available for ambiguous-routing case")


def _reordered_descriptor_runner():
    surface = build_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    partitions = list(descriptor_set.partitions)
    for partition_index, partition in enumerate(partitions):
        if partition.noise_count == 0 or partition.member_count < 2:
            continue
        members = list(partition.members)
        members[0], members[1] = members[1], members[0]
        partitions[partition_index] = replace(partition, members=tuple(members))
        tampered = replace(descriptor_set, partitions=tuple(partitions))
        return lambda: validate_partition_descriptor_set_against_surface(
            surface, tampered
        )
    raise ValueError("No mixed partition available for reordering negative case")


def build_unsupported_descriptor_cases() -> list[dict]:
    _, continuity_bridge = _build_continuity_bridge_request()
    return [
        _unsupported_case(
            "partition_qubits_too_small",
            lambda: preflight_descriptor_request(
                source_type="generated_hea",
                workload_id="phase2_xxz_hea_q4_continuity",
                bridge_metadata=continuity_bridge,
                max_partition_qubits=1,
            ),
        ),
        _unsupported_case("dropped_operation", _drop_last_descriptor_member_runner()),
        _unsupported_case("hidden_noise_placement", _hide_noise_member_runner()),
        _unsupported_case("incomplete_remapping", _incomplete_remap_runner()),
        _unsupported_case(
            "ambiguous_parameter_routing", _ambiguous_parameter_routing_runner()
        ),
        _unsupported_case(
            "reordering_across_noise_boundaries", _reordered_descriptor_runner()
        ),
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    unsupported_cases = sum(case["status"] == "unsupported" for case in cases)
    unexpected_passes = sum(case["status"] == "unexpected_pass" for case in cases)
    fallback_count = sum(bool(case["fallback_used"]) for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if unsupported_cases == len(cases)
        and unexpected_passes == 0
        and fallback_count == 0
        else "fail",
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "unsupported_cases": unsupported_cases,
            "unexpected_passes": unexpected_passes,
            "fallback_count": fallback_count,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Unsupported descriptor bundle missing required fields: {}".format(
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
        help="Directory to write the unsupported descriptor bundle into.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_unsupported_descriptor_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, output_dir=args.output_dir)

    if not args.quiet:
        for case in cases:
            print("{case_name}: {status}".format(**case))
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
