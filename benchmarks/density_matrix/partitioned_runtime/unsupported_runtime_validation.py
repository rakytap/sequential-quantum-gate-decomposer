#!/usr/bin/env python3
"""Validation: Phase 3 Task 3 Story 7 unsupported runtime boundary.

Exercises representative runtime-stage unsupported or incomplete execution
conditions and records the stable no-fallback outcome for each one.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/unsupported_runtime_validation.py
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

from benchmarks.density_matrix.partitioned_runtime.common import (
    build_initial_parameters,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    build_microcase_descriptor_set,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_SCHEMA_VERSION,
    execute_partitioned_density,
)

SUITE_NAME = "phase3_partitioned_runtime_unsupported_runtime"
ARTIFACT_FILENAME = "unsupported_runtime_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "unsupported_runtime"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
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
                "supported_runtime_case_recorded": False,
            }
        )
        return payload
    return {
        "case_name": case_name,
        "status": "unexpected_pass",
        "fallback_used": False,
        "supported_runtime_case_recorded": False,
        "reason": "unsupported runtime case unexpectedly passed",
    }


def _replace_member(descriptor_set, *, predicate, replacer):
    partitions = list(descriptor_set.partitions)
    for partition_index, partition in enumerate(partitions):
        members = list(partition.members)
        for member_index, member in enumerate(members):
            if predicate(member):
                members[member_index] = replacer(member)
                partitions[partition_index] = replace(partition, members=tuple(members))
                return replace(descriptor_set, partitions=tuple(partitions))
    raise ValueError("Unable to build requested unsupported runtime case")


def _wrong_requested_mode_runner():
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(continuity_vqe)
    bad_descriptor = replace(descriptor_set, requested_mode="state_vector")
    parameters = build_initial_parameters(bad_descriptor.parameter_count)
    return lambda: execute_partitioned_density(bad_descriptor, parameters)


def _parameter_count_mismatch_runner():
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(continuity_vqe)
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    return lambda: execute_partitioned_density(descriptor_set, parameters[:-1])


def _unsupported_gate_name_runner():
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    bad_descriptor = _replace_member(
        descriptor_set,
        predicate=lambda member: member.kind == "gate",
        replacer=lambda member: replace(member, name="CZ"),
    )
    parameters = build_initial_parameters(bad_descriptor.parameter_count)
    return lambda: execute_partitioned_density(bad_descriptor, parameters)


def _unsupported_noise_name_runner():
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    bad_descriptor = _replace_member(
        descriptor_set,
        predicate=lambda member: member.kind == "noise",
        replacer=lambda member: replace(member, name="depolarizing"),
    )
    parameters = build_initial_parameters(bad_descriptor.parameter_count)
    return lambda: execute_partitioned_density(bad_descriptor, parameters)


def _gate_fixed_value_runner():
    descriptor_set = build_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    bad_descriptor = _replace_member(
        descriptor_set,
        predicate=lambda member: member.kind == "gate",
        replacer=lambda member: replace(member, fixed_value=0.1),
    )
    parameters = build_initial_parameters(bad_descriptor.parameter_count)
    return lambda: execute_partitioned_density(bad_descriptor, parameters)


def build_cases() -> list[dict]:
    return [
        _unsupported_case("wrong_requested_mode", _wrong_requested_mode_runner()),
        _unsupported_case(
            "parameter_count_mismatch", _parameter_count_mismatch_runner()
        ),
        _unsupported_case("unsupported_gate_name", _unsupported_gate_name_runner()),
        _unsupported_case(
            "unsupported_noise_name", _unsupported_noise_name_runner()
        ),
        _unsupported_case("gate_fixed_value", _gate_fixed_value_runner()),
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
        "runtime_schema_version": PHASE3_RUNTIME_SCHEMA_VERSION,
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
            "Unsupported runtime bundle missing required fields: {}".format(
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
        help="Directory to write the unsupported runtime bundle into.",
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
            print("{case_name}: {status}".format(**case))
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
