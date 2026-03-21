#!/usr/bin/env python3
"""Validation: Phase 3 Task 3 Story 4 runtime semantic preservation.

Exercises representative supported cases that stress remapping, parameter
routing, and explicit noise placement, then checks that the partitioned runtime
matches the sequential descriptor-driven reference exactly within the frozen
tolerance.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/runtime_semantics_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    build_initial_parameters,
    execute_partitioned_with_reference,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    build_microcase_descriptor_set,
    iter_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set
from squander.partitioning.noisy_runtime import PHASE3_RUNTIME_SCHEMA_VERSION

SUITE_NAME = "phase3_partitioned_runtime_runtime_semantics"
ARTIFACT_FILENAME = "runtime_semantics_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "runtime_semantics"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)


def _semantic_case(*, case_name: str, case_kind: str, descriptor_set, metadata: dict | None = None):
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    runtime_result, _, density_metrics = execute_partitioned_with_reference(
        descriptor_set, parameters
    )
    runtime_payload = runtime_result.to_dict(include_density_matrix=False)
    case = {
        "case_name": case_name,
        "case_kind": case_kind,
        "status": "pass"
        if density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.rho_is_valid
        else "fail",
        "runtime_schema_version": runtime_payload["runtime_schema_version"],
        "planner_schema_version": runtime_payload["planner_schema_version"],
        "descriptor_schema_version": runtime_payload["descriptor_schema_version"],
        "workload_id": runtime_payload["workload_id"],
        "qbit_num": runtime_payload["qbit_num"],
        "partition_count": runtime_payload["summary"]["partition_count"],
        "remapped_partition_count": runtime_payload["summary"]["remapped_partition_count"],
        "parameter_routing_segment_count": runtime_payload["summary"][
            "parameter_routing_segment_count"
        ],
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
        "rho_is_valid": runtime_payload["summary"]["rho_is_valid"],
        "trace_deviation": runtime_payload["summary"]["trace_deviation"],
        "runtime_semantics_pass": (
            runtime_payload["runtime_schema_version"] == PHASE3_RUNTIME_SCHEMA_VERSION
            and density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and runtime_payload["summary"]["parameter_routing_segment_count"] > 0
        ),
        "runtime_partitions": runtime_payload["partitions"],
        "metadata": dict(metadata) if metadata is not None else {},
    }
    return case


def build_cases() -> list[dict]:
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    continuity_case = _semantic_case(
        case_name="phase2_xxz_hea_q4_continuity",
        case_kind="continuity",
        descriptor_set=build_phase3_continuity_partition_descriptor_set(continuity_vqe),
    )
    boundary_microcase = _semantic_case(
        case_name="microcase_4q_partition_boundary_triplet",
        case_kind="microcase",
        descriptor_set=build_microcase_descriptor_set(
            "microcase_4q_partition_boundary_triplet"
        ),
    )
    structured_metadata, structured_descriptor_set = next(
        iter(iter_structured_descriptor_sets())
    )
    structured_case = _semantic_case(
        case_name=structured_metadata["workload_id"],
        case_kind="structured_family",
        descriptor_set=structured_descriptor_set,
        metadata=structured_metadata,
    )
    return [continuity_case, boundary_microcase, structured_case]


def build_artifact_bundle(cases: list[dict]) -> dict:
    semantics_passes = sum(case["runtime_semantics_pass"] for case in cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if semantics_passes == len(cases) and passed_cases == len(cases)
        else "fail",
        "runtime_schema_version": PHASE3_RUNTIME_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "runtime_semantics_passes": semantics_passes,
            "cases_with_remap": sum(case["remapped_partition_count"] > 0 for case in cases),
            "cases_with_parameter_routing": sum(
                case["parameter_routing_segment_count"] > 0 for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Runtime semantics bundle missing required fields: {}".format(
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
        help="Directory to write the runtime semantics bundle into.",
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
                "{case_name}: dRho={frobenius_norm_diff:.3e}, remap={remapped_partition_count}, routing={parameter_routing_segment_count}, pass={runtime_semantics_pass}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
