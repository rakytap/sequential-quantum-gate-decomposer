#!/usr/bin/env python3
"""Validation: Phase 3 Task 3 Story 6 runtime audit surface.

Emits one shared runtime-audit record across supported workload classes and
checks that provenance, runtime-path labels, summary fields, and partition
record shapes remain stable.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/runtime_audit_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import (
    build_phase3_story1_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    iter_story2_microcase_descriptor_sets,
    iter_story2_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_SCHEMA_VERSION,
    build_runtime_audit_record,
    execute_partitioned_density,
)

SUITE_NAME = "phase3_task3_story6_runtime_audit"
ARTIFACT_FILENAME = "runtime_audit_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "phase3_task3"
    / "story6_audit"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)


def _audit_case(descriptor_set, metadata: dict) -> dict:
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    runtime_result = execute_partitioned_density(descriptor_set, parameters)
    audit = build_runtime_audit_record(runtime_result, metadata=metadata)
    audit["case_name"] = audit["provenance"]["workload_id"]
    return audit


def build_cases() -> list[dict]:
    continuity_vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    continuity_audit = _audit_case(
        build_phase3_continuity_partition_descriptor_set(continuity_vqe),
        {"case_kind": "continuity"},
    )
    micro_metadata, micro_descriptor_set = next(iter(iter_story2_microcase_descriptor_sets()))
    micro_audit = _audit_case(micro_descriptor_set, micro_metadata)
    structured_metadata, structured_descriptor_set = next(
        iter(iter_story2_structured_descriptor_sets())
    )
    structured_audit = _audit_case(structured_descriptor_set, structured_metadata)
    return [continuity_audit, micro_audit, structured_audit]


def build_artifact_bundle(cases: list[dict]) -> dict:
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }
    runtime_paths = {case["runtime_path"] for case in cases}
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if len(top_level_keys) == 1
        and len(summary_key_sets) == 1
        and len(partition_key_sets) == 1
        and len(runtime_paths) == 1
        else "fail",
        "runtime_schema_version": PHASE3_RUNTIME_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "top_level_schema_count": len(top_level_keys),
            "summary_schema_count": len(summary_key_sets),
            "partition_schema_count": len(partition_key_sets),
            "runtime_path_count": len(runtime_paths),
            "continuity_cases": sum(
                case["metadata"].get("case_kind") == "continuity" for case in cases
            ),
            "methods_cases": sum(
                case["metadata"].get("case_kind")
                in {"microcase", "structured_family"}
                for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Runtime audit bundle missing required fields: {}".format(
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
        help="Directory to write the runtime audit bundle into.",
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
                "{case_name}: runtime_path={runtime_path}, partitions={partition_count}".format(
                    case_name=case["case_name"],
                    runtime_path=case["runtime_path"],
                    partition_count=case["summary"]["partition_count"],
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
