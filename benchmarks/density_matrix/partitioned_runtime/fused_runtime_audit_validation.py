#!/usr/bin/env python3
"""Validation: Phase 3 Task 4 Story 6 fused runtime audit surface.

Emits one shared fused-capable runtime-audit record across representative
supported cases and checks that fused-region and summary schemas remain stable.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/fused_runtime_audit_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from benchmarks.density_matrix.partitioned_runtime.fusion_case_selection import (
    iter_fusion_continuity_cases,
    iter_fusion_microcase_cases,
    iter_fusion_structured_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    PHASE3_RUNTIME_SCHEMA_VERSION,
    build_runtime_audit_record,
    execute_partitioned_density_fused,
)

SUITE_NAME = "phase3_partitioned_runtime_fused_runtime_audit"
ARTIFACT_FILENAME = "fused_runtime_audit_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "story6_fused_audit"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)


def _first_case(case_iter):
    return next(iter(case_iter))


def _audit_case(metadata: dict, descriptor_set, parameters) -> dict:
    result = execute_partitioned_density_fused(descriptor_set, parameters)
    audit = build_runtime_audit_record(result, metadata={"case_kind": metadata["case_kind"]})
    audit["case_name"] = metadata["case_name"] if "case_name" in metadata else metadata["workload_id"]
    return audit


def build_cases() -> list[dict]:
    continuity_metadata, continuity_descriptor_set, continuity_parameters, _ = _first_case(
        iter_fusion_continuity_cases()
    )
    micro_metadata, micro_descriptor_set, micro_parameters = _first_case(
        iter_fusion_microcase_cases()
    )
    structured_metadata, structured_descriptor_set, structured_parameters = _first_case(
        iter_fusion_structured_cases()
    )
    return [
        _audit_case(continuity_metadata, continuity_descriptor_set, continuity_parameters),
        _audit_case(micro_metadata, micro_descriptor_set, micro_parameters),
        _audit_case(structured_metadata, structured_descriptor_set, structured_parameters),
    ]


def build_artifact_bundle(cases: list[dict]) -> dict:
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    partition_key_sets = {
        frozenset(case["partitions"][0].keys()) for case in cases if case["partitions"]
    }
    fused_region_key_sets = {
        frozenset(case["fused_regions"][0].keys())
        for case in cases
        if case["fused_regions"]
    }
    runtime_paths = {case["runtime_path"] for case in cases}
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if len(top_level_keys) == 1
        and len(summary_key_sets) == 1
        and len(partition_key_sets) == 1
        and len(fused_region_key_sets) == 1
        and runtime_paths
        <= {
            PHASE3_RUNTIME_PATH_BASELINE,
            PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
        }
        else "fail",
        "runtime_schema_version": PHASE3_RUNTIME_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "top_level_schema_count": len(top_level_keys),
            "summary_schema_count": len(summary_key_sets),
            "partition_schema_count": len(partition_key_sets),
            "fused_region_schema_count": len(fused_region_key_sets),
            "runtime_path_count": len(runtime_paths),
            "actual_fused_cases": sum(
                case["summary"]["actual_fused_execution"] for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 4 Story 6 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 4 Story 6 bundle into.",
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
                "{case_name}: runtime_path={runtime_path}, fused_regions={fused_region_count}".format(
                    case_name=case["case_name"],
                    runtime_path=case["runtime_path"],
                    fused_region_count=case["summary"]["fused_region_count"],
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
