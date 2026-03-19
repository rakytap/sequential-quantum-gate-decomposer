#!/usr/bin/env python3
"""Validation: Phase 3 Task 4 Story 2 structured fused runtime slice.

Finds one representative 8-qubit and one representative 10-qubit structured
workload that execute through the real fused unitary-island runtime path.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/structured_fused_runtime_validation.py
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
    execute_fused_with_reference,
)
from benchmarks.density_matrix.partitioned_runtime.task4_case_selection import (
    iter_task4_structured_cases,
)
from benchmarks.density_matrix.planner_surface.common import build_software_metadata
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    PHASE3_RUNTIME_SCHEMA_VERSION,
)

SUITE_NAME = "phase3_task4_story2_structured_fused_runtime"
ARTIFACT_FILENAME = "structured_fused_runtime_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "phase3_task4"
    / "story2_structured_fused_runtime"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)


def _fused_case(metadata: dict, descriptor_set, parameters) -> dict:
    runtime_result, _, density_metrics = execute_fused_with_reference(
        descriptor_set, parameters
    )
    payload = runtime_result.to_dict(include_density_matrix=False)
    return {
        "case_name": metadata["workload_id"],
        "case_kind": metadata["case_kind"],
        "family_name": metadata["family_name"],
        "qbit_num": metadata["qbit_num"],
        "noise_pattern": metadata["noise_pattern"],
        "runtime_schema_version": payload["runtime_schema_version"],
        "runtime_path": payload["runtime_path"],
        "partition_count": payload["summary"]["partition_count"],
        "fused_region_count": payload["summary"]["fused_region_count"],
        "supported_unfused_region_count": payload["summary"][
            "supported_unfused_region_count"
        ],
        "deferred_region_count": payload["summary"]["deferred_region_count"],
        "fused_gate_count": payload["summary"]["fused_gate_count"],
        "actual_fused_execution": payload["summary"]["actual_fused_execution"],
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
        "structured_fused_runtime_pass": (
            payload["runtime_schema_version"] == PHASE3_RUNTIME_SCHEMA_VERSION
            and payload["runtime_path"] == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
            and payload["summary"]["actual_fused_execution"] is True
            and payload["summary"]["fused_region_count"] > 0
            and density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        ),
        "metadata": dict(metadata),
    }


def build_cases() -> list[dict]:
    selected_by_qubits: dict[int, dict] = {}
    for metadata, descriptor_set, parameters in iter_task4_structured_cases():
        if metadata["qbit_num"] in selected_by_qubits:
            continue
        case = _fused_case(metadata, descriptor_set, parameters)
        if case["structured_fused_runtime_pass"]:
            selected_by_qubits[metadata["qbit_num"]] = case
        if set(selected_by_qubits) == {8, 10}:
            break
    missing = {8, 10} - set(selected_by_qubits)
    if missing:
        raise RuntimeError(
            "Missing representative structured fused cases for qubits: {}".format(
                sorted(missing)
            )
        )
    return [selected_by_qubits[8], selected_by_qubits[10]]


def build_artifact_bundle(cases: list[dict]) -> dict:
    fused_passes = sum(case["structured_fused_runtime_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if fused_passes == len(cases) else "fail",
        "runtime_schema_version": PHASE3_RUNTIME_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "structured_fused_runtime_passes": fused_passes,
            "required_qubits": [8, 10],
            "fused_regions": sum(case["fused_region_count"] for case in cases),
            "supported_unfused_regions": sum(
                case["supported_unfused_region_count"] for case in cases
            ),
            "deferred_regions": sum(case["deferred_region_count"] for case in cases),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Task 4 Story 2 bundle missing required fields: {}".format(
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
        help="Directory to write the Task 4 Story 2 bundle into.",
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
                "{case_name}: fused_regions={fused_region_count}, dRho={frobenius_norm_diff:.3e}, pass={structured_fused_runtime_pass}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
