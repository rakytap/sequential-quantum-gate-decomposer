#!/usr/bin/env python3
"""Continuity planner surface validation.

Builds the required 4, 6, 8, and 10 qubit Phase 2 continuity-anchor workloads
and verifies that each one reaches the canonical Phase 3 planner surface through
the shared bridge-based continuity entry path.

Run with:
    python benchmarks/density_matrix/planner_surface/continuity_surface_validation.py
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
    DEFAULT_ANSATZ,
    DEFAULT_INNER_BLOCKS,
    DEFAULT_LAYERS,
    PRIMARY_BACKEND,
    build_case_metadata,
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from squander.partitioning.noisy_planner import (
    PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY,
    PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY,
    build_phase3_continuity_planner_surface,
)

SUITE_NAME = "phase3_planner_surface_continuity_surface"
ARTIFACT_FILENAME = "continuity_surface_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "planner_surface"
)
REQUIRED_QUBITS = (4, 6, 8, 10)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "requested_mode",
    "schema_version",
    "software",
    "summary",
    "cases",
)


def build_case_result(qbit_num: int) -> dict:
    vqe, _, topology = build_phase2_continuity_vqe(qbit_num)
    surface = build_phase3_continuity_planner_surface(vqe)
    payload = surface.to_dict()

    case = build_case_metadata(
        qbit_num=qbit_num,
        topology=topology,
        density_noise=[dict(item) for item in vqe.density_noise],
    )
    case.update(
        {
            "case_name": "phase2_xxz_hea_q{}_continuity".format(qbit_num),
            "case_kind": "phase3_planner_surface_continuity_surface",
            "status": "pass",
            "requested_mode": payload["requested_mode"],
            "schema_version": payload["schema_version"],
            "source_type": payload["source_type"],
            "entry_route": payload["entry_route"],
            "workload_family": payload["workload_family"],
            "workload_id": payload["workload_id"],
            "parameter_count": payload["parameter_count"],
            "operation_count": payload["operation_count"],
            "gate_count": payload["gate_count"],
            "noise_count": payload["noise_count"],
            "gate_sequence": [
                op["name"] for op in payload["operations"] if op["operation_class"] == "GateOperation"
            ],
            "noise_sequence": [
                op["name"] for op in payload["operations"] if op["operation_class"] == "NoiseOperation"
            ],
            "continuity_anchor_pass": (
                payload["source_type"] == "generated_hea"
                and payload["entry_route"] == PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
                and payload["workload_family"] == PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
                and payload["operation_count"] == payload["gate_count"] + payload["noise_count"]
            ),
            "planner_operations": payload["operations"],
        }
    )
    return case


def build_artifact_bundle(cases: list[dict]) -> dict:
    passed_cases = sum(case["status"] == "pass" for case in cases)
    continuity_passes = sum(case["continuity_anchor_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if passed_cases == len(cases) and continuity_passes == len(cases) else "fail",
        "requested_mode": "partitioned_density",
        "schema_version": "phase3_canonical_noisy_planner_v1",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "continuity_anchor_passes": continuity_passes,
            "required_qubits": list(REQUIRED_QUBITS),
            "required_pass_rate": 1.0,
            "actual_pass_rate": passed_cases / len(cases) if cases else 0.0,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError("Continuity surface bundle missing required fields: {}".format(", ".join(missing)))
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def run_validation(verbose: bool = True) -> list[dict]:
    results = [build_case_result(qbit_num) for qbit_num in REQUIRED_QUBITS]
    if verbose:
        print("Continuity planner surface validation:")
        for case in results:
            print(
                "  q={qbit_num}: ops={operation_count}, gates={gate_count}, noise={noise_count}, continuity_pass={continuity_anchor_pass}".format(
                    **case
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the continuity surface bundle into.",
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
