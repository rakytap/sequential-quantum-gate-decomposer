#!/usr/bin/env python3
"""Mandatory workload runtime validation for the shared partitioned runtime surface.

Executes the required microcases plus one structured case per mandatory family
and size through the same partitioned runtime entry used by the continuity
anchor workloads.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/mandatory_workload_runtime_validation.py
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
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_structured_descriptor_set,
    iter_microcase_descriptor_sets,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set
from squander.partitioning.noisy_runtime import PHASE3_RUNTIME_PATH_BASELINE

SUITE_NAME = "phase3_partitioned_runtime_workload_runtime"
ARTIFACT_FILENAME = "mandatory_workload_runtime_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "mandatory_workload"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "software",
    "summary",
    "cases",
)
STRUCTURED_VALIDATION_NOISE_PATTERN = "sparse"


def _runtime_case(
    *,
    case_name: str,
    case_kind: str,
    descriptor_set,
    metadata: dict | None = None,
) -> dict:
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
        "requested_mode": runtime_payload["requested_mode"],
        "source_type": runtime_payload["source_type"],
        "workload_id": runtime_payload["workload_id"],
        "qbit_num": runtime_payload["qbit_num"],
        "parameter_count": runtime_payload["parameter_count"],
        "runtime_path": runtime_payload["runtime_path"],
        "partition_count": runtime_payload["summary"]["partition_count"],
        "descriptor_member_count": runtime_payload["summary"]["descriptor_member_count"],
        "gate_count": runtime_payload["summary"]["gate_count"],
        "noise_count": runtime_payload["summary"]["noise_count"],
        "max_partition_span": runtime_payload["summary"]["max_partition_span"],
        "partition_member_counts": runtime_payload["summary"]["partition_member_counts"],
        "exact_output_present": runtime_payload["summary"]["exact_output_present"],
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
        "shared_runtime_pass": (
            runtime_payload["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
            and runtime_payload["summary"]["partition_count"] > 0
            and runtime_payload["summary"]["exact_output_present"] is True
            and density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        ),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    return case


def build_cases() -> list[dict]:
    continuity_vqe, _, _ = build_phase2_continuity_vqe(4)
    cases = [
        _runtime_case(
            case_name="phase2_xxz_hea_q4_continuity",
            case_kind="continuity",
            descriptor_set=build_phase3_continuity_partition_descriptor_set(continuity_vqe),
        )
    ]
    for metadata, descriptor_set in iter_microcase_descriptor_sets():
        cases.append(
            _runtime_case(
                case_name=metadata["case_name"],
                case_kind="microcase",
                descriptor_set=descriptor_set,
                metadata=metadata,
            )
        )
    for family_name in STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            metadata = {
                "family_name": family_name,
                "qbit_num": qbit_num,
                "noise_pattern": STRUCTURED_VALIDATION_NOISE_PATTERN,
                "workload_id": "{}_q{}_{}_seed20260318".format(
                    family_name, qbit_num, STRUCTURED_VALIDATION_NOISE_PATTERN
                ),
            }
            cases.append(
                _runtime_case(
                    case_name=metadata["workload_id"],
                    case_kind="structured_family",
                    descriptor_set=build_structured_descriptor_set(
                        family_name,
                        qbit_num=qbit_num,
                        noise_pattern=STRUCTURED_VALIDATION_NOISE_PATTERN,
                    ),
                    metadata=metadata,
                )
            )
    return cases


def build_artifact_bundle(cases: list[dict]) -> dict:
    shared_runtime_passes = sum(case["shared_runtime_pass"] for case in cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if shared_runtime_passes == len(cases) and passed_cases == len(cases)
        else "fail",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "shared_runtime_passes": shared_runtime_passes,
            "continuity_cases": sum(case["case_kind"] == "continuity" for case in cases),
            "microcases": sum(case["case_kind"] == "microcase" for case in cases),
            "structured_cases": sum(
                case["case_kind"] == "structured_family" for case in cases
            ),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Mandatory runtime bundle missing required fields: {}".format(
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
        help="Directory to write the mandatory workload runtime bundle into.",
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
                "{case_name}: q={qbit_num}, partitions={partition_count}, dRho={frobenius_norm_diff:.3e}, pass={shared_runtime_pass}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
