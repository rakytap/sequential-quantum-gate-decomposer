#!/usr/bin/env python3
"""Validation: Phase 3 Task 3 Story 1 continuity runtime slice.

Builds the required 4, 6, 8, and 10 qubit Phase 2 continuity-anchor workloads
and verifies that each one executes through the first positive Task 3
partitioned density runtime path from validated Task 2 descriptors.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/continuity_runtime_validation.py
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
    PHASE3_RUNTIME_ENERGY_TOL,
    build_density_comparison_metrics,
    build_initial_parameters,
    density_energy,
    execute_partitioned_with_reference,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_case_metadata,
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    build_phase3_continuity_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_SCHEMA_VERSION,
)

SUITE_NAME = "phase3_partitioned_runtime_continuity_runtime"
ARTIFACT_FILENAME = "continuity_runtime_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "continuity_runtime"
)
REQUIRED_QUBITS = (4, 6, 8, 10)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "runtime_schema_version",
    "software",
    "summary",
    "cases",
)


def build_case_result(qbit_num: int) -> dict:
    vqe, hamiltonian, topology = build_phase2_continuity_vqe(qbit_num)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    runtime_result, reference_density, density_metrics = execute_partitioned_with_reference(
        descriptor_set, parameters
    )
    runtime_payload = runtime_result.to_dict(include_density_matrix=False)
    partitioned_energy_real, partitioned_energy_imag = density_energy(
        hamiltonian, runtime_result.density_matrix_numpy()
    )
    reference_energy_real, reference_energy_imag = density_energy(
        hamiltonian, reference_density.to_numpy()
    )
    continuity_energy = float(vqe.Optimization_Problem(parameters))
    absolute_energy_error = float(abs(partitioned_energy_real - continuity_energy))
    energy_pass = absolute_energy_error <= PHASE3_RUNTIME_ENERGY_TOL
    density_pass = (
        density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    )
    case = build_case_metadata(
        qbit_num=qbit_num,
        topology=topology,
        density_noise=[dict(item) for item in vqe.density_noise],
    )
    case.update(
        {
            "case_name": f"phase2_xxz_hea_q{qbit_num}_continuity",
            "case_kind": "phase3_partitioned_runtime_continuity_runtime",
            "status": "pass" if energy_pass and density_pass and runtime_result.rho_is_valid else "fail",
            "requested_mode": runtime_payload["requested_mode"],
            "runtime_schema_version": runtime_payload["runtime_schema_version"],
            "planner_schema_version": runtime_payload["planner_schema_version"],
            "descriptor_schema_version": runtime_payload["descriptor_schema_version"],
            "source_type": runtime_payload["source_type"],
            "entry_route": runtime_payload["entry_route"],
            "workload_family": runtime_payload["workload_family"],
            "workload_id": runtime_payload["workload_id"],
            "parameter_count": runtime_payload["parameter_count"],
            "runtime_path": runtime_payload["runtime_path"],
            "partition_count": runtime_payload["summary"]["partition_count"],
            "descriptor_member_count": runtime_payload["summary"]["descriptor_member_count"],
            "gate_count": runtime_payload["summary"]["gate_count"],
            "noise_count": runtime_payload["summary"]["noise_count"],
            "max_partition_span": runtime_payload["summary"]["max_partition_span"],
            "partition_member_counts": runtime_payload["summary"]["partition_member_counts"],
            "remapped_partition_count": runtime_payload["summary"]["remapped_partition_count"],
            "parameter_routing_segment_count": runtime_payload["summary"][
                "parameter_routing_segment_count"
            ],
            "fallback_used": runtime_payload["summary"]["fallback_used"],
            "exact_output_present": runtime_payload["summary"]["exact_output_present"],
            "rho_is_valid": runtime_payload["summary"]["rho_is_valid"],
            "trace_deviation": runtime_payload["summary"]["trace_deviation"],
            "partitioned_energy_real": partitioned_energy_real,
            "partitioned_energy_imag": partitioned_energy_imag,
            "reference_energy_real": reference_energy_real,
            "reference_energy_imag": reference_energy_imag,
            "continuity_energy_real": continuity_energy,
            "absolute_energy_error": absolute_energy_error,
            "energy_pass": energy_pass,
            "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
            "max_abs_diff": density_metrics["max_abs_diff"],
            "density_pass": density_pass,
            "runtime_ms": runtime_payload["summary"]["runtime_ms"],
            "peak_rss_kb": runtime_payload["summary"]["peak_rss_kb"],
            "runtime_partitions": runtime_payload["partitions"],
            "continuity_runtime_pass": (
                runtime_payload["requested_mode"] == PARTITIONED_DENSITY_MODE
                and runtime_payload["runtime_schema_version"] == PHASE3_RUNTIME_SCHEMA_VERSION
                and runtime_payload["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
                and runtime_payload["summary"]["partition_count"] > 0
                and runtime_payload["summary"]["fallback_used"] is False
                and runtime_payload["summary"]["exact_output_present"] is True
                and energy_pass
                and density_pass
            ),
        }
    )
    return case


def build_artifact_bundle(cases: list[dict]) -> dict:
    passed_cases = sum(case["status"] == "pass" for case in cases)
    continuity_runtime_passes = sum(case["continuity_runtime_pass"] for case in cases)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if passed_cases == len(cases) and continuity_runtime_passes == len(cases)
        else "fail",
        "runtime_schema_version": PHASE3_RUNTIME_SCHEMA_VERSION,
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "continuity_runtime_passes": continuity_runtime_passes,
            "required_qubits": list(REQUIRED_QUBITS),
            "required_pass_rate": 1.0,
            "actual_pass_rate": passed_cases / len(cases) if cases else 0.0,
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Continuity runtime bundle missing required fields: {}".format(
                ", ".join(missing)
            )
        )
    return bundle


def write_artifact_bundle(bundle: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ARTIFACT_FILENAME
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def run_validation(verbose: bool = True) -> list[dict]:
    results = [build_case_result(qbit_num) for qbit_num in REQUIRED_QUBITS]
    if verbose:
        print("Phase 3 Task 3 Story 1 continuity runtime:")
        for case in results:
            print(
                "  q={qbit_num}: partitions={partition_count}, runtime_path={runtime_path}, dE={absolute_energy_error:.3e}, dRho={frobenius_norm_diff:.3e}, pass={continuity_runtime_pass}".format(
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
        help="Directory to write the continuity runtime bundle into.",
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
