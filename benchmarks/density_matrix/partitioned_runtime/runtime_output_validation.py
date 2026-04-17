#!/usr/bin/env python3
"""Runtime output validation: exact-output schema and comparison-ready payloads.

Emits supported runtime outputs in the shared comparison-ready shape used by
later exact-baseline checks. Small cases include the explicit density matrix in
the emitted exact-output record; larger cases keep the same schema but omit the
full matrix payload.

Run with:
    python benchmarks/density_matrix/partitioned_runtime/runtime_output_validation.py
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
    PHASE3_RUNTIME_ENERGY_TOL,
    build_initial_parameters,
    density_energy,
    execute_partitioned_with_reference,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    build_microcase_descriptor_set,
    build_structured_descriptor_set,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set

SUITE_NAME = "phase3_partitioned_runtime_runtime_output"
ARTIFACT_FILENAME = "runtime_output_bundle.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "partitioned_runtime"
    / "runtime_output"
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "software",
    "summary",
    "cases",
)


def _include_density_matrix(qbit_num: int) -> bool:
    return qbit_num <= 4


def _output_case(
    *,
    case_name: str,
    case_kind: str,
    descriptor_set,
    hamiltonian=None,
    continuity_energy_real: float | None = None,
) -> dict:
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    runtime_result, _, _ = execute_partitioned_with_reference(descriptor_set, parameters)
    runtime_payload = runtime_result.to_dict(
        include_density_matrix=_include_density_matrix(descriptor_set.qbit_num)
    )
    case = {
        "case_name": case_name,
        "case_kind": case_kind,
        "status": "pass",
        "workload_id": runtime_payload["workload_id"],
        "qbit_num": runtime_payload["qbit_num"],
        "runtime_path": runtime_payload["runtime_path"],
        "exact_output_present": runtime_payload["summary"]["exact_output_present"],
        "matrix_included": runtime_payload["exact_output"]["matrix_included"],
        "exact_output": runtime_payload["exact_output"],
        "result_output_pass": (
            runtime_payload["summary"]["exact_output_present"] is True
            and "shape" in runtime_payload["exact_output"]
            and "trace_real" in runtime_payload["exact_output"]
            and "trace_imag" in runtime_payload["exact_output"]
            and "rho_is_valid" in runtime_payload["exact_output"]
        ),
    }
    if hamiltonian is not None:
        partitioned_energy_real, partitioned_energy_imag = density_energy(
            hamiltonian, runtime_result.density_matrix_numpy()
        )
        case.update(
            {
                "partitioned_energy_real": partitioned_energy_real,
                "partitioned_energy_imag": partitioned_energy_imag,
                "continuity_energy_real": continuity_energy_real,
                "absolute_energy_error": float(
                    abs(partitioned_energy_real - continuity_energy_real)
                ),
                "continuity_output_pass": (
                    float(abs(partitioned_energy_real - continuity_energy_real))
                    <= PHASE3_RUNTIME_ENERGY_TOL
                ),
            }
        )
    return case


def build_cases() -> list[dict]:
    continuity_vqe, hamiltonian, _ = build_phase2_continuity_vqe(4)
    continuity_parameters = build_initial_parameters(continuity_vqe.get_Parameter_Num())
    continuity_case = _output_case(
        case_name="phase2_xxz_hea_q4_continuity",
        case_kind="continuity",
        descriptor_set=build_phase3_continuity_partition_descriptor_set(continuity_vqe),
        hamiltonian=hamiltonian,
        continuity_energy_real=float(continuity_vqe.Optimization_Problem(continuity_parameters)),
    )
    microcase_case = _output_case(
        case_name="microcase_4q_partition_boundary_triplet",
        case_kind="microcase",
        descriptor_set=build_microcase_descriptor_set(
            "microcase_4q_partition_boundary_triplet"
        ),
    )
    structured_case = _output_case(
        case_name="layered_nearest_neighbor_q8_sparse_seed20260318",
        case_kind="structured_family",
        descriptor_set=build_structured_descriptor_set(
            "layered_nearest_neighbor",
            qbit_num=8,
            noise_pattern="sparse",
        ),
    )
    return [continuity_case, microcase_case, structured_case]


def build_artifact_bundle(cases: list[dict]) -> dict:
    output_passes = sum(case["result_output_pass"] for case in cases)
    continuity_output_passes = sum(
        case.get("continuity_output_pass", False) for case in cases
    )
    exact_output_key_sets = {frozenset(case["exact_output"].keys()) for case in cases}
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if output_passes == len(cases)
        and continuity_output_passes == 1
        and len(exact_output_key_sets) == 1
        else "fail",
        "software": build_software_metadata(),
        "summary": {
            "total_cases": len(cases),
            "result_output_passes": output_passes,
            "continuity_output_passes": continuity_output_passes,
            "matrix_payload_cases": sum(case["matrix_included"] for case in cases),
            "exact_output_schema_count": len(exact_output_key_sets),
        },
        "cases": cases,
    }
    missing = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing:
        raise ValueError(
            "Runtime output bundle missing required fields: {}".format(
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
        help="Directory to write the runtime output bundle into.",
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
                "{case_name}: matrix_included={matrix_included}, output_pass={result_output_pass}".format(
                    **case
                )
            )
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
