#!/usr/bin/env python3
"""Validation: Task 4 Story 1 required local-noise positive slice.

Runs one supported positive case per required Phase 2 local-noise model and
records reviewable bridge metadata proving that the requested local model is
what actually reaches the VQE-facing density path.

This script is intentionally narrower than the broader Task 2 and Task 4
workflow validation package:
- it proves each required local model executes on the supported generated-HEA
  density path,
- it records model identity, placement, and fixed value metadata,
- it marks every emitted case as part of the mandatory Task 4 baseline,
- and it rejects whole-register depolarizing as a stand-in for the required
  local-noise baseline.

Run with:
    python benchmarks/density_matrix/noise_support/required_local_noise_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
from qiskit_aer import AerSimulator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
    DEFAULT_ANSATZ,
    DEFAULT_INNER_BLOCKS,
    DEFAULT_LAYERS,
    PRIMARY_BACKEND,
    REFERENCE_BACKEND,
    build_case_metadata,
    build_software_metadata,
    build_reference_bridge_metadata,
    build_hamiltonian_metadata,
    build_initial_parameters,
    build_vqe,
    density_energy,
    insert_reference_noise,
)
from benchmarks.density_matrix.noise_support.support_tiers import (
    SUPPORT_TIER_VOCABULARY,
    build_required_case_classification,
    build_support_tier_summary,
)

SUITE_NAME = "required_local_noise_validation"
ARTIFACT_FILENAME = "required_local_noise_bundle.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "noise_support"
REQUIRED_LOCAL_NOISE_MODELS = (
    "local_depolarizing",
    "amplitude_damping",
    "phase_damping",
)
REQUIRED_LOCAL_NOISE_CASES = (
    {
        "case_name": "required_local_noise_4q_local_depolarizing_positive",
        "qbit_num": 4,
        "requested_noise_channel": "local_depolarizing",
        "expected_target": 0,
        "expected_after_gate_index": 0,
        "expected_value": 0.1,
        "density_noise": [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 0,
                "error_rate": 0.1,
            }
        ],
    },
    {
        "case_name": "required_local_noise_4q_amplitude_damping_positive",
        "qbit_num": 4,
        "requested_noise_channel": "amplitude_damping",
        "expected_target": 1,
        "expected_after_gate_index": 2,
        "expected_value": 0.05,
        "density_noise": [
            {
                "channel": "amplitude_damping",
                "target": 1,
                "after_gate_index": 2,
                "gamma": 0.05,
            }
        ],
    },
    {
        "case_name": "required_local_noise_4q_phase_damping_positive",
        "qbit_num": 4,
        "requested_noise_channel": "phase_damping",
        "expected_target": 0,
        "expected_after_gate_index": 4,
        "expected_value": 0.07,
        "density_noise": [
            {
                "channel": "phase_damping",
                "target": 0,
                "after_gate_index": 4,
                "lambda": 0.07,
            }
        ],
    },
)


def build_requirement_metadata():
    return {
        "source_type": "generated_hea",
        "required_local_noise_models": list(REQUIRED_LOCAL_NOISE_MODELS),
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "case_names": [case["case_name"] for case in REQUIRED_LOCAL_NOISE_CASES],
        "workflow_qubits": sorted({case["qbit_num"] for case in REQUIRED_LOCAL_NOISE_CASES}),
        "required_bridge_fields": [
            "bridge_source_type",
            "bridge_operation_count",
            "bridge_noise_count",
            "bridge_noise_sequence",
            "bridge_noise_targets",
            "bridge_noise_after_gate_indices",
            "bridge_noise_fixed_values",
        ],
    }


def validate_case_artifact(case):
    required_fields = (
        "case_name",
        "status",
        "backend",
        "qbit_num",
        "topology",
        "ansatz",
        "layers",
        "inner_blocks",
        "density_noise",
        "requested_noise_channel",
        "bridge_source_type",
        "bridge_noise_sequence",
        "bridge_noise_targets",
        "bridge_noise_after_gate_indices",
        "bridge_noise_fixed_values",
        "positive_slice_pass",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 4 Story 1 case artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if case["status"] not in {"pass", "fail"}:
        raise ValueError(
            "Task 4 Story 1 case artifact has unsupported status '{}'".format(
                case["status"]
            )
        )

    if case["backend"] != PRIMARY_BACKEND:
        raise ValueError(
            "Task 4 Story 1 case artifact has unexpected backend '{}'".format(
                case["backend"]
            )
        )


def run_required_local_noise_case(case):
    density_noise = [dict(item) for item in case["density_noise"]]
    vqe, hamiltonian, topology = build_vqe(
        case["qbit_num"],
        density_noise=density_noise,
    )
    params = build_initial_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(params)
    bridge_metadata = build_reference_bridge_metadata(vqe)

    squander_energy = float(vqe.Optimization_Problem(params))

    noisy_qiskit_circuit = insert_reference_noise(vqe.get_Qiskit_Circuit(), density_noise)
    simulator = AerSimulator(method="density_matrix")
    result = simulator.run(noisy_qiskit_circuit, shots=1).result()
    aer_rho = np.asarray(result.data()["density_matrix"])
    aer_energy_real, aer_energy_imag = density_energy(hamiltonian, aer_rho)

    requested_channel = case["requested_noise_channel"]
    noise_sequence = bridge_metadata["bridge_noise_sequence"]
    noise_targets = bridge_metadata["bridge_noise_targets"]
    noise_after_gate_indices = bridge_metadata["bridge_noise_after_gate_indices"]
    noise_fixed_values = bridge_metadata["bridge_noise_fixed_values"]

    bridge_source_pass = bridge_metadata["bridge_source_type"] == "generated_hea"
    bridge_noise_count_pass = bridge_metadata["bridge_noise_count"] == 1
    bridge_noise_model_pass = noise_sequence == [requested_channel]
    bridge_noise_target_pass = noise_targets == [case["expected_target"]]
    bridge_noise_position_pass = noise_after_gate_indices == [
        case["expected_after_gate_index"]
    ]
    bridge_noise_value_pass = bool(
        len(noise_fixed_values) == 1
        and np.isclose(
            noise_fixed_values[0],
            case["expected_value"],
            atol=1e-12,
        )
    )
    no_whole_register_substitute_pass = all(
        noise_name != "depolarizing" for noise_name in noise_sequence
    )
    energy_is_finite = bool(np.isfinite(squander_energy))
    positive_slice_pass = all(
        (
            bridge_source_pass,
            bridge_noise_count_pass,
            bridge_noise_model_pass,
            bridge_noise_target_pass,
            bridge_noise_position_pass,
            bridge_noise_value_pass,
            no_whole_register_substitute_pass,
            energy_is_finite,
        )
    )

    artifact = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=case["qbit_num"],
        topology=topology,
        density_noise=density_noise,
        ansatz=DEFAULT_ANSATZ,
        layers=DEFAULT_LAYERS,
        inner_blocks=DEFAULT_INNER_BLOCKS,
        reference_backend=REFERENCE_BACKEND,
        hamiltonian=build_hamiltonian_metadata(),
    )
    artifact.update(
        {
            "case_name": case["case_name"],
            "status": "pass" if positive_slice_pass else "fail",
            "requested_noise_channel": requested_channel,
            "parameter_vector": params.tolist(),
            "squander_energy": squander_energy,
            "aer_energy_real": aer_energy_real,
            "aer_energy_imag": aer_energy_imag,
            "absolute_energy_error": abs(squander_energy - aer_energy_real),
            "bridge_source_pass": bridge_source_pass,
            "bridge_noise_count_pass": bridge_noise_count_pass,
            "bridge_noise_model_pass": bridge_noise_model_pass,
            "bridge_noise_target_pass": bridge_noise_target_pass,
            "bridge_noise_position_pass": bridge_noise_position_pass,
            "bridge_noise_value_pass": bridge_noise_value_pass,
            "no_whole_register_substitute_pass": no_whole_register_substitute_pass,
            "energy_is_finite": energy_is_finite,
            "positive_slice_pass": positive_slice_pass,
            **bridge_metadata,
            **build_required_case_classification(),
        }
    )
    validate_case_artifact(artifact)
    return artifact


def run_validation(*, verbose: bool = False):
    results = []
    for case in REQUIRED_LOCAL_NOISE_CASES:
        result = run_required_local_noise_case(case)
        results.append(result)
        if verbose:
            print(
                "{} [{}] requested={} sequence={}".format(
                    result["case_name"],
                    result["status"],
                    result["requested_noise_channel"],
                    result["bridge_noise_sequence"],
                )
            )
    return results


def build_artifact_bundle(results):
    results = list(results)
    passed_cases = sum(case["status"] == "pass" for case in results)
    total_cases = len(results)
    pass_rate = (passed_cases / total_cases) if total_cases else 0.0
    support_tier_summary = build_support_tier_summary(results)

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if support_tier_summary["mandatory_baseline_completed"]
        and passed_cases == total_cases
        and total_cases
        else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "pass_rate": pass_rate,
            **support_tier_summary,
            "whole_register_substitute_failures": sum(
                not case["no_whole_register_substitute_pass"] for case in results
            ),
            "reviewable_bridge_cases": sum(
                case["bridge_noise_model_pass"]
                and case["bridge_noise_target_pass"]
                and case["bridge_noise_position_pass"]
                and case["bridge_noise_value_pass"]
                for case in results
            ),
        },
        "cases": results,
    }
    return bundle


def write_artifact_bundle(path: Path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Task 4 Story 1 required local-noise validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the emitted JSON bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_validation(verbose=not args.quiet)
    bundle = build_artifact_bundle(results)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["passed_cases"],
            bundle["summary"]["total_cases"],
        )
    )


if __name__ == "__main__":
    main()
