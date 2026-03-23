#!/usr/bin/env python3
"""Validation: bridge micro-validation matrix.

Validates the supported VQE-side density bridge on deterministic 1 to 3 qubit
microcases. The goal is not observable-threshold closure, but proof that the
required generated-`HEA` bridge surface lowers cleanly into the supported
`NoisyCircuit` vocabulary and remains execution-ready on small cases.

Run with:
    python benchmarks/density_matrix/bridge_scope/bridge_validation.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.VQE.shot_noise_measurement import generate_zz_xx_hamiltonian
from benchmarks.density_matrix.workflow_evidence.exact_density_vqe_validation import (
    DEFAULT_ANSATZ,
    DEFAULT_INNER_BLOCKS,
    DEFAULT_LAYERS,
    PRIMARY_BACKEND,
    build_case_metadata,
    build_software_metadata,
    build_reference_bridge_metadata,
    build_optimizer_config,
    build_hamiltonian_metadata,
    build_initial_parameters,
    build_open_chain_topology,
)
from squander import Variational_Quantum_Eigensolver
from squander.partitioning.noisy_planner import (
    PLANNER_OP_KIND_GATE,
    PLANNER_OP_KIND_NOISE,
)

SUITE_NAME = "bridge_micro_validation"
ARTIFACT_FILENAME = "bridge_micro_validation_bundle.json"
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "requirements",
    "summary",
    "software",
    "cases",
)

MANDATORY_BRIDGE_MICROCASES = (
    {
        "case_name": "bridge_1q_local_depolarizing_after_u3",
        "qbit_num": 1,
        "purpose": "Verify the generated-HEA bridge lowers a required U3-only microcase with local depolarizing noise.",
        "required_gate_families": ["U3"],
        "required_noise_models": ["local_depolarizing"],
        "density_noise": [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 0,
                "error_rate": 0.1,
            }
        ],
        "requires_cnot": False,
    },
    {
        "case_name": "bridge_1q_amplitude_damping_after_u3",
        "qbit_num": 1,
        "purpose": "Verify the generated-HEA bridge lowers a required U3-only microcase with amplitude damping noise.",
        "required_gate_families": ["U3"],
        "required_noise_models": ["amplitude_damping"],
        "density_noise": [
            {
                "channel": "amplitude_damping",
                "target": 0,
                "after_gate_index": 0,
                "gamma": 0.05,
            }
        ],
        "requires_cnot": False,
    },
    {
        "case_name": "bridge_1q_phase_damping_after_u3",
        "qbit_num": 1,
        "purpose": "Verify the generated-HEA bridge lowers a required U3-only microcase with phase damping noise.",
        "required_gate_families": ["U3"],
        "required_noise_models": ["phase_damping"],
        "density_noise": [
            {
                "channel": "phase_damping",
                "target": 0,
                "after_gate_index": 0,
                "lambda": 0.07,
            }
        ],
        "requires_cnot": False,
    },
    {
        "case_name": "bridge_2q_cnot_with_local_depolarizing",
        "qbit_num": 2,
        "purpose": "Verify the required CNOT bridge slice lowers cleanly on a 2-qubit anchor microcase.",
        "required_gate_families": ["U3", "CNOT"],
        "required_noise_models": ["local_depolarizing"],
        "density_noise": [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 2,
                "error_rate": 0.1,
            }
        ],
        "requires_cnot": True,
    },
    {
        "case_name": "bridge_3q_mixed_local_noise_sequence",
        "qbit_num": 3,
        "purpose": "Verify mixed required local-noise insertion remains ordered and auditable on a 3-qubit bridge microcase.",
        "required_gate_families": ["U3", "CNOT"],
        "required_noise_models": [
            "local_depolarizing",
            "amplitude_damping",
            "phase_damping",
        ],
        "density_noise": [
            {
                "channel": "local_depolarizing",
                "target": 0,
                "after_gate_index": 0,
                "error_rate": 0.1,
            },
            {
                "channel": "amplitude_damping",
                "target": 1,
                "after_gate_index": 1,
                "gamma": 0.05,
            },
            {
                "channel": "phase_damping",
                "target": 0,
                "after_gate_index": 2,
                "lambda": 0.07,
            },
        ],
        "requires_cnot": True,
    },
)


def build_requirement_metadata():
    return {
        "source_type": "generated_hea",
        "required_gate_families": ["U3", "CNOT"],
        "required_noise_models": [
            "local_depolarizing",
            "amplitude_damping",
            "phase_damping",
        ],
        "microcase_qubits": [1, 2, 3],
        "required_pass_rate": 1.0,
        "canonical_bridge_fields": [
            "bridge_source_type",
            "bridge_parameter_count",
            "bridge_operation_count",
            "bridge_gate_count",
            "bridge_noise_count",
            "bridge_operations",
        ],
    }


def build_bridge_vqe(qbit_num: int, density_noise):
    topology = build_open_chain_topology(qbit_num)
    hamiltonian, _ = generate_zz_xx_hamiltonian(
        n_qubits=qbit_num,
        h=0.5,
        topology=topology,
        Jz=1.0,
        Jx=1.0,
        Jy=1.0,
    )
    vqe = Variational_Quantum_Eigensolver(
        hamiltonian,
        qbit_num,
        build_optimizer_config(),
        backend=PRIMARY_BACKEND,
        density_noise=density_noise,
    )
    vqe.set_Ansatz(DEFAULT_ANSATZ)
    vqe.Generate_Circuit(layers=DEFAULT_LAYERS, inner_blocks=DEFAULT_INNER_BLOCKS)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(parameters)
    return vqe, hamiltonian, topology, parameters


def build_expected_bridge_operations(vqe):
    base_circuit = vqe.get_Qiskit_Circuit()
    noise_by_gate = {}
    for noise_spec in vqe.density_noise:
        noise_by_gate.setdefault(noise_spec["after_gate_index"], []).append(noise_spec)

    expected = []
    param_start = 0
    for gate_index, instruction in enumerate(base_circuit.data):
        qubit_indices = [base_circuit.find_bit(qubit).index for qubit in instruction.qubits]
        gate_name = instruction.operation.name

        if gate_name == "u":
            expected.append(
                {
                    "index": len(expected),
                    "kind": PLANNER_OP_KIND_GATE,
                    "name": "U3",
                    "is_unitary": True,
                    "source_gate_index": gate_index,
                    "target_qbit": qubit_indices[0],
                    "control_qbit": None,
                    "param_count": 3,
                    "param_start": param_start,
                    "fixed_value": None,
                }
            )
            param_start += 3
        elif gate_name in {"cx", "cnot"}:
            expected.append(
                {
                    "index": len(expected),
                    "kind": PLANNER_OP_KIND_GATE,
                    "name": "CNOT",
                    "is_unitary": True,
                    "source_gate_index": gate_index,
                    "target_qbit": qubit_indices[1],
                    "control_qbit": qubit_indices[0],
                    "param_count": 0,
                    "param_start": param_start,
                    "fixed_value": None,
                }
            )
        else:
            raise ValueError(
                f"Unsupported bridge micro-validation gate in expected bridge: {gate_name}"
            )

        for noise_spec in noise_by_gate.get(gate_index, []):
            expected.append(
                {
                    "index": len(expected),
                    "kind": PLANNER_OP_KIND_NOISE,
                    "name": noise_spec["channel"],
                    "is_unitary": False,
                    "source_gate_index": gate_index,
                    "target_qbit": noise_spec["target"],
                    "control_qbit": None,
                    "param_count": 0,
                    "param_start": param_start,
                    "fixed_value": noise_spec["value"],
                }
            )

    return expected


def bridge_operations_match(actual_operations, expected_operations):
    if len(actual_operations) != len(expected_operations):
        return False

    for actual, expected in zip(actual_operations, expected_operations):
        for key in (
            "index",
            "kind",
            "name",
            "is_unitary",
            "source_gate_index",
            "target_qbit",
            "control_qbit",
            "param_count",
            "param_start",
        ):
            if actual.get(key) != expected.get(key):
                return False

        actual_fixed = actual.get("fixed_value")
        expected_fixed = expected.get("fixed_value")
        if expected_fixed is None:
            if actual_fixed is not None:
                return False
        elif actual_fixed is None or not math.isclose(actual_fixed, expected_fixed, rel_tol=0.0, abs_tol=1e-12):
            return False

    return True


def _case_base(case, topology):
    metadata = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=case["qbit_num"],
        topology=topology,
        density_noise=case["density_noise"],
        hamiltonian=build_hamiltonian_metadata(),
    )
    metadata.update(
        {
            "case_name": case["case_name"],
            "case_kind": "bridge_micro_validation",
            "purpose": case["purpose"],
            "required_gate_families": case["required_gate_families"],
            "required_noise_models": case["required_noise_models"],
        }
    )
    return metadata


def validate_bridge_microcase(case, verbose=True):
    vqe, _, topology, parameters = build_bridge_vqe(case["qbit_num"], case["density_noise"])
    bridge = vqe.describe_density_bridge()
    expected_operations = build_expected_bridge_operations(vqe)
    energy = float(vqe.Optimization_Problem(parameters))

    gate_names = [op["name"] for op in bridge["operations"] if op["kind"] == "gate"]
    noise_names = [op["name"] for op in bridge["operations"] if op["kind"] == "noise"]
    source_pass = bridge["source_type"] == "generated_hea"
    gate_pass = "U3" in gate_names and (
        (not case["requires_cnot"]) or ("CNOT" in gate_names)
    )
    noise_pass = all(
        required_model in noise_names for required_model in case["required_noise_models"]
    )
    operation_match_pass = bridge_operations_match(
        bridge["operations"], expected_operations
    )
    execution_ready = bool(np.isfinite(energy))
    bridge_pass = (
        source_pass
        and gate_pass
        and noise_pass
        and operation_match_pass
        and execution_ready
    )

    result = _case_base(case, topology)
    result.update(
        {
            "status": "pass" if bridge_pass else "fail",
            "parameter_vector": parameters.tolist(),
            "execution_energy": energy,
            "execution_ready": execution_ready,
            "source_pass": source_pass,
            "gate_pass": gate_pass,
            "noise_pass": noise_pass,
            "operation_match_pass": operation_match_pass,
            "expected_operation_count": len(expected_operations),
            "expected_bridge_operations": expected_operations,
            **build_reference_bridge_metadata(vqe),
        }
    )

    if verbose:
        print(
            "  {case_name:<46} q={qbit_num} ops={ops} noise={noise} status={status}".format(
                case_name=result["case_name"],
                qbit_num=result["qbit_num"],
                ops=result["bridge_operation_count"],
                noise=result["bridge_noise_count"],
                status=result["status"].upper(),
            )
        )

    return result


def capture_bridge_microcase(case, verbose=True):
    try:
        return validate_bridge_microcase(case, verbose=verbose)
    except Exception as exc:
        topology = build_open_chain_topology(case["qbit_num"])
        result = _case_base(case, topology)
        result.update(
            {
                "status": "fail",
                "execution_ready": False,
                "source_pass": False,
                "gate_pass": False,
                "noise_pass": False,
                "operation_match_pass": False,
                "error_message": str(exc),
            }
        )
        if verbose:
            print(
                "  {case_name:<46} q={qbit_num} ERROR {message}".format(
                    case_name=result["case_name"],
                    qbit_num=result["qbit_num"],
                    message=result["error_message"],
                )
            )
        return result


def run_validation(verbose=True):
    print("=" * 78)
    print("  Bridge Micro-Validation [{}]".format(PRIMARY_BACKEND))
    print("=" * 78)

    results = []
    for case in MANDATORY_BRIDGE_MICROCASES:
        results.append(capture_bridge_microcase(case, verbose=verbose))
    return results


def build_artifact_bundle(results):
    passed = sum(1 for result in results if result["status"] == "pass")
    total = len(results)
    pass_rate = 0.0 if total == 0 else passed / total
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if pass_rate == 1.0 else "fail",
        "backend": PRIMARY_BACKEND,
        "requirements": build_requirement_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": total - passed,
            "pass_rate": pass_rate,
            "cases_per_qbit": {
                str(qbit_num): sum(
                    1 for result in results if result["qbit_num"] == qbit_num
                )
                for qbit_num in sorted({case["qbit_num"] for case in MANDATORY_BRIDGE_MICROCASES})
            },
        },
        "cases": results,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def print_summary(bundle):
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for result in bundle["cases"]:
        details = []
        if not result.get("source_pass", False):
            details.append("source")
        if not result.get("gate_pass", False):
            details.append("gate")
        if not result.get("noise_pass", False):
            details.append("noise")
        if not result.get("operation_match_pass", False):
            details.append("shape")
        if not result.get("execution_ready", False):
            details.append("exec")
        if "error_message" in result:
            details.append("error")

        print(
            "  {case_name:<46} q={qbit_num} status={status}{detail_suffix}".format(
                case_name=result["case_name"],
                qbit_num=result["qbit_num"],
                status=result["status"].upper(),
                detail_suffix="" if not details else " (" + ",".join(details) + ")",
            )
        )

    print("\n" + "-" * 78)
    print(
        "  Total: {passed}/{total} mandatory bridge microcases passed".format(
            passed=bundle["summary"]["passed_cases"],
            total=bundle["summary"]["total_cases"],
        )
    )
    if bundle["status"] == "pass":
        print("\n  ALL TESTS PASSED - Bridge micro-validation gate is closed.")
    else:
        print("\n  Some mandatory bridge microcases failed - bridge micro-validation is not yet closed.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the bridge micro-validation JSON artifact bundle.",
    )
    args = parser.parse_args()

    results = run_validation()
    bundle = build_artifact_bundle(results)
    print_summary(bundle)

    if args.output_dir is not None:
        write_artifact_bundle(args.output_dir / ARTIFACT_FILENAME, bundle)

    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
