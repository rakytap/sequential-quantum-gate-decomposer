#!/usr/bin/env python3
"""Validation: Task 3 Story 3 unsupported bridge cases.

Captures representative unsupported bridge requests as structured artifacts. The
goal is to prove that unsupported circuit-source, lowering, and noise-boundary
cases fail before execution and surface a stable first unsupported condition.

Run with:
    python benchmarks/density_matrix/task3_story3_unsupported_bridge_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.VQE.shot_noise_measurement import generate_zz_xx_hamiltonian
from benchmarks.density_matrix.story2_vqe_density_validation import (
    DEFAULT_ANSATZ,
    DEFAULT_INNER_BLOCKS,
    DEFAULT_LAYERS,
    PRIMARY_BACKEND,
    build_case_metadata,
    build_software_metadata,
    build_story2_config,
    build_story2_hamiltonian_metadata,
    build_story2_parameters,
    build_open_chain_topology,
)
from squander import Variational_Quantum_Eigensolver
from squander.gates.qgd_Circuit import qgd_Circuit

SUITE_NAME = "task3_story3_unsupported_bridge_validation"
ARTIFACT_FILENAME = "task3_story3_unsupported_bridge_bundle.json"
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "requirements",
    "summary",
    "software",
    "cases",
)


def build_requirement_metadata():
    return {
        "expected_status": "unsupported",
        "required_categories": [
            "circuit_source",
            "lowering_path",
            "noise_insertion",
            "noise_type",
        ],
        "canonical_bridge_fields": [
            "bridge_source_type",
            "unsupported_category",
            "first_unsupported_condition",
            "unsupported_reason",
            "source_pass",
            "gate_pass",
            "noise_pass",
            "operation_match_pass",
            "execution_ready",
        ],
    }


def build_density_vqe(qbit_num: int, density_noise=None, *, ansatz: str = DEFAULT_ANSATZ):
    if density_noise is None:
        density_noise = []
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
        build_story2_config(),
        backend=PRIMARY_BACKEND,
        density_noise=density_noise,
    )
    vqe.set_Ansatz(ansatz)
    return vqe, topology


def _base_case_metadata(
    *,
    case_name: str,
    qbit_num: int,
    topology,
    density_noise,
    purpose: str,
    unsupported_category: str,
    first_unsupported_condition: str,
    bridge_source_type: str,
    ansatz: str = DEFAULT_ANSATZ,
):
    metadata = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=qbit_num,
        topology=topology,
        density_noise=density_noise,
        ansatz=ansatz,
        hamiltonian=build_story2_hamiltonian_metadata(),
    )
    metadata.update(
        {
            "case_name": case_name,
            "case_kind": "task3_bridge_unsupported_validation",
            "purpose": purpose,
            "bridge_source_type": bridge_source_type,
            "unsupported_category": unsupported_category,
            "first_unsupported_condition": first_unsupported_condition,
            "source_pass": False,
            "gate_pass": False,
            "noise_pass": False,
            "operation_match_pass": False,
            "execution_ready": False,
        }
    )
    return metadata


def case_custom_gate_structure_source():
    qbit_num = 2
    vqe, topology = build_density_vqe(qbit_num)
    custom_circuit = qgd_Circuit(qbit_num)
    custom_circuit.add_H(0)
    custom_circuit.add_CNOT(1, 0)
    vqe.set_Gate_Structure(custom_circuit)

    metadata = _base_case_metadata(
        case_name="task3_story3_unsupported_custom_gate_structure_source",
        qbit_num=qbit_num,
        topology=topology,
        density_noise=[],
        purpose="Verify a custom manual circuit source fails before bridge execution with a source-category unsupported result.",
        unsupported_category="circuit_source",
        first_unsupported_condition="custom_gate_structure",
        bridge_source_type="custom_gate_structure",
    )

    def runner():
        vqe.describe_density_bridge()

    return metadata, runner, "unsupported circuit source in density backend path: custom_gate_structure"


def case_hea_zyz_lowering():
    qbit_num = 2
    vqe, topology = build_density_vqe(
        qbit_num,
        density_noise=[],
        ansatz="HEA_ZYZ",
    )
    vqe.Generate_Circuit(DEFAULT_LAYERS, DEFAULT_INNER_BLOCKS)
    parameters = build_story2_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(parameters)

    metadata = _base_case_metadata(
        case_name="task3_story3_unsupported_hea_zyz_lowering",
        qbit_num=qbit_num,
        topology=topology,
        density_noise=[],
        purpose="Verify the unsupported HEA_ZYZ lowering path fails before bridge execution.",
        unsupported_category="lowering_path",
        first_unsupported_condition="generated_hea_zyz",
        bridge_source_type="generated_hea_zyz",
        ansatz="HEA_ZYZ",
    )

    def runner():
        vqe.describe_density_bridge()

    return metadata, runner, "currently supports only the HEA ansatz"


def case_invalid_after_gate_index():
    qbit_num = 2
    density_noise = [
        {
            "channel": "local_depolarizing",
            "target": 0,
            "after_gate_index": 999,
            "error_rate": 0.1,
        }
    ]
    vqe, topology = build_density_vqe(qbit_num, density_noise=density_noise)
    vqe.Generate_Circuit(DEFAULT_LAYERS, DEFAULT_INNER_BLOCKS)

    metadata = _base_case_metadata(
        case_name="task3_story3_unsupported_after_gate_index",
        qbit_num=qbit_num,
        topology=topology,
        density_noise=density_noise,
        purpose="Verify invalid ordered noise insertion fails before density execution.",
        unsupported_category="noise_insertion",
        first_unsupported_condition="after_gate_index",
        bridge_source_type="generated_hea",
    )
    metadata["source_pass"] = True
    metadata["gate_pass"] = True

    def runner():
        vqe.describe_density_bridge()

    return metadata, runner, "after_gate_index exceeds generated gate count"


def case_invalid_noise_channel():
    qbit_num = 1
    topology = build_open_chain_topology(qbit_num)
    requested_density_noise = [
        {
            "channel": "readout_noise",
            "target": 0,
            "after_gate_index": 0,
            "value": 0.1,
        }
    ]
    metadata = _base_case_metadata(
        case_name="task3_story3_unsupported_noise_channel",
        qbit_num=qbit_num,
        topology=topology,
        density_noise=requested_density_noise,
        purpose="Verify unsupported density-noise channel names fail during bridge configuration.",
        unsupported_category="noise_type",
        first_unsupported_condition="unsupported_density_noise_channel",
        bridge_source_type="unset",
    )

    def runner():
        Variational_Quantum_Eigensolver(
            generate_zz_xx_hamiltonian(
                n_qubits=qbit_num,
                h=0.5,
                topology=topology,
                Jz=1.0,
                Jx=1.0,
                Jy=1.0,
            )[0],
            qbit_num,
            build_story2_config(),
            backend=PRIMARY_BACKEND,
            density_noise=requested_density_noise,
        )

    return metadata, runner, "Unsupported density-noise channel"


UNSUPPORTED_CASE_BUILDERS = (
    case_custom_gate_structure_source,
    case_hea_zyz_lowering,
    case_invalid_after_gate_index,
    case_invalid_noise_channel,
)


def capture_unsupported_case(case_builder, verbose=True):
    metadata, runner, expected_error_fragment = case_builder()
    try:
        runner()
        result = dict(metadata)
        result.update(
            {
                "status": "fail",
                "unsupported_reason": "Case executed successfully but was expected to fail.",
                "error_match_pass": False,
            }
        )
    except Exception as exc:
        result = dict(metadata)
        result.update(
            {
                "status": "unsupported",
                "unsupported_reason": str(exc),
                "error_match_pass": expected_error_fragment in str(exc),
            }
        )

    if verbose:
        print(
            "  {case_name:<52} category={category:<14} status={status}".format(
                case_name=result["case_name"],
                category=result["unsupported_category"],
                status=result["status"].upper(),
            )
        )

    return result


def run_validation(verbose=True):
    print("=" * 78)
    print("  Task 3 Story 3 Unsupported Bridge Validation [{}]".format(PRIMARY_BACKEND))
    print("=" * 78)
    return [capture_unsupported_case(builder, verbose=verbose) for builder in UNSUPPORTED_CASE_BUILDERS]


def build_artifact_bundle(results):
    unsupported = sum(1 for result in results if result["status"] == "unsupported")
    error_match = sum(1 for result in results if result.get("error_match_pass", False))
    total = len(results)
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if unsupported == total and error_match == total else "fail",
        "backend": PRIMARY_BACKEND,
        "requirements": build_requirement_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total,
            "unsupported_cases": unsupported,
            "failed_cases": total - unsupported,
            "error_match_count": error_match,
            "required_case_count": total,
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
        if result["status"] != "unsupported":
            details.append("status")
        if not result.get("error_match_pass", False):
            details.append("error_match")
        print(
            "  {case_name:<52} category={category:<14} status={status}{detail_suffix}".format(
                case_name=result["case_name"],
                category=result["unsupported_category"],
                status=result["status"].upper(),
                detail_suffix="" if not details else " (" + ",".join(details) + ")",
            )
        )

    print("\n" + "-" * 78)
    print(
        "  Total: {unsupported}/{total} representative unsupported bridge cases behaved as expected".format(
            unsupported=bundle["summary"]["unsupported_cases"],
            total=bundle["summary"]["total_cases"],
        )
    )
    if bundle["status"] == "pass":
        print("\n  ALL TESTS PASSED - Task 3 Story 3 unsupported bridge gate is closed.")
    else:
        print("\n  Some representative unsupported bridge cases did not behave as expected.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the Task 3 Story 3 JSON artifact bundle.",
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
