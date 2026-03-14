#!/usr/bin/env python3
"""Validation: Task 6 Story 1 canonical workflow contract.

Builds a machine-readable contract artifact for the canonical Phase 2 noisy
workflow. This is intentionally a thin contract layer:
- it freezes one stable workflow ID and contract version,
- it defines explicit input and output contract sections,
- it records supported / optional / deferred / unsupported workflow boundaries,
- and it links those contract sections back to the already delivered Task 5
  evidence inventory without re-running the full validation stack.

Run with:
    python benchmarks/density_matrix/task6_story1_workflow_contract_validation.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.task4_support_tiers import SUPPORT_TIER_VOCABULARY

PRIMARY_BACKEND = "density_matrix"
REFERENCE_BACKEND = "qiskit_aer_density_matrix"
DEFAULT_ANSATZ = "HEA"
DEFAULT_LAYERS = 1
DEFAULT_INNER_BLOCKS = 1
FIXED_PARAMETER_QUBITS = (4, 6)
STORY4_WORKFLOW_QUBITS = (4, 6, 8, 10)
STORY4_PARAMETER_SET_COUNT = 10

SUITE_NAME = "task6_story1_canonical_workflow_contract"
ARTIFACT_FILENAME = "story1_canonical_workflow_contract.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase2_task6"
)
TASK5_REFERENCE_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "phase2_task5"
    / "task5_story6_publication_bundle.json"
)
WORKFLOW_ID = "phase2_xxz_hea_density_matrix_anchor_workflow"
CONTRACT_VERSION = "v1"
STATUS_VOCABULARY = ("pass", "fail", "incomplete", "completed", "unsupported")
BOUNDARY_CLASS_NAMES = ("supported", "optional", "deferred", "unsupported")
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "workflow_id",
    "contract_version",
    "backend",
    "reference_backend",
    "requirements",
    "input_contract",
    "output_contract",
    "boundary_classification",
    "reference_artifacts",
    "software",
    "provenance",
    "summary",
)


def build_story2_noise():
    return [
        {
            "channel": "local_depolarizing",
            "target": 0,
            "after_gate_index": 0,
            "error_rate": 0.1,
        },
        {
            "channel": "amplitude_damping",
            "target": 1,
            "after_gate_index": 2,
            "gamma": 0.05,
        },
        {
            "channel": "phase_damping",
            "target": 0,
            "after_gate_index": 4,
            "lambda": 0.07,
        },
    ]


def build_open_chain_topology(qbit_num: int):
    return [(idx, idx + 1) for idx in range(qbit_num - 1)]


def build_story2_config():
    return {
        "max_inner_iterations": 4,
        "max_iterations": 1,
        "convergence_length": 2,
    }


def build_story2_hamiltonian_metadata():
    return {
        "Jx": 1.0,
        "Jy": 1.0,
        "Jz": 1.0,
        "h": 0.5,
    }


def _get_package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def build_software_metadata():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "qiskit": _get_package_version("qiskit"),
        "qiskit_aer": _get_package_version("qiskit-aer"),
    }


def get_git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _load_reference_task5_bundle(reference_path: Path = TASK5_REFERENCE_BUNDLE_PATH):
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    required_fields = ("suite_name", "status", "artifacts", "software", "provenance")
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "Task 5 reference bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )
    return payload


def build_requirement_metadata(reference_bundle):
    mandatory_reference_artifact_ids = [
        artifact["artifact_id"]
        for artifact in reference_bundle["artifacts"]
        if artifact.get("mandatory", False)
    ]
    return {
        "workflow_id": WORKFLOW_ID,
        "contract_version": CONTRACT_VERSION,
        "support_tier_vocabulary": list(SUPPORT_TIER_VOCABULARY),
        "status_vocabulary": list(STATUS_VOCABULARY),
        "end_to_end_qubits": list(FIXED_PARAMETER_QUBITS),
        "fixed_parameter_matrix_qubits": list(STORY4_WORKFLOW_QUBITS),
        "documented_anchor_qubit": 10,
        "fixed_parameter_sets_per_size": STORY4_PARAMETER_SET_COUNT,
        "required_input_fields": [
            "workflow_id",
            "contract_version",
            "hamiltonian_family",
            "hamiltonian_parameters",
            "qubit_inventory",
            "ansatz",
            "backend_selection",
            "noise_schedule_policy",
            "execution_modes",
        ],
        "required_output_fields": [
            "real_energy_semantics",
            "case_status_vocabulary",
            "bundle_status_vocabulary",
            "required_case_fields",
            "required_trace_fields",
            "required_bundle_fields",
            "stability_metrics",
        ],
        "required_boundary_classes": list(BOUNDARY_CLASS_NAMES),
        "required_reference_artifact_ids": mandatory_reference_artifact_ids,
        "required_reference_bundle_sources": [
            "task5_story6_publication_evidence",
        ],
    }


def build_input_contract():
    return {
        "workflow_id": WORKFLOW_ID,
        "contract_version": CONTRACT_VERSION,
        "workflow_family": "noisy_vqe_ground_state_estimation",
        "hamiltonian_family": "1d_xxz_with_local_z_field",
        "hamiltonian_parameters": build_story2_hamiltonian_metadata(),
        "qubit_inventory": {
            "end_to_end_qubits": list(FIXED_PARAMETER_QUBITS),
            "fixed_parameter_matrix_qubits": list(STORY4_WORKFLOW_QUBITS),
            "documented_anchor_qubit": 10,
            "topology_family": "open_chain",
            "example_topology_4q": build_open_chain_topology(4),
        },
        "ansatz": {
            "family": DEFAULT_ANSATZ,
            "layers": DEFAULT_LAYERS,
            "inner_blocks": DEFAULT_INNER_BLOCKS,
            "source_type": "generated_hea_circuit",
        },
        "backend_selection": {
            "selected_backend": PRIMARY_BACKEND,
            "selection_mode": "explicit",
            "reference_backend": REFERENCE_BACKEND,
            "silent_fallback_allowed": False,
        },
        "noise_schedule_policy": {
            "insertion_policy": "explicit_ordered_after_gate_index",
            "required_local_noise_models": [
                "local_depolarizing",
                "amplitude_damping",
                "phase_damping",
            ],
            "canonical_example_schedule": build_story2_noise(),
        },
        "execution_modes": {
            "fixed_parameter_matrix": {
                "required": True,
                "parameter_sets_per_size": STORY4_PARAMETER_SET_COUNT,
            },
            "bounded_optimization_trace": {
                "required": True,
                "canonical_trace_case_name": "story2_trace_4q",
                "optimizer_name": "COSINE",
                "optimizer_config": build_story2_config(),
            },
        },
    }


def build_output_contract():
    return {
        "real_energy_semantics": "Return exact Hermitian energy through Re Tr(H*rho).",
        "case_status_vocabulary": list(STATUS_VOCABULARY),
        "bundle_status_vocabulary": ["pass", "fail"],
        "required_case_fields": [
            "case_name",
            "status",
            "backend",
            "reference_backend",
            "qbit_num",
            "support_tier",
            "case_purpose",
            "counts_toward_mandatory_baseline",
            "workflow_completed",
            "absolute_energy_error",
            "energy_pass",
            "density_valid_pass",
            "trace_pass",
            "observable_pass",
            "bridge_supported_pass",
            "total_case_runtime_ms",
            "process_peak_rss_kb",
        ],
        "required_trace_fields": [
            "case_name",
            "status",
            "workflow_completed",
            "bridge_supported_pass",
            "optimizer",
            "parameter_count",
            "initial_energy",
            "final_energy",
            "energy_improvement",
            "total_trace_runtime_ms",
            "process_peak_rss_kb",
        ],
        "required_bundle_fields": [
            "suite_name",
            "status",
            "requirements",
            "thresholds",
            "software",
            "summary",
        ],
        "stability_metrics": [
            "workflow_completed",
            "total_case_runtime_ms",
            "total_trace_runtime_ms",
            "process_peak_rss_kb",
        ],
    }


def build_boundary_classification():
    return {
        "supported": {
            "backend_modes": [PRIMARY_BACKEND],
            "workflow_anchor": "xxz_plus_hea_plus_density_matrix",
            "bridge_source_type": "generated_hea_circuit",
            "required_gate_families": ["U3", "CNOT"],
            "required_local_noise_models": [
                "local_depolarizing",
                "amplitude_damping",
                "phase_damping",
            ],
            "execution_modes": [
                "fixed_parameter_matrix",
                "bounded_optimization_trace",
            ],
        },
        "optional": {
            "secondary_reference_baselines": [
                "one_additional_simulator_if_justified",
            ],
            "regression_or_stress_only": [
                "whole_register_depolarizing",
            ],
            "supplemental_workflow_evidence": [
                "extra_fixed_parameter_cases_not_counting_toward_mandatory_closure",
            ],
        },
        "deferred": {
            "noise_families": [
                "correlated_multi_qubit_noise",
                "readout_noise",
                "calibration_aware_noise",
                "non_markovian_noise",
            ],
            "workflow_extensions": [
                "broader_noisy_algorithm_families",
                "broader_hamiltonian_classes",
                "phase3_acceleration_claims",
                "phase4_optimizer_studies",
            ],
        },
        "unsupported": {
            "backend_behaviors": [
                "implicit_auto_backend_selection",
                "silent_density_to_state_vector_fallback",
            ],
            "workflow_conditions": [
                "unsupported_bridge_input",
                "unsupported_gate_or_noise_schedule",
                "unsupported_observable_request",
                "backend_incompatible_request",
            ],
        },
    }


def build_reference_artifacts(reference_bundle):
    artifacts = []
    for artifact in reference_bundle["artifacts"]:
        artifacts.append(
            {
                "artifact_id": artifact["artifact_id"],
                "artifact_class": artifact["artifact_class"],
                "mandatory": artifact["mandatory"],
                "path": artifact["path"],
                "status": artifact["status"],
                "expected_statuses": list(artifact["expected_statuses"]),
                "purpose": artifact["purpose"],
                "generation_command": artifact["generation_command"],
            }
        )
    return artifacts


def _contract_sections_complete(artifact):
    required_input_fields = artifact["requirements"]["required_input_fields"]
    required_output_fields = artifact["requirements"]["required_output_fields"]
    required_boundary_classes = set(artifact["requirements"]["required_boundary_classes"])
    return bool(
        artifact["workflow_id"]
        and artifact["contract_version"]
        and all(field in artifact["input_contract"] for field in required_input_fields)
        and all(field in artifact["output_contract"] for field in required_output_fields)
        and required_boundary_classes == set(artifact["boundary_classification"].keys())
        and artifact["reference_artifacts"]
    )


def build_artifact_bundle(reference_bundle):
    artifact = {
        "suite_name": SUITE_NAME,
        "status": "fail",
        "workflow_id": WORKFLOW_ID,
        "contract_version": CONTRACT_VERSION,
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "requirements": build_requirement_metadata(reference_bundle),
        "input_contract": build_input_contract(),
        "output_contract": build_output_contract(),
        "boundary_classification": build_boundary_classification(),
        "reference_artifacts": build_reference_artifacts(reference_bundle),
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task6_story1_workflow_contract_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "reference_task5_bundle_path": str(TASK5_REFERENCE_BUNDLE_PATH),
        },
        "summary": {},
    }
    artifact["summary"] = {
        "required_input_field_count": len(
            artifact["requirements"]["required_input_fields"]
        ),
        "required_output_field_count": len(
            artifact["requirements"]["required_output_fields"]
        ),
        "boundary_class_count": len(artifact["boundary_classification"]),
        "mandatory_reference_artifact_count": sum(
            entry["mandatory"] for entry in artifact["reference_artifacts"]
        ),
        "reference_artifact_count": len(artifact["reference_artifacts"]),
        "contract_sections_complete": _contract_sections_complete(artifact),
    }
    artifact["status"] = "pass" if artifact["summary"]["contract_sections_complete"] else "fail"
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 1 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if not artifact["workflow_id"]:
        raise ValueError("Task 6 Story 1 workflow_id must be non-empty")
    if not artifact["contract_version"]:
        raise ValueError("Task 6 Story 1 contract_version must be non-empty")

    if artifact["status"] not in {"pass", "fail"}:
        raise ValueError(
            "Task 6 Story 1 artifact has unsupported status '{}'".format(
                artifact["status"]
            )
        )

    required_boundary_classes = set(artifact["requirements"]["required_boundary_classes"])
    observed_boundary_classes = set(artifact["boundary_classification"].keys())
    if required_boundary_classes != observed_boundary_classes:
        raise ValueError(
            "Task 6 Story 1 boundary classes mismatch: expected {}, observed {}".format(
                sorted(required_boundary_classes), sorted(observed_boundary_classes)
            )
        )

    for field in artifact["requirements"]["required_input_fields"]:
        if field not in artifact["input_contract"]:
            raise ValueError(
                "Task 6 Story 1 input_contract is missing required field '{}'".format(
                    field
                )
            )

    for field in artifact["requirements"]["required_output_fields"]:
        if field not in artifact["output_contract"]:
            raise ValueError(
                "Task 6 Story 1 output_contract is missing required field '{}'".format(
                    field
                )
            )

    if not artifact["reference_artifacts"]:
        raise ValueError("Task 6 Story 1 requires at least one reference artifact")

    required_reference_fields = (
        "artifact_id",
        "artifact_class",
        "mandatory",
        "path",
        "status",
        "expected_statuses",
        "purpose",
        "generation_command",
    )
    for entry in artifact["reference_artifacts"]:
        missing_reference_fields = [
            field for field in required_reference_fields if field not in entry
        ]
        if missing_reference_fields:
            raise ValueError(
                "Task 6 Story 1 reference artifact is missing fields: {}".format(
                    ", ".join(missing_reference_fields)
                )
            )

    contract_sections_complete = _contract_sections_complete(artifact)
    if artifact["summary"]["contract_sections_complete"] != contract_sections_complete:
        raise ValueError(
            "Task 6 Story 1 contract_sections_complete summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_validation(*, verbose=False, reference_bundle_path: Path = TASK5_REFERENCE_BUNDLE_PATH):
    reference_bundle = _load_reference_task5_bundle(reference_bundle_path)
    artifact = build_artifact_bundle(reference_bundle)
    if verbose:
        print(
            "{} [{}] workflow_id={} reference_artifacts={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["workflow_id"],
                artifact["summary"]["reference_artifact_count"],
            )
        )
    return reference_bundle, artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 6 Story 1 JSON artifact bundle.",
    )
    parser.add_argument(
        "--reference-bundle-path",
        type=Path,
        default=TASK5_REFERENCE_BUNDLE_PATH,
        help="Path to the Task 5 publication bundle used as the reference inventory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, artifact = run_validation(
        verbose=not args.quiet,
        reference_bundle_path=args.reference_bundle_path,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, artifact)
    print(
        "Wrote {} with status {} ({})".format(
            output_path,
            artifact["status"],
            artifact["workflow_id"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
