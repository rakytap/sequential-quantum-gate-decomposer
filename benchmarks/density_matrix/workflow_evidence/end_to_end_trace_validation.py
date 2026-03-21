#!/usr/bin/env python3
"""Validation: Task 6 Story 2 end-to-end workflow plus trace gate.

Builds the Task 6 Story 2 execution gate from:
- the emitted Task 6 Story 1 canonical workflow contract,
- the committed Task 5 workflow baseline bundle, from which the canonical 4q and
  6q end-to-end cases are selected,
- and the committed Task 5 raw bounded optimization trace artifact.

This layer is intentionally thin:
- it freezes one 4q case, one 6q case, and one required trace as the Story 2
  gate,
- it ties those evidence items to the Story 1 workflow identity,
- and it makes missing or malformed required evidence fail explicitly.

Run with:
    python benchmarks/density_matrix/workflow_evidence/end_to_end_trace_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    CONTRACT_VERSION,
    DEFAULT_OUTPUT_DIR as CORRECTNESS_EVIDENCE_DEFAULT_OUTPUT_DIR,
    REFERENCE_BACKEND,
    STATUS_VOCABULARY,
    WORKFLOW_ID,
    build_software_metadata,
    get_git_revision,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)

SUITE_NAME = "end_to_end_trace_validation"
ARTIFACT_FILENAME = "end_to_end_trace_bundle.json"
DEFAULT_OUTPUT_DIR = CORRECTNESS_EVIDENCE_DEFAULT_OUTPUT_DIR
STORY1_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
PLANNER_CALIBRATION_WORKFLOW_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "validation_evidence"
    / "workflow_baseline_bundle.json"
)
PLANNER_CALIBRATION_TRACE_ARTIFACT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "validation_evidence"
    / "optimization_trace_4q.json"
)
MANDATORY_END_TO_END_CASE_NAMES = ("exact_regime_4q_set_00", "exact_regime_6q_set_00")
MANDATORY_END_TO_END_QUBITS = (4, 6)
TRACE_CASE_NAME = "optimization_trace_4q"
CANONICAL_END_TO_END_PARAMETER_SET_ID = "set_00"
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "workflow_id",
    "contract_version",
    "backend",
    "reference_backend",
    "requirements",
    "thresholds",
    "software",
    "provenance",
    "summary",
    "required_artifacts",
    "cases",
    "trace_artifact",
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_story1_contract(path: Path = STORY1_CONTRACT_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story1_artifact(artifact)
        return artifact
    _, artifact = run_story1_validation(verbose=False)
    return artifact


def get_required_end_to_end_qubits(story1_contract):
    return tuple(story1_contract["thresholds"]["required_end_to_end_qubits"])


def get_required_trace_case_name(story1_contract):
    return story1_contract["input_contract"]["execution_modes"][
        "bounded_optimization_trace"
    ]["canonical_trace_case_name"]


def build_mandatory_end_to_end_case_names(story1_contract):
    return tuple(
        f"exact_regime_{qbit_num}q_{CANONICAL_END_TO_END_PARAMETER_SET_ID}"
        for qbit_num in get_required_end_to_end_qubits(story1_contract)
    )


def build_requirement_metadata(story1_contract):
    mandatory_end_to_end_qubits = get_required_end_to_end_qubits(story1_contract)
    mandatory_case_names = build_mandatory_end_to_end_case_names(story1_contract)
    required_trace_case_name = get_required_trace_case_name(story1_contract)
    return {
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "mandatory_end_to_end_case_names": list(mandatory_case_names),
        "mandatory_end_to_end_qubits": list(mandatory_end_to_end_qubits),
        "required_trace_case_name": required_trace_case_name,
        "status_vocabulary": list(STATUS_VOCABULARY),
        "required_bundle_sources": [
            story1_contract["suite_name"],
            "workflow_baseline_validation",
            "trace_anchor_validation",
        ],
    }


def build_threshold_metadata(story1_contract, workflow_bundle):
    workflow_thresholds = workflow_bundle["thresholds"]
    contract_thresholds = story1_contract["thresholds"]
    return {
        "absolute_energy_error": contract_thresholds["absolute_energy_error"],
        "rho_is_valid_tol": contract_thresholds["rho_is_valid_tol"],
        "trace_deviation": contract_thresholds["trace_deviation"],
        "observable_imag_abs": contract_thresholds["observable_imag_abs"],
        "required_pass_rate": contract_thresholds["required_pass_rate"],
        "required_end_to_end_qubits": list(
            contract_thresholds["required_end_to_end_qubits"]
        ),
        "required_trace_case_name": get_required_trace_case_name(story1_contract),
        "workflow_thresholds_match_contract": all(
            workflow_thresholds[field] == contract_thresholds[field]
            for field in (
                "absolute_energy_error",
                "rho_is_valid_tol",
                "trace_deviation",
                "observable_imag_abs",
                "required_pass_rate",
            )
        ),
    }


def validate_end_to_end_case_payload(case):
    required_fields = (
        "case_name",
        "status",
        "backend",
        "reference_backend",
        "qbit_num",
        "workflow_completed",
        "parameter_set_id",
        "absolute_energy_error",
        "energy_pass",
        "density_valid_pass",
        "trace_pass",
        "observable_pass",
        "bridge_supported_pass",
        "support_tier",
        "case_purpose",
        "counts_toward_mandatory_baseline",
        "workflow_id",
        "contract_version",
        "story6_case_role",
        "required_workflow_case",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 2 end-to-end case is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def validate_trace_artifact(trace_artifact):
    required_fields = (
        "case_name",
        "status",
        "backend",
        "qbit_num",
        "workflow_completed",
        "bridge_supported_pass",
        "optimizer",
        "parameter_count",
        "initial_energy",
        "final_energy",
        "energy_improvement",
        "total_trace_runtime_ms",
        "process_peak_rss_kb",
        "workflow_id",
        "contract_version",
        "story6_case_role",
        "required_workflow_trace",
    )
    missing_fields = [
        field for field in required_fields if field not in trace_artifact
    ]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 2 trace artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_case_identity_summary(cases, mandatory_case_names, mandatory_qubits):
    case_names = [case["case_name"] for case in cases]
    counts = Counter(case_names)
    duplicate_case_names = sorted(
        case_name for case_name, count in counts.items() if count > 1
    )
    missing_mandatory_case_names = sorted(
        set(mandatory_case_names) - set(case_names)
    )
    unexpected_case_names = sorted(
        set(case_names) - set(mandatory_case_names)
    )
    observed_qubits = sorted(case["qbit_num"] for case in cases)
    missing_mandatory_qubits = sorted(set(mandatory_qubits) - set(observed_qubits))
    unexpected_mandatory_qubits = sorted(
        set(observed_qubits) - set(mandatory_qubits)
    )
    stable_case_ids_present = (
        not duplicate_case_names
        and not missing_mandatory_case_names
        and not unexpected_case_names
        and not missing_mandatory_qubits
        and not unexpected_mandatory_qubits
        and len(case_names) == len(mandatory_case_names)
    )
    return {
        "observed_case_names": case_names,
        "observed_qubits": observed_qubits,
        "missing_mandatory_case_names": missing_mandatory_case_names,
        "missing_mandatory_qubits": missing_mandatory_qubits,
        "duplicate_case_names": duplicate_case_names,
        "unexpected_case_names": unexpected_case_names,
        "unexpected_mandatory_qubits": unexpected_mandatory_qubits,
        "stable_case_ids_present": stable_case_ids_present,
    }


def _enrich_end_to_end_case(case, story1_contract):
    case = dict(case)
    case["workflow_id"] = story1_contract["workflow_id"]
    case["contract_version"] = story1_contract["contract_version"]
    case["story6_case_role"] = "required_end_to_end"
    case["required_workflow_case"] = True
    return case


def _enrich_trace_artifact(trace_artifact, story1_contract):
    trace_artifact = dict(trace_artifact)
    trace_artifact["workflow_id"] = story1_contract["workflow_id"]
    trace_artifact["contract_version"] = story1_contract["contract_version"]
    trace_artifact["story6_case_role"] = "required_trace"
    trace_artifact["required_workflow_trace"] = True
    return trace_artifact


def _extract_mandatory_cases(workflow_bundle, story1_contract):
    selected_cases = []
    for case_name in build_mandatory_end_to_end_case_names(story1_contract):
        matching_cases = [
            case for case in workflow_bundle["cases"] if case["case_name"] == case_name
        ]
        if matching_cases:
            selected_cases.append(_enrich_end_to_end_case(matching_cases[0], story1_contract))
    return selected_cases


def build_artifact_bundle(story1_contract, workflow_bundle, trace_artifact):
    mandatory_case_names = build_mandatory_end_to_end_case_names(story1_contract)
    mandatory_end_to_end_qubits = get_required_end_to_end_qubits(story1_contract)
    required_trace_case_name = get_required_trace_case_name(story1_contract)
    cases = _extract_mandatory_cases(workflow_bundle, story1_contract)
    for case in cases:
        validate_end_to_end_case_payload(case)

    trace_artifact = _enrich_trace_artifact(trace_artifact, story1_contract)
    validate_trace_artifact(trace_artifact)

    case_identity = build_case_identity_summary(
        cases, mandatory_case_names, mandatory_end_to_end_qubits
    )
    threshold_metadata = build_threshold_metadata(story1_contract, workflow_bundle)
    total_cases = len(cases)
    passed_end_to_end_cases = sum(
        bool(
            case["status"] == "pass"
            and case["workflow_completed"]
            and case["bridge_supported_pass"]
        )
        for case in cases
    )
    unsupported_cases = sum(case["status"] == "unsupported" for case in cases)
    all_cases_required = all(case["support_tier"] == "required" for case in cases)
    all_cases_mandatory = all(
        case["counts_toward_mandatory_baseline"] for case in cases
    )
    all_cases_match_contract = all(
        case["workflow_id"] == story1_contract["workflow_id"]
        and case["contract_version"] == story1_contract["contract_version"]
        for case in cases
    )
    end_to_end_qubits_match_contract = sorted(
        case["qbit_num"] for case in cases
    ) == sorted(mandatory_end_to_end_qubits)
    required_trace_present = trace_artifact["case_name"] == required_trace_case_name and bool(
        trace_artifact["required_workflow_trace"]
    )
    required_trace_completed = bool(
        trace_artifact["status"] == "completed"
        and trace_artifact["workflow_completed"]
    )
    required_trace_bridge_supported = bool(
        trace_artifact["bridge_supported_pass"]
    )
    trace_case_name_matches_contract = (
        trace_artifact["case_name"] == required_trace_case_name
    )
    trace_matches_contract = bool(
        trace_artifact["workflow_id"] == story1_contract["workflow_id"]
        and trace_artifact["contract_version"] == story1_contract["contract_version"]
    )
    end_to_end_gate_completed = bool(
        story1_contract["status"] == "pass"
        and case_identity["stable_case_ids_present"]
        and passed_end_to_end_cases == len(mandatory_case_names)
        and unsupported_cases == 0
        and all_cases_required
        and all_cases_mandatory
        and all_cases_match_contract
        and end_to_end_qubits_match_contract
        and required_trace_present
        and required_trace_completed
        and required_trace_bridge_supported
        and trace_case_name_matches_contract
        and trace_matches_contract
        and threshold_metadata["workflow_thresholds_match_contract"]
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if end_to_end_gate_completed else "fail",
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "backend": story1_contract["backend"],
        "reference_backend": story1_contract["reference_backend"],
        "requirements": build_requirement_metadata(story1_contract),
        "thresholds": threshold_metadata,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "end_to_end_trace_validation_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_contract_path": str(STORY1_CONTRACT_PATH),
            "planner_calibration_workflow_bundle_path": str(PLANNER_CALIBRATION_WORKFLOW_BUNDLE_PATH),
            "planner_calibration_trace_artifact_path": str(PLANNER_CALIBRATION_TRACE_ARTIFACT_PATH),
        },
        "summary": {
            "total_end_to_end_cases": total_cases,
            "passed_end_to_end_cases": passed_end_to_end_cases,
            "failed_end_to_end_cases": total_cases - passed_end_to_end_cases,
            "unsupported_end_to_end_cases": unsupported_cases,
            **case_identity,
            "all_cases_required": all_cases_required,
            "all_cases_count_toward_mandatory_baseline": all_cases_mandatory,
            "all_cases_match_contract": all_cases_match_contract,
            "end_to_end_qubits_match_contract": end_to_end_qubits_match_contract,
            "required_trace_case_name": trace_artifact["case_name"],
            "required_trace_present": required_trace_present,
            "required_trace_completed": required_trace_completed,
            "required_trace_bridge_supported": required_trace_bridge_supported,
            "trace_case_name_matches_contract": trace_case_name_matches_contract,
            "trace_matches_contract": trace_matches_contract,
            "workflow_thresholds_match_contract": threshold_metadata[
                "workflow_thresholds_match_contract"
            ],
            "end_to_end_gate_completed": end_to_end_gate_completed,
        },
        "required_artifacts": {
            "story1_contract": {
                "suite_name": story1_contract["suite_name"],
                "status": story1_contract["status"],
                "workflow_id": story1_contract["workflow_id"],
                "contract_version": story1_contract["contract_version"],
                "thresholds": story1_contract["thresholds"],
                "summary": story1_contract["summary"],
            },
            "planner_calibration_workflow_baseline_reference": {
                "suite_name": workflow_bundle["suite_name"],
                "status": workflow_bundle["status"],
                "summary": workflow_bundle["summary"],
            },
            "planner_calibration_trace_artifact": {
                "case_name": trace_artifact["case_name"],
                "status": trace_artifact["status"],
                "workflow_completed": trace_artifact["workflow_completed"],
            },
        },
        "cases": cases,
        "trace_artifact": trace_artifact,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 2 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Task 6 Story 2 bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Task 6 Story 2 bundle has unexpected contract_version '{}'".format(
                bundle["contract_version"]
            )
        )
    if (
        bundle["requirements"]["mandatory_end_to_end_qubits"]
        != bundle["thresholds"]["required_end_to_end_qubits"]
    ):
        raise ValueError(
            "Task 6 Story 2 bundle has inconsistent end-to-end qubit requirements"
        )
    if (
        bundle["requirements"]["required_trace_case_name"]
        != bundle["thresholds"]["required_trace_case_name"]
    ):
        raise ValueError(
            "Task 6 Story 2 bundle has inconsistent trace-case requirements"
        )
    if bundle["summary"]["stable_case_ids_present"] is False and bundle["status"] == "pass":
        raise ValueError("Task 6 Story 2 cannot pass without stable mandatory case IDs")
    if (
        bundle["summary"]["end_to_end_qubits_match_contract"] is False
        and bundle["status"] == "pass"
    ):
        raise ValueError(
            "Task 6 Story 2 cannot pass without matching Story 1 end-to-end qubit requirements"
        )
    if (
        bundle["summary"]["trace_case_name_matches_contract"] is False
        and bundle["status"] == "pass"
    ):
        raise ValueError(
            "Task 6 Story 2 cannot pass without matching Story 1 trace-case requirements"
        )
    if (
        bundle["summary"]["workflow_thresholds_match_contract"] is False
        and bundle["status"] == "pass"
    ):
        raise ValueError(
            "Task 6 Story 2 cannot pass when workflow thresholds drift from Story 1 contract metadata"
        )
    if bundle["summary"]["end_to_end_gate_completed"] != (bundle["status"] == "pass"):
        raise ValueError(
            "Task 6 Story 2 end_to_end_gate_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_validation(
    *,
    story1_contract_path: Path = STORY1_CONTRACT_PATH,
    workflow_bundle_path: Path = PLANNER_CALIBRATION_WORKFLOW_BUNDLE_PATH,
    trace_artifact_path: Path = PLANNER_CALIBRATION_TRACE_ARTIFACT_PATH,
    verbose=False,
):
    story1_contract = _load_story1_contract(story1_contract_path)
    workflow_bundle = _load_json(workflow_bundle_path)
    trace_artifact = _load_json(trace_artifact_path)
    bundle = build_artifact_bundle(story1_contract, workflow_bundle, trace_artifact)
    if verbose:
        print(
            "{} [{}] end_to_end={}/{} trace={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["passed_end_to_end_cases"],
                bundle["summary"]["total_end_to_end_cases"],
                bundle["summary"]["required_trace_completed"],
            )
        )
    return story1_contract, workflow_bundle, trace_artifact, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 6 Story 2 JSON artifact bundle.",
    )
    parser.add_argument(
        "--story1-contract-path",
        type=Path,
        default=STORY1_CONTRACT_PATH,
        help="Path to the Task 6 Story 1 canonical contract artifact.",
    )
    parser.add_argument(
        "--workflow-bundle-path",
        type=Path,
        default=PLANNER_CALIBRATION_WORKFLOW_BUNDLE_PATH,
        help="Path to the committed Task 5 workflow baseline bundle.",
    )
    parser.add_argument(
        "--trace-artifact-path",
        type=Path,
        default=PLANNER_CALIBRATION_TRACE_ARTIFACT_PATH,
        help="Path to the committed raw trace artifact used by Story 2.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, _, bundle = run_validation(
        story1_contract_path=args.story1_contract_path,
        workflow_bundle_path=args.workflow_bundle_path,
        trace_artifact_path=args.trace_artifact_path,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["passed_end_to_end_cases"],
            bundle["summary"]["total_end_to_end_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
