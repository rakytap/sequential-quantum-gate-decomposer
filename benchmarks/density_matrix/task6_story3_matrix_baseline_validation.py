#!/usr/bin/env python3
"""Validation: Task 6 Story 3 fixed-parameter matrix baseline.

Builds the Task 6 Story 3 matrix gate from:
- the emitted Task 6 Story 1 canonical workflow contract,
- the emitted Task 6 Story 2 end-to-end plus trace bundle,
- and the committed Task 5 workflow baseline bundle, which already contains the
  rich 4/6/8/10 fixed-parameter matrix evidence.

This layer is intentionally thin:
- it preserves one stable exact-regime matrix identity,
- it rebinds matrix cases to the canonical Task 6 workflow ID and version,
- and it fails explicitly when required matrix identity or 10-qubit anchor
  evidence is incomplete.

Run with:
    python benchmarks/density_matrix/task6_story3_matrix_baseline_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.task6_story1_workflow_contract_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    CONTRACT_VERSION,
    DEFAULT_OUTPUT_DIR as TASK6_DEFAULT_OUTPUT_DIR,
    REFERENCE_BACKEND,
    WORKFLOW_ID,
    build_software_metadata,
    get_git_revision,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)
from benchmarks.density_matrix.task6_story2_end_to_end_trace_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    validate_artifact_bundle as validate_story2_artifact,
)

SUITE_NAME = "task6_story3_matrix_baseline"
ARTIFACT_FILENAME = "story3_matrix_baseline_bundle.json"
DEFAULT_OUTPUT_DIR = TASK6_DEFAULT_OUTPUT_DIR
STORY1_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
TASK5_WORKFLOW_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "phase2_task5"
    / "story2_workflow_baseline_bundle.json"
)
MANDATORY_WORKFLOW_QUBITS = (4, 6, 8, 10)
PARAMETER_SET_COUNT = 10
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


def _load_story2_bundle(path: Path = STORY2_BUNDLE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story2_artifact(artifact)
        return artifact
    _, _, _, artifact = run_story2_validation(verbose=False)
    return artifact


def get_required_workflow_qubits(story1_contract):
    return tuple(story1_contract["thresholds"]["required_workflow_qubits"])


def get_required_parameter_set_count(story1_contract):
    return int(story1_contract["thresholds"]["fixed_parameter_sets_per_size"])


def get_documented_anchor_qubit(story1_contract):
    return int(story1_contract["thresholds"]["documented_anchor_qubit"])


def build_requirement_metadata(task5_workflow_bundle, story1_contract):
    requirements = task5_workflow_bundle["requirements"]
    return {
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "mandatory_workflow_qubits": list(get_required_workflow_qubits(story1_contract)),
        "fixed_parameter_sets_per_size": get_required_parameter_set_count(
            story1_contract
        ),
        "mandatory_parameter_set_ids": list(requirements["mandatory_parameter_set_ids"]),
        "required_case_names": list(requirements["mandatory_case_names"]),
        "documented_anchor_qubit": get_documented_anchor_qubit(story1_contract),
        "required_bundle_sources": [
            story1_contract["suite_name"],
            "task6_story2_end_to_end_trace",
            "task5_story2_exact_regime_workflow",
        ],
    }


def build_threshold_metadata(story1_contract, task5_workflow_bundle):
    contract_thresholds = story1_contract["thresholds"]
    workflow_thresholds = task5_workflow_bundle["thresholds"]
    threshold_fields = (
        "absolute_energy_error",
        "rho_is_valid_tol",
        "trace_deviation",
        "observable_imag_abs",
        "required_pass_rate",
        "required_workflow_qubits",
        "fixed_parameter_sets_per_size",
        "documented_anchor_qubit",
    )
    return {
        field: contract_thresholds[field] for field in threshold_fields
    }, all(
        workflow_thresholds[field] == contract_thresholds[field]
        for field in threshold_fields
        if field in workflow_thresholds
    )


def validate_case_payload(case):
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
        "required_story6_matrix_case",
    )
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 3 matrix case is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_case_identity_summary(cases, requirements):
    expected_case_names = set(requirements["required_case_names"])
    expected_parameter_set_ids = set(requirements["mandatory_parameter_set_ids"])

    observed_case_names = [case["case_name"] for case in cases]
    case_counts = Counter(observed_case_names)
    duplicate_case_names = sorted(
        case_name for case_name, count in case_counts.items() if count > 1
    )
    missing_mandatory_case_names = sorted(expected_case_names - set(observed_case_names))
    unexpected_case_names = sorted(set(observed_case_names) - expected_case_names)

    parameter_sets_per_qbit = defaultdict(list)
    for case in cases:
        parameter_sets_per_qbit[case["qbit_num"]].append(case["parameter_set_id"])

    missing_parameter_set_ids_by_qbit = {}
    duplicate_parameter_set_ids_by_qbit = {}
    cases_per_qbit = {}
    for qbit_num in requirements["mandatory_workflow_qubits"]:
        observed_ids = parameter_sets_per_qbit.get(qbit_num, [])
        counts = Counter(observed_ids)
        duplicate_ids = sorted(
            parameter_set_id
            for parameter_set_id, count in counts.items()
            if count > 1
        )
        missing_ids = sorted(expected_parameter_set_ids - set(observed_ids))
        duplicate_parameter_set_ids_by_qbit[str(qbit_num)] = duplicate_ids
        missing_parameter_set_ids_by_qbit[str(qbit_num)] = missing_ids
        cases_per_qbit[str(qbit_num)] = len(observed_ids)

    stable_case_ids_present = (
        not missing_mandatory_case_names
        and not duplicate_case_names
        and not unexpected_case_names
        and len(observed_case_names) == len(expected_case_names)
    )
    stable_parameter_set_ids_present = all(
        not missing_parameter_set_ids_by_qbit[str(qbit_num)]
        and not duplicate_parameter_set_ids_by_qbit[str(qbit_num)]
        for qbit_num in requirements["mandatory_workflow_qubits"]
    )

    return {
        "observed_case_names": observed_case_names,
        "missing_mandatory_case_names": missing_mandatory_case_names,
        "duplicate_case_names": duplicate_case_names,
        "unexpected_case_names": unexpected_case_names,
        "stable_case_ids_present": stable_case_ids_present,
        "cases_per_qbit": cases_per_qbit,
        "missing_parameter_set_ids_by_qbit": missing_parameter_set_ids_by_qbit,
        "duplicate_parameter_set_ids_by_qbit": duplicate_parameter_set_ids_by_qbit,
        "stable_parameter_set_ids_present": stable_parameter_set_ids_present,
    }


def _enrich_case(case, story1_contract):
    case = dict(case)
    case["workflow_id"] = story1_contract["workflow_id"]
    case["contract_version"] = story1_contract["contract_version"]
    case["story6_case_role"] = "required_fixed_parameter_matrix"
    case["required_story6_matrix_case"] = True
    return case


def build_artifact_bundle(story1_contract, story2_bundle, task5_workflow_bundle):
    requirements = build_requirement_metadata(task5_workflow_bundle, story1_contract)
    threshold_metadata, workflow_thresholds_match_contract = build_threshold_metadata(
        story1_contract, task5_workflow_bundle
    )
    cases = [_enrich_case(case, story1_contract) for case in task5_workflow_bundle["cases"]]
    for case in cases:
        validate_case_payload(case)

    case_identity = build_case_identity_summary(cases, requirements)
    total_cases = len(cases)
    passed_cases = sum(case["status"] == "pass" for case in cases)
    unsupported_cases = sum(case["status"] == "unsupported" for case in cases)
    bridge_supported_cases = sum(
        case.get("bridge_supported_pass", False) for case in cases
    )
    all_cases_required = all(case["support_tier"] == "required" for case in cases)
    all_cases_mandatory = all(
        case["counts_toward_mandatory_baseline"] for case in cases
    )
    all_cases_match_contract = all(
        case["workflow_id"] == story1_contract["workflow_id"]
        and case["contract_version"] == story1_contract["contract_version"]
        for case in cases
    )
    workflow_inventory_matches_contract = bool(
        task5_workflow_bundle["requirements"]["mandatory_workflow_qubits"]
        == requirements["mandatory_workflow_qubits"]
        and task5_workflow_bundle["requirements"]["fixed_parameter_sets_per_size"]
        == requirements["fixed_parameter_sets_per_size"]
    )
    documented_10q_anchor_present = any(
        case["qbit_num"] == requirements["documented_anchor_qubit"] for case in cases
    )
    matrix_gate_completed = bool(
        story1_contract["status"] == "pass"
        and story2_bundle["status"] == "pass"
        and task5_workflow_bundle["status"] == "pass"
        and case_identity["stable_case_ids_present"]
        and case_identity["stable_parameter_set_ids_present"]
        and passed_cases == total_cases
        and unsupported_cases == 0
        and bridge_supported_cases == total_cases
        and all_cases_required
        and all_cases_mandatory
        and all_cases_match_contract
        and workflow_inventory_matches_contract
        and workflow_thresholds_match_contract
        and documented_10q_anchor_present
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if matrix_gate_completed else "fail",
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "backend": story1_contract["backend"],
        "reference_backend": story1_contract["reference_backend"],
        "requirements": requirements,
        "thresholds": threshold_metadata,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task6_story3_matrix_baseline_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_contract_path": str(STORY1_CONTRACT_PATH),
            "story2_bundle_path": str(STORY2_BUNDLE_PATH),
            "task5_workflow_bundle_path": str(TASK5_WORKFLOW_BUNDLE_PATH),
        },
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "unsupported_cases": unsupported_cases,
            "bridge_supported_cases": bridge_supported_cases,
            "required_cases": total_cases,
            "required_passed_cases": passed_cases,
            "required_pass_rate": (passed_cases / total_cases) if total_cases else 0.0,
            "documented_10q_anchor_present": documented_10q_anchor_present,
            "workflow_inventory_matches_contract": workflow_inventory_matches_contract,
            "workflow_thresholds_match_contract": workflow_thresholds_match_contract,
            "all_cases_required": all_cases_required,
            "all_cases_count_toward_mandatory_baseline": all_cases_mandatory,
            "all_cases_match_contract": all_cases_match_contract,
            **case_identity,
            "matrix_gate_completed": matrix_gate_completed,
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
            "story2_end_to_end_trace": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            },
            "task5_story2_workflow_baseline": {
                "suite_name": task5_workflow_bundle["suite_name"],
                "status": task5_workflow_bundle["status"],
                "summary": task5_workflow_bundle["summary"],
            },
        },
        "cases": cases,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 3 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Task 6 Story 3 bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Task 6 Story 3 bundle has unexpected contract_version '{}'".format(
                bundle["contract_version"]
            )
        )
    if (
        bundle["requirements"]["mandatory_workflow_qubits"]
        != bundle["thresholds"]["required_workflow_qubits"]
    ):
        raise ValueError(
            "Task 6 Story 3 bundle has inconsistent workflow-qubit requirements"
        )
    if (
        bundle["requirements"]["fixed_parameter_sets_per_size"]
        != bundle["thresholds"]["fixed_parameter_sets_per_size"]
    ):
        raise ValueError(
            "Task 6 Story 3 bundle has inconsistent parameter-set-count requirements"
        )
    if (
        bundle["requirements"]["documented_anchor_qubit"]
        != bundle["thresholds"]["documented_anchor_qubit"]
    ):
        raise ValueError(
            "Task 6 Story 3 bundle has inconsistent documented-anchor requirements"
        )
    if bundle["summary"]["matrix_gate_completed"] != (bundle["status"] == "pass"):
        raise ValueError(
            "Task 6 Story 3 matrix_gate_completed summary is inconsistent"
        )
    for case in bundle["cases"]:
        validate_case_payload(case)
        if case["workflow_id"] != bundle["workflow_id"]:
            raise ValueError(
                "Task 6 Story 3 case '{}' does not match bundle workflow_id".format(
                    case["case_name"]
                )
            )
        if case["contract_version"] != bundle["contract_version"]:
            raise ValueError(
                "Task 6 Story 3 case '{}' does not match bundle contract_version".format(
                    case["case_name"]
                )
            )
    if not bundle["summary"]["all_cases_match_contract"] and bundle["status"] == "pass":
        raise ValueError(
            "Task 6 Story 3 cannot pass when matrix cases do not match the canonical contract"
        )
    if (
        bundle["summary"]["workflow_inventory_matches_contract"] is False
        and bundle["status"] == "pass"
    ):
        raise ValueError(
            "Task 6 Story 3 cannot pass when matrix inventory drifts from Story 1 contract metadata"
        )
    if (
        bundle["summary"]["workflow_thresholds_match_contract"] is False
        and bundle["status"] == "pass"
    ):
        raise ValueError(
            "Task 6 Story 3 cannot pass when matrix thresholds drift from Story 1 contract metadata"
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
    story2_bundle_path: Path = STORY2_BUNDLE_PATH,
    task5_workflow_bundle_path: Path = TASK5_WORKFLOW_BUNDLE_PATH,
    verbose=False,
):
    story1_contract = _load_story1_contract(story1_contract_path)
    story2_bundle = _load_story2_bundle(story2_bundle_path)
    task5_workflow_bundle = _load_json(task5_workflow_bundle_path)
    bundle = build_artifact_bundle(
        story1_contract,
        story2_bundle,
        task5_workflow_bundle,
    )
    if verbose:
        print(
            "{} [{}] matrix={}/{} 10q_anchor={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["required_passed_cases"],
                bundle["summary"]["required_cases"],
                bundle["summary"]["documented_10q_anchor_present"],
            )
        )
    return story1_contract, story2_bundle, task5_workflow_bundle, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 6 Story 3 JSON artifact bundle.",
    )
    parser.add_argument(
        "--story1-contract-path",
        type=Path,
        default=STORY1_CONTRACT_PATH,
        help="Path to the Task 6 Story 1 canonical workflow contract artifact.",
    )
    parser.add_argument(
        "--story2-bundle-path",
        type=Path,
        default=STORY2_BUNDLE_PATH,
        help="Path to the Task 6 Story 2 end-to-end plus trace bundle.",
    )
    parser.add_argument(
        "--task5-workflow-bundle-path",
        type=Path,
        default=TASK5_WORKFLOW_BUNDLE_PATH,
        help="Path to the committed Task 5 workflow baseline bundle.",
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
        story2_bundle_path=args.story2_bundle_path,
        task5_workflow_bundle_path=args.task5_workflow_bundle_path,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["required_passed_cases"],
            bundle["summary"]["required_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
