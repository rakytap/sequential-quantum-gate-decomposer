#!/usr/bin/env python3
"""Validation: Task 6 Story 4 unsupported workflow boundaries.

Builds the Task 6 Story 4 negative-evidence layer from:
- the emitted Task 6 Story 1 canonical workflow contract,
- the emitted Task 6 Story 2 end-to-end plus trace bundle,
- the emitted Task 6 Story 3 matrix baseline bundle,
- the committed Task 4 unsupported/deferred noise bundle,
- and the committed backend-mismatch unsupported case.

This layer is intentionally thin:
- it keeps unsupported and deferred cases explicit and machine-readable,
- it binds them to the same canonical workflow identity as the positive-path
  Task 6 bundles,
- and it fails explicitly when negative evidence is incomplete or ambiguous.

Run with:
    python benchmarks/density_matrix/workflow_evidence/unsupported_workflow_validation.py
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

from benchmarks.density_matrix.noise_support.support_tiers import (
    SUPPORT_TIER_DEFERRED,
    SUPPORT_TIER_UNSUPPORTED,
)
from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
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
from benchmarks.density_matrix.workflow_evidence.end_to_end_trace_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    validate_artifact_bundle as validate_story2_artifact,
)
from benchmarks.density_matrix.workflow_evidence.matrix_baseline_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)

SUITE_NAME = "unsupported_workflow_validation"
ARTIFACT_FILENAME = "unsupported_workflow_bundle.json"
DEFAULT_OUTPUT_DIR = TASK6_DEFAULT_OUTPUT_DIR
STORY1_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
STORY3_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / STORY3_ARTIFACT_FILENAME
TASK4_UNSUPPORTED_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "noise_support"
    / "unsupported_noise_bundle.json"
)
TASK2_BACKEND_MISMATCH_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "exact_density_validation"
    / "unsupported_state_vector_density_noise.json"
)
BACKEND_MISMATCH_CASE_NAME = "unsupported_state_vector_density_noise"
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


def _load_story3_bundle(path: Path = STORY3_BUNDLE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story3_artifact(artifact)
        return artifact
    _, _, _, artifact = run_story3_validation(verbose=False)
    return artifact


def get_required_unsupported_case_fields(story1_contract):
    return tuple(story1_contract["output_contract"]["required_unsupported_case_fields"])


def build_requirement_metadata(task4_unsupported_bundle, story1_contract):
    categories = sorted(
        set(case["unsupported_category"] for case in task4_unsupported_bundle["cases"])
        | {"backend_incompatible_request"}
    )
    boundary_classes = sorted(
        set(case["noise_boundary_class"] for case in task4_unsupported_bundle["cases"])
        | {"backend_mode"}
    )
    required_case_names = [
        case["case_name"] for case in task4_unsupported_bundle["cases"]
    ] + [BACKEND_MISMATCH_CASE_NAME]
    return {
        "workflow_id": WORKFLOW_ID,
        "contract_version": CONTRACT_VERSION,
        "required_case_names": required_case_names,
        "required_support_tiers": [
            SUPPORT_TIER_DEFERRED,
            SUPPORT_TIER_UNSUPPORTED,
        ],
        "required_case_fields": list(
            get_required_unsupported_case_fields(story1_contract)
        ),
        "required_categories": categories,
        "required_boundary_classes": boundary_classes,
        "required_bundle_sources": [
            "workflow_contract_validation",
            "end_to_end_trace_validation",
            "matrix_baseline_validation",
            "unsupported_noise_boundary",
            "exact_density_validation_backend_mismatch_case",
        ],
    }


def _augment_backend_mismatch_case(raw_case):
    case = dict(raw_case)
    case.update(
        {
            "reference_backend": REFERENCE_BACKEND,
            "support_tier": SUPPORT_TIER_UNSUPPORTED,
            "case_purpose": "unsupported_scope_guard",
            "counts_toward_mandatory_baseline": False,
            "source_unsupported_category": case.get("unsupported_category"),
            "unsupported_category": "backend_incompatible_request",
            "first_unsupported_condition": "state_vector_backend_density_noise",
            "noise_boundary_class": "backend_mode",
            "failure_stage": "density_anchor_preflight",
            "pre_execution_failure_pass": True,
            "silent_fallback_detected": False,
            "silent_substitution_detected": False,
            "unsupported_boundary_pass": case["status"] == "unsupported",
            "workflow_completed": False,
        }
    )
    return case


def _enrich_case(case):
    case = dict(case)
    case["workflow_id"] = WORKFLOW_ID
    case["contract_version"] = CONTRACT_VERSION
    case["story6_case_role"] = "unsupported_boundary"
    case["required_unsupported_case"] = True
    return case


def validate_case_payload(case, required_fields):
    missing_fields = [field for field in required_fields if field not in case]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 4 case is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def build_artifact_bundle(
    story1_contract,
    story2_bundle,
    story3_bundle,
    task4_unsupported_bundle,
    backend_mismatch_case,
):
    requirements = build_requirement_metadata(task4_unsupported_bundle, story1_contract)
    cases = [_enrich_case(case) for case in task4_unsupported_bundle["cases"]]
    cases.append(_enrich_case(_augment_backend_mismatch_case(backend_mismatch_case)))
    for case in cases:
        validate_case_payload(case, requirements["required_case_fields"])

    expected_case_names = set(requirements["required_case_names"])
    observed_case_names = [case["case_name"] for case in cases]
    case_counts = Counter(observed_case_names)
    duplicate_case_names = sorted(
        case_name for case_name, count in case_counts.items() if count > 1
    )
    missing_case_names = sorted(expected_case_names - set(observed_case_names))
    unexpected_case_names = sorted(set(observed_case_names) - expected_case_names)

    deferred_cases = sum(case["support_tier"] == SUPPORT_TIER_DEFERRED for case in cases)
    unsupported_cases = sum(
        case["support_tier"] == SUPPORT_TIER_UNSUPPORTED for case in cases
    )
    unsupported_status_cases = sum(case["status"] == "unsupported" for case in cases)
    mandatory_baseline_case_count = sum(
        case["counts_toward_mandatory_baseline"] for case in cases
    )
    pre_execution_failure_passed_cases = sum(
        case.get("pre_execution_failure_pass", False) for case in cases
    )
    no_silent_fallback_cases = sum(
        not case.get("silent_fallback_detected", False) for case in cases
    )
    no_silent_substitution_cases = sum(
        not case.get("silent_substitution_detected", False) for case in cases
    )
    first_condition_present_cases = sum(
        bool(case.get("first_unsupported_condition")) for case in cases
    )
    categories_present = sorted(set(case["unsupported_category"] for case in cases))
    boundary_classes_present = sorted(
        set(case["noise_boundary_class"] for case in cases)
    )
    backend_incompatible_case_present = any(
        case["case_name"] == BACKEND_MISMATCH_CASE_NAME for case in cases
    )
    all_cases_match_contract = all(
        case["workflow_id"] == story1_contract["workflow_id"]
        and case["contract_version"] == story1_contract["contract_version"]
        for case in cases
    )
    unsupported_gate_completed = bool(
        story1_contract["status"] == "pass"
        and story2_bundle["status"] == "pass"
        and story3_bundle["status"] == "pass"
        and task4_unsupported_bundle["status"] == "pass"
        and not duplicate_case_names
        and not missing_case_names
        and not unexpected_case_names
        and unsupported_status_cases == len(cases)
        and mandatory_baseline_case_count == 0
        and pre_execution_failure_passed_cases == len(cases)
        and no_silent_fallback_cases == len(cases)
        and no_silent_substitution_cases == len(cases)
        and first_condition_present_cases == len(cases)
        and backend_incompatible_case_present
        and all_cases_match_contract
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if unsupported_gate_completed else "fail",
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "backend": story1_contract["backend"],
        "reference_backend": story1_contract["reference_backend"],
        "requirements": requirements,
        "thresholds": {
            "expected_status": "unsupported",
            "silent_fallback_allowed": story1_contract["input_contract"][
                "backend_selection"
            ]["silent_fallback_allowed"],
            "silent_substitution_allowed": False,
            "mandatory_baseline_case_count": 0,
        },
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "unsupported_workflow_validation_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_contract_path": str(STORY1_CONTRACT_PATH),
            "story2_bundle_path": str(STORY2_BUNDLE_PATH),
            "story3_bundle_path": str(STORY3_BUNDLE_PATH),
            "task4_unsupported_bundle_path": str(TASK4_UNSUPPORTED_BUNDLE_PATH),
            "backend_mismatch_case_path": str(TASK2_BACKEND_MISMATCH_PATH),
        },
        "summary": {
            "total_cases": len(cases),
            "unsupported_status_cases": unsupported_status_cases,
            "unsupported_cases": unsupported_cases,
            "deferred_cases": deferred_cases,
            "mandatory_baseline_case_count": mandatory_baseline_case_count,
            "duplicate_case_names": duplicate_case_names,
            "missing_case_names": missing_case_names,
            "unexpected_case_names": unexpected_case_names,
            "pre_execution_failure_passed_cases": pre_execution_failure_passed_cases,
            "no_silent_fallback_cases": no_silent_fallback_cases,
            "no_silent_substitution_cases": no_silent_substitution_cases,
            "first_condition_present_cases": first_condition_present_cases,
            "categories_present": categories_present,
            "boundary_classes_present": boundary_classes_present,
            "backend_incompatible_case_present": backend_incompatible_case_present,
            "all_cases_match_contract": all_cases_match_contract,
            "unsupported_gate_completed": unsupported_gate_completed,
        },
        "required_artifacts": {
            "story1_contract": {
                "suite_name": story1_contract["suite_name"],
                "status": story1_contract["status"],
                "required_unsupported_case_fields": story1_contract["output_contract"][
                    "required_unsupported_case_fields"
                ],
                "summary": story1_contract["summary"],
            },
            "end_to_end_trace_reference": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            },
            "matrix_baseline_reference": {
                "suite_name": story3_bundle["suite_name"],
                "status": story3_bundle["status"],
                "summary": story3_bundle["summary"],
            },
            "task4_unsupported_noise_bundle": {
                "suite_name": task4_unsupported_bundle["suite_name"],
                "status": task4_unsupported_bundle["status"],
                "summary": task4_unsupported_bundle["summary"],
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
            "Task 6 Story 4 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Task 6 Story 4 bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Task 6 Story 4 bundle has unexpected contract_version '{}'".format(
                bundle["contract_version"]
            )
        )
    for case in bundle["cases"]:
        validate_case_payload(case, bundle["requirements"]["required_case_fields"])
        if case["workflow_id"] != bundle["workflow_id"]:
            raise ValueError(
                "Task 6 Story 4 case '{}' does not match bundle workflow_id".format(
                    case["case_name"]
                )
            )
    if bundle["summary"]["unsupported_gate_completed"] != (bundle["status"] == "pass"):
        raise ValueError(
            "Task 6 Story 4 unsupported_gate_completed summary is inconsistent"
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
    story3_bundle_path: Path = STORY3_BUNDLE_PATH,
    task4_unsupported_bundle_path: Path = TASK4_UNSUPPORTED_BUNDLE_PATH,
    backend_mismatch_case_path: Path = TASK2_BACKEND_MISMATCH_PATH,
    verbose=False,
):
    story1_contract = _load_story1_contract(story1_contract_path)
    story2_bundle = _load_story2_bundle(story2_bundle_path)
    story3_bundle = _load_story3_bundle(story3_bundle_path)
    task4_unsupported_bundle = _load_json(task4_unsupported_bundle_path)
    backend_mismatch_case = _load_json(backend_mismatch_case_path)
    bundle = build_artifact_bundle(
        story1_contract,
        story2_bundle,
        story3_bundle,
        task4_unsupported_bundle,
        backend_mismatch_case,
    )
    if verbose:
        print(
            "{} [{}] unsupported_cases={} backend_mismatch={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["unsupported_status_cases"],
                bundle["summary"]["backend_incompatible_case_present"],
            )
        )
    return (
        story1_contract,
        story2_bundle,
        story3_bundle,
        task4_unsupported_bundle,
        backend_mismatch_case,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 6 Story 4 JSON artifact bundle.",
    )
    parser.add_argument(
        "--story1-contract-path",
        type=Path,
        default=STORY1_CONTRACT_PATH,
        help="Path to the Task 6 Story 1 canonical contract artifact.",
    )
    parser.add_argument(
        "--story2-bundle-path",
        type=Path,
        default=STORY2_BUNDLE_PATH,
        help="Path to the Task 6 Story 2 end-to-end plus trace bundle.",
    )
    parser.add_argument(
        "--story3-bundle-path",
        type=Path,
        default=STORY3_BUNDLE_PATH,
        help="Path to the Task 6 Story 3 matrix baseline bundle.",
    )
    parser.add_argument(
        "--task4-unsupported-bundle-path",
        type=Path,
        default=TASK4_UNSUPPORTED_BUNDLE_PATH,
        help="Path to the committed Task 4 unsupported-noise bundle.",
    )
    parser.add_argument(
        "--backend-mismatch-case-path",
        type=Path,
        default=TASK2_BACKEND_MISMATCH_PATH,
        help="Path to the committed backend-mismatch unsupported case.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    *_, bundle = run_validation(
        story1_contract_path=args.story1_contract_path,
        story2_bundle_path=args.story2_bundle_path,
        story3_bundle_path=args.story3_bundle_path,
        task4_unsupported_bundle_path=args.task4_unsupported_bundle_path,
        backend_mismatch_case_path=args.backend_mismatch_case_path,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["unsupported_status_cases"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
