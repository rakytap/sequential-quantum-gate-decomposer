#!/usr/bin/env python3
"""Validation: Task 6 Story 5 interpretation guardrails.

Builds the Task 6 interpretation layer from:
- the emitted Task 6 Story 1 contract bundle,
- the emitted Task 6 Story 2 end-to-end plus trace bundle,
- the emitted Task 6 Story 3 matrix baseline bundle,
- the emitted Task 6 Story 4 unsupported-workflow bundle,
- and the committed Task 4 optional evidence bundle.

This layer is intentionally thin:
- it computes the main Task 6 completion signal only from mandatory complete
  supported evidence,
- it keeps optional evidence explicitly supplemental,
- it keeps unsupported/deferred evidence explicitly negative,
- and it treats missing mandatory evidence as incomplete rather than partial
  success.

Run with:
    python benchmarks/density_matrix/task6_story5_interpretation_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.task6_story1_workflow_contract_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    CONTRACT_VERSION,
    DEFAULT_OUTPUT_DIR as TASK6_DEFAULT_OUTPUT_DIR,
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
from benchmarks.density_matrix.task6_story3_matrix_baseline_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)
from benchmarks.density_matrix.task6_story4_unsupported_workflow_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)

SUITE_NAME = "task6_story5_interpretation"
ARTIFACT_FILENAME = "story5_interpretation_bundle.json"
DEFAULT_OUTPUT_DIR = TASK6_DEFAULT_OUTPUT_DIR
STORY1_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
STORY3_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / STORY3_ARTIFACT_FILENAME
STORY4_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / STORY4_ARTIFACT_FILENAME
OPTIONAL_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "phase2_task4"
    / "story3_optional_noise_classification_bundle.json"
)
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


def _load_story4_bundle(path: Path = STORY4_BUNDLE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story4_artifact(artifact)
        return artifact
    *_, artifact = run_story4_validation(verbose=False)
    return artifact


def build_requirement_metadata(story1_contract):
    return {
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "main_claim_rule": (
            "Only mandatory, complete, supported evidence may close the main "
            "Task 6 completion claim."
        ),
        "excluded_evidence_classes": [
            "optional",
            "deferred",
            "unsupported",
            "incomplete",
        ],
        "required_bundle_sources": [
            "task6_story1_canonical_workflow_contract",
            "task6_story2_end_to_end_trace",
            "task6_story3_matrix_baseline",
            "task6_story4_unsupported_workflow",
            "task4_story3_optional_noise_classification",
        ],
    }


def build_artifact_bundle(
    story1_contract,
    story2_bundle,
    story3_bundle,
    story4_bundle,
    optional_bundle,
):
    incomplete_mandatory_artifacts = []
    mandatory_artifacts = {
        "story1_contract": story1_contract,
        "story2_end_to_end_trace": story2_bundle,
        "story3_matrix_baseline": story3_bundle,
        "story4_unsupported_workflow": story4_bundle,
    }
    for artifact_id, artifact in mandatory_artifacts.items():
        if artifact["status"] != "pass":
            incomplete_mandatory_artifacts.append(artifact_id)

    story1_contract_complete = bool(
        story1_contract["summary"].get("contract_sections_complete", False)
    )
    story2_gate_complete = bool(
        story2_bundle["summary"].get("end_to_end_gate_completed", False)
    )
    story3_gate_complete = bool(
        story3_bundle["summary"].get("matrix_gate_completed", False)
    )
    story4_gate_complete = bool(
        story4_bundle["summary"].get("unsupported_gate_completed", False)
    )
    story4_case_field_alignment = (
        story4_bundle["requirements"]["required_case_fields"]
        == story1_contract["output_contract"]["required_unsupported_case_fields"]
    )
    mandatory_artifacts_complete = bool(
        not incomplete_mandatory_artifacts
        and story1_contract_complete
        and story2_gate_complete
        and story3_gate_complete
        and story4_gate_complete
    )
    optional_evidence_supplemental = bool(
        optional_bundle["status"] == "pass"
        and optional_bundle["summary"]["optional_cases_count_toward_mandatory_baseline"]
        == 0
    )
    unsupported_evidence_negative_only = bool(
        story4_bundle["status"] == "pass"
        and story4_bundle["summary"]["unsupported_status_cases"]
        == story4_bundle["summary"]["total_cases"]
        and story4_bundle["summary"]["mandatory_baseline_case_count"] == 0
        and story4_case_field_alignment
    )
    main_task6_claim_completed = bool(
        mandatory_artifacts_complete
        and optional_evidence_supplemental
        and unsupported_evidence_negative_only
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if main_task6_claim_completed else "fail",
        "workflow_id": story1_contract["workflow_id"],
        "contract_version": story1_contract["contract_version"],
        "backend": story1_contract["backend"],
        "reference_backend": story1_contract["reference_backend"],
        "requirements": build_requirement_metadata(story1_contract),
        "thresholds": {
            "mandatory_completion_rule": "all_mandatory_artifacts_pass",
            "optional_cases_count_toward_mandatory_baseline": 0,
            "mandatory_baseline_case_count_for_negative_evidence": 0,
        },
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task6_story5_interpretation_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_contract_path": str(STORY1_CONTRACT_PATH),
            "story2_bundle_path": str(STORY2_BUNDLE_PATH),
            "story3_bundle_path": str(STORY3_BUNDLE_PATH),
            "story4_bundle_path": str(STORY4_BUNDLE_PATH),
            "optional_bundle_path": str(OPTIONAL_BUNDLE_PATH),
        },
        "summary": {
            "mandatory_artifacts": list(mandatory_artifacts.keys()),
            "incomplete_mandatory_artifacts": incomplete_mandatory_artifacts,
            "story1_contract_complete": story1_contract_complete,
            "story2_gate_complete": story2_gate_complete,
            "story3_gate_complete": story3_gate_complete,
            "story4_gate_complete": story4_gate_complete,
            "story4_case_field_alignment": story4_case_field_alignment,
            "mandatory_artifacts_complete": mandatory_artifacts_complete,
            "optional_cases": optional_bundle["summary"]["optional_cases"],
            "optional_passed_cases": optional_bundle["summary"]["optional_passed_cases"],
            "optional_cases_count_toward_mandatory_baseline": optional_bundle[
                "summary"
            ]["optional_cases_count_toward_mandatory_baseline"],
            "optional_evidence_supplemental": optional_evidence_supplemental,
            "unsupported_status_cases": story4_bundle["summary"][
                "unsupported_status_cases"
            ],
            "unsupported_cases": story4_bundle["summary"]["unsupported_cases"],
            "deferred_cases": story4_bundle["summary"]["deferred_cases"],
            "unsupported_evidence_negative_only": unsupported_evidence_negative_only,
            "main_task6_claim_completed": main_task6_claim_completed,
        },
        "required_artifacts": {
            "story1_contract": {
                "suite_name": story1_contract["suite_name"],
                "status": story1_contract["status"],
                "summary": story1_contract["summary"],
            },
            "story2_end_to_end_trace": {
                "suite_name": story2_bundle["suite_name"],
                "status": story2_bundle["status"],
                "summary": story2_bundle["summary"],
            },
            "story3_matrix_baseline": {
                "suite_name": story3_bundle["suite_name"],
                "status": story3_bundle["status"],
                "summary": story3_bundle["summary"],
            },
            "story4_unsupported_workflow": {
                "suite_name": story4_bundle["suite_name"],
                "status": story4_bundle["status"],
                "summary": story4_bundle["summary"],
            },
            "task4_story3_optional_classification": {
                "suite_name": optional_bundle["suite_name"],
                "status": optional_bundle["status"],
                "summary": optional_bundle["summary"],
            },
        },
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 5 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Task 6 Story 5 bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Task 6 Story 5 bundle has unexpected contract_version '{}'".format(
                bundle["contract_version"]
            )
        )
    if bundle["summary"]["main_task6_claim_completed"] != (bundle["status"] == "pass"):
        raise ValueError(
            "Task 6 Story 5 main_task6_claim_completed summary is inconsistent"
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
    story4_bundle_path: Path = STORY4_BUNDLE_PATH,
    optional_bundle_path: Path = OPTIONAL_BUNDLE_PATH,
    verbose=False,
):
    story1_contract = _load_story1_contract(story1_contract_path)
    story2_bundle = _load_story2_bundle(story2_bundle_path)
    story3_bundle = _load_story3_bundle(story3_bundle_path)
    story4_bundle = _load_story4_bundle(story4_bundle_path)
    optional_bundle = _load_json(optional_bundle_path)
    bundle = build_artifact_bundle(
        story1_contract,
        story2_bundle,
        story3_bundle,
        story4_bundle,
        optional_bundle,
    )
    if verbose:
        print(
            "{} [{}] mandatory_complete={} optional_supplemental={} unsupported_negative_only={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["mandatory_artifacts_complete"],
                bundle["summary"]["optional_evidence_supplemental"],
                bundle["summary"]["unsupported_evidence_negative_only"],
            )
        )
    return (
        story1_contract,
        story2_bundle,
        story3_bundle,
        story4_bundle,
        optional_bundle,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 6 Story 5 JSON artifact bundle.",
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
        "--story3-bundle-path",
        type=Path,
        default=STORY3_BUNDLE_PATH,
        help="Path to the Task 6 Story 3 matrix baseline bundle.",
    )
    parser.add_argument(
        "--story4-bundle-path",
        type=Path,
        default=STORY4_BUNDLE_PATH,
        help="Path to the Task 6 Story 4 unsupported-workflow bundle.",
    )
    parser.add_argument(
        "--optional-bundle-path",
        type=Path,
        default=OPTIONAL_BUNDLE_PATH,
        help="Path to the committed optional evidence bundle.",
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
        story4_bundle_path=args.story4_bundle_path,
        optional_bundle_path=args.optional_bundle_path,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["main_task6_claim_completed"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
