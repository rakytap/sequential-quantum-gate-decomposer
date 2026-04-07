#!/usr/bin/env python3
"""Validation: workflow interpretation guardrails.

Builds the workflow-interpretation layer from:
- the emitted workflow-contract bundle,
- the emitted end-to-end trace bundle,
- the emitted matrix baseline bundle,
- the emitted unsupported-workflow bundle,
- and the committed optional evidence bundle.

This layer is intentionally thin:
- it computes the main workflow claim only from mandatory complete supported
  evidence,
- it keeps optional evidence explicitly supplemental,
- it keeps unsupported/deferred evidence explicitly negative,
- and it treats missing mandatory evidence as incomplete rather than partial
  success.

Run with:
    python benchmarks/density_matrix/workflow_evidence/workflow_interpretation_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.workflow_evidence.workflow_contract_validation import (
    ARTIFACT_FILENAME as WORKFLOW_CONTRACT_ARTIFACT_FILENAME,
    CONTRACT_VERSION,
    DEFAULT_OUTPUT_DIR as WORKFLOW_EVIDENCE_OUTPUT_DIR,
    WORKFLOW_ID,
    build_software_metadata,
    get_git_revision,
    run_validation as run_workflow_contract_validation,
    validate_artifact_bundle as validate_workflow_contract_artifact,
)
from benchmarks.density_matrix.workflow_evidence.end_to_end_trace_validation import (
    ARTIFACT_FILENAME as END_TO_END_TRACE_ARTIFACT_FILENAME,
    run_validation as run_end_to_end_trace_validation,
    validate_artifact_bundle as validate_end_to_end_trace_artifact,
)
from benchmarks.density_matrix.workflow_evidence.matrix_baseline_validation import (
    ARTIFACT_FILENAME as MATRIX_BASELINE_ARTIFACT_FILENAME,
    run_validation as run_matrix_baseline_validation,
    validate_artifact_bundle as validate_matrix_baseline_artifact,
)
from benchmarks.density_matrix.workflow_evidence.unsupported_workflow_validation import (
    ARTIFACT_FILENAME as UNSUPPORTED_WORKFLOW_ARTIFACT_FILENAME,
    run_validation as run_unsupported_workflow_validation,
    validate_artifact_bundle as validate_unsupported_workflow_artifact,
)

SUITE_NAME = "workflow_interpretation_validation"
ARTIFACT_FILENAME = "workflow_interpretation_bundle.json"
DEFAULT_OUTPUT_DIR = WORKFLOW_EVIDENCE_OUTPUT_DIR
WORKFLOW_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / WORKFLOW_CONTRACT_ARTIFACT_FILENAME
END_TO_END_TRACE_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / END_TO_END_TRACE_ARTIFACT_FILENAME
MATRIX_BASELINE_BUNDLE_PATH = DEFAULT_OUTPUT_DIR / MATRIX_BASELINE_ARTIFACT_FILENAME
UNSUPPORTED_WORKFLOW_BUNDLE_PATH = (
    DEFAULT_OUTPUT_DIR / UNSUPPORTED_WORKFLOW_ARTIFACT_FILENAME
)
OPTIONAL_NOISE_BUNDLE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "density_matrix"
    / "artifacts"
    / "noise_support"
    / "optional_noise_classification_bundle.json"
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


def _load_workflow_contract(path: Path = WORKFLOW_CONTRACT_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_workflow_contract_artifact(artifact)
        return artifact
    _, artifact = run_workflow_contract_validation(verbose=False)
    return artifact


def _load_end_to_end_trace_bundle(path: Path = END_TO_END_TRACE_BUNDLE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_end_to_end_trace_artifact(artifact)
        return artifact
    _, _, _, artifact = run_end_to_end_trace_validation(verbose=False)
    return artifact


def _load_matrix_baseline_bundle(path: Path = MATRIX_BASELINE_BUNDLE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_matrix_baseline_artifact(artifact)
        return artifact
    _, _, _, artifact = run_matrix_baseline_validation(verbose=False)
    return artifact


def _load_unsupported_workflow_bundle(path: Path = UNSUPPORTED_WORKFLOW_BUNDLE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_unsupported_workflow_artifact(artifact)
        return artifact
    *_, artifact = run_unsupported_workflow_validation(verbose=False)
    return artifact


def build_requirement_metadata(workflow_contract):
    return {
        "workflow_id": workflow_contract["workflow_id"],
        "contract_version": workflow_contract["contract_version"],
        "main_claim_rule": (
            "Only mandatory, complete, supported evidence may close the main "
            "workflow claim."
        ),
        "excluded_evidence_classes": [
            "optional",
            "deferred",
            "unsupported",
            "incomplete",
        ],
        "required_bundle_sources": [
            "workflow_contract_validation",
            "end_to_end_trace_validation",
            "matrix_baseline_validation",
            "unsupported_workflow_validation",
            "optional_noise_classification",
        ],
    }


def build_artifact_bundle(
    workflow_contract,
    end_to_end_trace_bundle,
    matrix_baseline_bundle,
    unsupported_workflow_bundle,
    optional_noise_bundle,
):
    incomplete_mandatory_artifacts = []
    mandatory_artifacts = {
        "workflow_contract": workflow_contract,
        "end_to_end_trace_reference": end_to_end_trace_bundle,
        "matrix_baseline_reference": matrix_baseline_bundle,
        "unsupported_workflow_reference": unsupported_workflow_bundle,
    }
    for artifact_id, artifact in mandatory_artifacts.items():
        if artifact["status"] != "pass":
            incomplete_mandatory_artifacts.append(artifact_id)

    workflow_contract_complete = bool(
        workflow_contract["summary"].get("contract_sections_complete", False)
    )
    end_to_end_gate_complete = bool(
        end_to_end_trace_bundle["summary"].get("end_to_end_gate_completed", False)
    )
    matrix_baseline_gate_complete = bool(
        matrix_baseline_bundle["summary"].get("matrix_gate_completed", False)
    )
    unsupported_workflow_gate_complete = bool(
        unsupported_workflow_bundle["summary"].get("unsupported_gate_completed", False)
    )
    unsupported_case_field_alignment = (
        unsupported_workflow_bundle["requirements"]["required_case_fields"]
        == workflow_contract["output_contract"]["required_unsupported_case_fields"]
    )
    mandatory_artifacts_complete = bool(
        not incomplete_mandatory_artifacts
        and workflow_contract_complete
        and end_to_end_gate_complete
        and matrix_baseline_gate_complete
        and unsupported_workflow_gate_complete
    )
    optional_evidence_supplemental = bool(
        optional_noise_bundle["status"] == "pass"
        and optional_noise_bundle["summary"][
            "optional_cases_count_toward_mandatory_baseline"
        ]
        == 0
    )
    unsupported_evidence_negative_only = bool(
        unsupported_workflow_bundle["status"] == "pass"
        and unsupported_workflow_bundle["summary"]["unsupported_status_cases"]
        == unsupported_workflow_bundle["summary"]["total_cases"]
        and unsupported_workflow_bundle["summary"]["mandatory_baseline_case_count"]
        == 0
        and unsupported_case_field_alignment
    )
    main_workflow_claim_completed = bool(
        mandatory_artifacts_complete
        and optional_evidence_supplemental
        and unsupported_evidence_negative_only
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if main_workflow_claim_completed else "fail",
        "workflow_id": workflow_contract["workflow_id"],
        "contract_version": workflow_contract["contract_version"],
        "backend": workflow_contract["backend"],
        "reference_backend": workflow_contract["reference_backend"],
        "requirements": build_requirement_metadata(workflow_contract),
        "thresholds": {
            "mandatory_completion_rule": "all_mandatory_artifacts_pass",
            "optional_cases_count_toward_mandatory_baseline": 0,
            "mandatory_baseline_case_count_for_negative_evidence": 0,
        },
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "workflow_evidence/workflow_interpretation_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "workflow_contract_path": str(WORKFLOW_CONTRACT_PATH),
            "end_to_end_trace_bundle_path": str(END_TO_END_TRACE_BUNDLE_PATH),
            "matrix_baseline_bundle_path": str(MATRIX_BASELINE_BUNDLE_PATH),
            "unsupported_workflow_bundle_path": str(
                UNSUPPORTED_WORKFLOW_BUNDLE_PATH
            ),
            "optional_noise_bundle_path": str(OPTIONAL_NOISE_BUNDLE_PATH),
        },
        "summary": {
            "mandatory_artifacts": list(mandatory_artifacts.keys()),
            "incomplete_mandatory_artifacts": incomplete_mandatory_artifacts,
            "workflow_contract_complete": workflow_contract_complete,
            "end_to_end_gate_complete": end_to_end_gate_complete,
            "matrix_baseline_gate_complete": matrix_baseline_gate_complete,
            "unsupported_workflow_gate_complete": unsupported_workflow_gate_complete,
            "unsupported_case_field_alignment": unsupported_case_field_alignment,
            "mandatory_artifacts_complete": mandatory_artifacts_complete,
            "optional_cases": optional_noise_bundle["summary"]["optional_cases"],
            "optional_passed_cases": optional_noise_bundle["summary"][
                "optional_passed_cases"
            ],
            "optional_cases_count_toward_mandatory_baseline": optional_noise_bundle[
                "summary"
            ]["optional_cases_count_toward_mandatory_baseline"],
            "optional_evidence_supplemental": optional_evidence_supplemental,
            "unsupported_status_cases": unsupported_workflow_bundle["summary"][
                "unsupported_status_cases"
            ],
            "unsupported_cases": unsupported_workflow_bundle["summary"][
                "unsupported_cases"
            ],
            "deferred_cases": unsupported_workflow_bundle["summary"]["deferred_cases"],
            "unsupported_evidence_negative_only": unsupported_evidence_negative_only,
            "main_workflow_claim_completed": main_workflow_claim_completed,
        },
        "required_artifacts": {
            "workflow_contract": {
                "suite_name": workflow_contract["suite_name"],
                "status": workflow_contract["status"],
                "summary": workflow_contract["summary"],
            },
            "end_to_end_trace_reference": {
                "suite_name": end_to_end_trace_bundle["suite_name"],
                "status": end_to_end_trace_bundle["status"],
                "summary": end_to_end_trace_bundle["summary"],
            },
            "matrix_baseline_reference": {
                "suite_name": matrix_baseline_bundle["suite_name"],
                "status": matrix_baseline_bundle["status"],
                "summary": matrix_baseline_bundle["summary"],
            },
            "unsupported_workflow_reference": {
                "suite_name": unsupported_workflow_bundle["suite_name"],
                "status": unsupported_workflow_bundle["status"],
                "summary": unsupported_workflow_bundle["summary"],
            },
            "optional_noise_reference": {
                "suite_name": optional_noise_bundle["suite_name"],
                "status": optional_noise_bundle["status"],
                "summary": optional_noise_bundle["summary"],
            },
        },
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Workflow-interpretation bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Workflow-interpretation bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Workflow-interpretation bundle has unexpected contract_version '{}'".format(
                bundle["contract_version"]
            )
        )
    if bundle["summary"]["main_workflow_claim_completed"] != (bundle["status"] == "pass"):
        raise ValueError(
            "Workflow-interpretation bundle main_workflow_claim_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_validation(
    *,
    workflow_contract_path: Path = WORKFLOW_CONTRACT_PATH,
    end_to_end_trace_bundle_path: Path = END_TO_END_TRACE_BUNDLE_PATH,
    matrix_baseline_bundle_path: Path = MATRIX_BASELINE_BUNDLE_PATH,
    unsupported_workflow_bundle_path: Path = UNSUPPORTED_WORKFLOW_BUNDLE_PATH,
    optional_noise_bundle_path: Path = OPTIONAL_NOISE_BUNDLE_PATH,
    verbose=False,
):
    workflow_contract = _load_workflow_contract(workflow_contract_path)
    end_to_end_trace_bundle = _load_end_to_end_trace_bundle(
        end_to_end_trace_bundle_path
    )
    matrix_baseline_bundle = _load_matrix_baseline_bundle(matrix_baseline_bundle_path)
    unsupported_workflow_bundle = _load_unsupported_workflow_bundle(
        unsupported_workflow_bundle_path
    )
    optional_noise_bundle = _load_json(optional_noise_bundle_path)
    bundle = build_artifact_bundle(
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        optional_noise_bundle,
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
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        optional_noise_bundle,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the workflow-interpretation JSON artifact bundle.",
    )
    parser.add_argument(
        "--workflow-contract-path",
        type=Path,
        default=WORKFLOW_CONTRACT_PATH,
        help="Path to the canonical workflow-contract artifact.",
    )
    parser.add_argument(
        "--end-to-end-trace-bundle-path",
        type=Path,
        default=END_TO_END_TRACE_BUNDLE_PATH,
        help="Path to the end-to-end trace bundle.",
    )
    parser.add_argument(
        "--matrix-baseline-bundle-path",
        type=Path,
        default=MATRIX_BASELINE_BUNDLE_PATH,
        help="Path to the matrix-baseline bundle.",
    )
    parser.add_argument(
        "--unsupported-workflow-bundle-path",
        type=Path,
        default=UNSUPPORTED_WORKFLOW_BUNDLE_PATH,
        help="Path to the unsupported-workflow bundle.",
    )
    parser.add_argument(
        "--optional-noise-bundle-path",
        type=Path,
        default=OPTIONAL_NOISE_BUNDLE_PATH,
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
        workflow_contract_path=args.workflow_contract_path,
        end_to_end_trace_bundle_path=args.end_to_end_trace_bundle_path,
        matrix_baseline_bundle_path=args.matrix_baseline_bundle_path,
        unsupported_workflow_bundle_path=args.unsupported_workflow_bundle_path,
        optional_noise_bundle_path=args.optional_noise_bundle_path,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["main_workflow_claim_completed"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
