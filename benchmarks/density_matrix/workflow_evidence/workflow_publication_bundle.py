#!/usr/bin/env python3
"""Validation: workflow publication bundle.

Builds the top-level workflow manifest by assembling the emitted workflow
evidence artifacts into one reproducible, machine-checkable package. The bundle
preserves canonical workflow identity and contract-version fields across the
packaged evidence layers.

Run with:
    python benchmarks/density_matrix/workflow_evidence/workflow_publication_bundle.py
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
    REFERENCE_BACKEND,
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
from benchmarks.density_matrix.workflow_evidence.workflow_interpretation_validation import (
    ARTIFACT_FILENAME as WORKFLOW_INTERPRETATION_ARTIFACT_FILENAME,
    run_validation as run_workflow_interpretation_validation,
    validate_artifact_bundle as validate_workflow_interpretation_artifact,
)

SUITE_NAME = "workflow_publication_evidence"
ARTIFACT_FILENAME = "workflow_publication_bundle.json"
DEFAULT_OUTPUT_DIR = WORKFLOW_EVIDENCE_OUTPUT_DIR
WORKFLOW_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / WORKFLOW_CONTRACT_ARTIFACT_FILENAME
END_TO_END_TRACE_PATH = DEFAULT_OUTPUT_DIR / END_TO_END_TRACE_ARTIFACT_FILENAME
MATRIX_BASELINE_PATH = DEFAULT_OUTPUT_DIR / MATRIX_BASELINE_ARTIFACT_FILENAME
UNSUPPORTED_WORKFLOW_PATH = DEFAULT_OUTPUT_DIR / UNSUPPORTED_WORKFLOW_ARTIFACT_FILENAME
WORKFLOW_INTERPRETATION_PATH = (
    DEFAULT_OUTPUT_DIR / WORKFLOW_INTERPRETATION_ARTIFACT_FILENAME
)
BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "workflow_id",
    "contract_version",
    "backend",
    "reference_backend",
    "software",
    "provenance",
    "summary",
    "artifacts",
)
REQUIRED_SEMANTIC_FLAGS = {
    "workflow_contract_bundle": ("contract_sections_complete",),
    "end_to_end_trace_bundle": (
        "end_to_end_gate_completed",
        "end_to_end_qubits_match_contract",
        "trace_case_name_matches_contract",
        "workflow_thresholds_match_contract",
    ),
    "matrix_baseline_bundle": (
        "matrix_gate_completed",
        "workflow_inventory_matches_contract",
        "workflow_thresholds_match_contract",
        "documented_10q_anchor_present",
    ),
    "unsupported_workflow_bundle": (
        "unsupported_gate_completed",
        "backend_incompatible_case_present",
    ),
    "workflow_interpretation_bundle": (
        "mandatory_artifacts_complete",
        "unsupported_evidence_negative_only",
        "unsupported_case_field_alignment",
        "main_workflow_claim_completed",
    ),
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_workflow_contract(path: Path = WORKFLOW_CONTRACT_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_workflow_contract_artifact(artifact)
        return artifact
    _, artifact = run_workflow_contract_validation(verbose=False)
    return artifact


def _load_end_to_end_trace(path: Path = END_TO_END_TRACE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_end_to_end_trace_artifact(artifact)
        return artifact
    _, _, _, artifact = run_end_to_end_trace_validation(verbose=False)
    return artifact


def _load_matrix_baseline(path: Path = MATRIX_BASELINE_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_matrix_baseline_artifact(artifact)
        return artifact
    _, _, _, artifact = run_matrix_baseline_validation(verbose=False)
    return artifact


def _load_unsupported_workflow(path: Path = UNSUPPORTED_WORKFLOW_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_unsupported_workflow_artifact(artifact)
        return artifact
    *_, artifact = run_unsupported_workflow_validation(verbose=False)
    return artifact


def _load_workflow_interpretation(path: Path = WORKFLOW_INTERPRETATION_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_workflow_interpretation_artifact(artifact)
        return artifact
    *_, artifact = run_workflow_interpretation_validation(verbose=False)
    return artifact


def _write_artifact(path: Path, artifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def _build_artifact_entry(
    *,
    artifact_id,
    artifact_class,
    mandatory,
    path,
    status,
    expected_statuses,
    purpose,
    generation_command,
    summary,
):
    return {
        "artifact_id": artifact_id,
        "artifact_class": artifact_class,
        "mandatory": mandatory,
        "path": path,
        "status": status,
        "expected_statuses": list(expected_statuses),
        "purpose": purpose,
        "generation_command": generation_command,
        "summary": dict(summary),
    }


def _artifact_semantics_complete(artifact):
    required_flags = REQUIRED_SEMANTIC_FLAGS.get(artifact["artifact_id"], ())
    return all(artifact["summary"].get(flag, False) for flag in required_flags)


def build_workflow_publication_bundle(
    output_dir: Path,
    *,
    workflow_contract,
    end_to_end_trace_bundle,
    matrix_baseline_bundle,
    unsupported_workflow_bundle,
    workflow_interpretation_bundle,
):
    output_dir = Path(output_dir)
    workflow_contract_command = (
        f"python benchmarks/density_matrix/workflow_evidence/workflow_contract_validation.py "
        f"--output-dir {output_dir}"
    )
    end_to_end_trace_command = (
        f"python benchmarks/density_matrix/workflow_evidence/end_to_end_trace_validation.py "
        f"--output-dir {output_dir}"
    )
    matrix_baseline_command = (
        f"python benchmarks/density_matrix/workflow_evidence/matrix_baseline_validation.py "
        f"--output-dir {output_dir}"
    )
    unsupported_workflow_command = (
        f"python benchmarks/density_matrix/workflow_evidence/unsupported_workflow_validation.py "
        f"--output-dir {output_dir}"
    )
    workflow_interpretation_command = (
        f"python benchmarks/density_matrix/workflow_evidence/workflow_interpretation_validation.py "
        f"--output-dir {output_dir}"
    )
    workflow_publication_command = (
        f"python benchmarks/density_matrix/workflow_evidence/workflow_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="workflow_contract_bundle",
            artifact_class="canonical_workflow_contract_bundle",
            mandatory=True,
            path=WORKFLOW_CONTRACT_ARTIFACT_FILENAME,
            status=workflow_contract["status"],
            expected_statuses=["pass"],
            purpose="Canonical workflow-contract artifact defining workflow identity, contract version, input/output sections, and boundary classes.",
            generation_command=workflow_contract_command,
            summary={
                "workflow_id": workflow_contract["workflow_id"],
                "contract_version": workflow_contract["contract_version"],
                "contract_sections_complete": workflow_contract["summary"][
                    "contract_sections_complete"
                ],
                "absolute_energy_error": workflow_contract["thresholds"][
                    "absolute_energy_error"
                ],
                "required_workflow_qubits": workflow_contract["thresholds"][
                    "required_workflow_qubits"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="end_to_end_trace_bundle",
            artifact_class="end_to_end_trace_bundle",
            mandatory=True,
            path=END_TO_END_TRACE_ARTIFACT_FILENAME,
            status=end_to_end_trace_bundle["status"],
            expected_statuses=["pass"],
            purpose="4q/6q end-to-end execution plus required bounded trace evidence.",
            generation_command=end_to_end_trace_command,
            summary={
                "total_end_to_end_cases": end_to_end_trace_bundle["summary"][
                    "total_end_to_end_cases"
                ],
                "passed_end_to_end_cases": end_to_end_trace_bundle["summary"][
                    "passed_end_to_end_cases"
                ],
                "required_trace_completed": end_to_end_trace_bundle["summary"][
                    "required_trace_completed"
                ],
                "end_to_end_gate_completed": end_to_end_trace_bundle["summary"][
                    "end_to_end_gate_completed"
                ],
                "end_to_end_qubits_match_contract": end_to_end_trace_bundle["summary"][
                    "end_to_end_qubits_match_contract"
                ],
                "trace_case_name_matches_contract": end_to_end_trace_bundle["summary"][
                    "trace_case_name_matches_contract"
                ],
                "workflow_thresholds_match_contract": end_to_end_trace_bundle["summary"][
                    "workflow_thresholds_match_contract"
                ],
                "workflow_id": end_to_end_trace_bundle["workflow_id"],
                "contract_version": end_to_end_trace_bundle["contract_version"],
            },
        ),
        _build_artifact_entry(
            artifact_id="matrix_baseline_bundle",
            artifact_class="matrix_baseline_bundle",
            mandatory=True,
            path=MATRIX_BASELINE_ARTIFACT_FILENAME,
            status=matrix_baseline_bundle["status"],
            expected_statuses=["pass"],
            purpose="Fixed-parameter 4/6/8/10 matrix baseline with explicit 10-qubit anchor presence.",
            generation_command=matrix_baseline_command,
            summary={
                "required_cases": matrix_baseline_bundle["summary"]["required_cases"],
                "required_passed_cases": matrix_baseline_bundle["summary"][
                    "required_passed_cases"
                ],
                "documented_10q_anchor_present": matrix_baseline_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "matrix_gate_completed": matrix_baseline_bundle["summary"][
                    "matrix_gate_completed"
                ],
                "workflow_inventory_matches_contract": matrix_baseline_bundle["summary"][
                    "workflow_inventory_matches_contract"
                ],
                "workflow_thresholds_match_contract": matrix_baseline_bundle["summary"][
                    "workflow_thresholds_match_contract"
                ],
                "workflow_id": matrix_baseline_bundle["workflow_id"],
                "contract_version": matrix_baseline_bundle["contract_version"],
            },
        ),
        _build_artifact_entry(
            artifact_id="unsupported_workflow_bundle",
            artifact_class="unsupported_workflow_bundle",
            mandatory=True,
            path=UNSUPPORTED_WORKFLOW_ARTIFACT_FILENAME,
            status=unsupported_workflow_bundle["status"],
            expected_statuses=["pass"],
            purpose="Deterministic unsupported/deferred workflow boundary evidence.",
            generation_command=unsupported_workflow_command,
            summary={
                "unsupported_status_cases": unsupported_workflow_bundle["summary"][
                    "unsupported_status_cases"
                ],
                "backend_incompatible_case_present": unsupported_workflow_bundle["summary"][
                    "backend_incompatible_case_present"
                ],
                "unsupported_gate_completed": unsupported_workflow_bundle["summary"][
                    "unsupported_gate_completed"
                ],
                "workflow_id": unsupported_workflow_bundle["workflow_id"],
                "contract_version": unsupported_workflow_bundle["contract_version"],
            },
        ),
        _build_artifact_entry(
            artifact_id="workflow_interpretation_bundle",
            artifact_class="interpretation_guardrail_bundle",
            mandatory=True,
            path=WORKFLOW_INTERPRETATION_ARTIFACT_FILENAME,
            status=workflow_interpretation_bundle["status"],
            expected_statuses=["pass"],
            purpose="Interpretation guardrails preventing optional, unsupported, or incomplete evidence from inflating the main claim.",
            generation_command=workflow_interpretation_command,
            summary={
                "mandatory_artifacts_complete": workflow_interpretation_bundle["summary"][
                    "mandatory_artifacts_complete"
                ],
                "optional_evidence_supplemental": workflow_interpretation_bundle["summary"][
                    "optional_evidence_supplemental"
                ],
                "unsupported_evidence_negative_only": workflow_interpretation_bundle["summary"][
                    "unsupported_evidence_negative_only"
                ],
                "unsupported_case_field_alignment": workflow_interpretation_bundle["summary"][
                    "unsupported_case_field_alignment"
                ],
                "main_workflow_claim_completed": workflow_interpretation_bundle["summary"][
                    "main_workflow_claim_completed"
                ],
                "workflow_id": workflow_interpretation_bundle["workflow_id"],
                "contract_version": workflow_interpretation_bundle["contract_version"],
            },
        ),
    ]

    mandatory_artifacts = [artifact for artifact in artifacts if artifact["mandatory"]]
    present_count = 0
    status_match_count = 0
    identity_match_count = 0
    semantic_match_count = 0
    for artifact in mandatory_artifacts:
        if (output_dir / artifact["path"]).exists():
            present_count += 1
        if artifact["status"] in artifact["expected_statuses"]:
            status_match_count += 1
        if artifact["summary"].get("workflow_id", WORKFLOW_ID) == WORKFLOW_ID and artifact[
            "summary"
        ].get("contract_version", CONTRACT_VERSION) == CONTRACT_VERSION:
            identity_match_count += 1
        if _artifact_semantics_complete(artifact):
            semantic_match_count += 1

    bundle_status = (
        "pass"
        if present_count == len(mandatory_artifacts)
        and status_match_count == len(mandatory_artifacts)
        and identity_match_count == len(mandatory_artifacts)
        and semantic_match_count == len(mandatory_artifacts)
        else "fail"
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": bundle_status,
        "workflow_id": WORKFLOW_ID,
        "contract_version": CONTRACT_VERSION,
        "backend": workflow_contract["backend"],
        "reference_backend": workflow_contract["reference_backend"],
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": workflow_publication_command,
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "workflow_contract_path": str(WORKFLOW_CONTRACT_PATH),
            "end_to_end_trace_path": str(END_TO_END_TRACE_PATH),
            "matrix_baseline_path": str(MATRIX_BASELINE_PATH),
            "unsupported_workflow_path": str(UNSUPPORTED_WORKFLOW_PATH),
            "workflow_interpretation_path": str(WORKFLOW_INTERPRETATION_PATH),
        },
        "summary": {
            "mandatory_artifact_count": len(mandatory_artifacts),
            "present_artifact_count": present_count,
            "status_match_count": status_match_count,
            "workflow_identity_match_count": identity_match_count,
            "semantic_match_count": semantic_match_count,
            "missing_artifact_count": len(mandatory_artifacts) - present_count,
            "mismatched_status_count": len(mandatory_artifacts) - status_match_count,
            "mismatched_identity_count": len(mandatory_artifacts) - identity_match_count,
            "mismatched_semantic_count": len(mandatory_artifacts)
            - semantic_match_count,
        },
        "artifacts": artifacts,
    }
    validate_workflow_publication_bundle(bundle, output_dir)
    return bundle


def validate_workflow_publication_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Workflow publication bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Workflow publication bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Workflow publication bundle has unexpected contract_version '{}'".format(
                bundle["contract_version"]
            )
        )

    required_ids = {
        "workflow_contract_bundle",
        "end_to_end_trace_bundle",
        "matrix_baseline_bundle",
        "unsupported_workflow_bundle",
        "workflow_interpretation_bundle",
    }
    artifact_ids = {artifact["artifact_id"] for artifact in bundle["artifacts"]}
    if required_ids - artifact_ids:
        raise ValueError(
            "Workflow publication bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(required_ids - artifact_ids))
            )
        )

    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                "Workflow publication bundle is missing artifact file: {}".format(
                    artifact["path"]
                )
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Workflow publication artifact {} has unexpected status {}".format(
                    artifact["artifact_id"], artifact["status"]
                )
            )
        if artifact["summary"].get("workflow_id", WORKFLOW_ID) != bundle["workflow_id"]:
            raise ValueError(
                "Workflow publication artifact {} has mismatched workflow_id".format(
                    artifact["artifact_id"]
                )
            )
        if artifact["summary"].get(
            "contract_version", CONTRACT_VERSION
        ) != bundle["contract_version"]:
            raise ValueError(
                "Workflow publication artifact {} has mismatched contract_version".format(
                    artifact["artifact_id"]
                )
            )
        if not _artifact_semantics_complete(artifact):
            raise ValueError(
                "Workflow publication artifact {} is missing required semantic closure flags".format(
                    artifact["artifact_id"]
                )
            )


def write_workflow_publication_bundle(output_path: Path, bundle):
    validate_workflow_publication_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def run_validation(
    *,
    workflow_contract_path: Path = WORKFLOW_CONTRACT_PATH,
    end_to_end_trace_path: Path = END_TO_END_TRACE_PATH,
    matrix_baseline_path: Path = MATRIX_BASELINE_PATH,
    unsupported_workflow_path: Path = UNSUPPORTED_WORKFLOW_PATH,
    workflow_interpretation_path: Path = WORKFLOW_INTERPRETATION_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_contract = _load_workflow_contract(workflow_contract_path)
    end_to_end_trace_bundle = _load_end_to_end_trace(end_to_end_trace_path)
    matrix_baseline_bundle = _load_matrix_baseline(matrix_baseline_path)
    unsupported_workflow_bundle = _load_unsupported_workflow(unsupported_workflow_path)
    workflow_interpretation_bundle = _load_workflow_interpretation(
        workflow_interpretation_path
    )
    _write_artifact(output_dir / WORKFLOW_CONTRACT_ARTIFACT_FILENAME, workflow_contract)
    _write_artifact(
        output_dir / END_TO_END_TRACE_ARTIFACT_FILENAME, end_to_end_trace_bundle
    )
    _write_artifact(output_dir / MATRIX_BASELINE_ARTIFACT_FILENAME, matrix_baseline_bundle)
    _write_artifact(
        output_dir / UNSUPPORTED_WORKFLOW_ARTIFACT_FILENAME,
        unsupported_workflow_bundle,
    )
    _write_artifact(
        output_dir / WORKFLOW_INTERPRETATION_ARTIFACT_FILENAME,
        workflow_interpretation_bundle,
    )
    bundle = build_workflow_publication_bundle(
        output_dir,
        workflow_contract=workflow_contract,
        end_to_end_trace_bundle=end_to_end_trace_bundle,
        matrix_baseline_bundle=matrix_baseline_bundle,
        unsupported_workflow_bundle=unsupported_workflow_bundle,
        workflow_interpretation_bundle=workflow_interpretation_bundle,
    )
    if verbose:
        print(
            "{} [{}] present={}/{} status_match={}/{} identity_match={}/{}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["present_artifact_count"],
                bundle["summary"]["mandatory_artifact_count"],
                bundle["summary"]["status_match_count"],
                bundle["summary"]["mandatory_artifact_count"],
                bundle["summary"]["workflow_identity_match_count"],
                bundle["summary"]["mandatory_artifact_count"],
            )
        )
    return (
        workflow_contract,
        end_to_end_trace_bundle,
        matrix_baseline_bundle,
        unsupported_workflow_bundle,
        workflow_interpretation_bundle,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the workflow-publication JSON artifact bundle.",
    )
    parser.add_argument(
        "--workflow-contract-path",
        type=Path,
        default=WORKFLOW_CONTRACT_PATH,
        help="Path to the canonical workflow-contract artifact.",
    )
    parser.add_argument(
        "--end-to-end-trace-path",
        type=Path,
        default=END_TO_END_TRACE_PATH,
        help="Path to the end-to-end trace bundle.",
    )
    parser.add_argument(
        "--matrix-baseline-path",
        type=Path,
        default=MATRIX_BASELINE_PATH,
        help="Path to the matrix-baseline bundle.",
    )
    parser.add_argument(
        "--unsupported-workflow-path",
        type=Path,
        default=UNSUPPORTED_WORKFLOW_PATH,
        help="Path to the unsupported-workflow bundle.",
    )
    parser.add_argument(
        "--workflow-interpretation-path",
        type=Path,
        default=WORKFLOW_INTERPRETATION_PATH,
        help="Path to the workflow-interpretation bundle.",
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
        end_to_end_trace_path=args.end_to_end_trace_path,
        matrix_baseline_path=args.matrix_baseline_path,
        unsupported_workflow_path=args.unsupported_workflow_path,
        workflow_interpretation_path=args.workflow_interpretation_path,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_workflow_publication_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["present_artifact_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
