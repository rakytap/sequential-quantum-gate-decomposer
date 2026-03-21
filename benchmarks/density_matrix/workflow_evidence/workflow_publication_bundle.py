#!/usr/bin/env python3
"""Validation: Task 6 Story 6 publication-ready evidence bundle.

Builds the top-level Task 6 manifest by assembling the emitted Story 1 to Story
5 artifacts into one reproducible, machine-checkable package. The bundle
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
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    CONTRACT_VERSION,
    DEFAULT_OUTPUT_DIR as CORRECTNESS_EVIDENCE_DEFAULT_OUTPUT_DIR,
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
from benchmarks.density_matrix.workflow_evidence.unsupported_workflow_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)
from benchmarks.density_matrix.workflow_evidence.workflow_interpretation_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    run_validation as run_story5_validation,
    validate_artifact_bundle as validate_story5_artifact,
)

SUITE_NAME = "workflow_publication_evidence"
ARTIFACT_FILENAME = "workflow_publication_bundle.json"
DEFAULT_OUTPUT_DIR = CORRECTNESS_EVIDENCE_DEFAULT_OUTPUT_DIR
STORY1_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
STORY3_PATH = DEFAULT_OUTPUT_DIR / STORY3_ARTIFACT_FILENAME
STORY4_PATH = DEFAULT_OUTPUT_DIR / STORY4_ARTIFACT_FILENAME
STORY5_PATH = DEFAULT_OUTPUT_DIR / STORY5_ARTIFACT_FILENAME
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


def _load_story1(path: Path = STORY1_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story1_artifact(artifact)
        return artifact
    _, artifact = run_story1_validation(verbose=False)
    return artifact


def _load_story2(path: Path = STORY2_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story2_artifact(artifact)
        return artifact
    _, _, _, artifact = run_story2_validation(verbose=False)
    return artifact


def _load_story3(path: Path = STORY3_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story3_artifact(artifact)
        return artifact
    _, _, _, artifact = run_story3_validation(verbose=False)
    return artifact


def _load_story4(path: Path = STORY4_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story4_artifact(artifact)
        return artifact
    *_, artifact = run_story4_validation(verbose=False)
    return artifact


def _load_story5(path: Path = STORY5_PATH):
    if path.exists():
        artifact = _load_json(path)
        validate_story5_artifact(artifact)
        return artifact
    *_, artifact = run_story5_validation(verbose=False)
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


def build_correctness_evidence_story6_bundle(
    output_dir: Path,
    *,
    story1_bundle,
    story2_bundle,
    story3_bundle,
    story4_bundle,
    story5_bundle,
):
    output_dir = Path(output_dir)
    story1_command = (
        f"python benchmarks/density_matrix/workflow_evidence/workflow_contract_validation.py "
        f"--output-dir {output_dir}"
    )
    story2_command = (
        f"python benchmarks/density_matrix/workflow_evidence/end_to_end_trace_validation.py "
        f"--output-dir {output_dir}"
    )
    story3_command = (
        f"python benchmarks/density_matrix/workflow_evidence/matrix_baseline_validation.py "
        f"--output-dir {output_dir}"
    )
    story4_command = (
        f"python benchmarks/density_matrix/workflow_evidence/unsupported_workflow_validation.py "
        f"--output-dir {output_dir}"
    )
    story5_command = (
        f"python benchmarks/density_matrix/workflow_evidence/workflow_interpretation_validation.py "
        f"--output-dir {output_dir}"
    )
    story6_command = (
        f"python benchmarks/density_matrix/workflow_evidence/workflow_publication_bundle.py "
        f"--output-dir {output_dir}"
    )

    artifacts = [
        _build_artifact_entry(
            artifact_id="workflow_contract_bundle",
            artifact_class="canonical_workflow_contract_bundle",
            mandatory=True,
            path=STORY1_ARTIFACT_FILENAME,
            status=story1_bundle["status"],
            expected_statuses=["pass"],
            purpose="Canonical Task 6 workflow contract artifact defining workflow identity, contract version, input/output sections, and boundary classes.",
            generation_command=story1_command,
            summary={
                "workflow_id": story1_bundle["workflow_id"],
                "contract_version": story1_bundle["contract_version"],
                "contract_sections_complete": story1_bundle["summary"]["contract_sections_complete"],
                "absolute_energy_error": story1_bundle["thresholds"][
                    "absolute_energy_error"
                ],
                "required_workflow_qubits": story1_bundle["thresholds"][
                    "required_workflow_qubits"
                ],
            },
        ),
        _build_artifact_entry(
            artifact_id="end_to_end_trace_bundle",
            artifact_class="end_to_end_trace_bundle",
            mandatory=True,
            path=STORY2_ARTIFACT_FILENAME,
            status=story2_bundle["status"],
            expected_statuses=["pass"],
            purpose="Task 6 Story 2 4q/6q end-to-end execution plus required bounded trace evidence.",
            generation_command=story2_command,
            summary={
                "total_end_to_end_cases": story2_bundle["summary"][
                    "total_end_to_end_cases"
                ],
                "passed_end_to_end_cases": story2_bundle["summary"][
                    "passed_end_to_end_cases"
                ],
                "required_trace_completed": story2_bundle["summary"][
                    "required_trace_completed"
                ],
                "end_to_end_gate_completed": story2_bundle["summary"][
                    "end_to_end_gate_completed"
                ],
                "end_to_end_qubits_match_contract": story2_bundle["summary"][
                    "end_to_end_qubits_match_contract"
                ],
                "trace_case_name_matches_contract": story2_bundle["summary"][
                    "trace_case_name_matches_contract"
                ],
                "workflow_thresholds_match_contract": story2_bundle["summary"][
                    "workflow_thresholds_match_contract"
                ],
                "workflow_id": story2_bundle["workflow_id"],
                "contract_version": story2_bundle["contract_version"],
            },
        ),
        _build_artifact_entry(
            artifact_id="matrix_baseline_bundle",
            artifact_class="matrix_baseline_bundle",
            mandatory=True,
            path=STORY3_ARTIFACT_FILENAME,
            status=story3_bundle["status"],
            expected_statuses=["pass"],
            purpose="Task 6 Story 3 fixed-parameter 4/6/8/10 matrix baseline with explicit 10-qubit anchor presence.",
            generation_command=story3_command,
            summary={
                "required_cases": story3_bundle["summary"]["required_cases"],
                "required_passed_cases": story3_bundle["summary"][
                    "required_passed_cases"
                ],
                "documented_10q_anchor_present": story3_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
                "matrix_gate_completed": story3_bundle["summary"][
                    "matrix_gate_completed"
                ],
                "workflow_inventory_matches_contract": story3_bundle["summary"][
                    "workflow_inventory_matches_contract"
                ],
                "workflow_thresholds_match_contract": story3_bundle["summary"][
                    "workflow_thresholds_match_contract"
                ],
                "workflow_id": story3_bundle["workflow_id"],
                "contract_version": story3_bundle["contract_version"],
            },
        ),
        _build_artifact_entry(
            artifact_id="unsupported_workflow_bundle",
            artifact_class="unsupported_workflow_bundle",
            mandatory=True,
            path=STORY4_ARTIFACT_FILENAME,
            status=story4_bundle["status"],
            expected_statuses=["pass"],
            purpose="Task 6 Story 4 deterministic unsupported/deferred workflow boundary evidence.",
            generation_command=story4_command,
            summary={
                "unsupported_status_cases": story4_bundle["summary"][
                    "unsupported_status_cases"
                ],
                "backend_incompatible_case_present": story4_bundle["summary"][
                    "backend_incompatible_case_present"
                ],
                "unsupported_gate_completed": story4_bundle["summary"][
                    "unsupported_gate_completed"
                ],
                "workflow_id": story4_bundle["workflow_id"],
                "contract_version": story4_bundle["contract_version"],
            },
        ),
        _build_artifact_entry(
            artifact_id="workflow_interpretation_bundle",
            artifact_class="interpretation_guardrail_bundle",
            mandatory=True,
            path=STORY5_ARTIFACT_FILENAME,
            status=story5_bundle["status"],
            expected_statuses=["pass"],
            purpose="Task 6 Story 5 interpretation guardrails preventing optional, unsupported, or incomplete evidence from inflating the main claim.",
            generation_command=story5_command,
            summary={
                "mandatory_artifacts_complete": story5_bundle["summary"][
                    "mandatory_artifacts_complete"
                ],
                "optional_evidence_supplemental": story5_bundle["summary"][
                    "optional_evidence_supplemental"
                ],
                "unsupported_evidence_negative_only": story5_bundle["summary"][
                    "unsupported_evidence_negative_only"
                ],
                "unsupported_case_field_alignment": story5_bundle["summary"][
                    "unsupported_case_field_alignment"
                ],
                "main_workflow_claim_completed": story5_bundle["summary"][
                    "main_workflow_claim_completed"
                ],
                "workflow_id": story5_bundle["workflow_id"],
                "contract_version": story5_bundle["contract_version"],
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
        "backend": story1_bundle["backend"],
        "reference_backend": story1_bundle["reference_backend"],
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": story6_command,
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_path": str(STORY1_PATH),
            "story2_path": str(STORY2_PATH),
            "story3_path": str(STORY3_PATH),
            "exact_regime_path": str(STORY4_PATH),
            "story5_path": str(STORY5_PATH),
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
    validate_correctness_evidence_story6_bundle(bundle, output_dir)
    return bundle


def validate_correctness_evidence_story6_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 6 Story 6 bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if bundle["workflow_id"] != WORKFLOW_ID:
        raise ValueError(
            "Task 6 Story 6 bundle has unexpected workflow_id '{}'".format(
                bundle["workflow_id"]
            )
        )
    if bundle["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            "Task 6 Story 6 bundle has unexpected contract_version '{}'".format(
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
            "Task 6 Story 6 bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(required_ids - artifact_ids))
            )
        )

    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                "Task 6 Story 6 bundle is missing artifact file: {}".format(
                    artifact["path"]
                )
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                "Task 6 Story 6 artifact {} has unexpected status {}".format(
                    artifact["artifact_id"], artifact["status"]
                )
            )
        if artifact["summary"].get("workflow_id", WORKFLOW_ID) != bundle["workflow_id"]:
            raise ValueError(
                "Task 6 Story 6 artifact {} has mismatched workflow_id".format(
                    artifact["artifact_id"]
                )
            )
        if artifact["summary"].get(
            "contract_version", CONTRACT_VERSION
        ) != bundle["contract_version"]:
            raise ValueError(
                "Task 6 Story 6 artifact {} has mismatched contract_version".format(
                    artifact["artifact_id"]
                )
            )
        if not _artifact_semantics_complete(artifact):
            raise ValueError(
                "Task 6 Story 6 artifact {} is missing required semantic closure flags".format(
                    artifact["artifact_id"]
                )
            )


def write_correctness_evidence_story6_bundle(output_path: Path, bundle):
    validate_correctness_evidence_story6_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def run_validation(
    *,
    story1_path: Path = STORY1_PATH,
    story2_path: Path = STORY2_PATH,
    story3_path: Path = STORY3_PATH,
    story4_path: Path = STORY4_PATH,
    story5_path: Path = STORY5_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    story1_bundle = _load_story1(story1_path)
    story2_bundle = _load_story2(story2_path)
    story3_bundle = _load_story3(story3_path)
    story4_bundle = _load_story4(story4_path)
    story5_bundle = _load_story5(story5_path)
    _write_artifact(output_dir / STORY1_ARTIFACT_FILENAME, story1_bundle)
    _write_artifact(output_dir / STORY2_ARTIFACT_FILENAME, story2_bundle)
    _write_artifact(output_dir / STORY3_ARTIFACT_FILENAME, story3_bundle)
    _write_artifact(output_dir / STORY4_ARTIFACT_FILENAME, story4_bundle)
    _write_artifact(output_dir / STORY5_ARTIFACT_FILENAME, story5_bundle)
    bundle = build_correctness_evidence_story6_bundle(
        output_dir,
        story1_bundle=story1_bundle,
        story2_bundle=story2_bundle,
        story3_bundle=story3_bundle,
        story4_bundle=story4_bundle,
        story5_bundle=story5_bundle,
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
    return story1_bundle, story2_bundle, story3_bundle, story4_bundle, story5_bundle, bundle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 6 Story 6 JSON artifact bundle.",
    )
    parser.add_argument(
        "--story1-path",
        type=Path,
        default=STORY1_PATH,
        help="Path to the Task 6 Story 1 canonical workflow contract artifact.",
    )
    parser.add_argument(
        "--story2-path",
        type=Path,
        default=STORY2_PATH,
        help="Path to the Task 6 Story 2 end-to-end plus trace bundle.",
    )
    parser.add_argument(
        "--story3-path",
        type=Path,
        default=STORY3_PATH,
        help="Path to the Task 6 Story 3 matrix baseline bundle.",
    )
    parser.add_argument(
        "--workflow-bundle-path",
        type=Path,
        default=STORY4_PATH,
        help="Path to the Task 6 Story 4 unsupported-workflow bundle.",
    )
    parser.add_argument(
        "--publication-bundle-path",
        type=Path,
        default=STORY5_PATH,
        help="Path to the Task 6 Story 5 interpretation bundle.",
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
        story1_path=args.story1_path,
        story2_path=args.story2_path,
        story3_path=args.story3_path,
        story4_path=args.workflow_bundle_path,
        story5_path=args.publication_bundle_path,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_correctness_evidence_story6_bundle(output_path, bundle)
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


# Compatibility aliases for legacy auxiliary validation imports.
build_task6_story6_bundle = build_correctness_evidence_story6_bundle
validate_task6_story6_bundle = validate_correctness_evidence_story6_bundle
write_task6_story6_bundle = write_correctness_evidence_story6_bundle


if __name__ == "__main__":
    main()
