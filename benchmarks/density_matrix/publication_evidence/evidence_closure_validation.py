#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 4 evidence-closure semantics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.claim_traceability_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)
from benchmarks.density_matrix.publication_evidence.common import (
    EVIDENCE_CLOSURE_RULE,
    EVIDENCE_CLOSURE_RULE_ALTERNATIVES,
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH,
    PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH,
    PERFORMANCE_EVIDENCE_POSITIVE_THRESHOLD_PATH,
    PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_or_build_artifact,
    load_text,
    missing_phrases,
    relative_to_repo,
    publication_evidence_output_dir,
    write_json,
)


SUITE_NAME = "phase3_publication_evidence_evidence_closure"
ARTIFACT_FILENAME = "evidence_closure_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("evidence_closure")
STORY3_PATH = (
    publication_evidence_output_dir("claim_traceability") / STORY3_ARTIFACT_FILENAME
)
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "story3_traceability",
    "evidence_inventory",
    "software",
    "provenance",
    "summary",
)


def _story3(path: Path = STORY3_PATH):
    return load_or_build_artifact(
        path,
        run_validation=run_story3_validation,
        validate_artifact_bundle=validate_story3_artifact,
    )


def build_evidence_inventory():
    correctness_evidence_summary = load_json(CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH)
    performance_evidence_summary = load_json(PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH)
    positive_threshold = load_json(PERFORMANCE_EVIDENCE_POSITIVE_THRESHOLD_PATH)
    diagnosis_bundle = load_json(PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH)

    closure_rule_entries = []
    for surface_id, surface in (
        ("abstract", MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_abstract"]),
        ("short_paper", MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper"]),
        ("short_paper_narrative", MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper_narrative"]),
        ("paper", MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"]),
    ):
        text = load_text(surface)
        closure_rule_entries.append(
            {
                "surface_id": surface_id,
                "path": relative_to_repo(surface),
                "missing_phrases": []
                if any(
                    missing_phrases(text, [phrase]) == []
                    for phrase in EVIDENCE_CLOSURE_RULE_ALTERNATIVES
                )
                else [EVIDENCE_CLOSURE_RULE],
            }
        )

    return [
        {
            "item_id": "correctness_evidence_main_correctness_claim",
            "label": "Task 6 correctness claim completes on counted supported evidence",
            "path": relative_to_repo(CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH),
            "bundle_status": correctness_evidence_summary["status"],
            "summary_field": "main_correctness_claim_completed",
            "summary_value": bool(
                correctness_evidence_summary["summary"]["main_correctness_claim_completed"]
            ),
        },
        {
            "item_id": "correctness_evidence_boundary_visibility",
            "label": "Task 6 boundary evidence remains explicit",
            "path": relative_to_repo(CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH),
            "bundle_status": "pass" if CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH.exists() else "fail",
            "summary_field": "unsupported_boundary_cases",
            "summary_value": int(correctness_evidence_summary["summary"]["unsupported_boundary_cases"])
            > 0,
        },
        {
            "item_id": "performance_evidence_main_benchmark_claim",
            "label": "Task 7 benchmark claim completes on supported evidence",
            "path": relative_to_repo(PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH),
            "bundle_status": performance_evidence_summary["status"],
            "summary_field": "main_benchmark_claim_completed",
            "summary_value": bool(
                performance_evidence_summary["summary"]["main_benchmark_claim_completed"]
            ),
        },
        {
            "item_id": "performance_evidence_threshold_or_diagnosis_rule",
            "label": "Task 7 closes through positive threshold or diagnosis path",
            "path": relative_to_repo(PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH),
            "bundle_status": performance_evidence_summary["status"],
            "summary_field": "positive_or_diagnosis_path_completed",
            "summary_value": bool(
                performance_evidence_summary["summary"]["positive_benchmark_claim_completed"]
                or performance_evidence_summary["summary"]["diagnosis_grounded_closure_completed"]
            ),
            "positive_threshold_status": positive_threshold["status"],
            "diagnosis_status": diagnosis_bundle["status"],
        },
        {
            "item_id": "paper2_closure_rule",
            "label": "Paper 2 closure rule stays explicit on publication surfaces",
            "surface_entries": closure_rule_entries,
        },
    ]


def build_artifact_bundle():
    story3_artifact = _story3()
    evidence_inventory = build_evidence_inventory()
    non_rule_entries = [
        entry for entry in evidence_inventory if entry["item_id"] != "paper2_closure_rule"
    ]
    all_evidence_items_present = all(
        entry["bundle_status"] == "pass" and entry["summary_value"] for entry in non_rule_entries
    )
    claim_closure_rule_present = all(
        not surface["missing_phrases"]
        for entry in evidence_inventory
        if entry["item_id"] == "paper2_closure_rule"
        for surface in entry["surface_entries"]
    )
    positive_or_diagnosis_path_completed = next(
        entry["summary_value"]
        for entry in evidence_inventory
        if entry["item_id"] == "performance_evidence_threshold_or_diagnosis_rule"
    )
    evidence_closure_completed = all(
        [
            story3_artifact["status"] == "pass",
            all_evidence_items_present,
            claim_closure_rule_present,
            positive_or_diagnosis_path_completed,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if evidence_closure_completed else "fail",
        "story3_traceability": {
            "suite_name": story3_artifact["suite_name"],
            "status": story3_artifact["status"],
            "path": relative_to_repo(STORY3_PATH),
            "summary": dict(story3_artifact["summary"]),
        },
        "evidence_inventory": evidence_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
                "evidence_closure_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story3_path": str(STORY3_PATH),
        },
        "summary": {
            "required_evidence_count": len(evidence_inventory),
            "all_evidence_items_present": all_evidence_items_present,
            "claim_closure_rule_present": claim_closure_rule_present,
            "positive_or_diagnosis_path_completed": (
                positive_or_diagnosis_path_completed
            ),
            "evidence_closure_completed": evidence_closure_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 4 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_item_ids = [item["item_id"] for item in artifact["evidence_inventory"]]
    required_item_ids = [
        "correctness_evidence_main_correctness_claim",
        "correctness_evidence_boundary_visibility",
        "performance_evidence_main_benchmark_claim",
        "performance_evidence_threshold_or_diagnosis_rule",
        "paper2_closure_rule",
    ]
    if observed_item_ids != required_item_ids:
        raise ValueError("evidence inventory mismatch")

    non_rule_entries = [
        entry
        for entry in artifact["evidence_inventory"]
        if entry["item_id"] != "paper2_closure_rule"
    ]
    all_evidence_items_present = all(
        entry["bundle_status"] == "pass" and entry["summary_value"] for entry in non_rule_entries
    )
    if artifact["summary"]["all_evidence_items_present"] != all_evidence_items_present:
        raise ValueError("all_evidence_items_present summary is inconsistent")

    claim_closure_rule_present = all(
        not surface["missing_phrases"]
        for entry in artifact["evidence_inventory"]
        if entry["item_id"] == "paper2_closure_rule"
        for surface in entry["surface_entries"]
    )
    if artifact["summary"]["claim_closure_rule_present"] != claim_closure_rule_present:
        raise ValueError("claim_closure_rule_present summary is inconsistent")

    positive_or_diagnosis_path_completed = next(
        entry["summary_value"]
        for entry in artifact["evidence_inventory"]
        if entry["item_id"] == "performance_evidence_threshold_or_diagnosis_rule"
    )
    if (
        artifact["summary"]["positive_or_diagnosis_path_completed"]
        != positive_or_diagnosis_path_completed
    ):
        raise ValueError("positive_or_diagnosis_path_completed summary is inconsistent")

    expected_completed = all(
        [
            artifact["story3_traceability"]["status"] == "pass",
            artifact["summary"]["all_evidence_items_present"],
            artifact["summary"]["claim_closure_rule_present"],
            artifact["summary"]["positive_or_diagnosis_path_completed"],
        ]
    )
    if artifact["summary"]["evidence_closure_completed"] != expected_completed:
        raise ValueError("evidence_closure_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 4 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] evidence_items={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["required_evidence_count"],
                artifact["summary"]["evidence_closure_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Phase 3 Task 8 Story 4 JSON artifact bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = run_validation(verbose=not args.quiet)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_artifact_bundle(output_path, artifact)
    print(
        "Wrote {} with status {} ({})".format(
            output_path,
            artifact["status"],
            artifact["summary"]["required_evidence_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
