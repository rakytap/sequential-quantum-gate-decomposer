#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 5 supported-path and benchmark honesty."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.common import (
    BOTTLENECK_PHRASE_ALTERNATIVES,
    CURRENT_DIAGNOSIS_PHRASE_ALTERNATIVES,
    NO_FALLBACK_RULE_ALTERNATIVES,
    PAPER_MAIN_CLAIM,
    PUBLICATION_SURFACES,
    SUPPORTED_PATH_BOUNDARY_ALTERNATIVES,
    build_software_metadata,
    build_surface_presence,
    get_git_revision,
    relative_to_repo,
    correctness_evidence_boundary_phrase_options,
    correctness_evidence_count_phrase_options,
    performance_evidence_count_phrase_options,
    performance_evidence_review_phrase_options,
    publication_evidence_output_dir,
    write_json,
    load_or_build_artifact,
)
from benchmarks.density_matrix.publication_evidence.evidence_closure_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)


SUITE_NAME = "phase3_publication_evidence_supported_path"
ARTIFACT_FILENAME = "supported_path_scope_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("supported_path")
STORY4_PATH = publication_evidence_output_dir("evidence_closure") / STORY4_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "evidence_closure",
    "surface_inventory",
    "software",
    "provenance",
    "summary",
)

BOUNDED_PLANNER_ALTERNATIVES = (
    "a benchmark-calibrated density-aware planning policy on a bounded candidate surface",
    "the supported planning claim is presently a benchmark-calibrated selection rule over a bounded noisy-planner candidate family",
    "the benchmark-facing planning result is a benchmark-calibrated policy over a bounded family of auditable `max_partition_qubits` span-budget settings",
    "the current benchmark-facing result calibrates auditable `max_partition_qubits` span-budget settings on the existing noisy planner surface",
)


def _story4(path: Path = STORY4_PATH):
    return load_or_build_artifact(
        path,
        run_validation=run_story4_validation,
        validate_artifact_bundle=validate_story4_artifact,
    )


def _required_groups():
    return [
        {
            "group_id": "supported_path_boundary",
            "phrases": list(SUPPORTED_PATH_BOUNDARY_ALTERNATIVES),
        },
        {
            "group_id": "no_fallback_rule",
            "phrases": list(NO_FALLBACK_RULE_ALTERNATIVES),
        },
        {
            "group_id": "bounded_planner_claim",
            "phrases": list(BOUNDED_PLANNER_ALTERNATIVES),
        },
        {
            "group_id": "correctness_evidence_count",
            "phrases": list(correctness_evidence_count_phrase_options()),
        },
        {
            "group_id": "correctness_evidence_boundary_count",
            "phrases": list(correctness_evidence_boundary_phrase_options()),
        },
        {
            "group_id": "performance_evidence_count",
            "phrases": list(performance_evidence_count_phrase_options()),
        },
        {
            "group_id": "performance_evidence_review_cases",
            "phrases": list(performance_evidence_review_phrase_options()),
        },
        {
            "group_id": "diagnosis_closure",
            "phrases": list(CURRENT_DIAGNOSIS_PHRASE_ALTERNATIVES),
        },
        {
            "group_id": "bottleneck_wording",
            "phrases": list(BOTTLENECK_PHRASE_ALTERNATIVES),
        },
    ]


def build_surface_inventory():
    inventory = []
    for surface_id in PUBLICATION_SURFACES:
        entry = build_surface_presence(
            surface_id,
            required_phrase_groups=_required_groups(),
        )
        missing_ids = {
            item["group_id"] for item in entry["missing_required_phrase_groups"]
        }
        entry["supported_path_boundary_present"] = "supported_path_boundary" not in missing_ids
        entry["no_fallback_rule_present"] = "no_fallback_rule" not in missing_ids
        entry["bounded_planner_claim_present"] = "bounded_planner_claim" not in missing_ids
        entry["correctness_evidence_counts_present"] = {
            "correctness_evidence_count": "correctness_evidence_count" not in missing_ids,
            "correctness_evidence_boundary_count": "correctness_evidence_boundary_count" not in missing_ids,
        }
        entry["performance_evidence_counts_present"] = {
            "performance_evidence_count": "performance_evidence_count" not in missing_ids,
            "performance_evidence_review_cases": "performance_evidence_review_cases" not in missing_ids,
        }
        entry["diagnosis_closure_present"] = "diagnosis_closure" not in missing_ids
        entry["bottleneck_wording_present"] = "bottleneck_wording" not in missing_ids
        entry["status"] = (
            "pass"
            if not entry["missing_required_phrase_groups"]
            else "fail"
        )
        inventory.append(entry)
    return inventory


def build_artifact_bundle():
    story4_artifact = _story4()
    surface_inventory = build_surface_inventory()
    all_supported_path_boundaries_present = all(
        entry["supported_path_boundary_present"] for entry in surface_inventory
    )
    all_no_fallback_rules_present = all(
        entry["no_fallback_rule_present"] for entry in surface_inventory
    )
    all_bounded_planner_claims_present = all(
        entry["bounded_planner_claim_present"] for entry in surface_inventory
    )
    all_correctness_evidence_count_surfaces_honest = all(
        all(entry["correctness_evidence_counts_present"].values()) for entry in surface_inventory
    )
    all_performance_evidence_count_surfaces_honest = all(
        all(entry["performance_evidence_counts_present"].values()) for entry in surface_inventory
    )
    all_diagnosis_limitations_present = all(
        entry["diagnosis_closure_present"] and entry["bottleneck_wording_present"]
        for entry in surface_inventory
    )
    supported_path_scope_completed = all(
        [
            story4_artifact["status"] == "pass",
            all_supported_path_boundaries_present,
            all_no_fallback_rules_present,
            all_bounded_planner_claims_present,
            all_correctness_evidence_count_surfaces_honest,
            all_performance_evidence_count_surfaces_honest,
            all_diagnosis_limitations_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if supported_path_scope_completed else "fail",
        "evidence_closure": {
            "suite_name": story4_artifact["suite_name"],
            "status": story4_artifact["status"],
            "path": relative_to_repo(STORY4_PATH),
            "summary": dict(story4_artifact["summary"]),
        },
        "surface_inventory": surface_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
                "supported_path_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story4_path": str(STORY4_PATH),
            "paper_main_claim_anchor": PAPER_MAIN_CLAIM,
        },
        "summary": {
            "surface_count": len(surface_inventory),
            "all_supported_path_boundaries_present": (
                all_supported_path_boundaries_present
            ),
            "all_no_fallback_rules_present": all_no_fallback_rules_present,
            "all_bounded_planner_claims_present": all_bounded_planner_claims_present,
            "all_correctness_evidence_count_surfaces_honest": all_correctness_evidence_count_surfaces_honest,
            "all_performance_evidence_count_surfaces_honest": all_performance_evidence_count_surfaces_honest,
            "all_diagnosis_limitations_present": all_diagnosis_limitations_present,
            "supported_path_scope_completed": supported_path_scope_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 5 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_surface_ids = [entry["surface_id"] for entry in artifact["surface_inventory"]]
    required_surface_ids = list(PUBLICATION_SURFACES.keys())
    if observed_surface_ids != required_surface_ids:
        raise ValueError("supported-path surface inventory mismatch")

    expected_flags = {
        "all_supported_path_boundaries_present": all(
            entry["supported_path_boundary_present"]
            for entry in artifact["surface_inventory"]
        ),
        "all_no_fallback_rules_present": all(
            entry["no_fallback_rule_present"] for entry in artifact["surface_inventory"]
        ),
        "all_bounded_planner_claims_present": all(
            entry["bounded_planner_claim_present"]
            for entry in artifact["surface_inventory"]
        ),
        "all_correctness_evidence_count_surfaces_honest": all(
            all(entry["correctness_evidence_counts_present"].values())
            for entry in artifact["surface_inventory"]
        ),
        "all_performance_evidence_count_surfaces_honest": all(
            all(entry["performance_evidence_counts_present"].values())
            for entry in artifact["surface_inventory"]
        ),
        "all_diagnosis_limitations_present": all(
            entry["diagnosis_closure_present"] and entry["bottleneck_wording_present"]
            for entry in artifact["surface_inventory"]
        ),
    }
    for field, value in expected_flags.items():
        if artifact["summary"][field] != value:
            raise ValueError(f"{field} summary is inconsistent")

    expected_completed = all(
        [
            artifact["evidence_closure"]["status"] == "pass",
            artifact["summary"]["all_supported_path_boundaries_present"],
            artifact["summary"]["all_no_fallback_rules_present"],
            artifact["summary"]["all_bounded_planner_claims_present"],
            artifact["summary"]["all_correctness_evidence_count_surfaces_honest"],
            artifact["summary"]["all_performance_evidence_count_surfaces_honest"],
            artifact["summary"]["all_diagnosis_limitations_present"],
        ]
    )
    if artifact["summary"]["supported_path_scope_completed"] != expected_completed:
        raise ValueError("supported_path_scope_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 5 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] surfaces={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["surface_count"],
                artifact["summary"]["supported_path_scope_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Phase 3 Task 8 Story 5 JSON artifact bundle.",
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
            artifact["summary"]["surface_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
