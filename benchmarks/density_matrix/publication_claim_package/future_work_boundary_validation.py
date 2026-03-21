#!/usr/bin/env python3
"""Validation: Task 8 Story 6 future-work and publication-ladder boundary.

Builds a machine-readable future-work boundary checker for Paper 1. This layer
is intentionally thin:
- it reuses the Story 5 supported-path scope output,
- it records the allowed later-phase positioning signals,
- it validates that Paper 1 stays a Phase 2 integration milestone,
- and it fails when later-phase work is phrased as current result.

Run with:
    python benchmarks/density_matrix/publication_claim_package/future_work_boundary_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_claim_package.doc_utils import (
    CANONICAL_NON_CLAIMS,
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    PHASE_POSITIONING_RULE,
    PUBLICATION_CLAIM_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    missing_phrases,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.publication_claim_package.supported_path_scope_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    run_validation as run_story5_validation,
    validate_artifact_bundle as validate_story5_artifact,
)


SUITE_NAME = "future_work_boundary_bundle"
ARTIFACT_FILENAME = "future_work_boundary_bundle.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
STORY5_PATH = DEFAULT_OUTPUT_DIR / STORY5_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "supported_path",
    "future_work_inventory",
    "software",
    "provenance",
    "summary",
)
MANDATORY_FUTURE_TOPICS = (
    {
        "topic_id": "phase_positioning",
        "label": "Phase 2 integration-paper positioning",
        "surface_phrases": {
            "phase2_abstract": [PHASE_POSITIONING_RULE],
            "phase2_short_paper": [PHASE_POSITIONING_RULE],
            "phase2_short_paper_narrative": [PHASE_POSITIONING_RULE],
            "phase2_paper": [PHASE_POSITIONING_RULE],
        },
    },
    {
        "topic_id": "phase3_partitioning_and_fusion",
        "label": "Phase 3 partitioning and fusion future work",
        "surface_phrases": {
            "phase2_abstract": [CANONICAL_NON_CLAIMS[0]],
            "phase2_short_paper": [CANONICAL_NON_CLAIMS[0], "Phase 3"],
            "phase2_short_paper_narrative": [CANONICAL_NON_CLAIMS[0], "Phase 3"],
            "phase2_paper": [CANONICAL_NON_CLAIMS[0], "Phase 3"],
        },
    },
    {
        "topic_id": "gradients_and_approximate_scaling",
        "label": "Gradients and approximate scaling stay future work",
        "surface_phrases": {
            "phase2_abstract": [CANONICAL_NON_CLAIMS[1]],
            "phase2_short_paper": [CANONICAL_NON_CLAIMS[1]],
            "phase2_short_paper_narrative": [CANONICAL_NON_CLAIMS[1]],
            "phase2_paper": [CANONICAL_NON_CLAIMS[1]],
        },
    },
    {
        "topic_id": "phase4_and_phase5_studies",
        "label": "Phase 4 and Phase 5 studies stay future work",
        "surface_phrases": {
            "phase2_abstract": [CANONICAL_NON_CLAIMS[4]],
            "phase2_short_paper": [CANONICAL_NON_CLAIMS[4], "Phase 4", "Phase 5"],
            "phase2_short_paper_narrative": [CANONICAL_NON_CLAIMS[4], "Phase 4", "Phase 5"],
            "phase2_paper": [CANONICAL_NON_CLAIMS[4], "Phase 4", "Phase 5"],
        },
    },
)


def _load_story5(path: Path = STORY5_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story5_artifact(artifact)
        return artifact
    return run_story5_validation(verbose=False)


def build_future_work_inventory():
    inventory = []
    for item in MANDATORY_FUTURE_TOPICS:
        surface_entries = []
        for doc_id, phrases in item["surface_phrases"].items():
            path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[doc_id]
            text = load_text(path)
            surface_entries.append(
                {
                    "doc_id": doc_id,
                    "path": relative_to_repo(path),
                    "phrases": list(phrases),
                    "missing_phrases": missing_phrases(text, phrases),
                }
            )
        inventory.append(
            {
                "topic_id": item["topic_id"],
                "label": item["label"],
                "surface_entries": surface_entries,
            }
        )
    return inventory


def build_artifact_bundle():
    story5_artifact = _load_story5()
    future_work_inventory = build_future_work_inventory()
    all_future_work_items_present = all(
        all(not surface["missing_phrases"] for surface in item["surface_entries"])
        for item in future_work_inventory
    )
    phase_positioning_present = all(
        not surface["missing_phrases"]
        for item in future_work_inventory
        if item["topic_id"] == "phase_positioning"
        for surface in item["surface_entries"]
    )
    future_work_boundary_completed = all(
        [
            story5_artifact["status"] == "pass",
            all_future_work_items_present,
            phase_positioning_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if future_work_boundary_completed else "fail",
        "supported_path": {
            "suite_name": story5_artifact["suite_name"],
            "status": story5_artifact["status"],
            "path": relative_to_repo(STORY5_PATH),
            "summary": dict(story5_artifact["summary"]),
        },
        "future_work_inventory": future_work_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "future_work_boundary_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story5_path": str(STORY5_PATH),
        },
        "summary": {
            "required_topic_count": len(MANDATORY_FUTURE_TOPICS),
            "all_future_work_items_present": all_future_work_items_present,
            "phase_positioning_present": phase_positioning_present,
            "future_work_boundary_completed": future_work_boundary_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 8 Story 6 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_topic_ids = [item["topic_id"] for item in artifact["future_work_inventory"]]
    required_topic_ids = [item["topic_id"] for item in MANDATORY_FUTURE_TOPICS]
    if observed_topic_ids != required_topic_ids:
        raise ValueError("future-work inventory mismatch")

    all_future_work_items_present = all(
        all(not surface["missing_phrases"] for surface in item["surface_entries"])
        for item in artifact["future_work_inventory"]
    )
    if artifact["summary"]["all_future_work_items_present"] != all_future_work_items_present:
        raise ValueError("all_future_work_items_present summary is inconsistent")

    phase_positioning_present = all(
        not surface["missing_phrases"]
        for item in artifact["future_work_inventory"]
        if item["topic_id"] == "phase_positioning"
        for surface in item["surface_entries"]
    )
    if artifact["summary"]["phase_positioning_present"] != phase_positioning_present:
        raise ValueError("phase_positioning_present summary is inconsistent")

    expected_completed = all(
        [
            artifact["supported_path"]["status"] == "pass",
            artifact["summary"]["all_future_work_items_present"],
            artifact["summary"]["phase_positioning_present"],
        ]
    )
    if artifact["summary"]["future_work_boundary_completed"] != expected_completed:
        raise ValueError("future_work_boundary_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Task 8 Story 6 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] topics={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["required_topic_count"],
                artifact["summary"]["future_work_boundary_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 8 Story 6 JSON artifact bundle.",
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
            artifact["summary"]["required_topic_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
