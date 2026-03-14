#!/usr/bin/env python3
"""Validation: Task 7 Story 5 future-work and non-goal boundary layer.

Builds a machine-readable checker for the current-versus-future boundary in the
Phase 2 documentation bundle. This layer is intentionally thin:
- it reuses the Story 4 evidence-bar clarification,
- it records the later-phase topics that Phase 2 must not overclaim,
- it ties those topics to roadmap-facing source docs,
- and it fails when later-phase work can be mistaken for a delivered Phase 2
  commitment.

Run with:
    python benchmarks/density_matrix/task7_story5_future_work_boundary_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.task7_doc_utils import (
    MANDATORY_PHASE2_DOCS,
    PHASE2_DOCUMENTATION_INDEX_PATH,
    TASK7_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    normalize_text,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.task7_story4_evidence_bar_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)


SUITE_NAME = "task7_story5_future_work_boundary"
ARTIFACT_FILENAME = "story5_future_work_boundary.json"
DEFAULT_OUTPUT_DIR = TASK7_OUTPUT_DIR
STORY4_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / STORY4_ARTIFACT_FILENAME
STORY5_SECTION_HEADING = "## Future Work And Non-Goals"
MANDATORY_FUTURE_TOPICS = (
    {
        "topic_id": "phase2_exact_integration",
        "label": "Phase 2 exact integration milestone",
        "entry_point_phrases": [
            "Phase 2 remains the exact noisy backend integration milestone",
        ],
        "source_doc_ids": ["planning_planning", "phase2_detailed_planning"],
        "source_phrases": [
            "Phase 2: Exact Noisy Training Backend Integration",
            "Phase 2 establishes the exact noisy backend as a usable scientific workflow",
        ],
    },
    {
        "topic_id": "phase3_partitioning_fusion",
        "label": "Phase 3 partitioning and fusion",
        "entry_point_phrases": [
            "Density-aware partitioning, fusion, and acceleration are future work for Phase 3",
        ],
        "source_doc_ids": ["planning_planning", "research_alignment"],
        "source_phrases": [
            "Phase 3: Density-Aware Partitioning, Fusion, And Acceleration",
            "Phase 3 | Full noise module and validation",
        ],
    },
    {
        "topic_id": "gradient_and_approximate_scaling",
        "label": "Gradients and approximate scaling",
        "entry_point_phrases": [
            "Gradient support and approximate scaling methods remain future work beyond the core Phase 2 milestone",
        ],
        "source_doc_ids": ["planning_planning", "phase2_detailed_planning"],
        "source_phrases": [
            "gradient-path completion for density-matrix optimization",
            "stochastic trajectories or MPDO-based approximation paths",
        ],
    },
    {
        "topic_id": "phase4_optimizer_work",
        "label": "Phase 4 optimizer studies",
        "entry_point_phrases": [
            "Broader noisy-VQA integration and optimizer comparisons are future work for Phase 4",
        ],
        "source_doc_ids": ["planning_planning", "research_alignment"],
        "source_phrases": [
            "Phase 4: Noisy VQA Workflows And Optimizer Studies",
            "Phase 4 | Noisy VQA integration and optimizer comparisons",
        ],
    },
    {
        "topic_id": "phase5_trainability",
        "label": "Phase 5 trainability analysis",
        "entry_point_phrases": [
            "Trainability analysis is future work for Phase 5",
        ],
        "source_doc_ids": ["planning_planning", "research_alignment"],
        "source_phrases": [
            "Phase 5: Trainability Analysis Under Realistic Noise",
            "Phase 5 | Trainability analysis under noise",
        ],
    },
    {
        "topic_id": "non_goal_boundary",
        "label": "Not current Phase 2 commitments",
        "entry_point_phrases": [
            "These topics are not current Phase 2 commitments.",
        ],
        "source_doc_ids": ["phase2_detailed_planning"],
        "source_phrases": [
            "must not be presented as Phase 2 commitments",
        ],
    },
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "entry_point",
    "requirements",
    "story4_evidence_bar",
    "future_work_inventory",
    "software",
    "provenance",
    "summary",
)


def _load_story4_artifact(path: Path = STORY4_ARTIFACT_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story4_artifact(artifact)
        return artifact
    artifact = run_story4_validation(verbose=False)
    return artifact


def _phrases_present(text: str, phrases: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(normalize_text(phrase) in normalized_text for phrase in phrases)


def build_future_work_inventory():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    inventory = []
    for item in MANDATORY_FUTURE_TOPICS:
        source_paths = [MANDATORY_PHASE2_DOCS[doc_id] for doc_id in item["source_doc_ids"]]
        source_text = "\n".join(load_text(path) for path in source_paths)
        inventory.append(
            {
                "topic_id": item["topic_id"],
                "label": item["label"],
                "source_doc_ids": list(item["source_doc_ids"]),
                "source_paths": [relative_to_repo(path) for path in source_paths],
                "entry_point_phrases": list(item["entry_point_phrases"]),
                "source_phrases": list(item["source_phrases"]),
                "entry_point_present": _phrases_present(
                    entry_text, item["entry_point_phrases"]
                ),
                "source_present": _phrases_present(source_text, item["source_phrases"]),
            }
        )
    return inventory


def build_artifact_bundle():
    story4_artifact = _load_story4_artifact()
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    future_work_inventory = build_future_work_inventory()

    section_heading_present = STORY5_SECTION_HEADING in entry_text
    all_future_work_items_present = all(
        item["entry_point_present"] and item["source_present"]
        for item in future_work_inventory
    )
    future_work_boundary_completed = bool(
        story4_artifact["status"] == "pass"
        and section_heading_present
        and all_future_work_items_present
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if future_work_boundary_completed else "fail",
        "entry_point": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "section_heading": STORY5_SECTION_HEADING,
            "section_heading_present": section_heading_present,
        },
        "requirements": {
            "required_topic_ids": [
                item["topic_id"] for item in MANDATORY_FUTURE_TOPICS
            ],
        },
        "story4_evidence_bar": {
            "suite_name": story4_artifact["suite_name"],
            "status": story4_artifact["status"],
            "summary": story4_artifact["summary"],
        },
        "future_work_inventory": future_work_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task7_story5_future_work_boundary_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story4_evidence_bar_path": str(STORY4_ARTIFACT_PATH),
            "entry_point_path": str(PHASE2_DOCUMENTATION_INDEX_PATH),
        },
        "summary": {
            "required_topic_count": len(future_work_inventory),
            "section_heading_present": section_heading_present,
            "all_future_work_items_present": all_future_work_items_present,
            "future_work_boundary_completed": future_work_boundary_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 7 Story 5 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_ids = [item["topic_id"] for item in artifact["future_work_inventory"]]
    required_ids = artifact["requirements"]["required_topic_ids"]
    if observed_ids != required_ids:
        raise ValueError(
            "Task 7 Story 5 future-work inventory mismatch: expected {}, observed {}".format(
                required_ids, observed_ids
            )
        )

    if artifact["summary"]["section_heading_present"] != artifact["entry_point"][
        "section_heading_present"
    ]:
        raise ValueError(
            "Task 7 Story 5 section_heading_present summary is inconsistent"
        )

    if artifact["summary"]["all_future_work_items_present"] != all(
        item["entry_point_present"] and item["source_present"]
        for item in artifact["future_work_inventory"]
    ):
        raise ValueError(
            "Task 7 Story 5 all_future_work_items_present summary is inconsistent"
        )

    if artifact["summary"]["future_work_boundary_completed"] != (
        artifact["status"] == "pass"
    ):
        raise ValueError(
            "Task 7 Story 5 future_work_boundary_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] future_topics={} complete={}".format(
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
        help="Directory for the Task 7 Story 5 JSON artifact bundle.",
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
