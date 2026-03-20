#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 7 future-work and publication-ladder boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.common import (
    MANDATORY_TASK8_DOCS,
    PHASE_POSITIONING_RULE_ALTERNATIVES,
    build_software_metadata,
    get_git_revision,
    load_or_build_artifact,
    load_text,
    phrase_group_present,
    relative_to_repo,
    task8_story_output_dir,
    write_json,
)
from benchmarks.density_matrix.publication_evidence.supported_path_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    run_validation as run_story5_validation,
    validate_artifact_bundle as validate_story5_artifact,
)


SUITE_NAME = "phase3_task8_story7_future_work_boundary"
ARTIFACT_FILENAME = "future_work_boundary_bundle.json"
DEFAULT_OUTPUT_DIR = task8_story_output_dir("story7_future_work_boundary")
STORY5_PATH = (
    task8_story_output_dir("story5_supported_path_scope")
    / STORY5_ARTIFACT_FILENAME
)
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "story5_supported_path_scope",
    "future_work_inventory",
    "software",
    "provenance",
    "summary",
)

MANDATORY_FUTURE_TOPICS = (
    {
        "topic_id": "phase_positioning",
        "label": "Phase 3 methods-paper positioning",
        "surface_phrase_groups": {
            "phase3_abstract": list(PHASE_POSITIONING_RULE_ALTERNATIVES),
            "phase3_short_paper": list(PHASE_POSITIONING_RULE_ALTERNATIVES),
            "phase3_short_paper_narrative": list(PHASE_POSITIONING_RULE_ALTERNATIVES),
            "phase3_paper": list(PHASE_POSITIONING_RULE_ALTERNATIVES),
        },
    },
    {
        "topic_id": "channel_native_future_work",
        "label": "Channel-native fusion stays future work",
        "surface_phrase_groups": {
            "phase3_abstract": [
                "fully channel-native or superoperator-native fused noisy blocks are follow-on work beyond the baseline Paper 2 claim",
            ],
            "phase3_short_paper": [
                "fully channel-native fused noisy blocks are future work beyond the baseline Paper 2 claim",
            ],
            "phase3_short_paper_narrative": [
                "fully channel-native fused noisy blocks are not part of the baseline Paper 2 claim",
            ],
            "phase3_paper": [
                "fully channel-native or superoperator-native fused noisy blocks are outside the baseline Paper 2 claim",
            ],
        },
    },
    {
        "topic_id": "workflow_growth_and_gradients_future_work",
        "label": "Broader workflow growth and gradients stay future work",
        "surface_phrase_groups": {
            "phase3_abstract": [
                "broader noisy VQE/VQA workflow growth, density-backend gradients, and optimizer-comparison studies remain Phase 4+ work",
            ],
            "phase3_short_paper": [
                "density-backend gradients, optimizer-comparison studies, and broader noisy VQE/VQA workflow growth are Phase 4+ work",
            ],
            "phase3_short_paper_narrative": [
                "broader noisy VQE/VQA feature growth and density-backend gradients are Phase 4+ work",
            ],
            "phase3_paper": [
                "broader noisy VQE/VQA workflow growth, density-backend gradients, and optimizer studies remain Phase 4+ work",
            ],
        },
    },
    {
        "topic_id": "approximate_scaling_future_work",
        "label": "Approximate scaling stays future work",
        "surface_phrase_groups": {
            "phase3_abstract": [
                "approximate scaling methods such as trajectories or MPDO-style approaches are outside the current Paper 2 claim",
            ],
            "phase3_short_paper": [
                "approximate scaling methods remain later work beyond the current Paper 2 claim",
            ],
            "phase3_short_paper_narrative": [
                "approximate scaling methods remain future branches rather than current Paper 2 results",
            ],
            "phase3_paper": [
                "approximate scaling methods such as trajectories or MPDO-style approaches are outside the current Paper 2 claim",
            ],
        },
    },
    {
        "topic_id": "benchmark_driven_follow_on_branch",
        "label": "Follow-on architecture remains benchmark-driven",
        "surface_phrase_groups": {
            "phase3_short_paper_narrative": [
                "benchmark-driven follow-ons rather than hidden dependencies",
            ],
            "phase3_paper": [
                "follow-on branch for more invasive channel-native fusion",
                "benchmark package shows that the native baseline still leaves the dominant bottleneck unresolved",
            ],
        },
    },
)


def _story5():
    return load_or_build_artifact(
        STORY5_PATH,
        run_validation=run_story5_validation,
        validate_artifact_bundle=validate_story5_artifact,
    )


def build_future_work_inventory():
    inventory = []
    for item in MANDATORY_FUTURE_TOPICS:
        surface_entries = []
        for doc_id, phrases in item["surface_phrase_groups"].items():
            path = MANDATORY_TASK8_DOCS[doc_id]
            text = load_text(path)
            surface_entries.append(
                {
                    "doc_id": doc_id,
                    "path": relative_to_repo(path),
                    "phrases": list(phrases),
                    "present": phrase_group_present(text, phrases),
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
    story5_artifact = _story5()
    future_work_inventory = build_future_work_inventory()
    all_future_work_items_present = all(
        all(surface["present"] for surface in item["surface_entries"])
        for item in future_work_inventory
    )
    phase_positioning_present = all(
        surface["present"]
        for item in future_work_inventory
        if item["topic_id"] == "phase_positioning"
        for surface in item["surface_entries"]
    )
    benchmark_driven_follow_on_present = all(
        surface["present"]
        for item in future_work_inventory
        if item["topic_id"] == "benchmark_driven_follow_on_branch"
        for surface in item["surface_entries"]
    )
    future_work_boundary_completed = all(
        [
            story5_artifact["status"] == "pass",
            all_future_work_items_present,
            phase_positioning_present,
            benchmark_driven_follow_on_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if future_work_boundary_completed else "fail",
        "story5_supported_path_scope": {
            "suite_name": story5_artifact["suite_name"],
            "status": story5_artifact["status"],
            "path": relative_to_repo(STORY5_PATH),
            "summary": dict(story5_artifact["summary"]),
        },
        "future_work_inventory": future_work_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
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
            "benchmark_driven_follow_on_present": benchmark_driven_follow_on_present,
            "future_work_boundary_completed": future_work_boundary_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 7 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_topic_ids = [item["topic_id"] for item in artifact["future_work_inventory"]]
    required_topic_ids = [item["topic_id"] for item in MANDATORY_FUTURE_TOPICS]
    if observed_topic_ids != required_topic_ids:
        raise ValueError("future-work inventory mismatch")

    all_future_work_items_present = all(
        all(surface["present"] for surface in item["surface_entries"])
        for item in artifact["future_work_inventory"]
    )
    if artifact["summary"]["all_future_work_items_present"] != all_future_work_items_present:
        raise ValueError("all_future_work_items_present summary is inconsistent")

    phase_positioning_present = all(
        surface["present"]
        for item in artifact["future_work_inventory"]
        if item["topic_id"] == "phase_positioning"
        for surface in item["surface_entries"]
    )
    if artifact["summary"]["phase_positioning_present"] != phase_positioning_present:
        raise ValueError("phase_positioning_present summary is inconsistent")

    benchmark_driven_follow_on_present = all(
        surface["present"]
        for item in artifact["future_work_inventory"]
        if item["topic_id"] == "benchmark_driven_follow_on_branch"
        for surface in item["surface_entries"]
    )
    if (
        artifact["summary"]["benchmark_driven_follow_on_present"]
        != benchmark_driven_follow_on_present
    ):
        raise ValueError("benchmark_driven_follow_on_present summary is inconsistent")

    expected_completed = all(
        [
            artifact["story5_supported_path_scope"]["status"] == "pass",
            artifact["summary"]["all_future_work_items_present"],
            artifact["summary"]["phase_positioning_present"],
            artifact["summary"]["benchmark_driven_follow_on_present"],
        ]
    )
    if artifact["summary"]["future_work_boundary_completed"] != expected_completed:
        raise ValueError("future_work_boundary_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 7 status does not match completion summary")


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
        help="Directory for the Phase 3 Task 8 Story 7 JSON artifact bundle.",
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
