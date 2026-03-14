#!/usr/bin/env python3
"""Validation: Task 7 Story 1 documentation contract reference map.

Builds a machine-readable map for the authoritative Phase 2 documentation path.
This layer is intentionally thin:
- it records the canonical Phase 2 document inventory and roles,
- it maps mandatory contract topics to authoritative sources,
- it validates one stable citable entry point for the Phase 2 contract,
- and it fails explicitly when topic coverage or source references are
  incomplete.

Run with:
    python benchmarks/density_matrix/task7_story1_contract_reference_validation.py
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
    load_text,
    relative_to_repo,
    write_json,
)


SUITE_NAME = "task7_story1_contract_reference_map"
ARTIFACT_FILENAME = "story1_contract_reference_map.json"
DEFAULT_OUTPUT_DIR = TASK7_OUTPUT_DIR
ENTRY_POINT_REQUIRED_HEADINGS = (
    "# Phase 2 Documentation Index",
    "## Source-of-Truth Hierarchy",
    "## Topic Map",
    "## Stable Evidence Entry Point",
    "## Reading Order",
)
MANDATORY_TOPIC_LABELS = (
    "Backend selection",
    "Observable scope",
    "Bridge scope",
    "Support matrix",
    "Workflow anchor",
    "Benchmark minimum",
    "Numeric thresholds",
    "Non-goals",
    "Publication evidence surface",
)
DOC_CLASS_ROLES = (
    {
        "document_class": "phase2_documentation_entry_point",
        "doc_ids": ["phase2_documentation_index"],
        "role": "stable starting point for navigation and citation",
    },
    {
        "document_class": "phase_contract",
        "doc_ids": ["phase2_detailed_planning"],
        "role": "full Phase 2 contract and frozen implementation scope",
    },
    {
        "document_class": "phase_decision_record",
        "doc_ids": ["phase2_adrs"],
        "role": "accepted decisions, rationale, consequences, and rejected alternatives",
    },
    {
        "document_class": "phase_readiness_closure",
        "doc_ids": ["phase2_checklist"],
        "role": "implementation-readiness closure and checklist verdict",
    },
    {
        "document_class": "task_contracts",
        "doc_ids": [
            "task1_mini_spec",
            "task2_mini_spec",
            "task3_mini_spec",
            "task4_mini_spec",
            "task5_mini_spec",
            "task6_mini_spec",
            "task7_mini_spec",
        ],
        "role": "task-level behavior contracts and acceptance evidence",
    },
    {
        "document_class": "workflow_evidence_surface",
        "doc_ids": ["task6_publication_bundle"],
        "role": "machine-readable evidence surface for the delivered workflow claim",
    },
    {
        "document_class": "publication_facing_summaries",
        "doc_ids": [
            "phase2_abstract",
            "phase2_short_paper",
            "phase2_paper",
        ],
        "role": "publication-facing framing of the delivered Phase 2 claim",
    },
    {
        "document_class": "roadmap_context",
        "doc_ids": [
            "planning_planning",
            "planning_publications",
            "research_alignment",
            "changelog",
        ],
        "role": "broader roadmap, publication ladder, and milestone wording context",
    },
)
MANDATORY_TOPICS = {
    "backend_selection": {
        "label": "Backend selection",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": [
            "phase2_adrs",
            "phase2_checklist",
            "task1_mini_spec",
        ],
        "layer": "phase contract + task contract",
    },
    "observable_scope": {
        "label": "Observable scope",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": ["phase2_adrs", "task2_mini_spec"],
        "layer": "phase contract + task contract",
    },
    "bridge_scope": {
        "label": "Bridge scope",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": ["phase2_adrs", "task3_mini_spec"],
        "layer": "phase contract + task contract",
    },
    "support_matrix": {
        "label": "Support matrix",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": ["phase2_adrs", "task4_mini_spec"],
        "layer": "phase contract + task contract",
    },
    "workflow_anchor": {
        "label": "Workflow anchor",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": ["phase2_adrs", "task6_mini_spec"],
        "layer": "phase contract + task contract",
    },
    "benchmark_minimum": {
        "label": "Benchmark minimum",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": [
            "phase2_adrs",
            "task5_mini_spec",
            "task6_publication_bundle",
        ],
        "layer": "phase contract + evidence surface",
    },
    "numeric_thresholds": {
        "label": "Numeric thresholds",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": [
            "phase2_adrs",
            "task5_mini_spec",
            "task6_mini_spec",
        ],
        "layer": "phase contract + task contract",
    },
    "non_goals": {
        "label": "Non-goals",
        "primary_doc_id": "phase2_detailed_planning",
        "supporting_doc_ids": [
            "planning_planning",
            "research_alignment",
            "changelog",
        ],
        "layer": "phase contract + roadmap context",
    },
    "publication_evidence_surface": {
        "label": "Publication evidence surface",
        "primary_doc_id": "task6_publication_bundle",
        "supporting_doc_ids": [
            "planning_publications",
            "phase2_abstract",
            "phase2_short_paper",
            "phase2_paper",
        ],
        "layer": "evidence surface + publication-facing",
    },
}
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "entry_point",
    "requirements",
    "document_inventory",
    "topic_map",
    "software",
    "provenance",
    "summary",
)


def _doc_entry(doc_id: str):
    path = MANDATORY_PHASE2_DOCS[doc_id]
    return {
        "doc_id": doc_id,
        "path": relative_to_repo(path),
        "exists": path.exists(),
    }


def build_document_inventory():
    inventory = []
    for item in DOC_CLASS_ROLES:
        inventory.append(
            {
                "document_class": item["document_class"],
                "role": item["role"],
                "documents": [_doc_entry(doc_id) for doc_id in item["doc_ids"]],
            }
        )
    return inventory


def build_topic_map():
    topic_map = []
    for topic_id, topic in MANDATORY_TOPICS.items():
        primary_path = MANDATORY_PHASE2_DOCS[topic["primary_doc_id"]]
        topic_map.append(
            {
                "topic_id": topic_id,
                "label": topic["label"],
                "primary_doc_id": topic["primary_doc_id"],
                "primary_path": relative_to_repo(primary_path),
                "primary_exists": primary_path.exists(),
                "supporting_doc_ids": list(topic["supporting_doc_ids"]),
                "supporting_paths": [
                    relative_to_repo(MANDATORY_PHASE2_DOCS[doc_id])
                    for doc_id in topic["supporting_doc_ids"]
                ],
                "supporting_exists": [
                    MANDATORY_PHASE2_DOCS[doc_id].exists()
                    for doc_id in topic["supporting_doc_ids"]
                ],
                "layer": topic["layer"],
            }
        )
    return topic_map


def build_entry_point_summary():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    missing_headings = [
        heading for heading in ENTRY_POINT_REQUIRED_HEADINGS if heading not in entry_text
    ]
    missing_topic_labels = [
        label for label in MANDATORY_TOPIC_LABELS if label not in entry_text
    ]
    missing_primary_paths = []
    for topic in MANDATORY_TOPICS.values():
        primary_path = relative_to_repo(MANDATORY_PHASE2_DOCS[topic["primary_doc_id"]])
        if primary_path not in entry_text:
            missing_primary_paths.append(primary_path)
    return {
        "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
        "exists": PHASE2_DOCUMENTATION_INDEX_PATH.exists(),
        "missing_headings": missing_headings,
        "missing_topic_labels": missing_topic_labels,
        "missing_primary_paths": missing_primary_paths,
    }


def build_artifact_bundle():
    document_inventory = build_document_inventory()
    topic_map = build_topic_map()
    entry_point = build_entry_point_summary()

    all_documents_exist = all(
        document["exists"]
        for inventory_class in document_inventory
        for document in inventory_class["documents"]
    )
    all_topics_present = (
        sorted(topic["topic_id"] for topic in topic_map) == sorted(MANDATORY_TOPICS.keys())
    )
    all_primary_references_exist = all(topic["primary_exists"] for topic in topic_map)
    all_supporting_references_exist = all(
        all(topic["supporting_exists"]) for topic in topic_map
    )
    entry_point_headings_complete = not entry_point["missing_headings"]
    entry_point_topics_complete = not entry_point["missing_topic_labels"]
    entry_point_primary_paths_complete = not entry_point["missing_primary_paths"]
    contract_reference_map_completed = bool(
        entry_point["exists"]
        and all_documents_exist
        and all_topics_present
        and all_primary_references_exist
        and all_supporting_references_exist
        and entry_point_headings_complete
        and entry_point_topics_complete
        and entry_point_primary_paths_complete
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if contract_reference_map_completed else "fail",
        "entry_point": entry_point,
        "requirements": {
            "required_entry_point_headings": list(ENTRY_POINT_REQUIRED_HEADINGS),
            "required_topic_labels": list(MANDATORY_TOPIC_LABELS),
            "required_topic_ids": sorted(MANDATORY_TOPICS.keys()),
        },
        "document_inventory": document_inventory,
        "topic_map": topic_map,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task7_story1_contract_reference_validation.py"
            ),
            "working_directory": str(DEFAULT_OUTPUT_DIR.parents[3]),
            "git_revision": get_git_revision(),
            "entry_point_path": str(PHASE2_DOCUMENTATION_INDEX_PATH),
        },
        "summary": {
            "document_class_count": len(document_inventory),
            "topic_count": len(topic_map),
            "all_documents_exist": all_documents_exist,
            "all_topics_present": all_topics_present,
            "all_primary_references_exist": all_primary_references_exist,
            "all_supporting_references_exist": all_supporting_references_exist,
            "entry_point_headings_complete": entry_point_headings_complete,
            "entry_point_topics_complete": entry_point_topics_complete,
            "entry_point_primary_paths_complete": entry_point_primary_paths_complete,
            "contract_reference_map_completed": contract_reference_map_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 7 Story 1 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_topic_ids = sorted(topic["topic_id"] for topic in artifact["topic_map"])
    required_topic_ids = sorted(artifact["requirements"]["required_topic_ids"])
    if observed_topic_ids != required_topic_ids:
        raise ValueError(
            "Task 7 Story 1 topic map mismatch: expected {}, observed {}".format(
                required_topic_ids, observed_topic_ids
            )
        )

    if artifact["summary"]["entry_point_headings_complete"] != (
        len(artifact["entry_point"]["missing_headings"]) == 0
    ):
        raise ValueError(
            "Task 7 Story 1 entry_point_headings_complete summary is inconsistent"
        )

    if artifact["summary"]["entry_point_topics_complete"] != (
        len(artifact["entry_point"]["missing_topic_labels"]) == 0
    ):
        raise ValueError(
            "Task 7 Story 1 entry_point_topics_complete summary is inconsistent"
        )

    if artifact["summary"]["entry_point_primary_paths_complete"] != (
        len(artifact["entry_point"]["missing_primary_paths"]) == 0
    ):
        raise ValueError(
            "Task 7 Story 1 entry_point_primary_paths_complete summary is inconsistent"
        )

    if not artifact["entry_point"]["exists"]:
        raise ValueError("Task 7 Story 1 entry point must exist")

    if not artifact["summary"]["all_documents_exist"]:
        raise ValueError("Task 7 Story 1 requires all referenced documents to exist")

    if not artifact["summary"]["all_primary_references_exist"]:
        raise ValueError("Task 7 Story 1 requires all topic primary references to exist")

    if not artifact["summary"]["all_supporting_references_exist"]:
        raise ValueError(
            "Task 7 Story 1 requires all topic supporting references to exist"
        )

    if artifact["summary"]["contract_reference_map_completed"] != (
        artifact["status"] == "pass"
    ):
        raise ValueError(
            "Task 7 Story 1 contract_reference_map_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] topics={} entry_point_complete={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["topic_count"],
                artifact["summary"]["contract_reference_map_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 7 Story 1 JSON artifact bundle.",
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
            artifact["summary"]["topic_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
