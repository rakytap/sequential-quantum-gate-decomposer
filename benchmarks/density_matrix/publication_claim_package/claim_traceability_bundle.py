#!/usr/bin/env python3
"""Validation: Task 8 Story 3 claim-to-source traceability.

Builds a machine-readable traceability bundle for Paper 1. This layer is
intentionally thin:
- it reuses the Story 1 claim package and Story 2 surface-alignment outputs,
- it maps major Paper 1 claims and section classes to authoritative sources,
- it preserves reviewer navigation through the Phase 2 documentation index and
  the Task 6 publication bundle,
- and it fails when major publication items lose a stable source path.

Run with:
    python benchmarks/density_matrix/publication_claim_package/claim_traceability_bundle.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_claim_package.doc_utils import (
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    PHASE2_DOCUMENTATION_INDEX_PATH,
    CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH,
    PUBLICATION_CLAIM_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.publication_claim_package.claim_package_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)
from benchmarks.density_matrix.publication_claim_package.publication_surface_alignment import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    validate_artifact_bundle as validate_story2_artifact,
)


SUITE_NAME = "claim_traceability_bundle"
ARTIFACT_FILENAME = "claim_traceability_bundle.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
STORY1_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "upstream_story_artifacts",
    "reviewer_entry_paths",
    "claim_traceability",
    "section_traceability",
    "software",
    "provenance",
    "summary",
)
CLAIM_TRACEABILITY_ITEMS = (
    {
        "claim_id": "paper1_main_claim",
        "label": "Paper 1 main claim",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["planning_publications", "phase2_paper"],
    },
    {
        "claim_id": "supported_workflow_scope",
        "label": "Supported workflow and scope claim",
        "surface_ids": ["short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "phase2_documentation_index",
        "supporting_doc_ids": ["correctness_evidence_mini_spec", "correctness_evidence_publication_bundle"],
    },
    {
        "claim_id": "evidence_closure_rule",
        "label": "Mandatory evidence closure rule",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["planner_calibration_mini_spec", "correctness_evidence_publication_bundle"],
    },
    {
        "claim_id": "limitations_and_non_claims",
        "label": "Limitations and explicit non-claims",
        "surface_ids": ["short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["research_alignment", "phase2_paper"],
    },
    {
        "claim_id": "future_work_handoff",
        "label": "Future-work handoff",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "planning_publications",
        "supporting_doc_ids": ["research_alignment", "changelog", "phase2_paper"],
    },
)
SECTION_TRACEABILITY_ITEMS = (
    {
        "section_id": "paper_abstract_summary",
        "label": "Full-paper abstract summary",
        "doc_id": "phase2_paper",
        "required_heading": "## Abstract Summary",
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["planning_publications", "publication_evidence_mini_spec"],
    },
    {
        "section_id": "paper_validation_methodology",
        "label": "Full-paper validation methodology section",
        "doc_id": "phase2_paper",
        "required_heading": "## 7. Validation Methodology",
        "primary_doc_id": "correctness_evidence_publication_bundle",
        "supporting_doc_ids": ["planner_calibration_mini_spec", "correctness_evidence_mini_spec"],
    },
    {
        "section_id": "paper_scope_boundaries",
        "label": "Full-paper scope boundaries section",
        "doc_id": "phase2_paper",
        "required_heading": "## 6. Scope Boundaries",
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["performance_evidence_documentation_bundle", "research_alignment"],
    },
    {
        "section_id": "short_paper_validation_baseline",
        "label": "Compact short-paper validation baseline section",
        "doc_id": "phase2_short_paper",
        "required_heading": "## 6. Validation and Benchmark Baseline (Delivered)",
        "primary_doc_id": "correctness_evidence_publication_bundle",
        "supporting_doc_ids": ["planner_calibration_mini_spec", "correctness_evidence_mini_spec"],
    },
    {
        "section_id": "short_paper_follow_on_phases",
        "label": "Compact short-paper follow-on phases section",
        "doc_id": "phase2_short_paper",
        "required_heading": "## 9. Follow-On Phases",
        "primary_doc_id": "planning_publications",
        "supporting_doc_ids": ["research_alignment", "changelog"],
    },
)


def _load_story1(path: Path = STORY1_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story1_artifact(artifact)
        return artifact
    return run_story1_validation(verbose=False)


def _load_story2(path: Path = STORY2_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story2_artifact(artifact)
        return artifact
    return run_story2_validation(verbose=False)


def build_claim_traceability():
    entries = []
    for item in CLAIM_TRACEABILITY_ITEMS:
        primary_path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[item["primary_doc_id"]]
        supporting_paths = [
            MANDATORY_PUBLICATION_EVIDENCE_DOCS[doc_id] for doc_id in item["supporting_doc_ids"]
        ]
        entries.append(
            {
                "claim_id": item["claim_id"],
                "label": item["label"],
                "surface_ids": list(item["surface_ids"]),
                "primary_doc_id": item["primary_doc_id"],
                "primary_path": relative_to_repo(primary_path),
                "primary_exists": primary_path.exists(),
                "supporting_doc_ids": list(item["supporting_doc_ids"]),
                "supporting_paths": [relative_to_repo(path) for path in supporting_paths],
                "supporting_exists": [path.exists() for path in supporting_paths],
            }
        )
    return entries


def build_section_traceability():
    entries = []
    for item in SECTION_TRACEABILITY_ITEMS:
        section_path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[item["doc_id"]]
        section_text = load_text(section_path)
        primary_path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[item["primary_doc_id"]]
        supporting_paths = [
            MANDATORY_PUBLICATION_EVIDENCE_DOCS[doc_id] for doc_id in item["supporting_doc_ids"]
        ]
        entries.append(
            {
                "section_id": item["section_id"],
                "label": item["label"],
                "doc_id": item["doc_id"],
                "section_path": relative_to_repo(section_path),
                "required_heading": item["required_heading"],
                "heading_present": item["required_heading"] in section_text,
                "primary_doc_id": item["primary_doc_id"],
                "primary_path": relative_to_repo(primary_path),
                "primary_exists": primary_path.exists(),
                "supporting_doc_ids": list(item["supporting_doc_ids"]),
                "supporting_paths": [relative_to_repo(path) for path in supporting_paths],
                "supporting_exists": [path.exists() for path in supporting_paths],
            }
        )
    return entries


def build_artifact_bundle():
    story1_artifact = _load_story1()
    story2_artifact = _load_story2()
    claim_traceability = build_claim_traceability()
    section_traceability = build_section_traceability()

    all_claim_sources_exist = all(
        entry["primary_exists"] and all(entry["supporting_exists"])
        for entry in claim_traceability
    )
    all_section_sources_exist = all(
        entry["primary_exists"] and all(entry["supporting_exists"])
        for entry in section_traceability
    )
    all_section_headings_present = all(
        entry["heading_present"] for entry in section_traceability
    )
    reviewer_entry_paths_complete = (
        PHASE2_DOCUMENTATION_INDEX_PATH.exists() and CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH.exists()
    )
    claim_traceability_completed = all(
        [
            story1_artifact["status"] == "pass",
            story2_artifact["status"] == "pass",
            all_claim_sources_exist,
            all_section_sources_exist,
            all_section_headings_present,
            reviewer_entry_paths_complete,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if claim_traceability_completed else "fail",
        "upstream_story_artifacts": [
            {
                "artifact_id": story1_artifact["suite_name"],
                "path": relative_to_repo(STORY1_PATH),
                "status": story1_artifact["status"],
                "summary": dict(story1_artifact["summary"]),
            },
            {
                "artifact_id": story2_artifact["suite_name"],
                "path": relative_to_repo(STORY2_PATH),
                "status": story2_artifact["status"],
                "summary": dict(story2_artifact["summary"]),
            },
        ],
        "reviewer_entry_paths": {
            "phase2_documentation_index": {
                "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
                "exists": PHASE2_DOCUMENTATION_INDEX_PATH.exists(),
            },
            "correctness_evidence_publication_bundle": {
                "path": relative_to_repo(CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH),
                "exists": CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH.exists(),
            },
        },
        "claim_traceability": claim_traceability,
        "section_traceability": section_traceability,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "claim_traceability_bundle.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_path": str(STORY1_PATH),
            "story2_path": str(STORY2_PATH),
            "phase2_documentation_index_path": str(PHASE2_DOCUMENTATION_INDEX_PATH),
            "correctness_evidence_publication_bundle_path": str(CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH),
        },
        "summary": {
            "claim_traceability_count": len(claim_traceability),
            "section_traceability_count": len(section_traceability),
            "all_claim_sources_exist": all_claim_sources_exist,
            "all_section_sources_exist": all_section_sources_exist,
            "all_section_headings_present": all_section_headings_present,
            "reviewer_entry_paths_complete": reviewer_entry_paths_complete,
            "claim_traceability_completed": claim_traceability_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 8 Story 3 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_claim_ids = [entry["claim_id"] for entry in artifact["claim_traceability"]]
    required_claim_ids = [entry["claim_id"] for entry in CLAIM_TRACEABILITY_ITEMS]
    if observed_claim_ids != required_claim_ids:
        raise ValueError("claim traceability inventory mismatch")

    observed_section_ids = [entry["section_id"] for entry in artifact["section_traceability"]]
    required_section_ids = [entry["section_id"] for entry in SECTION_TRACEABILITY_ITEMS]
    if observed_section_ids != required_section_ids:
        raise ValueError("section traceability inventory mismatch")

    all_claim_sources_exist = all(
        entry["primary_exists"] and all(entry["supporting_exists"])
        for entry in artifact["claim_traceability"]
    )
    if artifact["summary"]["all_claim_sources_exist"] != all_claim_sources_exist:
        raise ValueError("all_claim_sources_exist summary is inconsistent")

    all_section_sources_exist = all(
        entry["primary_exists"] and all(entry["supporting_exists"])
        for entry in artifact["section_traceability"]
    )
    if artifact["summary"]["all_section_sources_exist"] != all_section_sources_exist:
        raise ValueError("all_section_sources_exist summary is inconsistent")

    all_section_headings_present = all(
        entry["heading_present"] for entry in artifact["section_traceability"]
    )
    if artifact["summary"]["all_section_headings_present"] != all_section_headings_present:
        raise ValueError("all_section_headings_present summary is inconsistent")

    reviewer_entry_paths_complete = all(
        entry["exists"] for entry in artifact["reviewer_entry_paths"].values()
    )
    if artifact["summary"]["reviewer_entry_paths_complete"] != reviewer_entry_paths_complete:
        raise ValueError("reviewer_entry_paths_complete summary is inconsistent")

    expected_completed = all(
        [
            all(upstream["status"] == "pass" for upstream in artifact["upstream_story_artifacts"]),
            artifact["summary"]["all_claim_sources_exist"],
            artifact["summary"]["all_section_sources_exist"],
            artifact["summary"]["all_section_headings_present"],
            artifact["summary"]["reviewer_entry_paths_complete"],
        ]
    )
    if artifact["summary"]["claim_traceability_completed"] != expected_completed:
        raise ValueError("claim_traceability_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Task 8 Story 3 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] claims={} sections={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["claim_traceability_count"],
                artifact["summary"]["section_traceability_count"],
                artifact["summary"]["claim_traceability_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 8 Story 3 JSON artifact bundle.",
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
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            artifact["status"],
            artifact["summary"]["claim_traceability_count"],
            artifact["summary"]["section_traceability_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
