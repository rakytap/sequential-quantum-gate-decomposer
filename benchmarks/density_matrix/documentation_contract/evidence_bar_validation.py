#!/usr/bin/env python3
"""Validation: Task 7 Story 4 evidence-bar reference layer.

Builds a machine-readable checker for the mandatory Phase 2 evidence bar. This
layer is intentionally thin:
- it reuses the Story 3 support-surface clarification,
- it records the mandatory evidence package and its thresholds,
- it makes exclusion semantics explicit for optional, unsupported, and
  incomplete evidence,
- and it fails when the core claim can be overread from favorable subsets.

Run with:
    python benchmarks/density_matrix/documentation_contract/evidence_bar_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.documentation_contract.doc_utils import (
    MANDATORY_PHASE2_DOCS,
    PHASE2_DOCUMENTATION_INDEX_PATH,
    DOCUMENTATION_CONTRACT_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    normalize_text,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.documentation_contract.support_surface_reference_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)


SUITE_NAME = "evidence_bar_reference"
ARTIFACT_FILENAME = "evidence_bar_reference.json"
DEFAULT_OUTPUT_DIR = DOCUMENTATION_CONTRACT_OUTPUT_DIR
STORY3_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / STORY3_ARTIFACT_FILENAME
STORY4_SECTION_HEADING = "## Mandatory Evidence Bar"
MANDATORY_EVIDENCE_ITEMS = (
    {
        "evidence_id": "micro_validation_matrix",
        "label": "Micro-validation matrix",
        "entry_point_phrases": ["mandatory 1 to 3 qubit micro-validation matrix"],
        "primary_doc_id": "planner_calibration_mini_spec",
        "primary_source_phrases": [
            "The mandatory micro-validation layer covers 1 to 3 qubit circuits",
        ],
    },
    {
        "evidence_id": "workflow_matrix",
        "label": "Workflow matrix",
        "entry_point_phrases": [
            "mandatory 4 / 6 / 8 / 10 qubit fixed-parameter workflow matrix with 10",
            "parameter vectors per required size",
        ],
        "primary_doc_id": "planner_calibration_mini_spec",
        "primary_source_phrases": [
            "mandatory 4 / 6 / 8 / 10",
            "at least 10 fixed",
            "parameter vectors per mandatory workflow size",
        ],
    },
    {
        "evidence_id": "optimization_trace",
        "label": "Optimization trace",
        "entry_point_phrases": [
            "at least one reproducible 4- or 6-qubit optimization trace",
        ],
        "primary_doc_id": "correctness_evidence_mini_spec",
        "primary_source_phrases": [
            "At least one reproducible 4- or 6-qubit optimization trace",
        ],
    },
    {
        "evidence_id": "anchor_case",
        "label": "10-qubit anchor case",
        "entry_point_phrases": ["one documented 10-qubit anchor case"],
        "primary_doc_id": "correctness_evidence_mini_spec",
        "primary_source_phrases": [
            "At least one documented 10-qubit anchor evaluation case",
        ],
    },
    {
        "evidence_id": "runtime_peak_memory",
        "label": "Runtime and peak memory",
        "entry_point_phrases": [
            "runtime and peak-memory recording for mandatory workflow evidence",
        ],
        "primary_doc_id": "planner_calibration_mini_spec",
        "primary_source_phrases": [
            "The validation package must record workflow completion, runtime, and peak",
            "memory",
        ],
    },
    {
        "evidence_id": "reproducibility_bundle",
        "label": "Reproducibility bundle",
        "entry_point_phrases": [
            "the backend-explicit reproducibility bundle rooted in",
            "workflow_publication_bundle.json",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "complete workflow-facing publication bundle is archived at",
            "workflow_publication_bundle.json",
        ],
    },
    {
        "evidence_id": "micro_threshold",
        "label": "Micro-validation threshold",
        "entry_point_phrases": ["mandatory micro-validation accuracy: `<= 1e-10`"],
        "primary_doc_id": "planner_calibration_mini_spec",
        "primary_source_phrases": [
            "`<= 1e-10` maximum absolute energy error on the mandatory 1 to 3 qubit",
        ],
    },
    {
        "evidence_id": "workflow_threshold",
        "label": "Workflow threshold",
        "entry_point_phrases": ["mandatory workflow-matrix accuracy: `<= 1e-8`"],
        "primary_doc_id": "planner_calibration_mini_spec",
        "primary_source_phrases": [
            "`<= 1e-8` maximum absolute energy error on the mandatory 4 / 6 / 8 / 10",
        ],
    },
    {
        "evidence_id": "mandatory_pass_rate",
        "label": "Mandatory pass rate",
        "entry_point_phrases": ["`100%` pass rate on the mandatory evidence package"],
        "primary_doc_id": "planner_calibration_mini_spec",
        "primary_source_phrases": [
            "`100%` pass rate on the mandatory",
        ],
    },
    {
        "evidence_id": "main_claim_rule",
        "label": "Main claim closure rule",
        "entry_point_phrases": [
            "only mandatory, complete, supported evidence closes the core Phase 2 claim",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "only mandatory, complete, supported evidence may close the main Phase 2",
        ],
    },
    {
        "evidence_id": "optional_supplemental",
        "label": "Optional evidence stays supplemental",
        "entry_point_phrases": [
            "optional whole-register depolarizing remains supplemental",
        ],
        "primary_doc_id": "planning_publications",
        "primary_source_phrases": [
            "optional whole-register depolarizing remains supplemental",
        ],
    },
    {
        "evidence_id": "unsupported_excluded",
        "label": "Unsupported and incomplete evidence excluded",
        "entry_point_phrases": [
            "unsupported, deferred, or incomplete evidence remains excluded from the core claim",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "optional evidence remains supplemental and deferred or",
            "unsupported evidence remains boundary-only",
        ],
    },
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "entry_point",
    "requirements",
    "story3_support_surface",
    "evidence_inventory",
    "software",
    "provenance",
    "summary",
)


def _load_story3_artifact(path: Path = STORY3_ARTIFACT_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story3_artifact(artifact)
        return artifact
    artifact = run_story3_validation(verbose=False)
    return artifact


def _phrases_present(text: str, phrases: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(normalize_text(phrase) in normalized_text for phrase in phrases)


def build_evidence_inventory():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    inventory = []
    for item in MANDATORY_EVIDENCE_ITEMS:
        primary_path = MANDATORY_PHASE2_DOCS[item["primary_doc_id"]]
        primary_text = load_text(primary_path)
        inventory.append(
            {
                "evidence_id": item["evidence_id"],
                "label": item["label"],
                "primary_doc_id": item["primary_doc_id"],
                "primary_path": relative_to_repo(primary_path),
                "entry_point_phrases": list(item["entry_point_phrases"]),
                "primary_source_phrases": list(item["primary_source_phrases"]),
                "entry_point_present": _phrases_present(
                    entry_text, item["entry_point_phrases"]
                ),
                "primary_source_present": _phrases_present(
                    primary_text, item["primary_source_phrases"]
                ),
            }
        )
    return inventory


def build_artifact_bundle():
    story3_artifact = _load_story3_artifact()
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    evidence_inventory = build_evidence_inventory()

    section_heading_present = STORY4_SECTION_HEADING in entry_text
    all_evidence_items_present = all(
        item["entry_point_present"] and item["primary_source_present"]
        for item in evidence_inventory
    )
    evidence_bar_reference_completed = bool(
        story3_artifact["status"] == "pass"
        and section_heading_present
        and all_evidence_items_present
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if evidence_bar_reference_completed else "fail",
        "entry_point": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "section_heading": STORY4_SECTION_HEADING,
            "section_heading_present": section_heading_present,
        },
        "requirements": {
            "required_evidence_ids": [
                item["evidence_id"] for item in MANDATORY_EVIDENCE_ITEMS
            ],
        },
        "story3_support_surface": {
            "suite_name": story3_artifact["suite_name"],
            "status": story3_artifact["status"],
            "summary": story3_artifact["summary"],
        },
        "evidence_inventory": evidence_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "evidence_bar_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story3_support_surface_path": str(STORY3_ARTIFACT_PATH),
            "entry_point_path": str(PHASE2_DOCUMENTATION_INDEX_PATH),
        },
        "summary": {
            "required_evidence_count": len(evidence_inventory),
            "section_heading_present": section_heading_present,
            "all_evidence_items_present": all_evidence_items_present,
            "evidence_bar_reference_completed": evidence_bar_reference_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 7 Story 4 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_ids = [item["evidence_id"] for item in artifact["evidence_inventory"]]
    required_ids = artifact["requirements"]["required_evidence_ids"]
    if observed_ids != required_ids:
        raise ValueError(
            "Task 7 Story 4 evidence inventory mismatch: expected {}, observed {}".format(
                required_ids, observed_ids
            )
        )

    if artifact["summary"]["section_heading_present"] != artifact["entry_point"][
        "section_heading_present"
    ]:
        raise ValueError(
            "Task 7 Story 4 section_heading_present summary is inconsistent"
        )

    if artifact["summary"]["all_evidence_items_present"] != all(
        item["entry_point_present"] and item["primary_source_present"]
        for item in artifact["evidence_inventory"]
    ):
        raise ValueError(
            "Task 7 Story 4 all_evidence_items_present summary is inconsistent"
        )

    if artifact["summary"]["evidence_bar_reference_completed"] != (
        artifact["status"] == "pass"
    ):
        raise ValueError(
            "Task 7 Story 4 evidence_bar_reference_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] evidence_items={} complete={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["required_evidence_count"],
                artifact["summary"]["evidence_bar_reference_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 7 Story 4 JSON artifact bundle.",
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
