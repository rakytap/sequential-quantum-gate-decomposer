#!/usr/bin/env python3
"""Validation: Task 7 Story 2 supported entry and canonical workflow wording.

Builds a machine-readable documentation checker for the supported Phase 2 entry
surface and the canonical workflow wording. This layer is intentionally thin:
- it reuses the Story 1 document-entry surface,
- it records the mandatory Story 2 statements and their source links,
- it fails when supported-entry or exact-regime wording becomes ambiguous,
- and it keeps the reader-facing contract narrow to the frozen Phase 2 path.

Run with:
    python benchmarks/density_matrix/task7_story2_supported_entry_reference_validation.py
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
    normalize_text,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.task7_story1_contract_reference_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)


SUITE_NAME = "task7_story2_supported_entry_reference"
ARTIFACT_FILENAME = "story2_supported_entry_reference.json"
DEFAULT_OUTPUT_DIR = TASK7_OUTPUT_DIR
STORY1_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_SECTION_HEADING = "## Supported Entry And Canonical Workflow"
MANDATORY_STATEMENTS = (
    {
        "statement_id": "backend_default",
        "label": "Backend default",
        "text": "`state_vector` remains the default backend when no explicit backend is selected.",
        "entry_point_phrases": [
            "`state_vector` remains the default backend",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "`state_vector` remains the default",
        ],
    },
    {
        "statement_id": "explicit_density_selection",
        "label": "Explicit density-matrix selection",
        "text": "`density_matrix` must be selected explicitly for exact noisy mixed-state claims.",
        "entry_point_phrases": [
            "`density_matrix` must be selected explicitly",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "`density_matrix` must be selected explicitly",
        ],
    },
    {
        "statement_id": "no_fallback",
        "label": "No fallback",
        "text": "No implicit `auto` mode or silent fallback is part of the Phase 2 contract.",
        "entry_point_phrases": [
            "No implicit `auto` mode or silent fallback is part of the Phase 2 contract.",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "No implicit `auto` mode or silent fallback is part of the Phase 2 contract.",
        ],
    },
    {
        "statement_id": "canonical_workflow",
        "label": "Canonical workflow",
        "text": "The canonical supported Phase 2 workflow is noisy VQE ground-state estimation for a 1D XXZ spin chain with local `Z` field using the default `HEA` ansatz, explicit local noise insertion, and exact `Re Tr(H*rho)` evaluation.",
        "entry_point_phrases": [
            "The canonical supported Phase 2 workflow is noisy VQE ground-state estimation",
            "local `Z` field",
            "default `HEA` ansatz",
            "exact `Re Tr(H*rho)` evaluation",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "The Phase 2 anchor workflow is noisy VQE ground-state estimation",
            "local `Z` field",
            "default `HEA` ansatz",
            "exact energy evaluation via `Re Tr(H*rho)`",
        ],
    },
    {
        "statement_id": "end_to_end_4q_6q",
        "label": "4q and 6q end-to-end scope",
        "text": "Full end-to-end workflow execution is required at 4 and 6 qubits.",
        "entry_point_phrases": [
            "Full end-to-end workflow execution is required at 4 and 6 qubits.",
        ],
        "primary_doc_id": "task6_mini_spec",
        "primary_source_phrases": [
            "End-to-end workflow execution is mandatory at 4 and 6 qubits",
        ],
    },
    {
        "statement_id": "benchmark_8q_10q",
        "label": "8q and 10q benchmark scope",
        "text": "Benchmark-ready fixed-parameter evaluation is required at 8 and 10 qubits.",
        "entry_point_phrases": [
            "Benchmark-ready fixed-parameter evaluation is required at 8 and 10 qubits.",
        ],
        "primary_doc_id": "task6_mini_spec",
        "primary_source_phrases": [
            "Benchmark-ready fixed-parameter evaluation is mandatory at 8 and 10 qubits",
        ],
    },
    {
        "statement_id": "anchor_10q",
        "label": "10-qubit anchor",
        "text": "The documented 10-qubit case is the acceptance anchor for the current exact regime.",
        "entry_point_phrases": [
            "The documented 10-qubit case is the acceptance anchor for the current exact regime.",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "10 qubits treated as the acceptance anchor for the exact regime",
        ],
    },
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "entry_point",
    "requirements",
    "story1_reference_map",
    "required_statements",
    "software",
    "provenance",
    "summary",
)


def _load_story1_artifact(path: Path = STORY1_ARTIFACT_PATH):
    if path.exists():
        from benchmarks.density_matrix.task7_doc_utils import load_json

        artifact = load_json(path)
        validate_story1_artifact(artifact)
        return artifact
    artifact = run_story1_validation(verbose=False)
    return artifact


def _phrases_present(text: str, phrases: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(normalize_text(phrase) in normalized_text for phrase in phrases)


def build_required_statement_inventory():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    inventory = []
    for statement in MANDATORY_STATEMENTS:
        primary_path = MANDATORY_PHASE2_DOCS[statement["primary_doc_id"]]
        primary_text = load_text(primary_path)
        inventory.append(
            {
                "statement_id": statement["statement_id"],
                "label": statement["label"],
                "text": statement["text"],
                "primary_doc_id": statement["primary_doc_id"],
                "primary_path": relative_to_repo(primary_path),
                "entry_point_phrases": list(statement["entry_point_phrases"]),
                "primary_source_phrases": list(statement["primary_source_phrases"]),
                "entry_point_present": _phrases_present(
                    entry_text, statement["entry_point_phrases"]
                ),
                "primary_source_present": _phrases_present(
                    primary_text, statement["primary_source_phrases"]
                ),
            }
        )
    return inventory


def build_artifact_bundle():
    story1_artifact = _load_story1_artifact()
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    required_statements = build_required_statement_inventory()

    section_heading_present = STORY2_SECTION_HEADING in entry_text
    all_entry_point_statements_present = all(
        item["entry_point_present"] for item in required_statements
    )
    all_primary_sources_present = all(
        item["primary_source_present"] for item in required_statements
    )
    supported_entry_reference_completed = bool(
        story1_artifact["status"] == "pass"
        and section_heading_present
        and all_entry_point_statements_present
        and all_primary_sources_present
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if supported_entry_reference_completed else "fail",
        "entry_point": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "section_heading": STORY2_SECTION_HEADING,
            "section_heading_present": section_heading_present,
        },
        "requirements": {
            "required_statement_ids": [
                statement["statement_id"] for statement in MANDATORY_STATEMENTS
            ],
        },
        "story1_reference_map": {
            "suite_name": story1_artifact["suite_name"],
            "status": story1_artifact["status"],
            "entry_point_path": story1_artifact["entry_point"]["path"],
            "summary": story1_artifact["summary"],
        },
        "required_statements": required_statements,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task7_story2_supported_entry_reference_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_reference_map_path": str(STORY1_ARTIFACT_PATH),
            "entry_point_path": str(PHASE2_DOCUMENTATION_INDEX_PATH),
        },
        "summary": {
            "required_statement_count": len(required_statements),
            "section_heading_present": section_heading_present,
            "all_entry_point_statements_present": all_entry_point_statements_present,
            "all_primary_sources_present": all_primary_sources_present,
            "supported_entry_reference_completed": supported_entry_reference_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 7 Story 2 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_statement_ids = [item["statement_id"] for item in artifact["required_statements"]]
    required_statement_ids = artifact["requirements"]["required_statement_ids"]
    if observed_statement_ids != required_statement_ids:
        raise ValueError(
            "Task 7 Story 2 required statements mismatch: expected {}, observed {}".format(
                required_statement_ids, observed_statement_ids
            )
        )

    if artifact["summary"]["section_heading_present"] != artifact["entry_point"][
        "section_heading_present"
    ]:
        raise ValueError(
            "Task 7 Story 2 section_heading_present summary is inconsistent"
        )

    if artifact["summary"]["all_entry_point_statements_present"] != all(
        item["entry_point_present"] for item in artifact["required_statements"]
    ):
        raise ValueError(
            "Task 7 Story 2 all_entry_point_statements_present summary is inconsistent"
        )

    if artifact["summary"]["all_primary_sources_present"] != all(
        item["primary_source_present"] for item in artifact["required_statements"]
    ):
        raise ValueError(
            "Task 7 Story 2 all_primary_sources_present summary is inconsistent"
        )

    if artifact["summary"]["supported_entry_reference_completed"] != (
        artifact["status"] == "pass"
    ):
        raise ValueError(
            "Task 7 Story 2 supported_entry_reference_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] statements={} entry_point_complete={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["required_statement_count"],
                artifact["summary"]["supported_entry_reference_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 7 Story 2 JSON artifact bundle.",
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
            artifact["summary"]["required_statement_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
