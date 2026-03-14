#!/usr/bin/env python3
"""Validation: Task 7 Story 3 support-surface reference layer.

Builds a machine-readable checker for the guaranteed VQE-facing support surface.
This layer is intentionally thin:
- it reuses the Story 1 entry point and Story 2 wording surface,
- it records the required / optional / deferred / unsupported support boundary,
- it preserves a few explicit boundary examples,
- and it fails when broader capability is allowed to masquerade as guaranteed
  Phase 2 workflow support.

Run with:
    python benchmarks/density_matrix/task7_story3_support_surface_reference_validation.py
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
from benchmarks.density_matrix.task7_story2_supported_entry_reference_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    validate_artifact_bundle as validate_story2_artifact,
)


SUITE_NAME = "task7_story3_support_surface_reference"
ARTIFACT_FILENAME = "story3_support_surface_reference.json"
DEFAULT_OUTPUT_DIR = TASK7_OUTPUT_DIR
STORY2_ARTIFACT_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
STORY3_SECTION_HEADING = "## Guaranteed VQE-Facing Support Surface"
MANDATORY_SUPPORT_ITEMS = (
    {
        "support_id": "generated_hea_only",
        "label": "Generated HEA only",
        "entry_point_phrases": [
            "The guaranteed Phase 2 VQE-facing density path is generated `HEA` only.",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "The current guaranteed density path is generated `HEA` only.",
        ],
    },
    {
        "support_id": "required_gate_families",
        "label": "Required gate families",
        "entry_point_phrases": [
            "Required gate families: `U3`, `CNOT`",
        ],
        "primary_doc_id": "task4_mini_spec",
        "primary_source_phrases": [
            "The guaranteed mandatory gate surface remains `U3` and `CNOT`",
        ],
    },
    {
        "support_id": "required_local_noise_models",
        "label": "Required local noise models",
        "entry_point_phrases": [
            "Required local noise models: local depolarizing, local phase damping or dephasing, and local amplitude damping",
        ],
        "primary_doc_id": "task4_mini_spec",
        "primary_source_phrases": [
            "The mandatory Phase 2 local-noise baseline is:",
            "`local single-qubit depolarizing`, `local amplitude damping`, and",
            "`local phase damping / dephasing`",
        ],
    },
    {
        "support_id": "whole_register_optional",
        "label": "Whole-register depolarizing optional",
        "entry_point_phrases": [
            "Whole-register depolarizing remains optional and does not count as required local-noise support",
        ],
        "primary_doc_id": "task4_mini_spec",
        "primary_source_phrases": [
            "whole-register depolarizing is optional only as a regression or",
            "stress-test baseline",
        ],
    },
    {
        "support_id": "optional_extensions",
        "label": "Optional extensions remain non-mandatory",
        "entry_point_phrases": [
            "Generalized amplitude damping and coherent over-rotation remain optional benchmark extensions rather than delivered required support",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "generalized amplitude damping and coherent over-rotation remain",
            "undelivered extension ideas rather than part of the current publication claim",
        ],
    },
    {
        "support_id": "deferred_broader_capability",
        "label": "Deferred broader capability",
        "entry_point_phrases": [
            "Correlated multi-qubit noise, readout noise, calibration-aware models, non-Markovian noise, and broader manual circuit parity remain deferred",
        ],
        "primary_doc_id": "phase2_detailed_planning",
        "primary_source_phrases": [
            "correlated multi-qubit noise, readout noise as a density-backend feature,",
            "calibration-aware models, non-Markovian noise",
            "full `qgd_Circuit` parity",
        ],
    },
    {
        "support_id": "no_full_qgd_circuit_parity",
        "label": "No full qgd_Circuit parity",
        "entry_point_phrases": [
            "Full `qgd_Circuit` parity is not part of the Phase 2 guarantee",
        ],
        "primary_doc_id": "task3_mini_spec",
        "primary_source_phrases": [
            "full `qgd_Circuit` parity is not assumed",
        ],
    },
    {
        "support_id": "standalone_not_guaranteed",
        "label": "Standalone breadth is not workflow guarantee",
        "entry_point_phrases": [
            "Broader standalone `NoisyCircuit` capability does not automatically imply guaranteed VQE-facing support.",
        ],
        "primary_doc_id": "task4_mini_spec",
        "primary_source_phrases": [
            "Using standalone `NoisyCircuit` breadth to overstate the guaranteed VQE-facing",
            "Phase 2 support surface",
        ],
    },
)
MANDATORY_BOUNDARY_EXAMPLES = (
    "Full `qgd_Circuit` parity is not part of the Phase 2 guarantee.",
    "Broader standalone `NoisyCircuit` capability does not automatically imply guaranteed VQE-facing support.",
    "Whole-register depolarizing remains optional rather than required.",
)
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "entry_point",
    "requirements",
    "story2_supported_entry",
    "support_surface_inventory",
    "boundary_examples",
    "software",
    "provenance",
    "summary",
)


def _load_story2_artifact(path: Path = STORY2_ARTIFACT_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story2_artifact(artifact)
        return artifact
    artifact = run_story2_validation(verbose=False)
    return artifact


def _phrases_present(text: str, phrases: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(normalize_text(phrase) in normalized_text for phrase in phrases)


def build_support_surface_inventory():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    inventory = []
    for item in MANDATORY_SUPPORT_ITEMS:
        primary_path = MANDATORY_PHASE2_DOCS[item["primary_doc_id"]]
        primary_text = load_text(primary_path)
        inventory.append(
            {
                "support_id": item["support_id"],
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


def build_boundary_examples():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    examples = []
    for example in MANDATORY_BOUNDARY_EXAMPLES:
        examples.append(
            {
                "text": example,
                "entry_point_present": normalize_text(example) in normalize_text(entry_text),
            }
        )
    return examples


def build_artifact_bundle():
    story2_artifact = _load_story2_artifact()
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    support_surface_inventory = build_support_surface_inventory()
    boundary_examples = build_boundary_examples()

    section_heading_present = STORY3_SECTION_HEADING in entry_text
    all_support_items_present = all(
        item["entry_point_present"] and item["primary_source_present"]
        for item in support_surface_inventory
    )
    all_boundary_examples_present = all(
        example["entry_point_present"] for example in boundary_examples
    )
    support_surface_reference_completed = bool(
        story2_artifact["status"] == "pass"
        and section_heading_present
        and all_support_items_present
        and all_boundary_examples_present
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if support_surface_reference_completed else "fail",
        "entry_point": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "section_heading": STORY3_SECTION_HEADING,
            "section_heading_present": section_heading_present,
        },
        "requirements": {
            "required_support_ids": [
                item["support_id"] for item in MANDATORY_SUPPORT_ITEMS
            ],
            "required_boundary_examples": list(MANDATORY_BOUNDARY_EXAMPLES),
        },
        "story2_supported_entry": {
            "suite_name": story2_artifact["suite_name"],
            "status": story2_artifact["status"],
            "summary": story2_artifact["summary"],
        },
        "support_surface_inventory": support_surface_inventory,
        "boundary_examples": boundary_examples,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "task7_story3_support_surface_reference_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story2_supported_entry_path": str(STORY2_ARTIFACT_PATH),
            "entry_point_path": str(PHASE2_DOCUMENTATION_INDEX_PATH),
        },
        "summary": {
            "required_support_count": len(support_surface_inventory),
            "boundary_example_count": len(boundary_examples),
            "section_heading_present": section_heading_present,
            "all_support_items_present": all_support_items_present,
            "all_boundary_examples_present": all_boundary_examples_present,
            "support_surface_reference_completed": support_surface_reference_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 7 Story 3 artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_support_ids = [item["support_id"] for item in artifact["support_surface_inventory"]]
    required_support_ids = artifact["requirements"]["required_support_ids"]
    if observed_support_ids != required_support_ids:
        raise ValueError(
            "Task 7 Story 3 support inventory mismatch: expected {}, observed {}".format(
                required_support_ids, observed_support_ids
            )
        )

    if artifact["summary"]["section_heading_present"] != artifact["entry_point"][
        "section_heading_present"
    ]:
        raise ValueError(
            "Task 7 Story 3 section_heading_present summary is inconsistent"
        )

    if artifact["summary"]["all_support_items_present"] != all(
        item["entry_point_present"] and item["primary_source_present"]
        for item in artifact["support_surface_inventory"]
    ):
        raise ValueError(
            "Task 7 Story 3 all_support_items_present summary is inconsistent"
        )

    if artifact["summary"]["all_boundary_examples_present"] != all(
        example["entry_point_present"] for example in artifact["boundary_examples"]
    ):
        raise ValueError(
            "Task 7 Story 3 all_boundary_examples_present summary is inconsistent"
        )

    if artifact["summary"]["support_surface_reference_completed"] != (
        artifact["status"] == "pass"
    ):
        raise ValueError(
            "Task 7 Story 3 support_surface_reference_completed summary is inconsistent"
        )


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] support_items={} complete={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["required_support_count"],
                artifact["summary"]["support_surface_reference_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 7 Story 3 JSON artifact bundle.",
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
            artifact["summary"]["required_support_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
