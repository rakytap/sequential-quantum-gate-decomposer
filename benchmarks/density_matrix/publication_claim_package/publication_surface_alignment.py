#!/usr/bin/env python3
"""Validation: Task 8 Story 2 publication-surface alignment.

Builds a machine-readable checker for the Task 8 publication surfaces. This
layer is intentionally thin:
- it reuses the Story 1 claim package,
- it records the mandatory publication surfaces and their roles,
- it validates that each surface preserves the same claim boundary and closure
  rule,
- and it fails when one surface drifts from the others semantically.

Run with:
    python benchmarks/density_matrix/publication_claim_package/publication_surface_alignment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_claim_package.doc_utils import (
    CANONICAL_MAIN_CLAIM,
    CANONICAL_NON_CLAIMS,
    EVIDENCE_CLOSURE_RULE,
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    PUBLICATION_SURFACES,
    SURFACE_REQUIRED_HEADINGS,
    SURFACE_ROLE_PHRASES,
    PUBLICATION_CLAIM_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.publication_claim_package.claim_package_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)
from benchmarks.density_matrix.publication_claim_package.doc_utils import (
    PUBLICATION_CLAIM_OUTPUT_DIR as PUBLICATION_CLAIM_OUTPUT_DIR_FROM_UTILS,
    build_surface_presence,
    load_json,
)


SUITE_NAME = "publication_surface_alignment"
ARTIFACT_FILENAME = "publication_surface_alignment.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
STORY1_ARTIFACT_PATH = PUBLICATION_CLAIM_OUTPUT_DIR_FROM_UTILS / STORY1_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "claim_package",
    "surface_inventory",
    "software",
    "provenance",
    "summary",
)


def _load_story1_artifact(path: Path = STORY1_ARTIFACT_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story1_artifact(artifact)
        return artifact
    return run_story1_validation(verbose=False)


def build_surface_inventory():
    surfaces = []
    required_phrases = [
        CANONICAL_MAIN_CLAIM,
        EVIDENCE_CLOSURE_RULE,
        *CANONICAL_NON_CLAIMS,
    ]
    for surface_id in PUBLICATION_SURFACES:
        entry = build_surface_presence(
            surface_id,
            required_headings=SURFACE_REQUIRED_HEADINGS,
            required_phrases=required_phrases + [SURFACE_ROLE_PHRASES[surface_id]],
        )
        entry["role_phrase"] = SURFACE_ROLE_PHRASES[surface_id]
        entry["role_present"] = SURFACE_ROLE_PHRASES[surface_id] not in entry[
            "missing_required_phrases"
        ]
        entry["main_claim_present"] = CANONICAL_MAIN_CLAIM not in entry[
            "missing_required_phrases"
        ]
        entry["evidence_closure_rule_present"] = EVIDENCE_CLOSURE_RULE not in entry[
            "missing_required_phrases"
        ]
        entry["all_non_claims_present"] = all(
            phrase not in entry["missing_required_phrases"]
            for phrase in CANONICAL_NON_CLAIMS
        )
        entry["status"] = (
            "pass"
            if (
                not entry["missing_headings"]
                and not entry["missing_required_phrases"]
                and not entry["present_forbidden_phrases"]
            )
            else "fail"
        )
        surfaces.append(entry)
    return surfaces


def build_artifact_bundle():
    story1_artifact = _load_story1_artifact()
    surface_inventory = build_surface_inventory()
    all_surface_roles_present = all(entry["role_present"] for entry in surface_inventory)
    all_claim_headings_present = all(
        not entry["missing_headings"] for entry in surface_inventory
    )
    all_main_claims_present = all(
        entry["main_claim_present"] for entry in surface_inventory
    )
    all_non_claims_present = all(
        entry["all_non_claims_present"] for entry in surface_inventory
    )
    all_evidence_rules_present = all(
        entry["evidence_closure_rule_present"] for entry in surface_inventory
    )
    surface_alignment_completed = all(
        [
            story1_artifact["status"] == "pass",
            all_surface_roles_present,
            all_claim_headings_present,
            all_main_claims_present,
            all_non_claims_present,
            all_evidence_rules_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if surface_alignment_completed else "fail",
        "claim_package": {
            "suite_name": story1_artifact["suite_name"],
            "status": story1_artifact["status"],
            "path": relative_to_repo(STORY1_ARTIFACT_PATH),
            "summary": dict(story1_artifact["summary"]),
        },
        "surface_inventory": surface_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/"
                "publication_surface_alignment.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "claim_package_path": str(STORY1_ARTIFACT_PATH),
            "surface_paths": {
                surface_id: relative_to_repo(
                    MANDATORY_PUBLICATION_EVIDENCE_DOCS[PUBLICATION_SURFACES[surface_id]["doc_id"]]
                )
                for surface_id in PUBLICATION_SURFACES
            },
        },
        "summary": {
            "surface_count": len(surface_inventory),
            "all_surface_roles_present": all_surface_roles_present,
            "all_claim_headings_present": all_claim_headings_present,
            "all_main_claims_present": all_main_claims_present,
            "all_non_claims_present": all_non_claims_present,
            "all_evidence_rules_present": all_evidence_rules_present,
            "surface_alignment_completed": surface_alignment_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Task 8 Story 2 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_surface_ids = [entry["surface_id"] for entry in artifact["surface_inventory"]]
    required_surface_ids = list(PUBLICATION_SURFACES.keys())
    if observed_surface_ids != required_surface_ids:
        raise ValueError("publication surface inventory mismatch")

    all_surface_roles_present = all(
        entry.get("role_present", False) for entry in artifact["surface_inventory"]
    )
    if artifact["summary"]["all_surface_roles_present"] != all_surface_roles_present:
        raise ValueError("all_surface_roles_present summary is inconsistent")

    all_claim_headings_present = all(
        not entry["missing_headings"] for entry in artifact["surface_inventory"]
    )
    if artifact["summary"]["all_claim_headings_present"] != all_claim_headings_present:
        raise ValueError("all_claim_headings_present summary is inconsistent")

    all_main_claims_present = all(
        entry.get("main_claim_present", False)
        for entry in artifact["surface_inventory"]
    )
    if artifact["summary"]["all_main_claims_present"] != all_main_claims_present:
        raise ValueError("all_main_claims_present summary is inconsistent")

    all_non_claims_present = all(
        entry.get("all_non_claims_present", False)
        for entry in artifact["surface_inventory"]
    )
    if artifact["summary"]["all_non_claims_present"] != all_non_claims_present:
        raise ValueError("all_non_claims_present summary is inconsistent")

    all_evidence_rules_present = all(
        entry.get("evidence_closure_rule_present", False)
        for entry in artifact["surface_inventory"]
    )
    if artifact["summary"]["all_evidence_rules_present"] != all_evidence_rules_present:
        raise ValueError("all_evidence_rules_present summary is inconsistent")

    expected_completed = all(
        [
            artifact["claim_package"]["status"] == "pass",
            artifact["summary"]["all_surface_roles_present"],
            artifact["summary"]["all_claim_headings_present"],
            artifact["summary"]["all_main_claims_present"],
            artifact["summary"]["all_non_claims_present"],
            artifact["summary"]["all_evidence_rules_present"],
        ]
    )
    if artifact["summary"]["surface_alignment_completed"] != expected_completed:
        raise ValueError("surface_alignment_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Task 8 Story 2 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] surfaces={} aligned={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["surface_count"],
                artifact["summary"]["surface_alignment_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 8 Story 2 JSON artifact bundle.",
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
