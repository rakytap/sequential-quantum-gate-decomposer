#!/usr/bin/env python3
"""Validation: Task 8 Story 7 publication-package bundle.

Builds the top-level Task 8 bundle by assembling Story 1 to Story 6 outputs into
one machine-checkable publication surface. The bundle preserves the shared
reviewer-entry path, validates the lower-story semantic gates, records mandatory
file coverage, and checks the canonical Task 8 publication terminology
inventory.

Run with:
    python benchmarks/density_matrix/publication_claim_package/publication_claim_bundle.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_claim_package.doc_utils import (
    EVIDENCE_CLOSURE_RULE,
    EXACT_REGIME_BOUNDARY,
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    PHASE2_DOCUMENTATION_INDEX_PATH,
    PHASE_POSITIONING_RULE,
    PUBLICATION_SURFACES,
    SUPPORTED_PATH_BOUNDARY,
    CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH,
    PUBLICATION_CLAIM_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    normalize_text,
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
from benchmarks.density_matrix.publication_claim_package.claim_traceability_bundle import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)
from benchmarks.density_matrix.publication_claim_package.evidence_closure_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)
from benchmarks.density_matrix.publication_claim_package.supported_path_scope_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    run_validation as run_story5_validation,
    validate_artifact_bundle as validate_story5_artifact,
)
from benchmarks.density_matrix.publication_claim_package.future_work_boundary_validation import (
    ARTIFACT_FILENAME as STORY6_ARTIFACT_FILENAME,
    run_validation as run_story6_validation,
    validate_artifact_bundle as validate_story6_artifact,
)


SUITE_NAME = "publication_claim_bundle"
ARTIFACT_FILENAME = "publication_claim_bundle.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
STORY1_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
STORY3_PATH = DEFAULT_OUTPUT_DIR / STORY3_ARTIFACT_FILENAME
STORY4_PATH = DEFAULT_OUTPUT_DIR / STORY4_ARTIFACT_FILENAME
STORY5_PATH = DEFAULT_OUTPUT_DIR / STORY5_ARTIFACT_FILENAME
STORY6_PATH = DEFAULT_OUTPUT_DIR / STORY6_ARTIFACT_FILENAME
STORY_ARTIFACT_REQUIREMENTS = (
    {
        "artifact_id": "claim_package",
        "path": STORY1_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "claim_package_completed",
    },
    {
        "artifact_id": "publication_surface_alignment",
        "path": STORY2_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "surface_alignment_completed",
    },
    {
        "artifact_id": "claim_traceability_bundle",
        "path": STORY3_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "claim_traceability_completed",
    },
    {
        "artifact_id": "evidence_closure_bundle",
        "path": STORY4_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "evidence_closure_completed",
    },
    {
        "artifact_id": "supported_path_scope_bundle",
        "path": STORY5_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "supported_path_scope_completed",
    },
    {
        "artifact_id": "future_work_boundary_bundle",
        "path": STORY6_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "future_work_boundary_completed",
    },
)
REQUIRED_GLOSSARY_TERMS = (
    "`density_matrix`",
    "canonical noisy XXZ VQE workflow",
    EVIDENCE_CLOSURE_RULE,
    SUPPORTED_PATH_BOUNDARY,
    EXACT_REGIME_BOUNDARY,
    PHASE_POSITIONING_RULE,
)
BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "reviewer_entry_paths",
    "story_artifacts",
    "file_coverage",
    "terminology_inventory",
    "software",
    "provenance",
    "summary",
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


def _load_story3(path: Path = STORY3_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story3_artifact(artifact)
        return artifact
    return run_story3_validation(verbose=False)


def _load_story4(path: Path = STORY4_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story4_artifact(artifact)
        return artifact
    return run_story4_validation(verbose=False)


def _load_story5(path: Path = STORY5_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story5_artifact(artifact)
        return artifact
    return run_story5_validation(verbose=False)


def _load_story6(path: Path = STORY6_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_story6_artifact(artifact)
        return artifact
    return run_story6_validation(verbose=False)


def build_story_artifact_entries(
    *,
    story1_artifact,
    story2_artifact,
    story3_artifact,
    story4_artifact,
    story5_artifact,
    story6_artifact,
):
    artifact_map = {
        "claim_package": story1_artifact,
        "publication_surface_alignment": story2_artifact,
        "claim_traceability_bundle": story3_artifact,
        "evidence_closure_bundle": story4_artifact,
        "supported_path_scope_bundle": story5_artifact,
        "future_work_boundary_bundle": story6_artifact,
    }
    entries = []
    for item in STORY_ARTIFACT_REQUIREMENTS:
        payload = artifact_map[item["artifact_id"]]
        entries.append(
            {
                "artifact_id": item["artifact_id"],
                "path": relative_to_repo(item["path"]),
                "status": payload["status"],
                "expected_statuses": list(item["expected_statuses"]),
                "semantic_flag": item["semantic_flag"],
                "semantic_flag_passed": bool(
                    payload["summary"].get(item["semantic_flag"], False)
                ),
                "summary": dict(payload["summary"]),
            }
        )
    return entries


def build_file_coverage():
    coverage = []
    for doc_id, path in MANDATORY_PUBLICATION_EVIDENCE_DOCS.items():
        coverage.append(
            {
                "doc_id": doc_id,
                "path": relative_to_repo(path),
                "exists": path.exists(),
            }
        )
    return coverage


def build_terminology_inventory():
    combined_text = "\n".join(
        load_text(MANDATORY_PUBLICATION_EVIDENCE_DOCS[surface["doc_id"]])
        for surface in PUBLICATION_SURFACES.values()
    )
    combined_text += "\n" + load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    normalized = normalize_text(combined_text)
    return {
        "required_glossary_terms": list(REQUIRED_GLOSSARY_TERMS),
        "missing_glossary_terms": [
            term
            for term in REQUIRED_GLOSSARY_TERMS
            if normalize_text(term) not in normalized
        ],
    }


def build_publication_evidence_story7_bundle(
    *,
    story1_artifact,
    story2_artifact,
    story3_artifact,
    story4_artifact,
    story5_artifact,
    story6_artifact,
):
    story_artifacts = build_story_artifact_entries(
        story1_artifact=story1_artifact,
        story2_artifact=story2_artifact,
        story3_artifact=story3_artifact,
        story4_artifact=story4_artifact,
        story5_artifact=story5_artifact,
        story6_artifact=story6_artifact,
    )
    file_coverage = build_file_coverage()
    terminology_inventory = build_terminology_inventory()
    reviewer_entry_paths = {
        "phase2_documentation_index": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "exists": PHASE2_DOCUMENTATION_INDEX_PATH.exists(),
        },
        "correctness_evidence_publication_bundle": {
            "path": relative_to_repo(CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH),
            "exists": CORRECTNESS_EVIDENCE_PUBLICATION_BUNDLE_PATH.exists(),
        },
    }

    story_artifacts_complete = all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in story_artifacts
    )
    file_coverage_complete = all(entry["exists"] for entry in file_coverage)
    terminology_complete = not terminology_inventory["missing_glossary_terms"]
    reviewer_entry_paths_complete = all(
        entry["exists"] for entry in reviewer_entry_paths.values()
    )

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if (
            story_artifacts_complete
            and file_coverage_complete
            and terminology_complete
            and reviewer_entry_paths_complete
        )
        else "fail",
        "reviewer_entry_paths": reviewer_entry_paths,
        "story_artifacts": story_artifacts,
        "file_coverage": file_coverage,
        "terminology_inventory": terminology_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_claim_package/publication_claim_bundle.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_path": str(STORY1_PATH),
            "story2_path": str(STORY2_PATH),
            "story3_path": str(STORY3_PATH),
            "exact_regime_path": str(STORY4_PATH),
            "story5_path": str(STORY5_PATH),
            "story6_path": str(STORY6_PATH),
        },
        "summary": {
            "mandatory_story_artifact_count": len(story_artifacts),
            "story_artifacts_complete": story_artifacts_complete,
            "mandatory_file_count": len(file_coverage),
            "file_coverage_complete": file_coverage_complete,
            "glossary_term_count": len(REQUIRED_GLOSSARY_TERMS),
            "terminology_complete": terminology_complete,
            "reviewer_entry_paths_complete": reviewer_entry_paths_complete,
        },
    }
    validate_publication_evidence_story7_bundle(bundle)
    return bundle


def validate_publication_evidence_story7_bundle(bundle):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 8 Story 7 bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_artifact_ids = {
        item["artifact_id"] for item in STORY_ARTIFACT_REQUIREMENTS
    }
    observed_artifact_ids = {artifact["artifact_id"] for artifact in bundle["story_artifacts"]}
    if required_artifact_ids != observed_artifact_ids:
        raise ValueError(
            "Task 8 Story 7 bundle is missing required story artifact IDs: {}".format(
                ", ".join(sorted(required_artifact_ids - observed_artifact_ids))
            )
        )

    if not all(entry["exists"] for entry in bundle["reviewer_entry_paths"].values()):
        raise ValueError("Task 8 Story 7 reviewer entry paths must exist")

    if bundle["summary"]["story_artifacts_complete"] != all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in bundle["story_artifacts"]
    ):
        raise ValueError(
            "Task 8 Story 7 story_artifacts_complete summary is inconsistent"
        )

    if bundle["summary"]["file_coverage_complete"] != all(
        entry["exists"] for entry in bundle["file_coverage"]
    ):
        raise ValueError(
            "Task 8 Story 7 file_coverage_complete summary is inconsistent"
        )

    if bundle["summary"]["terminology_complete"] != (
        not bundle["terminology_inventory"]["missing_glossary_terms"]
    ):
        raise ValueError(
            "Task 8 Story 7 terminology_complete summary is inconsistent"
        )

    if bundle["summary"]["reviewer_entry_paths_complete"] != all(
        entry["exists"] for entry in bundle["reviewer_entry_paths"].values()
    ):
        raise ValueError(
            "Task 8 Story 7 reviewer_entry_paths_complete summary is inconsistent"
        )

    expected_status = (
        "pass"
        if (
            bundle["summary"]["story_artifacts_complete"]
            and bundle["summary"]["file_coverage_complete"]
            and bundle["summary"]["terminology_complete"]
            and bundle["summary"]["reviewer_entry_paths_complete"]
        )
        else "fail"
    )
    if bundle["status"] != expected_status:
        raise ValueError("Task 8 Story 7 status does not match bundle summary")


def write_publication_evidence_story7_bundle(output_path: Path, bundle):
    validate_publication_evidence_story7_bundle(bundle)
    write_json(output_path, bundle)


def run_validation(
    *,
    story1_path: Path = STORY1_PATH,
    story2_path: Path = STORY2_PATH,
    story3_path: Path = STORY3_PATH,
    story4_path: Path = STORY4_PATH,
    story5_path: Path = STORY5_PATH,
    story6_path: Path = STORY6_PATH,
    verbose=False,
):
    story1_artifact = _load_story1(story1_path)
    story2_artifact = _load_story2(story2_path)
    story3_artifact = _load_story3(story3_path)
    story4_artifact = _load_story4(story4_path)
    story5_artifact = _load_story5(story5_path)
    story6_artifact = _load_story6(story6_path)
    bundle = build_publication_evidence_story7_bundle(
        story1_artifact=story1_artifact,
        story2_artifact=story2_artifact,
        story3_artifact=story3_artifact,
        story4_artifact=story4_artifact,
        story5_artifact=story5_artifact,
        story6_artifact=story6_artifact,
    )
    if verbose:
        print(
            "{} [{}] story_artifacts_complete={} file_coverage_complete={} terminology_complete={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["story_artifacts_complete"],
                bundle["summary"]["file_coverage_complete"],
                bundle["summary"]["terminology_complete"],
            )
        )
    return (
        story1_artifact,
        story2_artifact,
        story3_artifact,
        story4_artifact,
        story5_artifact,
        story6_artifact,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 8 Story 7 JSON artifact bundle.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    *_, bundle = run_validation(verbose=not args.quiet)
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_publication_evidence_story7_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["mandatory_story_artifact_count"],
            bundle["summary"]["mandatory_file_count"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


# Compatibility aliases for legacy auxiliary validation imports.
build_task8_story7_bundle = build_publication_evidence_story7_bundle
validate_task8_story7_bundle = validate_publication_evidence_story7_bundle
write_task8_story7_bundle = write_publication_evidence_story7_bundle


if __name__ == "__main__":
    main()
