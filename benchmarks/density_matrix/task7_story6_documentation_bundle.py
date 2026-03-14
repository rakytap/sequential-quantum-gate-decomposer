#!/usr/bin/env python3
"""Validation: Task 7 Story 6 documentation-consistency bundle.

Builds the top-level Task 7 bundle by assembling Story 1 to Story 5 outputs into
one machine-checkable documentation surface. The bundle preserves the shared
entry-point path, validates the lower-story semantic gates, records mandatory
file coverage, and checks the canonical Phase 2 terminology inventory.

Run with:
    python benchmarks/density_matrix/task7_story6_documentation_bundle.py
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
from benchmarks.density_matrix.task7_story1_contract_reference_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)
from benchmarks.density_matrix.task7_story2_supported_entry_reference_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    validate_artifact_bundle as validate_story2_artifact,
)
from benchmarks.density_matrix.task7_story3_support_surface_reference_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)
from benchmarks.density_matrix.task7_story4_evidence_bar_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)
from benchmarks.density_matrix.task7_story5_future_work_boundary_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    run_validation as run_story5_validation,
    validate_artifact_bundle as validate_story5_artifact,
)


SUITE_NAME = "task7_story6_documentation_bundle"
ARTIFACT_FILENAME = "task7_story6_documentation_bundle.json"
DEFAULT_OUTPUT_DIR = TASK7_OUTPUT_DIR
STORY1_PATH = DEFAULT_OUTPUT_DIR / STORY1_ARTIFACT_FILENAME
STORY2_PATH = DEFAULT_OUTPUT_DIR / STORY2_ARTIFACT_FILENAME
STORY3_PATH = DEFAULT_OUTPUT_DIR / STORY3_ARTIFACT_FILENAME
STORY4_PATH = DEFAULT_OUTPUT_DIR / STORY4_ARTIFACT_FILENAME
STORY5_PATH = DEFAULT_OUTPUT_DIR / STORY5_ARTIFACT_FILENAME
STORY_ARTIFACT_REQUIREMENTS = (
    {
        "artifact_id": "task7_story1_contract_reference_map",
        "path": STORY1_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "contract_reference_map_completed",
    },
    {
        "artifact_id": "task7_story2_supported_entry_reference",
        "path": STORY2_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "supported_entry_reference_completed",
    },
    {
        "artifact_id": "task7_story3_support_surface_reference",
        "path": STORY3_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "support_surface_reference_completed",
    },
    {
        "artifact_id": "task7_story4_evidence_bar_reference",
        "path": STORY4_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "evidence_bar_reference_completed",
    },
    {
        "artifact_id": "task7_story5_future_work_boundary",
        "path": STORY5_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "future_work_boundary_completed",
    },
)
REQUIRED_GLOSSARY_TERMS = (
    "`state_vector`: the default backend",
    "`density_matrix`: the explicitly selected mixed-state backend",
    "canonical workflow: noisy XXZ VQE",
    "exact regime: full end-to-end workflow execution at 4 and 6 qubits",
    "acceptance anchor: the documented 10-qubit case",
    "required / optional / deferred / unsupported",
    "reproducibility bundle: the backend-explicit evidence package rooted in",
    "future work and non-goal",
)
BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "entry_point",
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


def build_story_artifact_entries(
    *,
    story1_artifact,
    story2_artifact,
    story3_artifact,
    story4_artifact,
    story5_artifact,
):
    artifact_map = {
        "task7_story1_contract_reference_map": story1_artifact,
        "task7_story2_supported_entry_reference": story2_artifact,
        "task7_story3_support_surface_reference": story3_artifact,
        "task7_story4_evidence_bar_reference": story4_artifact,
        "task7_story5_future_work_boundary": story5_artifact,
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
    for doc_id, path in MANDATORY_PHASE2_DOCS.items():
        coverage.append(
            {
                "doc_id": doc_id,
                "path": relative_to_repo(path),
                "exists": path.exists(),
            }
        )
    return coverage


def build_terminology_inventory():
    entry_text = load_text(PHASE2_DOCUMENTATION_INDEX_PATH)
    return {
        "entry_point_path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
        "required_glossary_terms": list(REQUIRED_GLOSSARY_TERMS),
        "missing_glossary_terms": [
            term
            for term in REQUIRED_GLOSSARY_TERMS
            if normalize_text(term) not in normalize_text(entry_text)
        ],
    }


def build_task7_story6_bundle(
    output_dir: Path,
    *,
    story1_artifact,
    story2_artifact,
    story3_artifact,
    story4_artifact,
    story5_artifact,
):
    story_artifacts = build_story_artifact_entries(
        story1_artifact=story1_artifact,
        story2_artifact=story2_artifact,
        story3_artifact=story3_artifact,
        story4_artifact=story4_artifact,
        story5_artifact=story5_artifact,
    )
    file_coverage = build_file_coverage()
    terminology_inventory = build_terminology_inventory()

    story_artifacts_complete = all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in story_artifacts
    )
    file_coverage_complete = all(entry["exists"] for entry in file_coverage)
    glossary_complete = not terminology_inventory["missing_glossary_terms"]

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if story_artifacts_complete and file_coverage_complete and glossary_complete
        else "fail",
        "entry_point": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "exists": PHASE2_DOCUMENTATION_INDEX_PATH.exists(),
        },
        "story_artifacts": story_artifacts,
        "file_coverage": file_coverage,
        "terminology_inventory": terminology_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/task7_story6_documentation_bundle.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story1_path": str(STORY1_PATH),
            "story2_path": str(STORY2_PATH),
            "story3_path": str(STORY3_PATH),
            "story4_path": str(STORY4_PATH),
            "story5_path": str(STORY5_PATH),
        },
        "summary": {
            "mandatory_story_artifact_count": len(story_artifacts),
            "story_artifacts_complete": story_artifacts_complete,
            "mandatory_file_count": len(file_coverage),
            "file_coverage_complete": file_coverage_complete,
            "glossary_term_count": len(REQUIRED_GLOSSARY_TERMS),
            "glossary_complete": glossary_complete,
        },
    }
    validate_task7_story6_bundle(bundle)
    return bundle


def validate_task7_story6_bundle(bundle):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Task 7 Story 6 bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_artifact_ids = {
        item["artifact_id"] for item in STORY_ARTIFACT_REQUIREMENTS
    }
    observed_artifact_ids = {artifact["artifact_id"] for artifact in bundle["story_artifacts"]}
    if required_artifact_ids != observed_artifact_ids:
        raise ValueError(
            "Task 7 Story 6 bundle is missing required story artifact IDs: {}".format(
                ", ".join(sorted(required_artifact_ids - observed_artifact_ids))
            )
        )

    if not bundle["entry_point"]["exists"]:
        raise ValueError("Task 7 Story 6 entry point must exist")

    if bundle["summary"]["story_artifacts_complete"] != all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in bundle["story_artifacts"]
    ):
        raise ValueError(
            "Task 7 Story 6 story_artifacts_complete summary is inconsistent"
        )

    if bundle["summary"]["file_coverage_complete"] != all(
        entry["exists"] for entry in bundle["file_coverage"]
    ):
        raise ValueError(
            "Task 7 Story 6 file_coverage_complete summary is inconsistent"
        )

    if bundle["summary"]["glossary_complete"] != (
        len(bundle["terminology_inventory"]["missing_glossary_terms"]) == 0
    ):
        raise ValueError("Task 7 Story 6 glossary_complete summary is inconsistent")

    if bundle["status"] != "pass" and (
        bundle["summary"]["story_artifacts_complete"]
        and bundle["summary"]["file_coverage_complete"]
        and bundle["summary"]["glossary_complete"]
    ):
        raise ValueError("Task 7 Story 6 bundle status is inconsistent")


def write_task7_story6_bundle(output_path: Path, bundle):
    validate_task7_story6_bundle(bundle)
    write_json(output_path, bundle)


def run_validation(
    *,
    story1_path: Path = STORY1_PATH,
    story2_path: Path = STORY2_PATH,
    story3_path: Path = STORY3_PATH,
    story4_path: Path = STORY4_PATH,
    story5_path: Path = STORY5_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose=False,
):
    story1_artifact = _load_story1(story1_path)
    story2_artifact = _load_story2(story2_path)
    story3_artifact = _load_story3(story3_path)
    story4_artifact = _load_story4(story4_path)
    story5_artifact = _load_story5(story5_path)
    bundle = build_task7_story6_bundle(
        output_dir,
        story1_artifact=story1_artifact,
        story2_artifact=story2_artifact,
        story3_artifact=story3_artifact,
        story4_artifact=story4_artifact,
        story5_artifact=story5_artifact,
    )
    if verbose:
        print(
            "{} [{}] story_artifacts_complete={} file_coverage_complete={} glossary_complete={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["story_artifacts_complete"],
                bundle["summary"]["file_coverage_complete"],
                bundle["summary"]["glossary_complete"],
            )
        )
    return (
        story1_artifact,
        story2_artifact,
        story3_artifact,
        story4_artifact,
        story5_artifact,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Task 7 Story 6 JSON artifact bundle.",
    )
    parser.add_argument(
        "--story1-path",
        type=Path,
        default=STORY1_PATH,
        help="Path to the Task 7 Story 1 contract-reference map artifact.",
    )
    parser.add_argument(
        "--story2-path",
        type=Path,
        default=STORY2_PATH,
        help="Path to the Task 7 Story 2 supported-entry artifact.",
    )
    parser.add_argument(
        "--story3-path",
        type=Path,
        default=STORY3_PATH,
        help="Path to the Task 7 Story 3 support-surface artifact.",
    )
    parser.add_argument(
        "--story4-path",
        type=Path,
        default=STORY4_PATH,
        help="Path to the Task 7 Story 4 evidence-bar artifact.",
    )
    parser.add_argument(
        "--story5-path",
        type=Path,
        default=STORY5_PATH,
        help="Path to the Task 7 Story 5 future-work artifact.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    *_, bundle = run_validation(
        story1_path=args.story1_path,
        story2_path=args.story2_path,
        story3_path=args.story3_path,
        story4_path=args.story4_path,
        story5_path=args.story5_path,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_task7_story6_bundle(output_path, bundle)
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


if __name__ == "__main__":
    main()
