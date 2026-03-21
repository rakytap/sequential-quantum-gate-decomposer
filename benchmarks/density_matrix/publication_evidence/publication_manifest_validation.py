#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 6 manifest-driven reviewer package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.claim_package_validation import (
    ARTIFACT_FILENAME as STORY1_ARTIFACT_FILENAME,
    run_validation as run_story1_validation,
    validate_artifact_bundle as validate_story1_artifact,
)
from benchmarks.density_matrix.publication_evidence.claim_traceability_validation import (
    ARTIFACT_FILENAME as STORY3_ARTIFACT_FILENAME,
    run_validation as run_story3_validation,
    validate_artifact_bundle as validate_story3_artifact,
)
from benchmarks.density_matrix.publication_evidence.common import (
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH,
    CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH,
    PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH,
    PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH,
    PERFORMANCE_EVIDENCE_POSITIVE_THRESHOLD_PATH,
    PERFORMANCE_EVIDENCE_SENSITIVITY_MATRIX_PATH,
    PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    build_software_metadata,
    get_git_revision,
    load_or_build_artifact,
    relative_to_repo,
    publication_evidence_output_dir,
    write_json,
)
from benchmarks.density_matrix.publication_evidence.evidence_closure_validation import (
    ARTIFACT_FILENAME as STORY4_ARTIFACT_FILENAME,
    run_validation as run_story4_validation,
    validate_artifact_bundle as validate_story4_artifact,
)
from benchmarks.density_matrix.publication_evidence.supported_path_validation import (
    ARTIFACT_FILENAME as STORY5_ARTIFACT_FILENAME,
    run_validation as run_story5_validation,
    validate_artifact_bundle as validate_story5_artifact,
)
from benchmarks.density_matrix.publication_evidence.surface_alignment_validation import (
    ARTIFACT_FILENAME as STORY2_ARTIFACT_FILENAME,
    run_validation as run_story2_validation,
    validate_artifact_bundle as validate_story2_artifact,
)


SUITE_NAME = "phase3_publication_evidence_manifest"
ARTIFACT_FILENAME = "publication_manifest_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("manifest")
STORY1_PATH = publication_evidence_output_dir("claim_package") / STORY1_ARTIFACT_FILENAME
STORY2_PATH = (
    publication_evidence_output_dir("surface_alignment")
    / STORY2_ARTIFACT_FILENAME
)
STORY3_PATH = (
    publication_evidence_output_dir("claim_traceability")
    / STORY3_ARTIFACT_FILENAME
)
STORY4_PATH = (
    publication_evidence_output_dir("evidence_closure")
    / STORY4_ARTIFACT_FILENAME
)
STORY5_PATH = (
    publication_evidence_output_dir("supported_path")
    / STORY5_ARTIFACT_FILENAME
)
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "story_artifacts",
    "publication_surfaces",
    "required_evidence_refs",
    "reviewer_entry_paths",
    "software",
    "provenance",
    "summary",
)
STORY_ARTIFACT_REQUIREMENTS = (
    {
        "artifact_id": "claim_package_bundle",
        "path": STORY1_PATH,
        "semantic_flag": "claim_package_completed",
    },
    {
        "artifact_id": "publication_surface_alignment_bundle",
        "path": STORY2_PATH,
        "semantic_flag": "surface_alignment_completed",
    },
    {
        "artifact_id": "claim_traceability_bundle",
        "path": STORY3_PATH,
        "semantic_flag": "claim_traceability_completed",
    },
    {
        "artifact_id": "evidence_closure_bundle",
        "path": STORY4_PATH,
        "semantic_flag": "evidence_closure_completed",
    },
    {
        "artifact_id": "supported_path_scope_bundle",
        "path": STORY5_PATH,
        "semantic_flag": "supported_path_scope_completed",
    },
)
REQUIRED_EVIDENCE_REFS = (
    {
        "artifact_id": "correctness_evidence_correctness_package_bundle",
        "path": CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH,
    },
    {
        "artifact_id": "correctness_evidence_unsupported_boundary_bundle",
        "path": CORRECTNESS_EVIDENCE_UNSUPPORTED_BOUNDARY_PATH,
    },
    {
        "artifact_id": "correctness_evidence_summary_consistency_bundle",
        "path": CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    },
    {
        "artifact_id": "performance_evidence_benchmark_package_bundle",
        "path": PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH,
    },
    {
        "artifact_id": "performance_evidence_diagnosis_bundle",
        "path": PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH,
    },
    {
        "artifact_id": "performance_evidence_positive_threshold_bundle",
        "path": PERFORMANCE_EVIDENCE_POSITIVE_THRESHOLD_PATH,
    },
    {
        "artifact_id": "performance_evidence_sensitivity_matrix_bundle",
        "path": PERFORMANCE_EVIDENCE_SENSITIVITY_MATRIX_PATH,
    },
    {
        "artifact_id": "performance_evidence_summary_consistency_bundle",
        "path": PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    },
)


def _story1():
    return load_or_build_artifact(
        STORY1_PATH,
        run_validation=run_story1_validation,
        validate_artifact_bundle=validate_story1_artifact,
    )


def _story2():
    return load_or_build_artifact(
        STORY2_PATH,
        run_validation=run_story2_validation,
        validate_artifact_bundle=validate_story2_artifact,
    )


def _story3():
    return load_or_build_artifact(
        STORY3_PATH,
        run_validation=run_story3_validation,
        validate_artifact_bundle=validate_story3_artifact,
    )


def _story4():
    return load_or_build_artifact(
        STORY4_PATH,
        run_validation=run_story4_validation,
        validate_artifact_bundle=validate_story4_artifact,
    )


def _story5():
    return load_or_build_artifact(
        STORY5_PATH,
        run_validation=run_story5_validation,
        validate_artifact_bundle=validate_story5_artifact,
    )


def build_story_artifact_entries(*, story1_artifact, story2_artifact, story3_artifact, story4_artifact, story5_artifact):
    artifact_map = {
        "claim_package_bundle": story1_artifact,
        "publication_surface_alignment_bundle": story2_artifact,
        "claim_traceability_bundle": story3_artifact,
        "evidence_closure_bundle": story4_artifact,
        "supported_path_scope_bundle": story5_artifact,
    }
    entries = []
    for requirement in STORY_ARTIFACT_REQUIREMENTS:
        payload = artifact_map[requirement["artifact_id"]]
        entries.append(
            {
                "artifact_id": requirement["artifact_id"],
                "path": relative_to_repo(requirement["path"]),
                "status": payload["status"],
                "semantic_flag": requirement["semantic_flag"],
                "semantic_flag_passed": bool(
                    payload["summary"].get(requirement["semantic_flag"], False)
                ),
                "summary": dict(payload["summary"]),
            }
        )
    return entries


def build_publication_surfaces():
    return [
        {
            "surface_id": "abstract",
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_abstract"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_abstract"].exists(),
        },
        {
            "surface_id": "short_paper",
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper"].exists(),
        },
        {
            "surface_id": "short_paper_narrative",
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper_narrative"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper_narrative"].exists(),
        },
        {
            "surface_id": "paper",
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"].exists(),
        },
    ]


def build_required_evidence_refs():
    return [
        {
            "artifact_id": requirement["artifact_id"],
            "path": relative_to_repo(requirement["path"]),
            "exists": requirement["path"].exists(),
        }
        for requirement in REQUIRED_EVIDENCE_REFS
    ]


def build_reviewer_entry_paths():
    return {
        "phase3_paper": {
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"].exists(),
        },
        "phase3_short_paper": {
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper"].exists(),
        },
        "phase3_short_paper_narrative": {
            "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper_narrative"]),
            "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_short_paper_narrative"].exists(),
        },
        "correctness_evidence_correctness_package": {
            "path": relative_to_repo(CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH),
            "exists": CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH.exists(),
        },
        "performance_evidence_benchmark_package": {
            "path": relative_to_repo(PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH),
            "exists": PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH.exists(),
        },
        "performance_evidence_diagnosis_bundle": {
            "path": relative_to_repo(PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH),
            "exists": PERFORMANCE_EVIDENCE_DIAGNOSIS_PATH.exists(),
        },
    }


def build_artifact_bundle():
    story1_artifact = _story1()
    story2_artifact = _story2()
    story3_artifact = _story3()
    story4_artifact = _story4()
    story5_artifact = _story5()
    story_artifacts = build_story_artifact_entries(
        story1_artifact=story1_artifact,
        story2_artifact=story2_artifact,
        story3_artifact=story3_artifact,
        story4_artifact=story4_artifact,
        story5_artifact=story5_artifact,
    )
    publication_surfaces = build_publication_surfaces()
    required_evidence_refs = build_required_evidence_refs()
    reviewer_entry_paths = build_reviewer_entry_paths()

    story_artifacts_complete = all(
        artifact["status"] == "pass" and artifact["semantic_flag_passed"]
        for artifact in story_artifacts
    )
    publication_surfaces_complete = all(entry["exists"] for entry in publication_surfaces)
    required_evidence_refs_complete = all(
        entry["exists"] for entry in required_evidence_refs
    )
    reviewer_entry_paths_complete = all(
        entry["exists"] for entry in reviewer_entry_paths.values()
    )
    publication_manifest_completed = all(
        [
            story_artifacts_complete,
            publication_surfaces_complete,
            required_evidence_refs_complete,
            reviewer_entry_paths_complete,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if publication_manifest_completed else "fail",
        "story_artifacts": story_artifacts,
        "publication_surfaces": publication_surfaces,
        "required_evidence_refs": required_evidence_refs,
        "reviewer_entry_paths": reviewer_entry_paths,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
                "publication_manifest_validation.py"
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
            "publication_surface_count": len(publication_surfaces),
            "required_evidence_ref_count": len(required_evidence_refs),
            "story_artifacts_complete": story_artifacts_complete,
            "publication_surfaces_complete": publication_surfaces_complete,
            "required_evidence_refs_complete": required_evidence_refs_complete,
            "reviewer_entry_paths_complete": reviewer_entry_paths_complete,
            "publication_manifest_completed": publication_manifest_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 6 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_artifact_ids = {
        requirement["artifact_id"] for requirement in STORY_ARTIFACT_REQUIREMENTS
    }
    observed_artifact_ids = {entry["artifact_id"] for entry in artifact["story_artifacts"]}
    if observed_artifact_ids != required_artifact_ids:
        raise ValueError("publication manifest story artifact inventory mismatch")

    required_ref_ids = {requirement["artifact_id"] for requirement in REQUIRED_EVIDENCE_REFS}
    observed_ref_ids = {entry["artifact_id"] for entry in artifact["required_evidence_refs"]}
    if observed_ref_ids != required_ref_ids:
        raise ValueError("publication manifest evidence reference inventory mismatch")

    expected_flags = {
        "story_artifacts_complete": all(
            entry["status"] == "pass" and entry["semantic_flag_passed"]
            for entry in artifact["story_artifacts"]
        ),
        "publication_surfaces_complete": all(
            entry["exists"] for entry in artifact["publication_surfaces"]
        ),
        "required_evidence_refs_complete": all(
            entry["exists"] for entry in artifact["required_evidence_refs"]
        ),
        "reviewer_entry_paths_complete": all(
            entry["exists"] for entry in artifact["reviewer_entry_paths"].values()
        ),
    }
    for field, value in expected_flags.items():
        if artifact["summary"][field] != value:
            raise ValueError(f"{field} summary is inconsistent")

    expected_completed = all(expected_flags.values())
    if artifact["summary"]["publication_manifest_completed"] != expected_completed:
        raise ValueError("publication_manifest_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 6 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] stories={} refs={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["mandatory_story_artifact_count"],
                artifact["summary"]["required_evidence_ref_count"],
                artifact["summary"]["publication_manifest_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Phase 3 Task 8 Story 6 JSON artifact bundle.",
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
            artifact["summary"]["mandatory_story_artifact_count"],
            artifact["summary"]["required_evidence_ref_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
