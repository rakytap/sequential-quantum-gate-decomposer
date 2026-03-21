#!/usr/bin/env python3
"""Validation: Manifest-driven reviewer entry paths and mandatory evidence anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.claim_package_validation import (
    ARTIFACT_FILENAME as CLAIM_PACKAGE_ARTIFACT_FILENAME,
    run_validation as run_claim_package_validation,
    validate_artifact_bundle as validate_claim_package_artifact,
)
from benchmarks.density_matrix.publication_evidence.claim_traceability_validation import (
    ARTIFACT_FILENAME as CLAIM_TRACEABILITY_ARTIFACT_FILENAME,
    run_validation as run_claim_traceability_validation,
    validate_artifact_bundle as validate_claim_traceability_artifact,
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
    ARTIFACT_FILENAME as EVIDENCE_CLOSURE_ARTIFACT_FILENAME,
    run_validation as run_evidence_closure_validation,
    validate_artifact_bundle as validate_evidence_closure_artifact,
)
from benchmarks.density_matrix.publication_evidence.supported_path_validation import (
    ARTIFACT_FILENAME as SUPPORTED_PATH_SCOPE_ARTIFACT_FILENAME,
    run_validation as run_supported_path_scope_validation,
    validate_artifact_bundle as validate_supported_path_scope_artifact,
)
from benchmarks.density_matrix.publication_evidence.surface_alignment_validation import (
    ARTIFACT_FILENAME as PUBLICATION_SURFACE_ALIGNMENT_ARTIFACT_FILENAME,
    run_validation as run_publication_surface_alignment_validation,
    validate_artifact_bundle as validate_publication_surface_alignment_artifact,
)


SUITE_NAME = "publication_manifest"
ARTIFACT_FILENAME = "publication_manifest_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("publication_manifest")
CLAIM_PACKAGE_BUNDLE_PATH = (
    publication_evidence_output_dir("claim_package") / CLAIM_PACKAGE_ARTIFACT_FILENAME
)
PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH = (
    publication_evidence_output_dir("publication_surface_alignment")
    / PUBLICATION_SURFACE_ALIGNMENT_ARTIFACT_FILENAME
)
CLAIM_TRACEABILITY_BUNDLE_PATH = (
    publication_evidence_output_dir("claim_traceability")
    / CLAIM_TRACEABILITY_ARTIFACT_FILENAME
)
EVIDENCE_CLOSURE_BUNDLE_PATH = (
    publication_evidence_output_dir("evidence_closure")
    / EVIDENCE_CLOSURE_ARTIFACT_FILENAME
)
SUPPORTED_PATH_SCOPE_BUNDLE_PATH = (
    publication_evidence_output_dir("supported_path_scope")
    / SUPPORTED_PATH_SCOPE_ARTIFACT_FILENAME
)
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "component_artifacts",
    "publication_surfaces",
    "required_evidence_refs",
    "reviewer_entry_paths",
    "software",
    "provenance",
    "summary",
)
COMPONENT_ARTIFACT_REQUIREMENTS = (
    {
        "artifact_id": "claim_package_bundle",
        "path": CLAIM_PACKAGE_BUNDLE_PATH,
        "semantic_flag": "claim_package_completed",
    },
    {
        "artifact_id": "publication_surface_alignment_bundle",
        "path": PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH,
        "semantic_flag": "publication_surface_alignment_completed",
    },
    {
        "artifact_id": "claim_traceability_bundle",
        "path": CLAIM_TRACEABILITY_BUNDLE_PATH,
        "semantic_flag": "claim_traceability_completed",
    },
    {
        "artifact_id": "evidence_closure_bundle",
        "path": EVIDENCE_CLOSURE_BUNDLE_PATH,
        "semantic_flag": "evidence_closure_completed",
    },
    {
        "artifact_id": "supported_path_scope_bundle",
        "path": SUPPORTED_PATH_SCOPE_BUNDLE_PATH,
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


def _load_claim_package_artifact():
    return load_or_build_artifact(
        CLAIM_PACKAGE_BUNDLE_PATH,
        run_validation=run_claim_package_validation,
        validate_artifact_bundle=validate_claim_package_artifact,
    )


def _load_publication_surface_alignment_artifact():
    return load_or_build_artifact(
        PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH,
        run_validation=run_publication_surface_alignment_validation,
        validate_artifact_bundle=validate_publication_surface_alignment_artifact,
    )


def _load_claim_traceability_artifact():
    return load_or_build_artifact(
        CLAIM_TRACEABILITY_BUNDLE_PATH,
        run_validation=run_claim_traceability_validation,
        validate_artifact_bundle=validate_claim_traceability_artifact,
    )


def _load_evidence_closure_artifact():
    return load_or_build_artifact(
        EVIDENCE_CLOSURE_BUNDLE_PATH,
        run_validation=run_evidence_closure_validation,
        validate_artifact_bundle=validate_evidence_closure_artifact,
    )


def _load_supported_path_scope_artifact():
    return load_or_build_artifact(
        SUPPORTED_PATH_SCOPE_BUNDLE_PATH,
        run_validation=run_supported_path_scope_validation,
        validate_artifact_bundle=validate_supported_path_scope_artifact,
    )


def build_component_artifact_entries(
    *,
    claim_package_artifact,
    publication_surface_alignment_artifact,
    claim_traceability_artifact,
    evidence_closure_artifact,
    supported_path_scope_artifact,
):
    artifact_map = {
        "claim_package_bundle": claim_package_artifact,
        "publication_surface_alignment_bundle": publication_surface_alignment_artifact,
        "claim_traceability_bundle": claim_traceability_artifact,
        "evidence_closure_bundle": evidence_closure_artifact,
        "supported_path_scope_bundle": supported_path_scope_artifact,
    }
    entries = []
    for requirement in COMPONENT_ARTIFACT_REQUIREMENTS:
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
    claim_package_artifact = _load_claim_package_artifact()
    publication_surface_alignment_artifact = _load_publication_surface_alignment_artifact()
    claim_traceability_artifact = _load_claim_traceability_artifact()
    evidence_closure_artifact = _load_evidence_closure_artifact()
    supported_path_scope_artifact = _load_supported_path_scope_artifact()
    component_artifacts = build_component_artifact_entries(
        claim_package_artifact=claim_package_artifact,
        publication_surface_alignment_artifact=publication_surface_alignment_artifact,
        claim_traceability_artifact=claim_traceability_artifact,
        evidence_closure_artifact=evidence_closure_artifact,
        supported_path_scope_artifact=supported_path_scope_artifact,
    )
    publication_surfaces = build_publication_surfaces()
    required_evidence_refs = build_required_evidence_refs()
    reviewer_entry_paths = build_reviewer_entry_paths()

    component_artifacts_complete = all(
        artifact["status"] == "pass" and artifact["semantic_flag_passed"]
        for artifact in component_artifacts
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
            component_artifacts_complete,
            publication_surfaces_complete,
            required_evidence_refs_complete,
            reviewer_entry_paths_complete,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if publication_manifest_completed else "fail",
        "component_artifacts": component_artifacts,
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
            "claim_package_bundle_path": str(CLAIM_PACKAGE_BUNDLE_PATH),
            "publication_surface_alignment_bundle_path": str(
                PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH
            ),
            "claim_traceability_bundle_path": str(CLAIM_TRACEABILITY_BUNDLE_PATH),
            "evidence_closure_bundle_path": str(EVIDENCE_CLOSURE_BUNDLE_PATH),
            "supported_path_scope_bundle_path": str(SUPPORTED_PATH_SCOPE_BUNDLE_PATH),
        },
        "summary": {
            "mandatory_component_artifact_count": len(component_artifacts),
            "publication_surface_count": len(publication_surfaces),
            "required_evidence_ref_count": len(required_evidence_refs),
            "component_artifacts_complete": component_artifacts_complete,
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
            "publication_manifest artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_artifact_ids = {
        requirement["artifact_id"] for requirement in COMPONENT_ARTIFACT_REQUIREMENTS
    }
    observed_artifact_ids = {entry["artifact_id"] for entry in artifact["component_artifacts"]}
    if observed_artifact_ids != required_artifact_ids:
        raise ValueError("publication manifest component artifact inventory mismatch")

    required_ref_ids = {requirement["artifact_id"] for requirement in REQUIRED_EVIDENCE_REFS}
    observed_ref_ids = {entry["artifact_id"] for entry in artifact["required_evidence_refs"]}
    if observed_ref_ids != required_ref_ids:
        raise ValueError("publication manifest evidence reference inventory mismatch")

    expected_flags = {
        "component_artifacts_complete": all(
            entry["status"] == "pass" and entry["semantic_flag_passed"]
            for entry in artifact["component_artifacts"]
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
        raise ValueError("publication_manifest status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] components={} refs={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["mandatory_component_artifact_count"],
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
        help="Directory for the publication_manifest JSON artifact bundle.",
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
            artifact["summary"]["mandatory_component_artifact_count"],
            artifact["summary"]["required_evidence_ref_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
