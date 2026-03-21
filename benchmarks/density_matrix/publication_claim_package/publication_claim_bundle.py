#!/usr/bin/env python3
"""Validation: Top-level publication claim bundle for Paper 1.

Assembles claim_package through future_work_boundary into one machine-checkable
publication surface. Preserves shared reviewer-entry paths, validates
lower-layer semantic gates, records mandatory file coverage, and checks
canonical publication terminology.

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
    WORKFLOW_PUBLICATION_BUNDLE_PATH,
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
    ARTIFACT_FILENAME as CLAIM_PACKAGE_FILENAME,
    run_validation as run_claim_package_validation,
    validate_artifact_bundle as validate_claim_package_artifact,
)
from benchmarks.density_matrix.publication_claim_package.publication_surface_alignment import (
    ARTIFACT_FILENAME as PUBLICATION_SURFACE_ALIGNMENT_FILENAME,
    run_validation as run_publication_surface_alignment_validation,
    validate_artifact_bundle as validate_publication_surface_alignment_artifact,
)
from benchmarks.density_matrix.publication_claim_package.claim_traceability_bundle import (
    ARTIFACT_FILENAME as CLAIM_TRACEABILITY_FILENAME,
    run_validation as run_claim_traceability_validation,
    validate_artifact_bundle as validate_claim_traceability_artifact,
)
from benchmarks.density_matrix.publication_claim_package.evidence_closure_validation import (
    ARTIFACT_FILENAME as EVIDENCE_CLOSURE_FILENAME,
    run_validation as run_evidence_closure_validation,
    validate_artifact_bundle as validate_evidence_closure_artifact,
)
from benchmarks.density_matrix.publication_claim_package.supported_path_scope_validation import (
    ARTIFACT_FILENAME as SUPPORTED_PATH_SCOPE_FILENAME,
    run_validation as run_supported_path_scope_validation,
    validate_artifact_bundle as validate_supported_path_scope_artifact,
)
from benchmarks.density_matrix.publication_claim_package.future_work_boundary_validation import (
    ARTIFACT_FILENAME as FUTURE_WORK_BOUNDARY_FILENAME,
    run_validation as run_future_work_boundary_validation,
    validate_artifact_bundle as validate_future_work_boundary_artifact,
)


SUITE_NAME = "publication_claim_bundle"
ARTIFACT_FILENAME = "publication_claim_bundle.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
CLAIM_PACKAGE_PATH = DEFAULT_OUTPUT_DIR / CLAIM_PACKAGE_FILENAME
PUBLICATION_SURFACE_ALIGNMENT_PATH = DEFAULT_OUTPUT_DIR / PUBLICATION_SURFACE_ALIGNMENT_FILENAME
CLAIM_TRACEABILITY_PATH = DEFAULT_OUTPUT_DIR / CLAIM_TRACEABILITY_FILENAME
EVIDENCE_CLOSURE_PATH = DEFAULT_OUTPUT_DIR / EVIDENCE_CLOSURE_FILENAME
SUPPORTED_PATH_SCOPE_PATH = DEFAULT_OUTPUT_DIR / SUPPORTED_PATH_SCOPE_FILENAME
FUTURE_WORK_BOUNDARY_PATH = DEFAULT_OUTPUT_DIR / FUTURE_WORK_BOUNDARY_FILENAME
COMPONENT_ARTIFACT_REQUIREMENTS = (
    {
        "artifact_id": "claim_package",
        "path": CLAIM_PACKAGE_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "claim_package_completed",
    },
    {
        "artifact_id": "publication_surface_alignment",
        "path": PUBLICATION_SURFACE_ALIGNMENT_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "surface_alignment_completed",
    },
    {
        "artifact_id": "claim_traceability",
        "path": CLAIM_TRACEABILITY_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "claim_traceability_completed",
    },
    {
        "artifact_id": "evidence_closure",
        "path": EVIDENCE_CLOSURE_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "evidence_closure_completed",
    },
    {
        "artifact_id": "supported_path_scope",
        "path": SUPPORTED_PATH_SCOPE_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "supported_path_scope_completed",
    },
    {
        "artifact_id": "future_work_boundary",
        "path": FUTURE_WORK_BOUNDARY_PATH,
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
    "component_artifacts",
    "file_coverage",
    "terminology_inventory",
    "software",
    "provenance",
    "summary",
)


def _load_claim_package(path: Path = CLAIM_PACKAGE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_claim_package_artifact(artifact)
        return artifact
    return run_claim_package_validation(verbose=False)


def _load_publication_surface_alignment(path: Path = PUBLICATION_SURFACE_ALIGNMENT_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_publication_surface_alignment_artifact(artifact)
        return artifact
    return run_publication_surface_alignment_validation(verbose=False)


def _load_claim_traceability(path: Path = CLAIM_TRACEABILITY_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_claim_traceability_artifact(artifact)
        return artifact
    return run_claim_traceability_validation(verbose=False)


def _load_evidence_closure(path: Path = EVIDENCE_CLOSURE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_evidence_closure_artifact(artifact)
        return artifact
    return run_evidence_closure_validation(verbose=False)


def _load_supported_path_scope(path: Path = SUPPORTED_PATH_SCOPE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_supported_path_scope_artifact(artifact)
        return artifact
    return run_supported_path_scope_validation(verbose=False)


def _load_future_work_boundary(path: Path = FUTURE_WORK_BOUNDARY_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_future_work_boundary_artifact(artifact)
        return artifact
    return run_future_work_boundary_validation(verbose=False)


def build_component_artifact_entries(
    *,
    claim_package_artifact,
    publication_surface_alignment_artifact,
    claim_traceability_artifact,
    evidence_closure_artifact,
    supported_path_scope_artifact,
    future_work_boundary_artifact,
):
    artifact_map = {
        "claim_package": claim_package_artifact,
        "publication_surface_alignment": publication_surface_alignment_artifact,
        "claim_traceability": claim_traceability_artifact,
        "evidence_closure": evidence_closure_artifact,
        "supported_path_scope": supported_path_scope_artifact,
        "future_work_boundary": future_work_boundary_artifact,
    }
    entries = []
    for item in COMPONENT_ARTIFACT_REQUIREMENTS:
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


def build_publication_claim_bundle(
    *,
    claim_package_artifact,
    publication_surface_alignment_artifact,
    claim_traceability_artifact,
    evidence_closure_artifact,
    supported_path_scope_artifact,
    future_work_boundary_artifact,
):
    component_artifacts = build_component_artifact_entries(
        claim_package_artifact=claim_package_artifact,
        publication_surface_alignment_artifact=publication_surface_alignment_artifact,
        claim_traceability_artifact=claim_traceability_artifact,
        evidence_closure_artifact=evidence_closure_artifact,
        supported_path_scope_artifact=supported_path_scope_artifact,
        future_work_boundary_artifact=future_work_boundary_artifact,
    )
    file_coverage = build_file_coverage()
    terminology_inventory = build_terminology_inventory()
    reviewer_entry_paths = {
        "phase2_documentation_index": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "exists": PHASE2_DOCUMENTATION_INDEX_PATH.exists(),
        },
        "workflow_publication_bundle": {
            "path": relative_to_repo(WORKFLOW_PUBLICATION_BUNDLE_PATH),
            "exists": WORKFLOW_PUBLICATION_BUNDLE_PATH.exists(),
        },
    }

    component_artifacts_complete = all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in component_artifacts
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
            component_artifacts_complete
            and file_coverage_complete
            and terminology_complete
            and reviewer_entry_paths_complete
        )
        else "fail",
        "reviewer_entry_paths": reviewer_entry_paths,
        "component_artifacts": component_artifacts,
        "file_coverage": file_coverage,
        "terminology_inventory": terminology_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_claim_package/publication_claim_bundle.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "claim_package_path": str(CLAIM_PACKAGE_PATH),
            "publication_surface_alignment_path": str(PUBLICATION_SURFACE_ALIGNMENT_PATH),
            "claim_traceability_path": str(CLAIM_TRACEABILITY_PATH),
            "evidence_closure_path": str(EVIDENCE_CLOSURE_PATH),
            "supported_path_scope_path": str(SUPPORTED_PATH_SCOPE_PATH),
            "future_work_boundary_path": str(FUTURE_WORK_BOUNDARY_PATH),
        },
        "summary": {
            "mandatory_component_count": len(component_artifacts),
            "component_artifacts_complete": component_artifacts_complete,
            "mandatory_file_count": len(file_coverage),
            "file_coverage_complete": file_coverage_complete,
            "glossary_term_count": len(REQUIRED_GLOSSARY_TERMS),
            "terminology_complete": terminology_complete,
            "reviewer_entry_paths_complete": reviewer_entry_paths_complete,
        },
    }
    validate_publication_claim_bundle(bundle)
    return bundle


def validate_publication_claim_bundle(bundle):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "publication_claim_bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_artifact_ids = {
        item["artifact_id"] for item in COMPONENT_ARTIFACT_REQUIREMENTS
    }
    observed_artifact_ids = {
        artifact["artifact_id"] for artifact in bundle["component_artifacts"]
    }
    if required_artifact_ids != observed_artifact_ids:
        raise ValueError(
            "publication_claim_bundle is missing required component artifact IDs: {}".format(
                ", ".join(sorted(required_artifact_ids - observed_artifact_ids))
            )
        )

    if not all(entry["exists"] for entry in bundle["reviewer_entry_paths"].values()):
        raise ValueError("publication_claim_bundle reviewer entry paths must exist")

    if bundle["summary"]["component_artifacts_complete"] != all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in bundle["component_artifacts"]
    ):
        raise ValueError(
            "publication_claim_bundle component_artifacts_complete summary is inconsistent"
        )

    if bundle["summary"]["file_coverage_complete"] != all(
        entry["exists"] for entry in bundle["file_coverage"]
    ):
        raise ValueError(
            "publication_claim_bundle file_coverage_complete summary is inconsistent"
        )

    if bundle["summary"]["terminology_complete"] != (
        not bundle["terminology_inventory"]["missing_glossary_terms"]
    ):
        raise ValueError(
            "publication_claim_bundle terminology_complete summary is inconsistent"
        )

    if bundle["summary"]["reviewer_entry_paths_complete"] != all(
        entry["exists"] for entry in bundle["reviewer_entry_paths"].values()
    ):
        raise ValueError(
            "publication_claim_bundle reviewer_entry_paths_complete summary is inconsistent"
        )

    expected_status = (
        "pass"
        if (
            bundle["summary"]["component_artifacts_complete"]
            and bundle["summary"]["file_coverage_complete"]
            and bundle["summary"]["terminology_complete"]
            and bundle["summary"]["reviewer_entry_paths_complete"]
        )
        else "fail"
    )
    if bundle["status"] != expected_status:
        raise ValueError("publication_claim_bundle status does not match bundle summary")


def write_publication_claim_bundle(output_path: Path, bundle):
    validate_publication_claim_bundle(bundle)
    write_json(output_path, bundle)


def run_validation(
    *,
    claim_package_path: Path = CLAIM_PACKAGE_PATH,
    publication_surface_alignment_path: Path = PUBLICATION_SURFACE_ALIGNMENT_PATH,
    claim_traceability_path: Path = CLAIM_TRACEABILITY_PATH,
    evidence_closure_path: Path = EVIDENCE_CLOSURE_PATH,
    supported_path_scope_path: Path = SUPPORTED_PATH_SCOPE_PATH,
    future_work_boundary_path: Path = FUTURE_WORK_BOUNDARY_PATH,
    verbose=False,
):
    claim_package_artifact = _load_claim_package(claim_package_path)
    publication_surface_alignment_artifact = _load_publication_surface_alignment(
        publication_surface_alignment_path
    )
    claim_traceability_artifact = _load_claim_traceability(claim_traceability_path)
    evidence_closure_artifact = _load_evidence_closure(evidence_closure_path)
    supported_path_scope_artifact = _load_supported_path_scope(
        supported_path_scope_path
    )
    future_work_boundary_artifact = _load_future_work_boundary(
        future_work_boundary_path
    )
    bundle = build_publication_claim_bundle(
        claim_package_artifact=claim_package_artifact,
        publication_surface_alignment_artifact=publication_surface_alignment_artifact,
        claim_traceability_artifact=claim_traceability_artifact,
        evidence_closure_artifact=evidence_closure_artifact,
        supported_path_scope_artifact=supported_path_scope_artifact,
        future_work_boundary_artifact=future_work_boundary_artifact,
    )
    if verbose:
        print(
            "{} [{}] component_artifacts_complete={} file_coverage_complete={} terminology_complete={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["component_artifacts_complete"],
                bundle["summary"]["file_coverage_complete"],
                bundle["summary"]["terminology_complete"],
            )
        )
    return (
        claim_package_artifact,
        publication_surface_alignment_artifact,
        claim_traceability_artifact,
        evidence_closure_artifact,
        supported_path_scope_artifact,
        future_work_boundary_artifact,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for publication_claim_bundle.json.",
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
    write_publication_claim_bundle(output_path, bundle)
    print(
        "Wrote {} with status {} ({}/{})".format(
            output_path,
            bundle["status"],
            bundle["summary"]["mandatory_component_count"],
            bundle["summary"]["mandatory_file_count"],
        )
    )
    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
