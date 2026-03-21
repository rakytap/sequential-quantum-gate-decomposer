#!/usr/bin/env python3
"""Validation: Claim-to-source traceability for publication surfaces."""

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
from benchmarks.density_matrix.publication_evidence.common import (
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH,
    PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH,
    build_software_metadata,
    get_git_revision,
    load_or_build_artifact,
    load_text,
    relative_to_repo,
    publication_evidence_output_dir,
    write_json,
)
from benchmarks.density_matrix.publication_evidence.surface_alignment_validation import (
    ARTIFACT_FILENAME as PUBLICATION_SURFACE_ALIGNMENT_ARTIFACT_FILENAME,
    run_validation as run_publication_surface_alignment_validation,
    validate_artifact_bundle as validate_publication_surface_alignment_artifact,
)


SUITE_NAME = "claim_traceability"
ARTIFACT_FILENAME = "claim_traceability_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("claim_traceability")
CLAIM_PACKAGE_BUNDLE_PATH = (
    publication_evidence_output_dir("claim_package") / CLAIM_PACKAGE_ARTIFACT_FILENAME
)
PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH = (
    publication_evidence_output_dir("publication_surface_alignment")
    / PUBLICATION_SURFACE_ALIGNMENT_ARTIFACT_FILENAME
)
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "upstream_reference_artifacts",
    "reviewer_entry_paths",
    "claim_traceability",
    "section_traceability",
    "software",
    "provenance",
    "summary",
)

CLAIM_TRACEABILITY_ITEMS = (
    {
        "claim_id": "paper2_main_claim",
        "label": "Paper 2 main claim",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["planning_publications", "phase3_paper"],
    },
    {
        "claim_id": "supported_path_and_no_fallback",
        "label": "Supported path and no-fallback boundary",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["planner_calibration_mini_spec", "phase3_short_paper"],
    },
    {
        "claim_id": "correctness_and_boundary_evidence",
        "label": "Correctness and unsupported-boundary evidence",
        "surface_ids": ["short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "correctness_evidence_correctness_package_bundle",
        "supporting_doc_ids": [
            "correctness_evidence_unsupported_boundary_bundle",
            "correctness_evidence_summary_consistency_bundle",
        ],
    },
    {
        "claim_id": "benchmark_and_diagnosis_surface",
        "label": "Benchmark and diagnosis evidence",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "performance_evidence_benchmark_package_bundle",
        "supporting_doc_ids": [
            "performance_evidence_diagnosis_bundle",
            "performance_evidence_summary_consistency_bundle",
        ],
    },
    {
        "claim_id": "future_work_handoff",
        "label": "Future-work and roadmap handoff",
        "surface_ids": ["abstract", "short_paper", "short_paper_narrative", "paper"],
        "primary_doc_id": "planning_publications",
        "supporting_doc_ids": ["phase3_paper", "phase3_short_paper_narrative"],
    },
)

SECTION_TRACEABILITY_ITEMS = (
    {
        "section_id": "paper_claim_boundary",
        "label": "Full-paper claim-boundary section",
        "doc_id": "phase3_paper",
        "required_heading": "## Paper 2 Claim Boundary",
        "primary_doc_id": "publication_evidence_mini_spec",
        "supporting_doc_ids": ["phase3_detailed_planning", "phase3_adrs"],
    },
    {
        "section_id": "paper_validation_methodology",
        "label": "Full-paper validation methodology section",
        "doc_id": "phase3_paper",
        "required_heading": "## 6. Validation Methodology",
        "primary_doc_id": "correctness_evidence_correctness_package_bundle",
        "supporting_doc_ids": ["correctness_evidence_summary_consistency_bundle", "correctness_evidence_mini_spec"],
    },
    {
        "section_id": "paper_benchmark_design",
        "label": "Full-paper benchmark design section",
        "doc_id": "phase3_paper",
        "required_heading": "## 7. Benchmark Design",
        "primary_doc_id": "performance_evidence_benchmark_package_bundle",
        "supporting_doc_ids": ["performance_evidence_summary_consistency_bundle", "performance_evidence_mini_spec"],
    },
    {
        "section_id": "short_paper_validation_surface",
        "label": "Technical short-paper validation surface",
        "doc_id": "phase3_short_paper",
        "required_heading": "## 4. Validation and Benchmark Surface",
        "primary_doc_id": "correctness_evidence_correctness_package_bundle",
        "supporting_doc_ids": ["performance_evidence_benchmark_package_bundle", "publication_evidence_mini_spec"],
    },
    {
        "section_id": "short_paper_follow_on_phases",
        "label": "Technical short-paper follow-on phases section",
        "doc_id": "phase3_short_paper",
        "required_heading": "## 8. Follow-On Phases",
        "primary_doc_id": "planning_publications",
        "supporting_doc_ids": ["phase3_detailed_planning", "phase3_paper"],
    },
)


def _load_claim_package_artifact(path: Path = CLAIM_PACKAGE_BUNDLE_PATH):
    return load_or_build_artifact(
        path,
        run_validation=run_claim_package_validation,
        validate_artifact_bundle=validate_claim_package_artifact,
    )


def _load_publication_surface_alignment_artifact(
    path: Path = PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH,
):
    return load_or_build_artifact(
        path,
        run_validation=run_publication_surface_alignment_validation,
        validate_artifact_bundle=validate_publication_surface_alignment_artifact,
    )


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
    claim_package_artifact = _load_claim_package_artifact()
    publication_surface_alignment_artifact = _load_publication_surface_alignment_artifact()
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
        MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"].exists()
        and CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH.exists()
        and PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH.exists()
    )
    claim_traceability_completed = all(
        [
            claim_package_artifact["status"] == "pass",
            publication_surface_alignment_artifact["status"] == "pass",
            all_claim_sources_exist,
            all_section_sources_exist,
            all_section_headings_present,
            reviewer_entry_paths_complete,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if claim_traceability_completed else "fail",
        "upstream_reference_artifacts": [
            {
                "artifact_id": claim_package_artifact["suite_name"],
                "path": relative_to_repo(CLAIM_PACKAGE_BUNDLE_PATH),
                "status": claim_package_artifact["status"],
                "summary": dict(claim_package_artifact["summary"]),
            },
            {
                "artifact_id": publication_surface_alignment_artifact["suite_name"],
                "path": relative_to_repo(PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH),
                "status": publication_surface_alignment_artifact["status"],
                "summary": dict(publication_surface_alignment_artifact["summary"]),
            },
        ],
        "reviewer_entry_paths": {
            "phase3_paper": {
                "path": relative_to_repo(MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"]),
                "exists": MANDATORY_PUBLICATION_EVIDENCE_DOCS["phase3_paper"].exists(),
            },
            "correctness_evidence_correctness_package": {
                "path": relative_to_repo(CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH),
                "exists": CORRECTNESS_EVIDENCE_CORRECTNESS_PACKAGE_PATH.exists(),
            },
            "performance_evidence_benchmark_package": {
                "path": relative_to_repo(PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH),
                "exists": PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_PATH.exists(),
            },
        },
        "claim_traceability": claim_traceability,
        "section_traceability": section_traceability,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
                "claim_traceability_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "claim_package_bundle_path": str(CLAIM_PACKAGE_BUNDLE_PATH),
            "publication_surface_alignment_bundle_path": str(
                PUBLICATION_SURFACE_ALIGNMENT_BUNDLE_PATH
            ),
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
            "claim_traceability artifact is missing required fields: {}".format(
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
            all(upstream["status"] == "pass" for upstream in artifact["upstream_reference_artifacts"]),
            artifact["summary"]["all_claim_sources_exist"],
            artifact["summary"]["all_section_sources_exist"],
            artifact["summary"]["all_section_headings_present"],
            artifact["summary"]["reviewer_entry_paths_complete"],
        ]
    )
    if artifact["summary"]["claim_traceability_completed"] != expected_completed:
        raise ValueError("claim_traceability_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("claim_traceability status does not match completion summary")


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
        help="Directory for the claim_traceability JSON artifact bundle.",
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
