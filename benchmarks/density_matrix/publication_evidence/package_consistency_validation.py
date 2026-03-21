#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 8 final package consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.common import (
    BOTTLENECK_PHRASE_ALTERNATIVES,
    CURRENT_DIAGNOSIS_PHRASE_ALTERNATIVES,
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    PUBLICATION_SURFACES,
    CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH,
    build_software_metadata,
    build_surface_presence,
    get_git_revision,
    load_json,
    load_or_build_artifact,
    load_text,
    normalize_text,
    relative_to_repo,
    correctness_evidence_boundary_phrase_options,
    correctness_evidence_count_phrase_options,
    performance_evidence_count_phrase_options,
    performance_evidence_review_phrase_options,
    publication_evidence_output_dir,
    write_json,
)
from benchmarks.density_matrix.publication_evidence.future_work_boundary_validation import (
    ARTIFACT_FILENAME as STORY7_ARTIFACT_FILENAME,
    run_validation as run_story7_validation,
    validate_artifact_bundle as validate_story7_artifact,
)
from benchmarks.density_matrix.publication_evidence.publication_manifest_validation import (
    ARTIFACT_FILENAME as STORY6_ARTIFACT_FILENAME,
    run_validation as run_story6_validation,
    validate_artifact_bundle as validate_story6_artifact,
)


SUITE_NAME = "phase3_publication_evidence_package_consistency"
ARTIFACT_FILENAME = "package_consistency_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("package_consistency")
STORY6_PATH = (
    publication_evidence_output_dir("manifest")
    / STORY6_ARTIFACT_FILENAME
)
STORY7_PATH = (
    publication_evidence_output_dir("future_work")
    / STORY7_ARTIFACT_FILENAME
)
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "manifest",
    "future_work",
    "reviewer_entry_paths",
    "terminology_inventory",
    "surface_inventory",
    "software",
    "provenance",
    "summary",
)

REQUIRED_GLOSSARY_TERMS = (
    "exact noisy mixed-state circuits",
    "canonical noisy mixed-state planner surface",
    "partitioned density runtime",
    "real fused path",
    "counted supported",
    "diagnosis-grounded",
    "future work",
    "Phase 4+ work",
)


def _story6():
    return load_or_build_artifact(
        STORY6_PATH,
        run_validation=run_story6_validation,
        validate_artifact_bundle=validate_story6_artifact,
    )


def _story7():
    return load_or_build_artifact(
        STORY7_PATH,
        run_validation=run_story7_validation,
        validate_artifact_bundle=validate_story7_artifact,
    )


def _required_groups():
    return [
        {
            "group_id": "correctness_evidence_count",
            "phrases": list(correctness_evidence_count_phrase_options()),
        },
        {
            "group_id": "correctness_evidence_boundary_count",
            "phrases": list(correctness_evidence_boundary_phrase_options()),
        },
        {
            "group_id": "performance_evidence_count",
            "phrases": list(performance_evidence_count_phrase_options()),
        },
        {
            "group_id": "performance_evidence_review_cases",
            "phrases": list(performance_evidence_review_phrase_options()),
        },
        {
            "group_id": "diagnosis_closure",
            "phrases": list(CURRENT_DIAGNOSIS_PHRASE_ALTERNATIVES),
        },
        {
            "group_id": "bottleneck_wording",
            "phrases": list(BOTTLENECK_PHRASE_ALTERNATIVES),
        },
    ]


def build_terminology_inventory():
    combined_text = "\n".join(
        load_text(MANDATORY_PUBLICATION_EVIDENCE_DOCS[surface["doc_id"]])
        for surface in PUBLICATION_SURFACES.values()
    )
    combined_text += "\n" + load_text(MANDATORY_PUBLICATION_EVIDENCE_DOCS["publication_evidence_mini_spec"])
    normalized = normalize_text(combined_text)
    return {
        "required_glossary_terms": list(REQUIRED_GLOSSARY_TERMS),
        "missing_glossary_terms": [
            term
            for term in REQUIRED_GLOSSARY_TERMS
            if normalize_text(term) not in normalized
        ],
    }


def build_surface_inventory():
    inventory = []
    for surface_id in PUBLICATION_SURFACES:
        entry = build_surface_presence(
            surface_id,
            required_phrase_groups=_required_groups(),
        )
        missing_ids = {
            item["group_id"] for item in entry["missing_required_phrase_groups"]
        }
        entry["count_references_present"] = all(
            group_id not in missing_ids
            for group_id in (
                "correctness_evidence_count",
                "correctness_evidence_boundary_count",
                "performance_evidence_count",
                "performance_evidence_review_cases",
            )
        )
        entry["diagnosis_limitations_present"] = all(
            group_id not in missing_ids
            for group_id in ("diagnosis_closure", "bottleneck_wording")
        )
        entry["status"] = (
            "pass"
            if not entry["missing_required_phrase_groups"]
            else "fail"
        )
        inventory.append(entry)
    return inventory


def build_artifact_bundle():
    story6_artifact = _story6()
    story7_artifact = _story7()
    terminology_inventory = build_terminology_inventory()
    surface_inventory = build_surface_inventory()
    reviewer_entry_paths_complete = all(
        entry["exists"] for entry in story6_artifact["reviewer_entry_paths"].values()
    )
    terminology_complete = not terminology_inventory["missing_glossary_terms"]
    count_consistency_complete = all(
        entry["count_references_present"] for entry in surface_inventory
    )
    limitation_summary_consistency_complete = all(
        entry["diagnosis_limitations_present"] for entry in surface_inventory
    )
    correctness_evidence_summary = load_json(CORRECTNESS_EVIDENCE_SUMMARY_CONSISTENCY_PATH)
    performance_evidence_summary = load_json(PERFORMANCE_EVIDENCE_SUMMARY_CONSISTENCY_PATH)
    package_consistency_completed = all(
        [
            story6_artifact["status"] == "pass",
            story7_artifact["status"] == "pass",
            reviewer_entry_paths_complete,
            terminology_complete,
            count_consistency_complete,
            limitation_summary_consistency_complete,
            correctness_evidence_summary["summary"]["summary_consistency_pass"],
            performance_evidence_summary["summary"]["summary_consistency_pass"],
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if package_consistency_completed else "fail",
        "manifest": {
            "suite_name": story6_artifact["suite_name"],
            "status": story6_artifact["status"],
            "path": relative_to_repo(STORY6_PATH),
            "summary": dict(story6_artifact["summary"]),
        },
        "future_work": {
            "suite_name": story7_artifact["suite_name"],
            "status": story7_artifact["status"],
            "path": relative_to_repo(STORY7_PATH),
            "summary": dict(story7_artifact["summary"]),
        },
        "reviewer_entry_paths": dict(story6_artifact["reviewer_entry_paths"]),
        "terminology_inventory": terminology_inventory,
        "surface_inventory": surface_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
                "package_consistency_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "story6_path": str(STORY6_PATH),
            "story7_path": str(STORY7_PATH),
        },
        "summary": {
            "surface_count": len(surface_inventory),
            "glossary_term_count": len(REQUIRED_GLOSSARY_TERMS),
            "reviewer_entry_paths_complete": reviewer_entry_paths_complete,
            "terminology_complete": terminology_complete,
            "count_consistency_complete": count_consistency_complete,
            "limitation_summary_consistency_complete": (
                limitation_summary_consistency_complete
            ),
            "correctness_evidence_summary_consistency_pass": bool(
                correctness_evidence_summary["summary"]["summary_consistency_pass"]
            ),
            "performance_evidence_summary_consistency_pass": bool(
                performance_evidence_summary["summary"]["summary_consistency_pass"]
            ),
            "package_consistency_completed": package_consistency_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 8 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    expected_surface_ids = list(PUBLICATION_SURFACES.keys())
    observed_surface_ids = [entry["surface_id"] for entry in artifact["surface_inventory"]]
    if observed_surface_ids != expected_surface_ids:
        raise ValueError("package consistency surface inventory mismatch")

    expected_flags = {
        "reviewer_entry_paths_complete": all(
            entry["exists"] for entry in artifact["reviewer_entry_paths"].values()
        ),
        "terminology_complete": not artifact["terminology_inventory"][
            "missing_glossary_terms"
        ],
        "count_consistency_complete": all(
            entry["count_references_present"] for entry in artifact["surface_inventory"]
        ),
        "limitation_summary_consistency_complete": all(
            entry["diagnosis_limitations_present"]
            for entry in artifact["surface_inventory"]
        ),
    }
    for field, value in expected_flags.items():
        if artifact["summary"][field] != value:
            raise ValueError(f"{field} summary is inconsistent")

    expected_completed = all(
        [
            artifact["manifest"]["status"] == "pass",
            artifact["future_work"]["status"] == "pass",
            artifact["summary"]["reviewer_entry_paths_complete"],
            artifact["summary"]["terminology_complete"],
            artifact["summary"]["count_consistency_complete"],
            artifact["summary"]["limitation_summary_consistency_complete"],
            artifact["summary"]["correctness_evidence_summary_consistency_pass"],
            artifact["summary"]["performance_evidence_summary_consistency_pass"],
        ]
    )
    if artifact["summary"]["package_consistency_completed"] != expected_completed:
        raise ValueError("package_consistency_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 8 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] surfaces={} glossary_terms={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["surface_count"],
                artifact["summary"]["glossary_term_count"],
                artifact["summary"]["package_consistency_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Phase 3 Task 8 Story 8 JSON artifact bundle.",
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
            artifact["summary"]["surface_count"],
            artifact["summary"]["glossary_term_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
