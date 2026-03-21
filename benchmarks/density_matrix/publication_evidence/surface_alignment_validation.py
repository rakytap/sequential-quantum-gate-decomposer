#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 2 publication-surface alignment."""

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
from benchmarks.density_matrix.publication_evidence.common import (
    CLAIM_HEADING,
    EVIDENCE_CLOSURE_RULE,
    EVIDENCE_CLOSURE_RULE_ALTERNATIVES,
    NO_FALLBACK_RULE_ALTERNATIVES,
    NARRATIVE_MAIN_CLAIM,
    PAPER_MAIN_CLAIM,
    PHASE_POSITIONING_RULE_ALTERNATIVES,
    PUBLICATION_SURFACES,
    ROLE_HEADING,
    SUPPORTED_PATH_BOUNDARY_ALTERNATIVES,
    SURFACE_ROLE_PHRASES,
    build_software_metadata,
    build_surface_presence,
    get_git_revision,
    load_or_build_artifact,
    relative_to_repo,
    publication_evidence_output_dir,
    write_json,
)


SUITE_NAME = "phase3_publication_evidence_surface_alignment"
ARTIFACT_FILENAME = "publication_surface_alignment_bundle.json"
DEFAULT_OUTPUT_DIR = publication_evidence_output_dir("surface_alignment")
STORY1_ARTIFACT_PATH = publication_evidence_output_dir("claim_package") / STORY1_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "claim_package",
    "surface_inventory",
    "software",
    "provenance",
    "summary",
)

CORE_NON_CLAIM_GROUPS = {
    "channel_native_non_claim": {
        "group_id": "channel_native_non_claim",
        "phrases": [
            "fully channel-native or superoperator-native fused noisy blocks are follow-on work beyond the baseline Paper 2 claim",
            "fully channel-native fused noisy blocks are future work beyond the baseline Paper 2 claim",
            "fully channel-native fused noisy blocks are not part of the baseline Paper 2 claim",
            "fully channel-native or superoperator-native fused noisy blocks are outside the baseline Paper 2 claim",
        ],
    },
    "workflow_growth_non_claim": {
        "group_id": "workflow_growth_non_claim",
        "phrases": [
            "broader noisy VQE/VQA workflow growth, density-backend gradients, and optimizer-comparison studies remain Phase 4+ work",
            "broader noisy VQE/VQA workflow growth, density-backend gradients, and optimizer studies remain Phase 4+ work",
            "broader noisy VQE/VQA feature growth and density-backend gradients are Phase 4+ work",
            "density-backend gradients, optimizer-comparison studies, and broader noisy VQE/VQA workflow growth are Phase 4+ work",
        ],
    },
    "approximate_scaling_non_claim": {
        "group_id": "approximate_scaling_non_claim",
        "phrases": [
            "approximate scaling methods such as trajectories or MPDO-style approaches are outside the current Paper 2 claim",
            "approximate scaling methods remain later work beyond the current Paper 2 claim",
            "approximate scaling methods remain future branches rather than current Paper 2 results",
        ],
    },
    "circuit_parity_non_claim": {
        "group_id": "circuit_parity_non_claim",
        "phrases": [
            "full `qgd_Circuit` parity is not part of the baseline Paper 2 claim",
            "full direct `qgd_Circuit` parity is not a current Paper 2 claim",
        ],
    },
}


def _story1_artifact(path: Path = STORY1_ARTIFACT_PATH):
    return load_or_build_artifact(
        path,
        run_validation=run_story1_validation,
        validate_artifact_bundle=validate_story1_artifact,
    )


def _required_groups_for_surface(surface_id: str):
    main_claim = PAPER_MAIN_CLAIM if surface_id != "short_paper_narrative" else NARRATIVE_MAIN_CLAIM
    return [
        {
            "group_id": "role_phrase",
            "phrases": [SURFACE_ROLE_PHRASES[surface_id]],
        },
        {
            "group_id": "main_claim",
            "phrases": [main_claim],
        },
        {
            "group_id": "supported_path_boundary",
            "phrases": list(SUPPORTED_PATH_BOUNDARY_ALTERNATIVES),
        },
        {
            "group_id": "no_fallback_rule",
            "phrases": list(NO_FALLBACK_RULE_ALTERNATIVES),
        },
        {
            "group_id": "evidence_closure_rule",
            "phrases": list(EVIDENCE_CLOSURE_RULE_ALTERNATIVES),
        },
        {
            "group_id": "phase_positioning",
            "phrases": list(PHASE_POSITIONING_RULE_ALTERNATIVES),
        },
        CORE_NON_CLAIM_GROUPS["channel_native_non_claim"],
        CORE_NON_CLAIM_GROUPS["workflow_growth_non_claim"],
        CORE_NON_CLAIM_GROUPS["approximate_scaling_non_claim"],
        CORE_NON_CLAIM_GROUPS["circuit_parity_non_claim"],
    ]


def build_surface_inventory():
    surfaces = []
    for surface_id in PUBLICATION_SURFACES:
        entry = build_surface_presence(
            surface_id,
            required_headings=(ROLE_HEADING, CLAIM_HEADING),
            required_phrase_groups=_required_groups_for_surface(surface_id),
        )
        missing_ids = {
            item["group_id"] for item in entry["missing_required_phrase_groups"]
        }
        entry["role_present"] = "role_phrase" not in missing_ids
        entry["main_claim_present"] = "main_claim" not in missing_ids
        entry["supported_path_boundary_present"] = "supported_path_boundary" not in missing_ids
        entry["no_fallback_rule_present"] = "no_fallback_rule" not in missing_ids
        entry["evidence_closure_rule_present"] = "evidence_closure_rule" not in missing_ids
        entry["phase_positioning_present"] = "phase_positioning" not in missing_ids
        entry["core_non_claims_present"] = all(
            group_id not in missing_ids
            for group_id in (
                "channel_native_non_claim",
                "workflow_growth_non_claim",
                "approximate_scaling_non_claim",
                "circuit_parity_non_claim",
            )
        )
        entry["status"] = (
            "pass"
            if (
                not entry["missing_headings"]
                and not entry["missing_required_phrase_groups"]
                and not entry["present_forbidden_phrase_groups"]
            )
            else "fail"
        )
        surfaces.append(entry)
    return surfaces


def build_artifact_bundle():
    story1_artifact = _story1_artifact()
    surface_inventory = build_surface_inventory()
    all_surface_roles_present = all(entry["role_present"] for entry in surface_inventory)
    all_claim_headings_present = all(
        not entry["missing_headings"] for entry in surface_inventory
    )
    all_main_claims_present = all(
        entry["main_claim_present"] for entry in surface_inventory
    )
    all_non_claims_present = all(
        entry["core_non_claims_present"] for entry in surface_inventory
    )
    all_supported_path_present = all(
        entry["supported_path_boundary_present"] for entry in surface_inventory
    )
    all_no_fallback_present = all(
        entry["no_fallback_rule_present"] for entry in surface_inventory
    )
    all_evidence_rules_present = all(
        entry["evidence_closure_rule_present"] for entry in surface_inventory
    )
    all_phase_positioning_present = all(
        entry["phase_positioning_present"] for entry in surface_inventory
    )
    surface_alignment_completed = all(
        [
            story1_artifact["status"] == "pass",
            all_surface_roles_present,
            all_claim_headings_present,
            all_main_claims_present,
            all_non_claims_present,
            all_supported_path_present,
            all_no_fallback_present,
            all_evidence_rules_present,
            all_phase_positioning_present,
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
                "python benchmarks/density_matrix/publication_evidence/"
                "surface_alignment_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "claim_package_path": str(STORY1_ARTIFACT_PATH),
        },
        "summary": {
            "surface_count": len(surface_inventory),
            "all_surface_roles_present": all_surface_roles_present,
            "all_claim_headings_present": all_claim_headings_present,
            "all_main_claims_present": all_main_claims_present,
            "all_non_claims_present": all_non_claims_present,
            "all_supported_path_present": all_supported_path_present,
            "all_no_fallback_present": all_no_fallback_present,
            "all_evidence_rules_present": all_evidence_rules_present,
            "all_phase_positioning_present": all_phase_positioning_present,
            "surface_alignment_completed": surface_alignment_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 2 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_surface_ids = [entry["surface_id"] for entry in artifact["surface_inventory"]]
    required_surface_ids = list(PUBLICATION_SURFACES.keys())
    if observed_surface_ids != required_surface_ids:
        raise ValueError("publication surface inventory mismatch")

    expected_flags = {
        "all_surface_roles_present": all(
            entry["role_present"] for entry in artifact["surface_inventory"]
        ),
        "all_claim_headings_present": all(
            not entry["missing_headings"] for entry in artifact["surface_inventory"]
        ),
        "all_main_claims_present": all(
            entry["main_claim_present"] for entry in artifact["surface_inventory"]
        ),
        "all_non_claims_present": all(
            entry["core_non_claims_present"] for entry in artifact["surface_inventory"]
        ),
        "all_supported_path_present": all(
            entry["supported_path_boundary_present"]
            for entry in artifact["surface_inventory"]
        ),
        "all_no_fallback_present": all(
            entry["no_fallback_rule_present"] for entry in artifact["surface_inventory"]
        ),
        "all_evidence_rules_present": all(
            entry["evidence_closure_rule_present"]
            for entry in artifact["surface_inventory"]
        ),
        "all_phase_positioning_present": all(
            entry["phase_positioning_present"] for entry in artifact["surface_inventory"]
        ),
    }
    for field, value in expected_flags.items():
        if artifact["summary"][field] != value:
            raise ValueError(f"{field} summary is inconsistent")

    expected_completed = all(
        [
            artifact["claim_package"]["status"] == "pass",
            artifact["summary"]["all_surface_roles_present"],
            artifact["summary"]["all_claim_headings_present"],
            artifact["summary"]["all_main_claims_present"],
            artifact["summary"]["all_non_claims_present"],
            artifact["summary"]["all_supported_path_present"],
            artifact["summary"]["all_no_fallback_present"],
            artifact["summary"]["all_evidence_rules_present"],
            artifact["summary"]["all_phase_positioning_present"],
        ]
    )
    if artifact["summary"]["surface_alignment_completed"] != expected_completed:
        raise ValueError("surface_alignment_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 2 status does not match completion summary")


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
        help="Directory for the Phase 3 Task 8 Story 2 JSON artifact bundle.",
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
