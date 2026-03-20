#!/usr/bin/env python3
"""Validation: Phase 3 Task 8 Story 1 claim package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.common import (
    CLAIM_HEADING,
    EVIDENCE_CLOSURE_RULE,
    MANDATORY_TASK8_DOCS,
    PAPER_MAIN_CLAIM,
    ROLE_HEADING,
    SURFACE_ROLE_PHRASES,
    build_software_metadata,
    get_git_revision,
    load_text,
    missing_phrases,
    relative_to_repo,
    task8_story_output_dir,
    write_json,
)


SUITE_NAME = "phase3_task8_story1_claim_package"
ARTIFACT_FILENAME = "claim_package_bundle.json"
DEFAULT_OUTPUT_DIR = task8_story_output_dir("story1_claim_package")
PRIMARY_SURFACE_PATH = MANDATORY_TASK8_DOCS["phase3_paper"]
PRIMARY_SOURCE_PATH = MANDATORY_TASK8_DOCS["task8_mini_spec"]
PUBLICATION_STRATEGY_PATH = MANDATORY_TASK8_DOCS["planning_publications"]
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "primary_surface",
    "claim_package",
    "source_docs",
    "software",
    "provenance",
    "summary",
)

SUPPORTING_CLAIM_ITEMS = (
    {
        "claim_id": "canonical_planner_surface",
        "label": "Canonical noisy planner surface",
        "text": "a canonical noisy mixed-state planner surface aligned with the `NoisyCircuit` execution contract",
        "source_phrases": [
            "canonical noisy mixed-state planner surface",
        ],
    },
    {
        "claim_id": "semantic_preservation",
        "label": "Exact semantic preservation",
        "text": "exact semantic-preservation rules for gate/noise ordering, parameter routing, and partition execution",
        "source_phrases": [
            "preserving exact gate/noise semantics",
            "semantic-preservation",
            "parameter-routing",
        ],
    },
    {
        "claim_id": "runtime_and_real_fused_path",
        "label": "Executable runtime and real fused path",
        "text": "an executable partitioned density runtime with at least one real fused execution mode on eligible substructures",
        "source_phrases": [
            "executable partitioned density runtime",
            "real fused path",
        ],
    },
    {
        "claim_id": "bounded_task5_planner_result",
        "label": "Bounded benchmark-calibrated planning result",
        "text": "a benchmark-calibrated density-aware planning policy on a bounded candidate surface",
        "source_phrases": [
            "bounded Task 5 calibrated planner claim",
            "benchmark-calibrated planning result",
            "`max_partition_qubits`",
        ],
    },
    {
        "claim_id": "machine_reviewable_package",
        "label": "Machine-reviewable correctness and benchmark package",
        "text": "a machine-reviewable correctness, benchmark, and reproducibility package, with the current benchmark closure now occurring through the diagnosis branch of the phase-level threshold-or-diagnosis rule",
        "source_phrases": [
            "machine-reviewable correctness and unsupported-boundary package",
            "representative benchmark and diagnosis package",
            "diagnosis branch rather than the positive-threshold branch",
        ],
    },
)

NON_CLAIM_ITEMS = (
    {
        "claim_id": "no_channel_native_baseline_claim",
        "label": "Channel-native fused noisy blocks stay out of baseline scope",
        "text": "fully channel-native or superoperator-native fused noisy blocks are outside the baseline Paper 2 claim",
        "source_phrases": [
            "fully channel-native or superoperator-native fused noisy blocks",
            "future work rather than current Paper 2 results",
        ],
    },
    {
        "claim_id": "no_broader_workflow_or_optimizer_claim",
        "label": "Broader workflow and optimizer studies stay out of scope",
        "text": "broader noisy VQE/VQA workflow growth, density-backend gradients, and optimizer studies remain Phase 4+ work",
        "source_phrases": [
            "broader noisy VQE/VQA workflow growth",
            "density-backend gradients",
            "optimizer studies",
        ],
    },
    {
        "claim_id": "no_approximate_scaling_claim",
        "label": "Approximate scaling stays out of scope",
        "text": "approximate scaling methods such as trajectories or MPDO-style approaches are outside the current Paper 2 claim",
        "source_phrases": [
            "approximate scaling methods",
            "future work rather than current Paper 2 results",
        ],
    },
    {
        "claim_id": "no_full_circuit_parity_claim",
        "label": "Full qgd_Circuit parity stays out of scope",
        "text": "full `qgd_Circuit` parity is not part of the baseline Paper 2 claim",
        "source_phrases": [
            "full direct `qgd_Circuit` parity",
            "future work rather than current Paper 2 results",
        ],
    },
    {
        "claim_id": "no_universal_speedup_claim",
        "label": "Universal speedup stays out of scope",
        "text": "universal speedup across every noisy workload is not required for Paper 2",
        "source_phrases": [
            "threshold-or-diagnosis interpretation",
            "generic speedup story",
        ],
    },
)


def _build_item_entries(items):
    source_text = load_text(PRIMARY_SOURCE_PATH)
    paper_text = load_text(PRIMARY_SURFACE_PATH)
    entries = []
    for item in items:
        entries.append(
            {
                "claim_id": item["claim_id"],
                "label": item["label"],
                "text": item["text"],
                "source_path": relative_to_repo(PRIMARY_SOURCE_PATH),
                "paper_path": relative_to_repo(PRIMARY_SURFACE_PATH),
                "source_phrases": list(item["source_phrases"]),
                "missing_source_phrases": missing_phrases(
                    source_text, item["source_phrases"]
                ),
                "missing_paper_phrases": missing_phrases(paper_text, [item["text"]]),
            }
        )
    return entries


def build_artifact_bundle():
    paper_text = load_text(PRIMARY_SURFACE_PATH)
    source_text = load_text(PRIMARY_SOURCE_PATH)
    supporting_claims = _build_item_entries(SUPPORTING_CLAIM_ITEMS)
    non_claims = _build_item_entries(NON_CLAIM_ITEMS)
    main_claim_missing_in_source = missing_phrases(
        source_text,
        [
            "stable main-claim boundary",
            "preserving exact gate/noise semantics",
            "representative noisy workloads",
        ],
    )
    main_claim_missing_in_paper = missing_phrases(paper_text, [PAPER_MAIN_CLAIM])
    evidence_rule_missing = missing_phrases(paper_text, [EVIDENCE_CLOSURE_RULE])

    all_source_phrases_present = (
        not main_claim_missing_in_source
        and all(not item["missing_source_phrases"] for item in supporting_claims)
        and all(not item["missing_source_phrases"] for item in non_claims)
    )
    all_supporting_claims_present = all(
        not item["missing_paper_phrases"] for item in supporting_claims
    )
    all_non_claims_present = all(not item["missing_paper_phrases"] for item in non_claims)
    main_claim_present_in_primary_surface = not main_claim_missing_in_paper
    evidence_closure_rule_present = not evidence_rule_missing
    primary_surface_has_role_heading = ROLE_HEADING in paper_text
    primary_surface_has_claim_heading = CLAIM_HEADING in paper_text
    primary_surface_has_role_phrase = not missing_phrases(
        paper_text, [SURFACE_ROLE_PHRASES["paper"]]
    )
    claim_package_completed = all(
        [
            primary_surface_has_role_heading,
            primary_surface_has_claim_heading,
            primary_surface_has_role_phrase,
            main_claim_present_in_primary_surface,
            all_supporting_claims_present,
            all_non_claims_present,
            evidence_closure_rule_present,
            all_source_phrases_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if claim_package_completed else "fail",
        "primary_surface": {
            "path": relative_to_repo(PRIMARY_SURFACE_PATH),
            "role_heading": ROLE_HEADING,
            "role_heading_present": primary_surface_has_role_heading,
            "claim_heading": CLAIM_HEADING,
            "claim_heading_present": primary_surface_has_claim_heading,
            "role_phrase": SURFACE_ROLE_PHRASES["paper"],
            "role_phrase_present": primary_surface_has_role_phrase,
        },
        "claim_package": {
            "main_claim": {
                "text": PAPER_MAIN_CLAIM,
                "source_path": relative_to_repo(PRIMARY_SOURCE_PATH),
                "paper_path": relative_to_repo(PRIMARY_SURFACE_PATH),
                "missing_source_phrases": main_claim_missing_in_source,
                "missing_paper_phrases": main_claim_missing_in_paper,
            },
            "supporting_claims": supporting_claims,
            "explicit_non_claims": non_claims,
            "evidence_closure_rule": {
                "text": EVIDENCE_CLOSURE_RULE,
                "paper_path": relative_to_repo(PRIMARY_SURFACE_PATH),
                "missing_paper_phrases": evidence_rule_missing,
            },
        },
        "source_docs": [
            {
                "doc_id": "task8_mini_spec",
                "path": relative_to_repo(PRIMARY_SOURCE_PATH),
                "exists": PRIMARY_SOURCE_PATH.exists(),
            },
            {
                "doc_id": "planning_publications",
                "path": relative_to_repo(PUBLICATION_STRATEGY_PATH),
                "exists": PUBLICATION_STRATEGY_PATH.exists(),
            },
            {
                "doc_id": "phase3_paper",
                "path": relative_to_repo(PRIMARY_SURFACE_PATH),
                "exists": PRIMARY_SURFACE_PATH.exists(),
            },
        ],
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_evidence/"
                "claim_package_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "primary_surface_path": str(PRIMARY_SURFACE_PATH),
            "primary_source_path": str(PRIMARY_SOURCE_PATH),
        },
        "summary": {
            "supporting_claim_count": len(SUPPORTING_CLAIM_ITEMS),
            "non_claim_count": len(NON_CLAIM_ITEMS),
            "primary_surface_has_role_heading": primary_surface_has_role_heading,
            "primary_surface_has_claim_heading": primary_surface_has_claim_heading,
            "primary_surface_has_role_phrase": primary_surface_has_role_phrase,
            "main_claim_present_in_primary_surface": (
                main_claim_present_in_primary_surface
            ),
            "all_supporting_claims_present": all_supporting_claims_present,
            "all_non_claims_present": all_non_claims_present,
            "evidence_closure_rule_present": evidence_closure_rule_present,
            "all_source_phrases_present": all_source_phrases_present,
            "claim_package_completed": claim_package_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "Phase 3 Task 8 Story 1 artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    supporting_claim_ids = [
        item["claim_id"] for item in artifact["claim_package"]["supporting_claims"]
    ]
    required_supporting_ids = [item["claim_id"] for item in SUPPORTING_CLAIM_ITEMS]
    if supporting_claim_ids != required_supporting_ids:
        raise ValueError("supporting claim inventory mismatch")

    non_claim_ids = [
        item["claim_id"] for item in artifact["claim_package"]["explicit_non_claims"]
    ]
    required_non_claim_ids = [item["claim_id"] for item in NON_CLAIM_ITEMS]
    if non_claim_ids != required_non_claim_ids:
        raise ValueError("explicit non-claim inventory mismatch")

    all_supporting_claims_present = all(
        not item["missing_paper_phrases"]
        for item in artifact["claim_package"]["supporting_claims"]
    )
    if artifact["summary"]["all_supporting_claims_present"] != all_supporting_claims_present:
        raise ValueError("all_supporting_claims_present summary is inconsistent")

    all_non_claims_present = all(
        not item["missing_paper_phrases"]
        for item in artifact["claim_package"]["explicit_non_claims"]
    )
    if artifact["summary"]["all_non_claims_present"] != all_non_claims_present:
        raise ValueError("all_non_claims_present summary is inconsistent")

    all_source_phrases_present = (
        not artifact["claim_package"]["main_claim"]["missing_source_phrases"]
        and all(
            not item["missing_source_phrases"]
            for item in artifact["claim_package"]["supporting_claims"]
        )
        and all(
            not item["missing_source_phrases"]
            for item in artifact["claim_package"]["explicit_non_claims"]
        )
    )
    if artifact["summary"]["all_source_phrases_present"] != all_source_phrases_present:
        raise ValueError("all_source_phrases_present summary is inconsistent")

    main_claim_present = not artifact["claim_package"]["main_claim"]["missing_paper_phrases"]
    if artifact["summary"]["main_claim_present_in_primary_surface"] != main_claim_present:
        raise ValueError("main_claim_present_in_primary_surface summary is inconsistent")

    evidence_rule_present = not artifact["claim_package"]["evidence_closure_rule"][
        "missing_paper_phrases"
    ]
    if artifact["summary"]["evidence_closure_rule_present"] != evidence_rule_present:
        raise ValueError("evidence_closure_rule_present summary is inconsistent")

    expected_completed = all(
        [
            artifact["summary"]["primary_surface_has_role_heading"],
            artifact["summary"]["primary_surface_has_claim_heading"],
            artifact["summary"]["primary_surface_has_role_phrase"],
            artifact["summary"]["main_claim_present_in_primary_surface"],
            artifact["summary"]["all_supporting_claims_present"],
            artifact["summary"]["all_non_claims_present"],
            artifact["summary"]["evidence_closure_rule_present"],
            artifact["summary"]["all_source_phrases_present"],
        ]
    )
    if artifact["summary"]["claim_package_completed"] != expected_completed:
        raise ValueError("claim_package_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("Phase 3 Task 8 Story 1 status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] supporting_claims={} non_claims={} claim_package_completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["supporting_claim_count"],
                artifact["summary"]["non_claim_count"],
                artifact["summary"]["claim_package_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the Phase 3 Task 8 Story 1 JSON artifact bundle.",
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
            artifact["summary"]["supporting_claim_count"],
            artifact["summary"]["non_claim_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
