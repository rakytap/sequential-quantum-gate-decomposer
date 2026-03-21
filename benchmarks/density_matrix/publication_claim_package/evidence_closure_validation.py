#!/usr/bin/env python3
"""Validation: Evidence-closure semantics for Paper 1.

Builds a machine-readable evidence-closure checker for Paper 1. This layer is
intentionally thin:
- it reuses the claim-traceability artifact,
- it records the mandatory evidence-floor items used by publication-facing docs,
- it validates the explicit closure rule for the main Paper 1 claim,
- and it fails when optional or incomplete evidence can masquerade as closure.

Run with:
    python benchmarks/density_matrix/publication_claim_package/evidence_closure_validation.py
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
    MANDATORY_PUBLICATION_EVIDENCE_DOCS,
    PUBLICATION_CLAIM_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    missing_phrases,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.publication_claim_package.claim_traceability_bundle import (
    ARTIFACT_FILENAME as CLAIM_TRACEABILITY_ARTIFACT_FILENAME,
    run_validation as run_claim_traceability_validation,
    validate_artifact_bundle as validate_claim_traceability_artifact,
)


SUITE_NAME = "evidence_closure"
ARTIFACT_FILENAME = "evidence_closure.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
CLAIM_TRACEABILITY_PATH = DEFAULT_OUTPUT_DIR / CLAIM_TRACEABILITY_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "upstream_claim_traceability",
    "evidence_inventory",
    "software",
    "provenance",
    "summary",
)
MANDATORY_EVIDENCE_ITEMS = (
    {
        "item_id": "micro_validation",
        "label": "1 to 3 qubit micro-validation",
        "surface_phrases": {
            "phase2_short_paper": ["1 to 3 qubit micro-validation"],
            "phase2_paper": ["1 to 3 qubit micro-validation matrix"],
        },
    },
    {
        "item_id": "workflow_matrix",
        "label": "4 / 6 / 8 / 10 workflow matrix",
        "surface_phrases": {
            "phase2_short_paper": ["4 / 6 / 8 / 10 qubit", "10 parameter vectors"],
            "phase2_short_paper_narrative": [
                "4/6/8/10-qubit workflow regime",
                "1e-8",
            ],
            "phase2_paper": ["4/6/8/10-qubit workflow-scale", "10 fixed parameter vectors"],
        },
    },
    {
        "item_id": "optimization_trace",
        "label": "Bounded optimization trace",
        "surface_phrases": {
            "phase2_abstract": ["bounded 4-qubit optimization trace"],
            "phase2_short_paper": ["bounded 4-qubit optimization trace"],
            "phase2_short_paper_narrative": ["end-to-end optimization trace on a 4-qubit"],
            "phase2_paper": ["one bounded 4-qubit optimization trace"],
        },
    },
    {
        "item_id": "runtime_and_peak_memory",
        "label": "Runtime and peak-memory recording",
        "surface_phrases": {
            "phase2_short_paper": ["runtime and peak-memory"],
            "phase2_short_paper_narrative": ["Peak resident memory", "Per-evaluation runtime"],
            "phase2_paper": ["runtime and peak-memory"],
        },
    },
    {
        "item_id": "reproducibility_bundle",
        "label": "Workflow-facing reproducibility bundle",
        "surface_phrases": {
            "phase2_abstract": ["publication bundle"],
            "phase2_short_paper": ["machine-checkable publication manifest"],
            "phase2_paper": ["workflow-facing validation package archived in"],
        },
    },
    {
        "item_id": "claim_closure_rule",
        "label": "Claim closure rule",
        "surface_phrases": {
            "phase2_abstract": [EVIDENCE_CLOSURE_RULE],
            "phase2_short_paper": [EVIDENCE_CLOSURE_RULE],
            "phase2_short_paper_narrative": [EVIDENCE_CLOSURE_RULE],
            "phase2_paper": [EVIDENCE_CLOSURE_RULE],
        },
    },
)


def _load_claim_traceability(path: Path = CLAIM_TRACEABILITY_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_claim_traceability_artifact(artifact)
        return artifact
    return run_claim_traceability_validation(verbose=False)


def build_evidence_inventory():
    inventory = []
    for item in MANDATORY_EVIDENCE_ITEMS:
        surface_entries = []
        for doc_id, phrases in item["surface_phrases"].items():
            path = MANDATORY_PUBLICATION_EVIDENCE_DOCS[doc_id]
            text = load_text(path)
            surface_entries.append(
                {
                    "doc_id": doc_id,
                    "path": relative_to_repo(path),
                    "phrases": list(phrases),
                    "missing_phrases": missing_phrases(text, phrases),
                }
            )
        inventory.append(
            {
                "item_id": item["item_id"],
                "label": item["label"],
                "surface_entries": surface_entries,
            }
        )
    return inventory


def build_artifact_bundle():
    claim_traceability_artifact = _load_claim_traceability()
    evidence_inventory = build_evidence_inventory()
    all_evidence_items_present = all(
        all(not surface["missing_phrases"] for surface in item["surface_entries"])
        for item in evidence_inventory
    )
    claim_closure_rule_present = all(
        not surface["missing_phrases"]
        for item in evidence_inventory
        if item["item_id"] == "claim_closure_rule"
        for surface in item["surface_entries"]
    )
    evidence_closure_completed = all(
        [
            claim_traceability_artifact["status"] == "pass",
            all_evidence_items_present,
            claim_closure_rule_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if evidence_closure_completed else "fail",
        "upstream_claim_traceability": {
            "suite_name": claim_traceability_artifact["suite_name"],
            "status": claim_traceability_artifact["status"],
            "path": relative_to_repo(CLAIM_TRACEABILITY_PATH),
            "summary": dict(claim_traceability_artifact["summary"]),
        },
        "evidence_inventory": evidence_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_claim_package/"
                "evidence_closure_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "claim_traceability_path": str(CLAIM_TRACEABILITY_PATH),
        },
        "summary": {
            "required_evidence_count": len(MANDATORY_EVIDENCE_ITEMS),
            "all_evidence_items_present": all_evidence_items_present,
            "claim_closure_rule_present": claim_closure_rule_present,
            "evidence_closure_completed": evidence_closure_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "evidence_closure artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_item_ids = [item["item_id"] for item in artifact["evidence_inventory"]]
    required_item_ids = [item["item_id"] for item in MANDATORY_EVIDENCE_ITEMS]
    if observed_item_ids != required_item_ids:
        raise ValueError("evidence inventory mismatch")

    all_evidence_items_present = all(
        all(not surface["missing_phrases"] for surface in item["surface_entries"])
        for item in artifact["evidence_inventory"]
    )
    if artifact["summary"]["all_evidence_items_present"] != all_evidence_items_present:
        raise ValueError("all_evidence_items_present summary is inconsistent")

    claim_closure_rule_present = all(
        not surface["missing_phrases"]
        for item in artifact["evidence_inventory"]
        if item["item_id"] == "claim_closure_rule"
        for surface in item["surface_entries"]
    )
    if artifact["summary"]["claim_closure_rule_present"] != claim_closure_rule_present:
        raise ValueError("claim_closure_rule_present summary is inconsistent")

    expected_completed = all(
        [
            artifact["upstream_claim_traceability"]["status"] == "pass",
            artifact["summary"]["all_evidence_items_present"],
            artifact["summary"]["claim_closure_rule_present"],
        ]
    )
    if artifact["summary"]["evidence_closure_completed"] != expected_completed:
        raise ValueError("evidence_closure_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("evidence_closure status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] evidence_items={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["required_evidence_count"],
                artifact["summary"]["evidence_closure_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the evidence_closure JSON artifact.",
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
            artifact["summary"]["required_evidence_count"],
        )
    )
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
