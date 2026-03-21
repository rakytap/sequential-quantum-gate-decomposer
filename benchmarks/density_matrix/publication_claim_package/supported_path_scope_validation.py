#!/usr/bin/env python3
"""Validation: Supported-path and exact-regime scope honesty for Paper 1.

Builds a machine-readable scope checker for the supported Paper 1 path. This
layer is intentionally thin:
- it reuses the evidence-closure artifact,
- it records the mandatory supported-path and exact-regime statements,
- it validates that publication-facing surfaces keep the path narrow and honest,
- and it fails when broader capability or broader scale is implied.

Run with:
    python benchmarks/density_matrix/publication_claim_package/supported_path_scope_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_claim_package.doc_utils import (
    CANONICAL_MAIN_CLAIM,
    CANONICAL_NON_CLAIMS,
    EXACT_REGIME_BOUNDARY,
    NO_FALLBACK_RULE,
    PUBLICATION_SURFACES,
    SUPPORTED_PATH_BOUNDARY,
    PUBLICATION_CLAIM_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    build_surface_presence,
    load_json,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.publication_claim_package.evidence_closure_validation import (
    ARTIFACT_FILENAME as EVIDENCE_CLOSURE_ARTIFACT_FILENAME,
    run_validation as run_evidence_closure_validation,
    validate_artifact_bundle as validate_evidence_closure_artifact,
)


SUITE_NAME = "supported_path_scope"
ARTIFACT_FILENAME = "supported_path_scope.json"
DEFAULT_OUTPUT_DIR = PUBLICATION_CLAIM_OUTPUT_DIR
EVIDENCE_CLOSURE_PATH = DEFAULT_OUTPUT_DIR / EVIDENCE_CLOSURE_ARTIFACT_FILENAME
ARTIFACT_FIELDS = (
    "suite_name",
    "status",
    "upstream_evidence_closure",
    "surface_inventory",
    "software",
    "provenance",
    "summary",
)
MANDATORY_SCOPE_PHRASES = (
    CANONICAL_MAIN_CLAIM,
    NO_FALLBACK_RULE,
    SUPPORTED_PATH_BOUNDARY,
    EXACT_REGIME_BOUNDARY,
    CANONICAL_NON_CLAIMS[3],
)


def _load_evidence_closure(path: Path = EVIDENCE_CLOSURE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_evidence_closure_artifact(artifact)
        return artifact
    return run_evidence_closure_validation(verbose=False)


def build_surface_inventory():
    inventory = []
    for surface_id in PUBLICATION_SURFACES:
        entry = build_surface_presence(
            surface_id,
            required_phrases=MANDATORY_SCOPE_PHRASES,
        )
        entry["main_claim_present"] = CANONICAL_MAIN_CLAIM not in entry[
            "missing_required_phrases"
        ]
        entry["no_fallback_present"] = NO_FALLBACK_RULE not in entry[
            "missing_required_phrases"
        ]
        entry["supported_path_boundary_present"] = SUPPORTED_PATH_BOUNDARY not in entry[
            "missing_required_phrases"
        ]
        entry["exact_regime_boundary_present"] = EXACT_REGIME_BOUNDARY not in entry[
            "missing_required_phrases"
        ]
        entry["parity_non_claim_present"] = CANONICAL_NON_CLAIMS[3] not in entry[
            "missing_required_phrases"
        ]
        entry["status"] = (
            "pass"
            if (
                entry["main_claim_present"]
                and entry["no_fallback_present"]
                and entry["supported_path_boundary_present"]
                and entry["exact_regime_boundary_present"]
                and entry["parity_non_claim_present"]
            )
            else "fail"
        )
        inventory.append(entry)
    return inventory


def build_artifact_bundle():
    evidence_closure_artifact = _load_evidence_closure()
    surface_inventory = build_surface_inventory()
    all_main_claims_present = all(
        entry["main_claim_present"] for entry in surface_inventory
    )
    all_no_fallback_present = all(
        entry["no_fallback_present"] for entry in surface_inventory
    )
    all_supported_path_boundaries_present = all(
        entry["supported_path_boundary_present"] for entry in surface_inventory
    )
    all_exact_regime_boundaries_present = all(
        entry["exact_regime_boundary_present"] for entry in surface_inventory
    )
    all_parity_non_claims_present = all(
        entry["parity_non_claim_present"] for entry in surface_inventory
    )
    supported_path_scope_completed = all(
        [
            evidence_closure_artifact["status"] == "pass",
            all_main_claims_present,
            all_no_fallback_present,
            all_supported_path_boundaries_present,
            all_exact_regime_boundaries_present,
            all_parity_non_claims_present,
        ]
    )

    artifact = {
        "suite_name": SUITE_NAME,
        "status": "pass" if supported_path_scope_completed else "fail",
        "upstream_evidence_closure": {
            "suite_name": evidence_closure_artifact["suite_name"],
            "status": evidence_closure_artifact["status"],
            "path": relative_to_repo(EVIDENCE_CLOSURE_PATH),
            "summary": dict(evidence_closure_artifact["summary"]),
        },
        "surface_inventory": surface_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/publication_claim_package/"
                "supported_path_scope_validation.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "evidence_closure_path": str(EVIDENCE_CLOSURE_PATH),
        },
        "summary": {
            "surface_count": len(surface_inventory),
            "all_main_claims_present": all_main_claims_present,
            "all_no_fallback_present": all_no_fallback_present,
            "all_supported_path_boundaries_present": (
                all_supported_path_boundaries_present
            ),
            "all_exact_regime_boundaries_present": (
                all_exact_regime_boundaries_present
            ),
            "all_parity_non_claims_present": all_parity_non_claims_present,
            "supported_path_scope_completed": supported_path_scope_completed,
        },
    }
    validate_artifact_bundle(artifact)
    return artifact


def validate_artifact_bundle(artifact):
    missing_fields = [field for field in ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(
            "supported_path_scope artifact is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    observed_surface_ids = [entry["surface_id"] for entry in artifact["surface_inventory"]]
    required_surface_ids = list(PUBLICATION_SURFACES.keys())
    if observed_surface_ids != required_surface_ids:
        raise ValueError("supported-path surface inventory mismatch")

    expected_flags = {
        "all_main_claims_present": all(
            entry["main_claim_present"] for entry in artifact["surface_inventory"]
        ),
        "all_no_fallback_present": all(
            entry["no_fallback_present"] for entry in artifact["surface_inventory"]
        ),
        "all_supported_path_boundaries_present": all(
            entry["supported_path_boundary_present"]
            for entry in artifact["surface_inventory"]
        ),
        "all_exact_regime_boundaries_present": all(
            entry["exact_regime_boundary_present"]
            for entry in artifact["surface_inventory"]
        ),
        "all_parity_non_claims_present": all(
            entry["parity_non_claim_present"] for entry in artifact["surface_inventory"]
        ),
    }
    for field, value in expected_flags.items():
        if artifact["summary"][field] != value:
            raise ValueError(f"{field} summary is inconsistent")

    expected_completed = all(
        [
            artifact["upstream_evidence_closure"]["status"] == "pass",
            artifact["summary"]["all_main_claims_present"],
            artifact["summary"]["all_no_fallback_present"],
            artifact["summary"]["all_supported_path_boundaries_present"],
            artifact["summary"]["all_exact_regime_boundaries_present"],
            artifact["summary"]["all_parity_non_claims_present"],
        ]
    )
    if artifact["summary"]["supported_path_scope_completed"] != expected_completed:
        raise ValueError("supported_path_scope_completed summary is inconsistent")
    if artifact["status"] != ("pass" if expected_completed else "fail"):
        raise ValueError("supported_path_scope status does not match completion summary")


def write_artifact_bundle(output_path: Path, artifact):
    validate_artifact_bundle(artifact)
    write_json(output_path, artifact)


def run_validation(*, verbose=False):
    artifact = build_artifact_bundle()
    if verbose:
        print(
            "{} [{}] surfaces={} completed={}".format(
                artifact["suite_name"],
                artifact["status"],
                artifact["summary"]["surface_count"],
                artifact["summary"]["supported_path_scope_completed"],
            )
        )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the supported_path_scope JSON artifact.",
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
