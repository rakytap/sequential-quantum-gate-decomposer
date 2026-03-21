#!/usr/bin/env python3
"""Validation: Top-level documentation contract bundle.

Assembles contract_reference_map through future_work_boundary into one
machine-checkable documentation surface. Preserves the shared entry-point path,
validates lower-layer semantic gates, records mandatory file coverage, and
checks the canonical Phase 2 terminology inventory.

Run with:
    python benchmarks/density_matrix/documentation_contract/documentation_contract_bundle.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.documentation_contract.doc_utils import (
    MANDATORY_PHASE2_DOCS,
    PHASE2_DOCUMENTATION_INDEX_PATH,
    DOCUMENTATION_CONTRACT_OUTPUT_DIR,
    build_software_metadata,
    get_git_revision,
    load_json,
    load_text,
    normalize_text,
    relative_to_repo,
    write_json,
)
from benchmarks.density_matrix.documentation_contract.contract_reference_validation import (
    ARTIFACT_FILENAME as CONTRACT_REFERENCE_MAP_FILENAME,
    run_validation as run_contract_reference_map_validation,
    validate_artifact_bundle as validate_contract_reference_map_artifact,
)
from benchmarks.density_matrix.documentation_contract.supported_entry_reference_validation import (
    ARTIFACT_FILENAME as SUPPORTED_ENTRY_REFERENCE_FILENAME,
    run_validation as run_supported_entry_reference_validation,
    validate_artifact_bundle as validate_supported_entry_reference_artifact,
)
from benchmarks.density_matrix.documentation_contract.support_surface_reference_validation import (
    ARTIFACT_FILENAME as SUPPORT_SURFACE_REFERENCE_FILENAME,
    run_validation as run_support_surface_reference_validation,
    validate_artifact_bundle as validate_support_surface_reference_artifact,
)
from benchmarks.density_matrix.documentation_contract.evidence_bar_validation import (
    ARTIFACT_FILENAME as EVIDENCE_BAR_REFERENCE_FILENAME,
    run_validation as run_evidence_bar_reference_validation,
    validate_artifact_bundle as validate_evidence_bar_reference_artifact,
)
from benchmarks.density_matrix.documentation_contract.future_work_boundary_validation import (
    ARTIFACT_FILENAME as FUTURE_WORK_BOUNDARY_FILENAME,
    run_validation as run_future_work_boundary_validation,
    validate_artifact_bundle as validate_future_work_boundary_artifact,
)


SUITE_NAME = "documentation_contract_bundle"
ARTIFACT_FILENAME = "documentation_contract_bundle.json"
DEFAULT_OUTPUT_DIR = DOCUMENTATION_CONTRACT_OUTPUT_DIR
CONTRACT_REFERENCE_MAP_PATH = DEFAULT_OUTPUT_DIR / CONTRACT_REFERENCE_MAP_FILENAME
SUPPORTED_ENTRY_REFERENCE_PATH = DEFAULT_OUTPUT_DIR / SUPPORTED_ENTRY_REFERENCE_FILENAME
SUPPORT_SURFACE_REFERENCE_PATH = DEFAULT_OUTPUT_DIR / SUPPORT_SURFACE_REFERENCE_FILENAME
EVIDENCE_BAR_REFERENCE_PATH = DEFAULT_OUTPUT_DIR / EVIDENCE_BAR_REFERENCE_FILENAME
FUTURE_WORK_BOUNDARY_PATH = DEFAULT_OUTPUT_DIR / FUTURE_WORK_BOUNDARY_FILENAME
COMPONENT_ARTIFACT_REQUIREMENTS = (
    {
        "artifact_id": "contract_reference_map",
        "path": CONTRACT_REFERENCE_MAP_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "contract_reference_map_completed",
    },
    {
        "artifact_id": "supported_entry_reference",
        "path": SUPPORTED_ENTRY_REFERENCE_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "supported_entry_reference_completed",
    },
    {
        "artifact_id": "support_surface_reference",
        "path": SUPPORT_SURFACE_REFERENCE_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "support_surface_reference_completed",
    },
    {
        "artifact_id": "evidence_bar_reference",
        "path": EVIDENCE_BAR_REFERENCE_PATH,
        "expected_statuses": ("pass",),
        "semantic_flag": "evidence_bar_reference_completed",
    },
    {
        "artifact_id": "future_work_boundary",
        "path": FUTURE_WORK_BOUNDARY_PATH,
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
    "component_artifacts",
    "file_coverage",
    "terminology_inventory",
    "software",
    "provenance",
    "summary",
)


def _load_contract_reference_map(path: Path = CONTRACT_REFERENCE_MAP_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_contract_reference_map_artifact(artifact)
        return artifact
    return run_contract_reference_map_validation(verbose=False)


def _load_supported_entry_reference(path: Path = SUPPORTED_ENTRY_REFERENCE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_supported_entry_reference_artifact(artifact)
        return artifact
    return run_supported_entry_reference_validation(verbose=False)


def _load_support_surface_reference(path: Path = SUPPORT_SURFACE_REFERENCE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_support_surface_reference_artifact(artifact)
        return artifact
    return run_support_surface_reference_validation(verbose=False)


def _load_evidence_bar_reference(path: Path = EVIDENCE_BAR_REFERENCE_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_evidence_bar_reference_artifact(artifact)
        return artifact
    return run_evidence_bar_reference_validation(verbose=False)


def _load_future_work_boundary(path: Path = FUTURE_WORK_BOUNDARY_PATH):
    if path.exists():
        artifact = load_json(path)
        validate_future_work_boundary_artifact(artifact)
        return artifact
    return run_future_work_boundary_validation(verbose=False)


def build_component_artifact_entries(
    *,
    contract_reference_map_artifact,
    supported_entry_reference_artifact,
    support_surface_reference_artifact,
    evidence_bar_reference_artifact,
    future_work_boundary_artifact,
):
    artifact_map = {
        "contract_reference_map": contract_reference_map_artifact,
        "supported_entry_reference": supported_entry_reference_artifact,
        "support_surface_reference": support_surface_reference_artifact,
        "evidence_bar_reference": evidence_bar_reference_artifact,
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


def build_documentation_contract_bundle(
    output_dir: Path,
    *,
    contract_reference_map_artifact,
    supported_entry_reference_artifact,
    support_surface_reference_artifact,
    evidence_bar_reference_artifact,
    future_work_boundary_artifact,
):
    component_artifacts = build_component_artifact_entries(
        contract_reference_map_artifact=contract_reference_map_artifact,
        supported_entry_reference_artifact=supported_entry_reference_artifact,
        support_surface_reference_artifact=support_surface_reference_artifact,
        evidence_bar_reference_artifact=evidence_bar_reference_artifact,
        future_work_boundary_artifact=future_work_boundary_artifact,
    )
    file_coverage = build_file_coverage()
    terminology_inventory = build_terminology_inventory()

    component_artifacts_complete = all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in component_artifacts
    )
    file_coverage_complete = all(entry["exists"] for entry in file_coverage)
    glossary_complete = not terminology_inventory["missing_glossary_terms"]

    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass"
        if component_artifacts_complete and file_coverage_complete and glossary_complete
        else "fail",
        "entry_point": {
            "path": relative_to_repo(PHASE2_DOCUMENTATION_INDEX_PATH),
            "exists": PHASE2_DOCUMENTATION_INDEX_PATH.exists(),
        },
        "component_artifacts": component_artifacts,
        "file_coverage": file_coverage,
        "terminology_inventory": terminology_inventory,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": (
                "python benchmarks/density_matrix/documentation_contract/documentation_contract_bundle.py"
            ),
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
            "contract_reference_map_path": str(CONTRACT_REFERENCE_MAP_PATH),
            "supported_entry_reference_path": str(SUPPORTED_ENTRY_REFERENCE_PATH),
            "support_surface_reference_path": str(SUPPORT_SURFACE_REFERENCE_PATH),
            "evidence_bar_reference_path": str(EVIDENCE_BAR_REFERENCE_PATH),
            "future_work_boundary_path": str(FUTURE_WORK_BOUNDARY_PATH),
        },
        "summary": {
            "mandatory_component_count": len(component_artifacts),
            "component_artifacts_complete": component_artifacts_complete,
            "mandatory_file_count": len(file_coverage),
            "file_coverage_complete": file_coverage_complete,
            "glossary_term_count": len(REQUIRED_GLOSSARY_TERMS),
            "glossary_complete": glossary_complete,
        },
    }
    validate_documentation_contract_bundle(bundle)
    return bundle


def validate_documentation_contract_bundle(bundle):
    missing_fields = [field for field in BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "documentation_contract_bundle is missing required fields: {}".format(
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
            "documentation_contract_bundle is missing required component artifact IDs: {}".format(
                ", ".join(sorted(required_artifact_ids - observed_artifact_ids))
            )
        )

    if not bundle["entry_point"]["exists"]:
        raise ValueError("documentation_contract_bundle entry point must exist")

    if bundle["summary"]["component_artifacts_complete"] != all(
        artifact["status"] in artifact["expected_statuses"]
        and artifact["semantic_flag_passed"]
        for artifact in bundle["component_artifacts"]
    ):
        raise ValueError(
            "documentation_contract_bundle component_artifacts_complete summary is inconsistent"
        )

    if bundle["summary"]["file_coverage_complete"] != all(
        entry["exists"] for entry in bundle["file_coverage"]
    ):
        raise ValueError(
            "documentation_contract_bundle file_coverage_complete summary is inconsistent"
        )

    if bundle["summary"]["glossary_complete"] != (
        len(bundle["terminology_inventory"]["missing_glossary_terms"]) == 0
    ):
        raise ValueError(
            "documentation_contract_bundle glossary_complete summary is inconsistent"
        )

    if bundle["status"] != "pass" and (
        bundle["summary"]["component_artifacts_complete"]
        and bundle["summary"]["file_coverage_complete"]
        and bundle["summary"]["glossary_complete"]
    ):
        raise ValueError("documentation_contract_bundle status is inconsistent")


def write_documentation_contract_bundle(output_path: Path, bundle):
    validate_documentation_contract_bundle(bundle)
    write_json(output_path, bundle)


def run_validation(
    *,
    contract_reference_map_path: Path = CONTRACT_REFERENCE_MAP_PATH,
    supported_entry_reference_path: Path = SUPPORTED_ENTRY_REFERENCE_PATH,
    support_surface_reference_path: Path = SUPPORT_SURFACE_REFERENCE_PATH,
    evidence_bar_reference_path: Path = EVIDENCE_BAR_REFERENCE_PATH,
    future_work_boundary_path: Path = FUTURE_WORK_BOUNDARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_reference_map_artifact = _load_contract_reference_map(
        contract_reference_map_path
    )
    supported_entry_reference_artifact = _load_supported_entry_reference(
        supported_entry_reference_path
    )
    support_surface_reference_artifact = _load_support_surface_reference(
        support_surface_reference_path
    )
    evidence_bar_reference_artifact = _load_evidence_bar_reference(
        evidence_bar_reference_path
    )
    future_work_boundary_artifact = _load_future_work_boundary(
        future_work_boundary_path
    )
    write_json(output_dir / CONTRACT_REFERENCE_MAP_FILENAME, contract_reference_map_artifact)
    write_json(
        output_dir / SUPPORTED_ENTRY_REFERENCE_FILENAME,
        supported_entry_reference_artifact,
    )
    write_json(
        output_dir / SUPPORT_SURFACE_REFERENCE_FILENAME,
        support_surface_reference_artifact,
    )
    write_json(
        output_dir / EVIDENCE_BAR_REFERENCE_FILENAME,
        evidence_bar_reference_artifact,
    )
    write_json(
        output_dir / FUTURE_WORK_BOUNDARY_FILENAME,
        future_work_boundary_artifact,
    )
    bundle = build_documentation_contract_bundle(
        output_dir,
        contract_reference_map_artifact=contract_reference_map_artifact,
        supported_entry_reference_artifact=supported_entry_reference_artifact,
        support_surface_reference_artifact=support_surface_reference_artifact,
        evidence_bar_reference_artifact=evidence_bar_reference_artifact,
        future_work_boundary_artifact=future_work_boundary_artifact,
    )
    if verbose:
        print(
            "{} [{}] component_artifacts_complete={} file_coverage_complete={} glossary_complete={}".format(
                bundle["suite_name"],
                bundle["status"],
                bundle["summary"]["component_artifacts_complete"],
                bundle["summary"]["file_coverage_complete"],
                bundle["summary"]["glossary_complete"],
            )
        )
    return (
        contract_reference_map_artifact,
        supported_entry_reference_artifact,
        support_surface_reference_artifact,
        evidence_bar_reference_artifact,
        future_work_boundary_artifact,
        bundle,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for documentation_contract JSON artifacts.",
    )
    parser.add_argument(
        "--contract-reference-map-path",
        type=Path,
        default=CONTRACT_REFERENCE_MAP_PATH,
        help="Path to the contract_reference_map artifact.",
    )
    parser.add_argument(
        "--supported-entry-reference-path",
        type=Path,
        default=SUPPORTED_ENTRY_REFERENCE_PATH,
        help="Path to the supported_entry_reference artifact.",
    )
    parser.add_argument(
        "--support-surface-reference-path",
        type=Path,
        default=SUPPORT_SURFACE_REFERENCE_PATH,
        help="Path to the support_surface_reference artifact.",
    )
    parser.add_argument(
        "--evidence-bar-reference-path",
        type=Path,
        default=EVIDENCE_BAR_REFERENCE_PATH,
        help="Path to the evidence_bar_reference artifact.",
    )
    parser.add_argument(
        "--future-work-boundary-path",
        type=Path,
        default=FUTURE_WORK_BOUNDARY_PATH,
        help="Path to the future_work_boundary artifact.",
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
        contract_reference_map_path=args.contract_reference_map_path,
        supported_entry_reference_path=args.supported_entry_reference_path,
        support_surface_reference_path=args.support_surface_reference_path,
        evidence_bar_reference_path=args.evidence_bar_reference_path,
        future_work_boundary_path=args.future_work_boundary_path,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    output_path = args.output_dir / ARTIFACT_FILENAME
    write_documentation_contract_bundle(output_path, bundle)
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
