import copy
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.publication_evidence.claim_package_validation import (
    NON_CLAIM_ITEMS,
    SUPPORTING_CLAIM_ITEMS,
    run_validation as run_claim_package_validation,
    validate_artifact_bundle as validate_claim_package_artifact,
)
from benchmarks.density_matrix.publication_evidence.surface_alignment_validation import (
    PUBLICATION_SURFACES,
    run_validation as run_publication_surface_alignment_validation,
)
from benchmarks.density_matrix.publication_evidence.claim_traceability_validation import (
    CLAIM_TRACEABILITY_ITEMS,
    SECTION_TRACEABILITY_ITEMS,
    run_validation as run_claim_traceability_validation,
)
from benchmarks.density_matrix.publication_evidence.evidence_closure_validation import (
    run_validation as run_evidence_closure_validation,
)
from benchmarks.density_matrix.publication_evidence.supported_path_validation import (
    run_validation as run_supported_path_scope_validation,
)
from benchmarks.density_matrix.publication_evidence.publication_manifest_validation import (
    run_validation as run_publication_manifest_validation,
    validate_artifact_bundle as validate_publication_manifest_artifact,
)
from benchmarks.density_matrix.publication_evidence.future_work_boundary_validation import (
    MANDATORY_FUTURE_TOPICS,
    run_validation as run_future_work_boundary_validation,
)
from benchmarks.density_matrix.publication_evidence.package_consistency_validation import (
    REQUIRED_GLOSSARY_TERMS,
    run_validation as run_package_consistency_validation,
    validate_artifact_bundle as validate_package_consistency_artifact,
)


def test_publication_evidence_claim_package_schema_module_level():
    artifact = run_claim_package_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["supporting_claim_count"] == len(SUPPORTING_CLAIM_ITEMS)
    assert artifact["summary"]["non_claim_count"] == len(NON_CLAIM_ITEMS)
    assert artifact["summary"]["main_claim_present_in_primary_surface"] is True
    assert artifact["summary"]["all_supporting_claims_present"] is True
    assert artifact["summary"]["all_non_claims_present"] is True
    assert artifact["summary"]["claim_package_completed"] is True


def test_publication_evidence_claim_package_non_claim_inventory_mismatch_fails_validation_module_level():
    artifact = run_claim_package_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["claim_package"]["explicit_non_claims"] = broken_artifact[
        "claim_package"
    ]["explicit_non_claims"][:-1]

    with pytest.raises(ValueError, match="explicit non-claim inventory mismatch"):
        validate_claim_package_artifact(broken_artifact)


def test_publication_evidence_publication_surface_alignment_schema_module_level():
    artifact = run_publication_surface_alignment_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["surface_count"] == len(PUBLICATION_SURFACES)
    assert artifact["summary"]["all_surface_roles_present"] is True
    assert artifact["summary"]["all_main_claims_present"] is True
    assert artifact["summary"]["all_non_claims_present"] is True
    assert artifact["summary"]["publication_surface_alignment_completed"] is True


def test_publication_evidence_claim_traceability_schema_module_level():
    artifact = run_claim_traceability_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["claim_traceability_count"] == len(
        CLAIM_TRACEABILITY_ITEMS
    )
    assert artifact["summary"]["section_traceability_count"] == len(
        SECTION_TRACEABILITY_ITEMS
    )
    assert artifact["summary"]["all_claim_sources_exist"] is True
    assert artifact["summary"]["all_section_sources_exist"] is True
    assert artifact["summary"]["claim_traceability_completed"] is True


def test_publication_evidence_evidence_closure_schema_module_level():
    artifact = run_evidence_closure_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["required_evidence_count"] == 5
    assert artifact["summary"]["all_evidence_items_present"] is True
    assert artifact["summary"]["claim_closure_rule_present"] is True
    assert artifact["summary"]["positive_or_diagnosis_path_completed"] is True
    assert artifact["summary"]["evidence_closure_completed"] is True


def test_publication_evidence_supported_path_scope_schema_module_level():
    artifact = run_supported_path_scope_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["surface_count"] == len(PUBLICATION_SURFACES)
    assert artifact["summary"]["all_supported_path_boundaries_present"] is True
    assert artifact["summary"]["all_no_fallback_rules_present"] is True
    assert artifact["summary"]["all_bounded_planner_claims_present"] is True
    assert artifact["summary"]["all_correctness_evidence_count_surfaces_honest"] is True
    assert artifact["summary"]["all_performance_evidence_count_surfaces_honest"] is True
    assert artifact["summary"]["supported_path_scope_completed"] is True


def test_publication_evidence_publication_manifest_schema_module_level():
    artifact = run_publication_manifest_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["mandatory_component_artifact_count"] == 5
    assert artifact["summary"]["required_evidence_ref_count"] == 8
    assert artifact["summary"]["component_artifacts_complete"] is True
    assert artifact["summary"]["required_evidence_refs_complete"] is True
    assert artifact["summary"]["reviewer_entry_paths_complete"] is True
    assert artifact["summary"]["publication_manifest_completed"] is True


def test_publication_evidence_publication_manifest_missing_required_evidence_ref_fails_validation_module_level():
    artifact = run_publication_manifest_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["required_evidence_refs"] = broken_artifact["required_evidence_refs"][:-1]

    with pytest.raises(
        ValueError, match="publication manifest evidence reference inventory mismatch"
    ):
        validate_publication_manifest_artifact(broken_artifact)


def test_publication_evidence_future_work_boundary_schema_module_level():
    artifact = run_future_work_boundary_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["required_topic_count"] == len(MANDATORY_FUTURE_TOPICS)
    assert artifact["summary"]["all_future_work_items_present"] is True
    assert artifact["summary"]["phase_positioning_present"] is True
    assert artifact["summary"]["benchmark_driven_follow_on_present"] is True
    assert artifact["summary"]["future_work_boundary_completed"] is True


def test_publication_evidence_package_consistency_schema_module_level():
    artifact = run_package_consistency_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["surface_count"] == len(PUBLICATION_SURFACES)
    assert artifact["summary"]["glossary_term_count"] == len(REQUIRED_GLOSSARY_TERMS)
    assert artifact["summary"]["reviewer_entry_paths_complete"] is True
    assert artifact["summary"]["terminology_complete"] is True
    assert artifact["summary"]["count_consistency_complete"] is True
    assert artifact["summary"]["limitation_summary_consistency_complete"] is True
    assert artifact["summary"]["package_consistency_completed"] is True


def test_publication_evidence_package_consistency_surface_inventory_mismatch_fails_validation_module_level():
    artifact = run_package_consistency_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["surface_inventory"] = broken_artifact["surface_inventory"][:-1]

    with pytest.raises(ValueError, match="package consistency surface inventory mismatch"):
        validate_package_consistency_artifact(broken_artifact)
