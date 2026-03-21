import copy

import pytest


def test_claim_package_schema_module_level():
    """claim_package artifact schema."""
    from benchmarks.density_matrix.publication_claim_package.claim_package_validation import (
        NON_CLAIM_ITEMS,
        SUPPORTING_CLAIM_ITEMS,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["supporting_claim_count"] == len(SUPPORTING_CLAIM_ITEMS)
    assert artifact["summary"]["non_claim_count"] == len(NON_CLAIM_ITEMS)
    assert artifact["summary"]["primary_surface_has_role_heading"] is True
    assert artifact["summary"]["primary_surface_has_claim_heading"] is True
    assert artifact["summary"]["main_claim_present_in_primary_surface"] is True
    assert artifact["summary"]["all_supporting_claims_present"] is True
    assert artifact["summary"]["all_non_claims_present"] is True
    assert artifact["summary"]["evidence_closure_rule_present"] is True
    assert artifact["summary"]["claim_package_completed"] is True


def test_claim_package_non_claim_inventory_mismatch_fails_validation_module_level():
    """Non-claim inventory drift fails claim_package validation."""
    from benchmarks.density_matrix.publication_claim_package.claim_package_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["claim_package"]["explicit_non_claims"] = broken_artifact[
        "claim_package"
    ]["explicit_non_claims"][:-1]

    with pytest.raises(ValueError, match="explicit non-claim inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_publication_surface_alignment_schema_module_level():
    """publication_surface_alignment artifact schema."""
    from benchmarks.density_matrix.publication_claim_package.publication_surface_alignment import (
        PUBLICATION_SURFACES,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["surface_count"] == len(PUBLICATION_SURFACES)
    assert artifact["summary"]["all_surface_roles_present"] is True
    assert artifact["summary"]["all_claim_headings_present"] is True
    assert artifact["summary"]["all_main_claims_present"] is True
    assert artifact["summary"]["all_non_claims_present"] is True
    assert artifact["summary"]["all_evidence_rules_present"] is True
    assert artifact["summary"]["surface_alignment_completed"] is True


def test_publication_surface_alignment_surface_inventory_mismatch_fails_validation_module_level():
    """Missing publication surfaces fail publication_surface_alignment validation."""
    from benchmarks.density_matrix.publication_claim_package.publication_surface_alignment import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["surface_inventory"] = broken_artifact["surface_inventory"][:-1]

    with pytest.raises(ValueError, match="publication surface inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_claim_traceability_schema_module_level():
    """claim_traceability artifact schema."""
    from benchmarks.density_matrix.publication_claim_package.claim_traceability_bundle import (
        CLAIM_TRACEABILITY_ITEMS,
        SECTION_TRACEABILITY_ITEMS,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["claim_traceability_count"] == len(
        CLAIM_TRACEABILITY_ITEMS
    )
    assert artifact["summary"]["section_traceability_count"] == len(
        SECTION_TRACEABILITY_ITEMS
    )
    assert artifact["summary"]["all_claim_sources_exist"] is True
    assert artifact["summary"]["all_section_sources_exist"] is True
    assert artifact["summary"]["all_section_headings_present"] is True
    assert artifact["summary"]["reviewer_entry_paths_complete"] is True
    assert artifact["summary"]["claim_traceability_completed"] is True


def test_evidence_closure_schema_module_level():
    """evidence_closure artifact schema."""
    from benchmarks.density_matrix.publication_claim_package.evidence_closure_validation import (
        MANDATORY_EVIDENCE_ITEMS,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["required_evidence_count"] == len(
        MANDATORY_EVIDENCE_ITEMS
    )
    assert artifact["summary"]["all_evidence_items_present"] is True
    assert artifact["summary"]["claim_closure_rule_present"] is True
    assert artifact["summary"]["evidence_closure_completed"] is True


def test_supported_path_scope_schema_module_level():
    """supported_path_scope artifact schema."""
    from benchmarks.density_matrix.publication_claim_package.supported_path_scope_validation import (
        PUBLICATION_SURFACES,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["surface_count"] == len(PUBLICATION_SURFACES)
    assert artifact["summary"]["all_main_claims_present"] is True
    assert artifact["summary"]["all_no_fallback_present"] is True
    assert artifact["summary"]["all_supported_path_boundaries_present"] is True
    assert artifact["summary"]["all_exact_regime_boundaries_present"] is True
    assert artifact["summary"]["all_parity_non_claims_present"] is True
    assert artifact["summary"]["supported_path_scope_completed"] is True


def test_future_work_boundary_publication_schema_module_level():
    """future_work_boundary artifact schema (publication claim package)."""
    from benchmarks.density_matrix.publication_claim_package.future_work_boundary_validation import (
        MANDATORY_FUTURE_TOPICS,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["required_topic_count"] == len(MANDATORY_FUTURE_TOPICS)
    assert artifact["summary"]["all_future_work_items_present"] is True
    assert artifact["summary"]["phase_positioning_present"] is True
    assert artifact["summary"]["future_work_boundary_completed"] is True


def test_publication_claim_bundle_schema_module_level():
    """publication_claim_bundle aggregates lower publication claim layers."""
    from benchmarks.density_matrix.publication_claim_package.publication_claim_bundle import run_validation

    (
        claim_package_artifact,
        publication_surface_alignment_artifact,
        claim_traceability_artifact,
        evidence_closure_artifact,
        supported_path_scope_artifact,
        future_work_boundary_artifact,
        bundle,
    ) = run_validation(verbose=False)

    assert claim_package_artifact["status"] == "pass"
    assert publication_surface_alignment_artifact["status"] == "pass"
    assert claim_traceability_artifact["status"] == "pass"
    assert evidence_closure_artifact["status"] == "pass"
    assert supported_path_scope_artifact["status"] == "pass"
    assert future_work_boundary_artifact["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["summary"]["mandatory_component_count"] == 6
    assert bundle["summary"]["component_artifacts_complete"] is True
    assert bundle["summary"]["file_coverage_complete"] is True
    assert bundle["summary"]["terminology_complete"] is True
    assert bundle["summary"]["reviewer_entry_paths_complete"] is True
    assert len(bundle["component_artifacts"]) == 6


def test_publication_claim_bundle_missing_artifact_entry_fails_validation_module_level():
    """Missing mandatory component rows fail publication_claim_bundle validation."""
    from benchmarks.density_matrix.publication_claim_package.publication_claim_bundle import (
        run_validation,
        validate_publication_claim_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["component_artifacts"] = broken_bundle["component_artifacts"][:-1]

    with pytest.raises(ValueError, match="missing required component artifact IDs"):
        validate_publication_claim_bundle(broken_bundle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
