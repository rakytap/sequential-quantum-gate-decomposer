import copy

import pytest


def test_task8_story1_claim_package_schema_module_level():
    """Test the Task 8 Story 1 claim-package schema."""
    from benchmarks.density_matrix.task8_story1_claim_package_validation import (
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


def test_task8_story1_non_claim_inventory_mismatch_fails_validation_module_level():
    """Test that non-claim inventory drift fails Task 8 Story 1 validation."""
    from benchmarks.density_matrix.task8_story1_claim_package_validation import (
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


def test_task8_story2_publication_surface_alignment_schema_module_level():
    """Test the Task 8 Story 2 publication-surface alignment schema."""
    from benchmarks.density_matrix.task8_story2_publication_surface_alignment import (
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


def test_task8_story2_surface_inventory_mismatch_fails_validation_module_level():
    """Test that missing publication surfaces fail Task 8 Story 2 validation."""
    from benchmarks.density_matrix.task8_story2_publication_surface_alignment import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["surface_inventory"] = broken_artifact["surface_inventory"][:-1]

    with pytest.raises(ValueError, match="publication surface inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_task8_story3_claim_traceability_schema_module_level():
    """Test the Task 8 Story 3 claim-traceability schema."""
    from benchmarks.density_matrix.task8_story3_claim_traceability_bundle import (
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


def test_task8_story4_evidence_closure_schema_module_level():
    """Test the Task 8 Story 4 evidence-closure schema."""
    from benchmarks.density_matrix.task8_story4_evidence_closure_validation import (
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


def test_task8_story5_supported_path_scope_schema_module_level():
    """Test the Task 8 Story 5 supported-path scope schema."""
    from benchmarks.density_matrix.task8_story5_supported_path_scope_validation import (
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


def test_task8_story6_future_work_boundary_schema_module_level():
    """Test the Task 8 Story 6 future-work boundary schema."""
    from benchmarks.density_matrix.task8_story6_future_work_boundary_validation import (
        MANDATORY_FUTURE_TOPICS,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["summary"]["required_topic_count"] == len(MANDATORY_FUTURE_TOPICS)
    assert artifact["summary"]["all_future_work_items_present"] is True
    assert artifact["summary"]["phase_positioning_present"] is True
    assert artifact["summary"]["future_work_boundary_completed"] is True


def test_task8_story7_publication_bundle_schema_module_level():
    """Test the Task 8 Story 7 publication bundle schema."""
    from benchmarks.density_matrix.task8_story7_publication_bundle import run_validation

    (
        story1_artifact,
        story2_artifact,
        story3_artifact,
        story4_artifact,
        story5_artifact,
        story6_artifact,
        bundle,
    ) = run_validation(verbose=False)

    assert story1_artifact["status"] == "pass"
    assert story2_artifact["status"] == "pass"
    assert story3_artifact["status"] == "pass"
    assert story4_artifact["status"] == "pass"
    assert story5_artifact["status"] == "pass"
    assert story6_artifact["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["summary"]["mandatory_story_artifact_count"] == 6
    assert bundle["summary"]["story_artifacts_complete"] is True
    assert bundle["summary"]["file_coverage_complete"] is True
    assert bundle["summary"]["terminology_complete"] is True
    assert bundle["summary"]["reviewer_entry_paths_complete"] is True
    assert len(bundle["story_artifacts"]) == 6


def test_task8_story7_missing_artifact_entry_fails_validation_module_level():
    """Test that missing mandatory story artifacts fail Task 8 Story 7 validation."""
    from benchmarks.density_matrix.task8_story7_publication_bundle import (
        run_validation,
        validate_task8_story7_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["story_artifacts"] = broken_bundle["story_artifacts"][:-1]

    with pytest.raises(ValueError, match="missing required story artifact IDs"):
        validate_task8_story7_bundle(broken_bundle)
