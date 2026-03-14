import copy

import pytest


def test_task7_story1_contract_reference_map_schema_module_level():
    """Test the Task 7 Story 1 contract-reference map schema."""
    from benchmarks.density_matrix.task7_story1_contract_reference_validation import (
        MANDATORY_TOPICS,
        PHASE2_DOCUMENTATION_INDEX_PATH,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["path"].endswith("PHASE_2_DOCUMENTATION_INDEX.md")
    assert artifact["entry_point"]["exists"] is True
    assert artifact["summary"]["topic_count"] == len(MANDATORY_TOPICS)
    assert artifact["summary"]["all_documents_exist"] is True
    assert artifact["summary"]["all_topics_present"] is True
    assert artifact["summary"]["entry_point_headings_complete"] is True
    assert artifact["summary"]["entry_point_topics_complete"] is True
    assert artifact["summary"]["entry_point_primary_paths_complete"] is True
    assert artifact["summary"]["contract_reference_map_completed"] is True
    assert artifact["provenance"]["entry_point_path"] == str(PHASE2_DOCUMENTATION_INDEX_PATH)


def test_task7_story1_missing_topic_blocks_validation_module_level():
    """Test that missing mandatory topics fail Task 7 Story 1 validation."""
    from benchmarks.density_matrix.task7_story1_contract_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["topic_map"] = broken_artifact["topic_map"][:-1]

    with pytest.raises(ValueError, match="topic map mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_task7_story1_missing_entry_point_heading_fails_validation_module_level():
    """Test that missing entry-point headings fail Task 7 Story 1 validation."""
    from benchmarks.density_matrix.task7_story1_contract_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["entry_point"]["missing_headings"] = ["## Topic Map"]
    broken_artifact["summary"]["entry_point_headings_complete"] = True

    with pytest.raises(
        ValueError, match="entry_point_headings_complete summary is inconsistent"
    ):
        validate_artifact_bundle(broken_artifact)


def test_task7_story2_supported_entry_reference_schema_module_level():
    """Test the Task 7 Story 2 supported-entry reference schema."""
    from benchmarks.density_matrix.task7_story2_supported_entry_reference_validation import (
        MANDATORY_STATEMENTS,
        STORY2_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == STORY2_SECTION_HEADING
    assert artifact["summary"]["required_statement_count"] == len(MANDATORY_STATEMENTS)
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_entry_point_statements_present"] is True
    assert artifact["summary"]["all_primary_sources_present"] is True
    assert artifact["summary"]["supported_entry_reference_completed"] is True


def test_task7_story2_missing_statement_fails_validation_module_level():
    """Test that missing required statements fail Task 7 Story 2 validation."""
    from benchmarks.density_matrix.task7_story2_supported_entry_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["required_statements"][0]["entry_point_present"] = False
    broken_artifact["summary"]["all_entry_point_statements_present"] = True

    with pytest.raises(
        ValueError,
        match="all_entry_point_statements_present summary is inconsistent",
    ):
        validate_artifact_bundle(broken_artifact)


def test_task7_story2_statement_inventory_mismatch_fails_validation_module_level():
    """Test that statement inventory drift fails Task 7 Story 2 validation."""
    from benchmarks.density_matrix.task7_story2_supported_entry_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["required_statements"] = broken_artifact["required_statements"][:-1]

    with pytest.raises(ValueError, match="required statements mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_task7_story3_support_surface_reference_schema_module_level():
    """Test the Task 7 Story 3 support-surface reference schema."""
    from benchmarks.density_matrix.task7_story3_support_surface_reference_validation import (
        MANDATORY_BOUNDARY_EXAMPLES,
        MANDATORY_SUPPORT_ITEMS,
        STORY3_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == STORY3_SECTION_HEADING
    assert artifact["summary"]["required_support_count"] == len(MANDATORY_SUPPORT_ITEMS)
    assert artifact["summary"]["boundary_example_count"] == len(
        MANDATORY_BOUNDARY_EXAMPLES
    )
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_support_items_present"] is True
    assert artifact["summary"]["all_boundary_examples_present"] is True
    assert artifact["summary"]["support_surface_reference_completed"] is True


def test_task7_story3_missing_support_item_fails_validation_module_level():
    """Test that missing support items fail Task 7 Story 3 validation."""
    from benchmarks.density_matrix.task7_story3_support_surface_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["support_surface_inventory"][0]["entry_point_present"] = False
    broken_artifact["summary"]["all_support_items_present"] = True

    with pytest.raises(
        ValueError, match="all_support_items_present summary is inconsistent"
    ):
        validate_artifact_bundle(broken_artifact)


def test_task7_story3_support_inventory_mismatch_fails_validation_module_level():
    """Test that support inventory drift fails Task 7 Story 3 validation."""
    from benchmarks.density_matrix.task7_story3_support_surface_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["support_surface_inventory"] = broken_artifact[
        "support_surface_inventory"
    ][:-1]

    with pytest.raises(ValueError, match="support inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_task7_story4_evidence_bar_reference_schema_module_level():
    """Test the Task 7 Story 4 evidence-bar reference schema."""
    from benchmarks.density_matrix.task7_story4_evidence_bar_validation import (
        MANDATORY_EVIDENCE_ITEMS,
        STORY4_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == STORY4_SECTION_HEADING
    assert artifact["summary"]["required_evidence_count"] == len(MANDATORY_EVIDENCE_ITEMS)
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_evidence_items_present"] is True
    assert artifact["summary"]["evidence_bar_reference_completed"] is True


def test_task7_story4_missing_evidence_item_fails_validation_module_level():
    """Test that missing evidence items fail Task 7 Story 4 validation."""
    from benchmarks.density_matrix.task7_story4_evidence_bar_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["evidence_inventory"][0]["entry_point_present"] = False
    broken_artifact["summary"]["all_evidence_items_present"] = True

    with pytest.raises(
        ValueError, match="all_evidence_items_present summary is inconsistent"
    ):
        validate_artifact_bundle(broken_artifact)


def test_task7_story4_evidence_inventory_mismatch_fails_validation_module_level():
    """Test that evidence inventory drift fails Task 7 Story 4 validation."""
    from benchmarks.density_matrix.task7_story4_evidence_bar_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["evidence_inventory"] = broken_artifact["evidence_inventory"][:-1]

    with pytest.raises(ValueError, match="evidence inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_task7_story5_future_work_boundary_schema_module_level():
    """Test the Task 7 Story 5 future-work boundary schema."""
    from benchmarks.density_matrix.task7_story5_future_work_boundary_validation import (
        MANDATORY_FUTURE_TOPICS,
        STORY5_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == STORY5_SECTION_HEADING
    assert artifact["summary"]["required_topic_count"] == len(MANDATORY_FUTURE_TOPICS)
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_future_work_items_present"] is True
    assert artifact["summary"]["future_work_boundary_completed"] is True


def test_task7_story5_missing_future_work_item_fails_validation_module_level():
    """Test that missing future-work labeling fails Task 7 Story 5 validation."""
    from benchmarks.density_matrix.task7_story5_future_work_boundary_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["future_work_inventory"][0]["entry_point_present"] = False
    broken_artifact["summary"]["all_future_work_items_present"] = True

    with pytest.raises(
        ValueError, match="all_future_work_items_present summary is inconsistent"
    ):
        validate_artifact_bundle(broken_artifact)


def test_task7_story5_future_work_inventory_mismatch_fails_validation_module_level():
    """Test that future-work inventory drift fails Task 7 Story 5 validation."""
    from benchmarks.density_matrix.task7_story5_future_work_boundary_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["future_work_inventory"] = broken_artifact["future_work_inventory"][
        :-1
    ]

    with pytest.raises(ValueError, match="future-work inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_task7_story6_documentation_bundle_schema_module_level():
    """Test the Task 7 Story 6 documentation bundle schema."""
    from benchmarks.density_matrix.task7_story6_documentation_bundle import (
        run_validation,
    )

    (
        story1_artifact,
        story2_artifact,
        story3_artifact,
        story4_artifact,
        story5_artifact,
        bundle,
    ) = run_validation(verbose=False)

    assert story1_artifact["status"] == "pass"
    assert story2_artifact["status"] == "pass"
    assert story3_artifact["status"] == "pass"
    assert story4_artifact["status"] == "pass"
    assert story5_artifact["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["summary"]["mandatory_story_artifact_count"] == 5
    assert bundle["summary"]["story_artifacts_complete"] is True
    assert bundle["summary"]["file_coverage_complete"] is True
    assert bundle["summary"]["glossary_complete"] is True
    assert len(bundle["story_artifacts"]) == 5


def test_task7_story6_missing_artifact_entry_fails_validation_module_level():
    """Test that missing mandatory story artifacts fail Task 7 Story 6 validation."""
    from benchmarks.density_matrix.task7_story6_documentation_bundle import (
        run_validation,
        validate_task7_story6_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["story_artifacts"] = broken_bundle["story_artifacts"][:-1]

    with pytest.raises(ValueError, match="missing required story artifact IDs"):
        validate_task7_story6_bundle(broken_bundle)


def test_task7_story6_missing_semantic_flag_fails_validation_module_level():
    """Test that missing lower-story semantic closure fails Task 7 Story 6 validation."""
    from benchmarks.density_matrix.task7_story6_documentation_bundle import (
        run_validation,
        validate_task7_story6_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["story_artifacts"][1]["semantic_flag_passed"] = False
    broken_bundle["summary"]["story_artifacts_complete"] = True

    with pytest.raises(ValueError, match="story_artifacts_complete summary is inconsistent"):
        validate_task7_story6_bundle(broken_bundle)


def test_task7_story6_missing_glossary_term_fails_validation_module_level():
    """Test that missing glossary terms fail Task 7 Story 6 validation."""
    from benchmarks.density_matrix.task7_story6_documentation_bundle import (
        run_validation,
        validate_task7_story6_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["terminology_inventory"]["missing_glossary_terms"] = [
        "future work and non-goal"
    ]
    broken_bundle["summary"]["glossary_complete"] = True

    with pytest.raises(ValueError, match="glossary_complete summary is inconsistent"):
        validate_task7_story6_bundle(broken_bundle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
