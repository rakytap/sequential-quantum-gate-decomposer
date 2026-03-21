import copy

import pytest


def test_contract_reference_map_schema_module_level():
    """contract_reference_map artifact schema and live validation."""
    from benchmarks.density_matrix.documentation_contract.contract_reference_validation import (
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


def test_contract_reference_map_missing_topic_blocks_validation_module_level():
    """Missing mandatory topics fail contract_reference_map validation."""
    from benchmarks.density_matrix.documentation_contract.contract_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["topic_map"] = broken_artifact["topic_map"][:-1]

    with pytest.raises(ValueError, match="topic map mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_contract_reference_map_missing_entry_point_heading_fails_validation_module_level():
    """Missing entry-point headings fail contract_reference_map validation."""
    from benchmarks.density_matrix.documentation_contract.contract_reference_validation import (
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


def test_supported_entry_reference_schema_module_level():
    """supported_entry_reference artifact schema."""
    from benchmarks.density_matrix.documentation_contract.supported_entry_reference_validation import (
        MANDATORY_STATEMENTS,
        SUPPORTED_ENTRY_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == SUPPORTED_ENTRY_SECTION_HEADING
    assert artifact["summary"]["required_statement_count"] == len(MANDATORY_STATEMENTS)
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_entry_point_statements_present"] is True
    assert artifact["summary"]["all_primary_sources_present"] is True
    assert artifact["summary"]["supported_entry_reference_completed"] is True


def test_supported_entry_reference_missing_statement_fails_validation_module_level():
    """Missing required statements fail supported_entry_reference validation."""
    from benchmarks.density_matrix.documentation_contract.supported_entry_reference_validation import (
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


def test_supported_entry_reference_statement_inventory_mismatch_fails_validation_module_level():
    """Statement inventory drift fails supported_entry_reference validation."""
    from benchmarks.density_matrix.documentation_contract.supported_entry_reference_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["required_statements"] = broken_artifact["required_statements"][:-1]

    with pytest.raises(ValueError, match="required statements mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_support_surface_reference_schema_module_level():
    """support_surface_reference artifact schema."""
    from benchmarks.density_matrix.documentation_contract.support_surface_reference_validation import (
        MANDATORY_BOUNDARY_EXAMPLES,
        MANDATORY_SUPPORT_ITEMS,
        SUPPORT_SURFACE_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == SUPPORT_SURFACE_SECTION_HEADING
    assert artifact["summary"]["required_support_count"] == len(MANDATORY_SUPPORT_ITEMS)
    assert artifact["summary"]["boundary_example_count"] == len(
        MANDATORY_BOUNDARY_EXAMPLES
    )
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_support_items_present"] is True
    assert artifact["summary"]["all_boundary_examples_present"] is True
    assert artifact["summary"]["support_surface_reference_completed"] is True


def test_support_surface_reference_missing_support_item_fails_validation_module_level():
    """Missing support items fail support_surface_reference validation."""
    from benchmarks.density_matrix.documentation_contract.support_surface_reference_validation import (
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


def test_support_surface_reference_support_inventory_mismatch_fails_validation_module_level():
    """Support inventory drift fails support_surface_reference validation."""
    from benchmarks.density_matrix.documentation_contract.support_surface_reference_validation import (
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


def test_evidence_bar_reference_schema_module_level():
    """evidence_bar_reference artifact schema."""
    from benchmarks.density_matrix.documentation_contract.evidence_bar_validation import (
        MANDATORY_EVIDENCE_ITEMS,
        EVIDENCE_BAR_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == EVIDENCE_BAR_SECTION_HEADING
    assert artifact["summary"]["required_evidence_count"] == len(MANDATORY_EVIDENCE_ITEMS)
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_evidence_items_present"] is True
    assert artifact["summary"]["evidence_bar_reference_completed"] is True


def test_evidence_bar_reference_missing_evidence_item_fails_validation_module_level():
    """Missing evidence items fail evidence_bar_reference validation."""
    from benchmarks.density_matrix.documentation_contract.evidence_bar_validation import (
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


def test_evidence_bar_reference_evidence_inventory_mismatch_fails_validation_module_level():
    """Evidence inventory drift fails evidence_bar_reference validation."""
    from benchmarks.density_matrix.documentation_contract.evidence_bar_validation import (
        run_validation,
        validate_artifact_bundle,
    )

    artifact = run_validation(verbose=False)
    broken_artifact = copy.deepcopy(artifact)
    broken_artifact["evidence_inventory"] = broken_artifact["evidence_inventory"][:-1]

    with pytest.raises(ValueError, match="evidence inventory mismatch"):
        validate_artifact_bundle(broken_artifact)


def test_future_work_boundary_schema_module_level():
    """future_work_boundary artifact schema (documentation contract)."""
    from benchmarks.density_matrix.documentation_contract.future_work_boundary_validation import (
        MANDATORY_FUTURE_TOPICS,
        FUTURE_WORK_SECTION_HEADING,
        run_validation,
    )

    artifact = run_validation(verbose=False)

    assert artifact["status"] == "pass"
    assert artifact["entry_point"]["section_heading"] == FUTURE_WORK_SECTION_HEADING
    assert artifact["summary"]["required_topic_count"] == len(MANDATORY_FUTURE_TOPICS)
    assert artifact["summary"]["section_heading_present"] is True
    assert artifact["summary"]["all_future_work_items_present"] is True
    assert artifact["summary"]["future_work_boundary_completed"] is True


def test_future_work_boundary_missing_future_work_item_fails_validation_module_level():
    """Missing future-work labeling fails future_work_boundary validation."""
    from benchmarks.density_matrix.documentation_contract.future_work_boundary_validation import (
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


def test_future_work_boundary_future_work_inventory_mismatch_fails_validation_module_level():
    """Future-work inventory drift fails future_work_boundary validation."""
    from benchmarks.density_matrix.documentation_contract.future_work_boundary_validation import (
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


def test_documentation_contract_bundle_schema_module_level():
    """documentation_contract_bundle aggregates lower documentation contract layers."""
    from benchmarks.density_matrix.documentation_contract.documentation_contract_bundle import (
        run_validation,
    )

    (
        contract_reference_map_artifact,
        supported_entry_reference_artifact,
        support_surface_reference_artifact,
        evidence_bar_reference_artifact,
        future_work_boundary_artifact,
        bundle,
    ) = run_validation(verbose=False)

    assert contract_reference_map_artifact["status"] == "pass"
    assert supported_entry_reference_artifact["status"] == "pass"
    assert support_surface_reference_artifact["status"] == "pass"
    assert evidence_bar_reference_artifact["status"] == "pass"
    assert future_work_boundary_artifact["status"] == "pass"
    assert bundle["status"] == "pass"
    assert bundle["summary"]["mandatory_component_count"] == 5
    assert bundle["summary"]["component_artifacts_complete"] is True
    assert bundle["summary"]["file_coverage_complete"] is True
    assert bundle["summary"]["glossary_complete"] is True
    assert len(bundle["component_artifacts"]) == 5


def test_documentation_contract_bundle_missing_artifact_entry_fails_validation_module_level():
    """Missing mandatory component rows fail documentation_contract_bundle validation."""
    from benchmarks.density_matrix.documentation_contract.documentation_contract_bundle import (
        run_validation,
        validate_documentation_contract_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["component_artifacts"] = broken_bundle["component_artifacts"][:-1]

    with pytest.raises(ValueError, match="missing required component artifact IDs"):
        validate_documentation_contract_bundle(broken_bundle)


def test_documentation_contract_bundle_missing_semantic_flag_fails_validation_module_level():
    """Inconsistent semantic closure flags fail documentation_contract_bundle validation."""
    from benchmarks.density_matrix.documentation_contract.documentation_contract_bundle import (
        run_validation,
        validate_documentation_contract_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["component_artifacts"][1]["semantic_flag_passed"] = False
    broken_bundle["summary"]["component_artifacts_complete"] = True

    with pytest.raises(
        ValueError, match="component_artifacts_complete summary is inconsistent"
    ):
        validate_documentation_contract_bundle(broken_bundle)


def test_documentation_contract_bundle_missing_glossary_term_fails_validation_module_level():
    """Missing glossary terms fail documentation_contract_bundle validation."""
    from benchmarks.density_matrix.documentation_contract.documentation_contract_bundle import (
        run_validation,
        validate_documentation_contract_bundle,
    )

    *_, bundle = run_validation(verbose=False)
    broken_bundle = copy.deepcopy(bundle)
    broken_bundle["terminology_inventory"]["missing_glossary_terms"] = [
        "future work and non-goal"
    ]
    broken_bundle["summary"]["glossary_complete"] = True

    with pytest.raises(ValueError, match="glossary_complete summary is inconsistent"):
        validate_documentation_contract_bundle(broken_bundle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
