"""Shared table-driven expectations for evidence bundle shape checks."""


def assert_performance_benchmark_matrix_bundle_core(bundle: dict) -> None:
    assert bundle["status"] == "pass"
    assert "record_schema_version" not in bundle
    summary = bundle["summary"]
    assert summary["total_cases"] == 34
    assert summary["continuity_cases"] == 4
    assert summary["structured_cases"] == 30
    assert summary["representative_review_cases"] == 6


def assert_correctness_unsupported_boundary_bundle_core(
    bundle: dict, *, negative_record_schema_version: str
) -> None:
    assert bundle["status"] == "pass"
    assert bundle["negative_record_schema_version"] == negative_record_schema_version
    summary = bundle["summary"]
    assert summary["planner_entry_cases"] >= 1
    assert summary["descriptor_generation_cases"] >= 1
    assert summary["runtime_stage_cases"] >= 1


def assert_correctness_full_package_bundle(
    bundle: dict, *, schema_version: str, unsupported_case_count: int
) -> None:
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == schema_version
    summary = bundle["summary"]
    assert summary["total_cases"] == 25
    assert summary["counted_supported_cases"] == 25
    assert summary["unsupported_boundary_cases"] == unsupported_case_count


def assert_performance_full_benchmark_package_bundle(
    bundle: dict, *, schema_version: str, negative_case_count: int
) -> None:
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == schema_version
    summary = bundle["summary"]
    assert summary["total_cases"] == 34
    assert summary["representative_review_cases"] == 6
    assert summary["correctness_evidence_boundary_cases"] == negative_case_count
