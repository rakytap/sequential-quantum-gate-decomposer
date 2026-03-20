from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    TASK7_BENCHMARK_PACKAGE_SCHEMA_VERSION,
    TASK7_CASE_SCHEMA_VERSION,
    TASK7_PRIMARY_STRUCTURED_SEED,
    TASK7_SUMMARY_SCHEMA_VERSION,
)
from benchmarks.density_matrix.performance_evidence.benchmark_matrix_validation import (
    build_artifact_bundle as build_story1_bundle,
    build_cases as build_story1_cases,
)
from benchmarks.density_matrix.performance_evidence.counted_supported_validation import (
    build_artifact_bundle as build_story2_bundle,
    build_cases as build_story2_cases,
)
from benchmarks.density_matrix.performance_evidence.positive_threshold_validation import (
    build_artifact_bundle as build_story3_bundle,
    build_cases as build_story3_cases,
)
from benchmarks.density_matrix.performance_evidence.sensitivity_matrix_validation import (
    build_artifact_bundle as build_story4_bundle,
    build_cases as build_story4_cases,
)
from benchmarks.density_matrix.performance_evidence.metric_surface_validation import (
    build_artifact_bundle as build_story5_bundle,
    build_cases as build_story5_cases,
)
from benchmarks.density_matrix.performance_evidence.diagnosis_validation import (
    build_artifact_bundle as build_story6_bundle,
    build_cases as build_story6_cases,
)
from benchmarks.density_matrix.performance_evidence.benchmark_bundle_validation import (
    build_artifact_bundle as build_story7_bundle,
)
from benchmarks.density_matrix.performance_evidence.summary_consistency_validation import (
    build_artifact_bundle as build_story8_bundle,
)


def test_phase3_task7_story1_benchmark_matrix_covers_required_inventory():
    cases = build_story1_cases()
    assert len(cases) == 34
    assert sum(case["case_kind"] == "continuity" for case in cases) == 4
    assert sum(case["case_kind"] == "structured_family" for case in cases) == 30
    assert sum(case["representative_review_case"] for case in cases) == 6
    assert {
        case["seed"] for case in cases if case["case_kind"] == "structured_family"
    } == {
        TASK7_PRIMARY_STRUCTURED_SEED,
        TASK7_PRIMARY_STRUCTURED_SEED + 1,
        TASK7_PRIMARY_STRUCTURED_SEED + 2,
    }
    assert {
        case["noise_pattern"] for case in cases if case["noise_pattern"] is not None
    } == {"sparse", "periodic", "dense"}


def test_phase3_task7_story1_bundle_core_fields_are_stable():
    bundle = build_story1_bundle(build_story1_cases())
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["continuity_cases"] == 4
    assert bundle["summary"]["structured_cases"] == 30
    assert bundle["summary"]["representative_review_cases"] == 6


@pytest.fixture(scope="module")
def story2_cases():
    return build_story2_cases()


def test_phase3_task7_story2_counted_supported_gate_passes_full_matrix(story2_cases):
    assert len(story2_cases) == 34
    assert all(case["counted_supported_benchmark_case"] for case in story2_cases)
    assert all(case["record_schema_version"] == TASK7_CASE_SCHEMA_VERSION for case in story2_cases)
    assert all(case["supported_runtime_case"] for case in story2_cases)


def test_phase3_task7_story2_bundle_core_fields_are_stable(story2_cases):
    bundle = build_story2_bundle(story2_cases)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == TASK7_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["counted_supported_cases"] == 34
    assert bundle["summary"]["excluded_cases"] == 0


def test_phase3_task7_story3_positive_review_surface_is_bounded_and_auditable():
    cases = build_story3_cases()
    assert len(cases) == 6
    assert all(case["representative_review_case"] for case in cases)
    assert all(case["sequential_median_runtime_ms"] is not None for case in cases)
    assert all(case["fused_median_runtime_ms"] is not None for case in cases)


def test_phase3_task7_story3_bundle_core_fields_are_stable():
    bundle = build_story3_bundle(build_story3_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == TASK7_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 6
    assert len(bundle["summary"]["review_groups"]) == 6


def test_phase3_task7_story4_sensitivity_surface_covers_required_groups():
    cases = build_story4_cases()
    assert len(cases) == 30
    assert {case["family_name"] for case in cases} == {
        "layered_nearest_neighbor",
        "seeded_random_layered",
        "partition_stress_ladder",
    }
    assert {case["qbit_num"] for case in cases} == {8, 10}


def test_phase3_task7_story4_bundle_core_fields_are_stable():
    bundle = build_story4_bundle(build_story4_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == TASK7_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 30
    assert bundle["summary"]["full_noise_groups"] == 6
    assert bundle["summary"]["full_seed_groups"] == 6


@pytest.fixture(scope="module")
def story5_cases():
    return build_story5_cases()


def test_phase3_task7_story5_metric_surface_is_complete(story5_cases):
    assert len(story5_cases) == 34
    assert all(case["runtime_ms"] is not None for case in story5_cases)
    assert all(case["peak_rss_kb"] is not None for case in story5_cases)
    assert all(case["planning_time_ms"] is not None for case in story5_cases)


def test_phase3_task7_story5_bundle_core_fields_are_stable(story5_cases):
    bundle = build_story5_bundle(story5_cases)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["representative_review_cases"] == 6
    assert bundle["summary"]["median_timed_cases"] == 6


def test_phase3_task7_story6_diagnosis_surface_is_explicit():
    cases = build_story6_cases()
    assert len(cases) >= 1
    assert all(case["diagnosis_only_case"] for case in cases)
    assert all(case["diagnosis_reasons"] for case in cases)


def test_phase3_task7_story6_bundle_core_fields_are_stable():
    bundle = build_story6_bundle(build_story6_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == TASK7_CASE_SCHEMA_VERSION
    assert bundle["summary"]["task6_boundary_cases"] >= 1


def test_phase3_task7_story7_benchmark_package_is_complete():
    bundle = build_story7_bundle()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == TASK7_BENCHMARK_PACKAGE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["representative_review_cases"] == 6
    assert bundle["summary"]["task6_boundary_cases"] == len(bundle["negative_cases"])


def test_phase3_task7_story8_summary_consistency_closes_from_positive_or_diagnosis_path():
    bundle = build_story8_bundle()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == TASK7_SUMMARY_SCHEMA_VERSION
    assert bundle["summary"]["summary_consistency_pass"] is True
    assert bundle["summary"]["main_benchmark_claim_completed"] is True
    assert (
        bundle["summary"]["positive_benchmark_claim_completed"]
        or bundle["summary"]["diagnosis_grounded_closure_completed"]
    )
