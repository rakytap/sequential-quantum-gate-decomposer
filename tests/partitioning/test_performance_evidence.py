from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED,
    PERFORMANCE_EVIDENCE_SUMMARY_SCHEMA_VERSION,
)
from benchmarks.density_matrix.performance_evidence.benchmark_matrix_validation import (
    build_benchmark_matrix_bundle,
    build_benchmark_matrix_cases,
)
from benchmarks.density_matrix.performance_evidence.counted_supported_validation import (
    build_counted_supported_bundle,
    build_counted_supported_cases,
)
from benchmarks.density_matrix.performance_evidence.positive_threshold_validation import (
    build_positive_threshold_bundle,
    build_positive_threshold_cases,
)
from benchmarks.density_matrix.performance_evidence.sensitivity_matrix_validation import (
    build_sensitivity_matrix_bundle,
    build_sensitivity_matrix_cases,
)
from benchmarks.density_matrix.performance_evidence.metric_surface_validation import (
    build_metric_surface_bundle,
    build_metric_surface_cases,
)
from benchmarks.density_matrix.performance_evidence.diagnosis_validation import (
    build_diagnosis_bundle,
    build_diagnosis_cases,
)
from benchmarks.density_matrix.performance_evidence.benchmark_bundle_validation import (
    build_performance_evidence_benchmark_package,
)
from benchmarks.density_matrix.performance_evidence.summary_consistency_validation import (
    build_summary_consistency_bundle,
)


def test_performance_evidence_benchmark_matrix_covers_required_inventory():
    cases = build_benchmark_matrix_cases()
    assert len(cases) == 34
    assert sum(case["case_kind"] == "continuity" for case in cases) == 4
    assert sum(case["case_kind"] == "structured_family" for case in cases) == 30
    assert sum(case["representative_review_case"] for case in cases) == 6
    assert {
        case["seed"] for case in cases if case["case_kind"] == "structured_family"
    } == {
        PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED,
        PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED + 1,
        PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED + 2,
    }
    assert {
        case["noise_pattern"] for case in cases if case["noise_pattern"] is not None
    } == {"sparse", "periodic", "dense"}


def test_performance_evidence_benchmark_matrix_bundle_core_fields_are_stable():
    bundle = build_benchmark_matrix_bundle(build_benchmark_matrix_cases())
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["continuity_cases"] == 4
    assert bundle["summary"]["structured_cases"] == 30
    assert bundle["summary"]["representative_review_cases"] == 6


@pytest.fixture(scope="module")
def counted_supported_cases_fixture():
    return build_counted_supported_cases()


def test_performance_evidence_counted_supported_gate_passes_full_matrix(
    counted_supported_cases_fixture,
):
    cases = counted_supported_cases_fixture
    assert len(cases) == 34
    assert all(case["counted_supported_benchmark_case"] for case in cases)
    assert all(case["record_schema_version"] == PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION for case in cases)
    assert all(case["supported_runtime_case"] for case in cases)


def test_performance_evidence_counted_supported_bundle_core_fields_are_stable(
    counted_supported_cases_fixture,
):
    bundle = build_counted_supported_bundle(counted_supported_cases_fixture)
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["counted_supported_cases"] == 34
    assert bundle["summary"]["excluded_cases"] == 0


def test_performance_evidence_positive_threshold_review_surface_is_bounded_and_auditable():
    cases = build_positive_threshold_cases()
    assert len(cases) == 6
    assert all(case["representative_review_case"] for case in cases)
    assert all(case["sequential_median_runtime_ms"] is not None for case in cases)
    assert all(case["fused_median_runtime_ms"] is not None for case in cases)


def test_performance_evidence_positive_threshold_bundle_core_fields_are_stable():
    bundle = build_positive_threshold_bundle(build_positive_threshold_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 6
    assert len(bundle["summary"]["review_groups"]) == 6


def test_performance_evidence_sensitivity_matrix_covers_required_groups():
    cases = build_sensitivity_matrix_cases()
    assert len(cases) == 30
    assert {case["family_name"] for case in cases} == {
        "layered_nearest_neighbor",
        "seeded_random_layered",
        "partition_stress_ladder",
    }
    assert {case["qbit_num"] for case in cases} == {8, 10}


def test_performance_evidence_sensitivity_matrix_bundle_core_fields_are_stable():
    bundle = build_sensitivity_matrix_bundle(build_sensitivity_matrix_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 30
    assert bundle["summary"]["full_noise_groups"] == 6
    assert bundle["summary"]["full_seed_groups"] == 6


@pytest.fixture(scope="module")
def metric_surface_cases_fixture():
    return build_metric_surface_cases()


def test_performance_evidence_metric_surface_is_complete(metric_surface_cases_fixture):
    cases = metric_surface_cases_fixture
    assert len(cases) == 34
    assert all(case["runtime_ms"] is not None for case in cases)
    assert all(case["peak_rss_kb"] is not None for case in cases)
    assert all(case["planning_time_ms"] is not None for case in cases)


def test_performance_evidence_metric_surface_bundle_core_fields_are_stable(
    metric_surface_cases_fixture,
):
    bundle = build_metric_surface_bundle(metric_surface_cases_fixture)
    assert bundle["status"] == "pass"
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["representative_review_cases"] == 6
    assert bundle["summary"]["median_timed_cases"] == 6


def test_performance_evidence_diagnosis_surface_is_explicit():
    cases = build_diagnosis_cases()
    assert len(cases) >= 1
    assert all(case["diagnosis_only_case"] for case in cases)
    assert all(case["diagnosis_reasons"] for case in cases)


def test_performance_evidence_diagnosis_bundle_core_fields_are_stable():
    bundle = build_diagnosis_bundle(build_diagnosis_cases())
    assert bundle["status"] == "pass"
    assert bundle["record_schema_version"] == PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION
    assert bundle["summary"]["correctness_evidence_boundary_cases"] >= 1


def test_performance_evidence_benchmark_package_is_complete():
    bundle = build_performance_evidence_benchmark_package()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION
    assert bundle["summary"]["total_cases"] == 34
    assert bundle["summary"]["representative_review_cases"] == 6
    assert bundle["summary"]["correctness_evidence_boundary_cases"] == len(bundle["negative_cases"])


def test_performance_evidence_summary_consistency_closes_from_positive_or_diagnosis_path():
    bundle = build_summary_consistency_bundle()
    assert bundle["status"] == "pass"
    assert bundle["schema_version"] == PERFORMANCE_EVIDENCE_SUMMARY_SCHEMA_VERSION
    assert bundle["summary"]["summary_consistency_pass"] is True
    assert bundle["summary"]["main_benchmark_claim_completed"] is True
    assert (
        bundle["summary"]["positive_benchmark_claim_completed"]
        or bundle["summary"]["diagnosis_grounded_closure_completed"]
    )
