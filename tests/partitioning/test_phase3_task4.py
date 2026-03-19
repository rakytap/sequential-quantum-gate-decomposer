from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.partitioned_runtime.fused_classification_validation import (
    build_cases as build_story5_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_eligibility_validation import (
    build_cases as build_story1_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_performance_validation import (
    build_cases as build_story7_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_runtime_audit_validation import (
    build_cases as build_story6_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_semantics_validation import (
    build_cases as build_story4_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_surface_reuse_validation import (
    build_cases as build_story3_cases,
)
from benchmarks.density_matrix.partitioned_runtime.structured_fused_runtime_validation import (
    build_cases as build_story2_cases,
)
from benchmarks.density_matrix.partitioned_runtime.task4_case_selection import (
    iter_task4_structured_cases,
)
from squander.partitioning.noisy_runtime import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_FUSED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    execute_partitioned_density,
    execute_partitioned_density_fused,
)


def _first_structured_case():
    return next(iter(iter_task4_structured_cases()))


def test_phase3_task4_story1_baseline_surface_exposes_eligible_unitary_regions():
    cases = build_story1_cases()
    assert len(cases) == 3
    assert {case["case_kind"] for case in cases} == {
        "continuity",
        "microcase",
        "structured_family",
    }
    for case in cases:
        assert case["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
        assert case["eligible_unitary_region_count"] > 0


def test_phase3_task4_story2_fused_runtime_executes_representative_structured_cases():
    cases = build_story2_cases()
    assert {case["qbit_num"] for case in cases} == {8, 10}
    for case in cases:
        assert case["structured_fused_runtime_pass"] is True
        assert case["runtime_path"] == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
        assert case["fused_region_count"] > 0


def test_phase3_task4_story2_direct_fused_runtime_path_differs_from_baseline_when_exercised():
    metadata, descriptor_set, parameters = _first_structured_case()
    baseline_result = execute_partitioned_density(descriptor_set, parameters)
    fused_result = execute_partitioned_density_fused(descriptor_set, parameters)

    assert baseline_result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert fused_result.actual_fused_execution is True
    assert fused_result.runtime_path == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
    assert fused_result.fused_region_count > 0


def test_phase3_task4_story3_fused_surface_cases_share_audit_shape():
    cases = build_story3_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    fused_region_key_sets = {
        frozenset(case["fused_regions"][0].keys()) for case in cases if case["fused_regions"]
    }

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(fused_region_key_sets) == 1


def test_phase3_task4_story4_fused_cases_match_sequential_reference():
    cases = build_story4_cases()
    assert len(cases) >= 2
    for case in cases:
        assert case["fused_semantics_pass"] is True
        assert case["actual_fused_execution"] is True
        assert case["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase3_task4_story4_direct_fused_result_matches_sequential_reference():
    metadata, descriptor_set, parameters = _first_structured_case()
    result, _, density_metrics = execute_fused_with_reference(descriptor_set, parameters)

    assert result.actual_fused_execution is True
    assert density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase3_task4_story5_classification_matrix_exposes_all_required_categories():
    cases = build_story5_cases()
    observed = {classification for case in cases for classification in case["classifications"]}

    assert observed >= {
        PHASE3_FUSION_CLASS_FUSED,
        PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
        PHASE3_FUSION_CLASS_DEFERRED,
    }
    for case in cases:
        assert case["classification_pass"] is True


def test_phase3_task4_story6_fused_audit_cases_share_shape():
    cases = build_story6_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    fused_region_key_sets = {
        frozenset(case["fused_regions"][0].keys()) for case in cases if case["fused_regions"]
    }
    runtime_paths = {case["runtime_path"] for case in cases}

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(fused_region_key_sets) == 1
    assert runtime_paths <= {
        PHASE3_RUNTIME_PATH_BASELINE,
        PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    }


@pytest.mark.slow
def test_phase3_task4_story7_threshold_or_diagnosis_rule_closes():
    cases = build_story7_cases()
    assert {case["qbit_num"] for case in cases} == {8, 10}
    for case in cases:
        assert case["actual_fused_execution"] is True
        assert case["threshold_or_diagnosis_pass"] is True
        assert case["correctness_pass"] is True
