from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    execute_fused_with_reference,
)
from benchmarks.density_matrix.partitioned_runtime.fused_classification_validation import (
    build_cases as build_fused_classification_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_eligibility_validation import (
    build_cases as build_fused_eligibility_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_performance_validation import (
    build_cases as build_fused_performance_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_runtime_audit_validation import (
    build_cases as build_fused_runtime_audit_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_semantics_validation import (
    build_cases as build_fused_semantics_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fused_surface_reuse_validation import (
    build_cases as build_fused_surface_reuse_cases,
)
from benchmarks.density_matrix.partitioned_runtime.structured_fused_runtime_validation import (
    build_cases as build_structured_fused_runtime_cases,
)
from benchmarks.density_matrix.partitioned_runtime.fusion_case_selection import (
    iter_fusion_structured_cases,
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
    return next(iter(iter_fusion_structured_cases()))


def test_partitioned_runtime_fused_eligibility_surface_exposes_eligible_unitary_regions():
    cases = build_fused_eligibility_cases()
    assert len(cases) == 3
    assert {case["case_kind"] for case in cases} == {
        "continuity",
        "microcase",
        "structured_family",
    }
    for case in cases:
        assert case["runtime_path"] == PHASE3_RUNTIME_PATH_BASELINE
        assert case["eligible_unitary_region_count"] > 0


def test_partitioned_runtime_structured_fused_runtime_executes_representative_cases():
    cases = build_structured_fused_runtime_cases()
    assert {case["qbit_num"] for case in cases} == {8, 10}
    for case in cases:
        assert case["structured_fused_runtime_pass"] is True
        assert case["runtime_path"] == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
        assert case["fused_region_count"] > 0


def test_partitioned_runtime_direct_fused_runtime_path_differs_from_baseline_when_exercised():
    metadata, descriptor_set, parameters = _first_structured_case()
    baseline_result = execute_partitioned_density(descriptor_set, parameters)
    fused_result = execute_partitioned_density_fused(descriptor_set, parameters)

    assert baseline_result.runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert baseline_result.requested_runtime_path == PHASE3_RUNTIME_PATH_BASELINE
    assert fused_result.actual_fused_execution is True
    assert fused_result.runtime_path == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
    assert fused_result.requested_runtime_path == PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
    assert fused_result.fused_region_count > 0


def test_partitioned_runtime_fused_and_unfused_density_matrices_match():
    """Fused kernels and sequential NoisyCircuit lowering share gate semantics."""
    _, descriptor_set, parameters = _first_structured_case()
    fused = execute_partitioned_density(descriptor_set, parameters, allow_fusion=True)
    unfused = execute_partitioned_density(descriptor_set, parameters, allow_fusion=False)
    np.testing.assert_allclose(
        fused.density_matrix_numpy(),
        unfused.density_matrix_numpy(),
        atol=PHASE3_RUNTIME_DENSITY_TOL,
        rtol=0.0,
    )


def test_partitioned_runtime_fused_surface_reuse_cases_share_audit_shape():
    cases = build_fused_surface_reuse_cases()
    top_level_keys = {frozenset(case.keys()) for case in cases}
    summary_key_sets = {frozenset(case["summary"].keys()) for case in cases}
    fused_region_key_sets = {
        frozenset(case["fused_regions"][0].keys()) for case in cases if case["fused_regions"]
    }

    assert len(cases) == 3
    assert len(top_level_keys) == 1
    assert len(summary_key_sets) == 1
    assert len(fused_region_key_sets) == 1


def test_partitioned_runtime_fused_semantics_cases_match_sequential_reference():
    cases = build_fused_semantics_cases()
    assert len(cases) >= 2
    for case in cases:
        assert case["fused_semantics_pass"] is True
        assert case["actual_fused_execution"] is True
        assert case["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_partitioned_runtime_direct_fused_result_matches_sequential_reference():
    metadata, descriptor_set, parameters = _first_structured_case()
    result, _, density_metrics = execute_fused_with_reference(descriptor_set, parameters)

    assert result.actual_fused_execution is True
    assert density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_partitioned_runtime_fusion_classification_matrix_exposes_all_required_categories():
    cases = build_fused_classification_cases()
    observed = {classification for case in cases for classification in case["classifications"]}

    assert observed >= {
        PHASE3_FUSION_CLASS_FUSED,
        PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
        PHASE3_FUSION_CLASS_DEFERRED,
    }
    for case in cases:
        assert case["classification_pass"] is True


def test_partitioned_runtime_fused_runtime_audit_cases_share_shape():
    cases = build_fused_runtime_audit_cases()
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
def test_partitioned_runtime_fused_performance_threshold_or_diagnosis_rule_closes():
    cases = build_fused_performance_cases()
    assert {case["qbit_num"] for case in cases} == {8, 10}
    for case in cases:
        assert case["actual_fused_execution"] is True
        assert case["threshold_or_diagnosis_pass"] is True
        assert case["correctness_pass"] is True
