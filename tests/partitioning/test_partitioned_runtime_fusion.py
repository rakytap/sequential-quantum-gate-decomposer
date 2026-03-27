from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander.partitioning.noisy_runtime import (
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    execute_partitioned_density,
    execute_partitioned_density_fused,
)
from tests.partitioning.fixtures.fusion_cases import iter_fusion_structured_cases
from tests.partitioning.fixtures.runtime import (
    PHASE3_RUNTIME_DENSITY_TOL,
    execute_fused_with_reference,
)


def _first_structured_case():
    return next(iter(iter_fusion_structured_cases()))


def test_partitioned_runtime_direct_fused_runtime_path_differs_from_baseline_when_exercised():
    _metadata, descriptor_set, parameters = _first_structured_case()
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


def test_partitioned_runtime_direct_fused_result_matches_sequential_reference():
    _metadata, descriptor_set, parameters = _first_structured_case()
    result, _, density_metrics = execute_fused_with_reference(descriptor_set, parameters)

    assert result.actual_fused_execution is True
    assert density_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert density_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
