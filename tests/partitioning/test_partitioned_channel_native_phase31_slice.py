"""Phase 3.1 slice: channel-native runtime vs sequential reference (1q microcase)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF,
    PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    execute_partitioned_density_channel_native,
    execute_sequential_density_reference,
)
from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError
from tests.partitioning.fixtures.runtime import (
    PHASE3_RUNTIME_DENSITY_TOL,
    build_initial_parameters,
    build_density_comparison_metrics,
)
from tests.partitioning.fixtures.workloads import (
    _u3,
    build_phase31_microcase_descriptor_set,
)


def test_phase31_channel_native_1q_microcase_matches_sequential_reference():
    descriptor_set = build_phase31_microcase_descriptor_set(
        "phase31_microcase_1q_u3_local_noise_chain"
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    result = execute_partitioned_density_channel_native(descriptor_set, parameters)

    assert result.runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
    assert result.requested_runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
    assert result.actual_fused_execution is True
    assert result.fused_region_count == 1
    assert len(result.fused_regions) == 1
    assert (
        result.fused_regions[0].candidate_kind == PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF
    )
    metrics = build_density_comparison_metrics(
        result.density_matrix, reference
    )
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
    assert result.rho_is_valid is True


def test_phase31_channel_native_parametric_noise_clamped_matches_sequential():
    """Out-of-range parametric noise rate is clamped like NoisyCircuit / noise_operation.cpp."""
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=1,
        source_type="microcase_builder",
        workload_id="slice_parametric_noise_clamp",
        operation_specs=[
            _u3(0),
            {
                "kind": "noise",
                "name": "local_depolarizing",
                "target_qbit": 0,
                "source_gate_index": 0,
                "param_count": 1,
            },
        ],
    )
    descriptor_set = build_partition_descriptor_set(surface)
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    parameters[3] = 1.5
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    result = execute_partitioned_density_channel_native(descriptor_set, parameters)
    metrics = build_density_comparison_metrics(result.density_matrix, reference)
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase31_channel_native_rejects_pure_unitary_motif():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=1,
        source_type="microcase_builder",
        workload_id="slice_negative_pure_u3_only",
        operation_specs=[_u3(0)],
    )
    descriptor_set = build_partition_descriptor_set(surface)
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        execute_partitioned_density_channel_native(descriptor_set, parameters)
    assert excinfo.value.first_unsupported_condition == "channel_native_noise_presence"
