"""Focused tests for the opt-in Phase 3.1 hybrid cost-model gate."""

from __future__ import annotations

from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    execute_partitioned_density_channel_native,
    execute_partitioned_density_channel_native_hybrid,
    execute_sequential_density_reference,
)
from squander.partitioning.noisy_runtime_cost_model import (
    PHASE31_KRAUS_EXPANSION_MODEL_ID,
    PHASE31_KRAUS_EXPANSION_MODEL_VERSION,
    MotifCostFeatures,
    Phase31KrausExpansionCostModelV0,
    extract_motif_cost_features,
)
from tests.partitioning.fixtures.runtime import (
    PHASE3_RUNTIME_DENSITY_TOL,
    build_density_comparison_metrics,
    build_initial_parameters,
)
from tests.partitioning.fixtures.workloads import _cnot, _noise, _noise_value, _u3


def _build_rank_surface_descriptor_set():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=2,
        source_type="microcase_builder",
        workload_id="cost_model_rank_surface",
        operation_specs=[
            _u3(0),
            _cnot(1, 0),
            _noise(
                "local_depolarizing",
                0,
                1,
                _noise_value("local_depolarizing"),
            ),
            _noise(
                "amplitude_damping",
                1,
                1,
                _noise_value("amplitude_damping"),
            ),
            _noise("phase_damping", 0, 1, _noise_value("phase_damping")),
        ],
    )
    return build_partition_descriptor_set(surface)


def _build_skip_descriptor_set():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=1,
        source_type="microcase_builder",
        workload_id="cost_model_skip_motif",
        operation_specs=[
            _u3(0),
            _noise(
                "local_depolarizing",
                0,
                0,
                _noise_value("local_depolarizing"),
            ),
            _noise(
                "amplitude_damping",
                0,
                0,
                _noise_value("amplitude_damping"),
            ),
        ],
    )
    return build_partition_descriptor_set(surface)


def test_extract_motif_cost_features_uses_frozen_per_operation_rank_bounds():
    descriptor_set = _build_rank_surface_descriptor_set()
    partition = descriptor_set.partitions[0]

    features = extract_motif_cost_features(
        descriptor_set, partition, local_support=(0, 1)
    )

    assert features.motif_length == 5
    assert features.gate_count == 2
    assert features.noise_count == 3
    assert features.predicted_kraus_count_upper == 1 * 1 * 4 * 2 * 2
    assert features.predicted_sequential_kraus_count_sum == 1 + 1 + 4 + 2 + 2


def test_v0_skips_only_when_composed_apply_cost_exceeds_sequential_cost():
    model = Phase31KrausExpansionCostModelV0()
    skip = model.decide(
        MotifCostFeatures(
            support_qubit_count=2,
            motif_length=3,
            gate_count=1,
            noise_count=2,
            predicted_kraus_count_upper=8,
            predicted_sequential_kraus_count_sum=7,
            qbit_num=3,
        )
    )
    tie = model.decide(
        MotifCostFeatures(
            support_qubit_count=1,
            motif_length=2,
            gate_count=0,
            noise_count=2,
            predicted_kraus_count_upper=4,
            predicted_sequential_kraus_count_sum=4,
            qbit_num=1,
        )
    )

    assert skip.decision == "skip_to_phase3"
    assert skip.route_reason == "cost_model_skip_kraus_expansion"
    assert skip.predicted_apply_cost == 8 * 256
    assert skip.predicted_baseline_cost == 7 * 256
    assert skip.model_id == PHASE31_KRAUS_EXPANSION_MODEL_ID
    assert skip.model_version == PHASE31_KRAUS_EXPANSION_MODEL_VERSION
    assert tie.decision == "fuse_channel_native"


def test_hybrid_cost_model_flag_off_preserves_existing_route_and_audit_shape():
    descriptor_set = _build_skip_descriptor_set()
    parameters = build_initial_parameters(descriptor_set.parameter_count)

    result = execute_partitioned_density_channel_native_hybrid(
        descriptor_set, parameters
    )
    partition_payload = result.partitions[0].to_dict(descriptor_set)

    assert result.partitions[0].partition_runtime_class == "phase31_channel_native"
    assert (
        result.partitions[0].partition_route_reason
        == "eligible_channel_native_motif"
    )
    assert "cost_model_id" not in partition_payload
    assert "cost_model_decision" not in partition_payload


def test_hybrid_cost_model_flag_on_skips_with_audit_and_preserves_exactness():
    descriptor_set = _build_skip_descriptor_set()
    parameters = build_initial_parameters(descriptor_set.parameter_count)

    result = execute_partitioned_density_channel_native_hybrid(
        descriptor_set,
        parameters,
        enable_channel_native_cost_model=True,
    )
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    metrics = build_density_comparison_metrics(result.density_matrix, reference)
    record = result.partitions[0]
    payload = record.to_dict(descriptor_set)

    assert record.partition_runtime_class in (
        "phase3_unitary_island_fused",
        "phase3_supported_unfused",
    )
    assert record.partition_route_reason == "cost_model_skip_kraus_expansion"
    assert payload["cost_model_id"] == PHASE31_KRAUS_EXPANSION_MODEL_ID
    assert payload["cost_model_version"] == PHASE31_KRAUS_EXPANSION_MODEL_VERSION
    assert payload["predicted_kraus_count_upper"] == 8
    assert payload["predicted_apply_cost"] > payload["predicted_baseline_cost"]
    assert payload["cost_model_decision"] == "skip_to_phase3"
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_strict_channel_native_path_never_invokes_cost_model(monkeypatch):
    descriptor_set = _build_skip_descriptor_set()
    parameters = build_initial_parameters(descriptor_set.parameter_count)

    class _FailIfCalled:
        def decide(self, features):
            raise AssertionError("strict path called the hybrid cost model")

    monkeypatch.setattr(
        "squander.partitioning.noisy_runtime_cost_model.DEFAULT_CHANNEL_NATIVE_COST_MODEL",
        _FailIfCalled(),
    )
    result = execute_partitioned_density_channel_native(descriptor_set, parameters)

    assert result.actual_fused_execution is True
