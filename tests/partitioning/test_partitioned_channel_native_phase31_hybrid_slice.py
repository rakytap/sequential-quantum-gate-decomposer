"""Phase 3.1 hybrid whole-workload path: runtime (P31-S07-E01) and counted continuity (P31-S08 + P31-S10).

**P31-S07-E01 (runtime / routing):** ``phase31_channel_native_hybrid``,
``execute_partitioned_density_channel_native_hybrid``, partition-level route
reasons, unsupported-by-both failure, and structured smoke.

**Counted hybrid continuity gates (Task 3):** ``P31-S08-E01`` —
``phase2_xxz_hea_q4_continuity``; ``P31-S10-E01`` — ``phase2_xxz_hea_q6_continuity``.
Each asserts full-density exactness vs sequential oracle and a frozen aggregated
route summary (partition count, runtime-class counts, route-reason counts).

**Correctness package (``P31-S10-E02``):** Stage-A sibling builders and
``phase31_validation_pipeline.py`` emit the bounded six-case package under
``artifacts/correctness_evidence/phase31_stage_a/`` (default Phase 3 pipeline
unchanged).

**Still deferred:** counted structured performance matrix (``P31-ADR-010``);
default-pipeline Stage-B switch; publication closure.

Strict motif-proof tests stay in ``test_partitioned_channel_native_phase31_*slice.py``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
    build_phase3_continuity_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID,
    NoisyRuntimeExecutionResult,
    execute_partitioned_density_channel_native_hybrid,
    execute_sequential_density_reference,
)
from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError
from tests.partitioning.fixtures.continuity import build_phase2_continuity_vqe
from tests.partitioning.fixtures.runtime import (
    PHASE3_RUNTIME_DENSITY_TOL,
    build_density_comparison_metrics,
    build_initial_parameters,
)
from tests.partitioning.fixtures.workloads import (
    _noise_value,
    build_phase31_structured_descriptor_set,
)

# Frozen P31-ADR-012 hybrid route-reason vocabulary (subset used by classifier).
_FROZEN_HYBRID_ROUTE_REASONS = frozenset(
    {
        "eligible_channel_native_motif",
        "pure_unitary_partition",
        "channel_native_noise_presence",
        "channel_native_qubit_span",
        "channel_native_support_surface",
    }
)

# Counted hybrid continuity anchor — aggregated route summary for current planner
# partitioning (update deliberately if partitioning changes).
_COUNTED_HYBRID_CONTINUITY_Q4_WORKLOAD_ID = "phase2_xxz_hea_q4_continuity"
_EXPECTED_Q4_HYBRID_PARTITION_COUNT = 5
_EXPECTED_Q4_RUNTIME_CLASS_COUNTS = {
    "phase31_channel_native": 2,
    "phase3_unitary_island_fused": 3,
}
_EXPECTED_Q4_ROUTE_REASON_COUNTS = {
    "eligible_channel_native_motif": 2,
    "pure_unitary_partition": 3,
}

_COUNTED_HYBRID_CONTINUITY_Q6_WORKLOAD_ID = "phase2_xxz_hea_q6_continuity"
_EXPECTED_Q6_HYBRID_PARTITION_COUNT = 7
_EXPECTED_Q6_RUNTIME_CLASS_COUNTS = {
    "phase31_channel_native": 2,
    "phase3_unitary_island_fused": 5,
}
_EXPECTED_Q6_ROUTE_REASON_COUNTS = {
    "eligible_channel_native_motif": 2,
    "pure_unitary_partition": 5,
}


def _hybrid_partition_route_summary(
    result: NoisyRuntimeExecutionResult,
) -> dict[str, Any]:
    """Aggregate hybrid route metadata from ``result.partitions`` (test-local)."""
    classes = [rec.partition_runtime_class for rec in result.partitions]
    reasons = [rec.partition_route_reason for rec in result.partitions]
    return {
        "partition_count": len(result.partitions),
        "runtime_class_counts": dict(Counter(c for c in classes if c is not None)),
        "route_reason_counts": dict(Counter(r for r in reasons if r is not None)),
    }


# --- P31-S08-E01: counted hybrid continuity correctness gate ---


def test_phase31_s08_e01_counted_hybrid_continuity_q4_matches_sequential_oracle():
    vqe, _, _ = build_phase2_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    assert descriptor_set.workload_id == _COUNTED_HYBRID_CONTINUITY_Q4_WORKLOAD_ID

    parameters = build_initial_parameters(descriptor_set.parameter_count)
    hybrid = execute_partitioned_density_channel_native_hybrid(descriptor_set, parameters)
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    metrics = build_density_comparison_metrics(hybrid.density_matrix, reference)

    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert hybrid.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
    assert hybrid.rho_is_valid is True
    assert hybrid.runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID
    assert hybrid.requested_runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID

    summary = _hybrid_partition_route_summary(hybrid)
    assert summary["partition_count"] == _EXPECTED_Q4_HYBRID_PARTITION_COUNT
    assert summary["runtime_class_counts"] == _EXPECTED_Q4_RUNTIME_CLASS_COUNTS
    assert summary["route_reason_counts"] == _EXPECTED_Q4_ROUTE_REASON_COUNTS

    classes = [rec.partition_runtime_class for rec in hybrid.partitions]
    assert any(c == "phase31_channel_native" for c in classes)
    assert any(
        c in ("phase3_unitary_island_fused", "phase3_supported_unfused") for c in classes
    )

    reasons = [rec.partition_route_reason for rec in hybrid.partitions]
    assert all(r in _FROZEN_HYBRID_ROUTE_REASONS for r in reasons)


# --- P31-S10-E01: second counted hybrid continuity anchor (q6) ---


def test_phase31_s10_e01_counted_hybrid_continuity_q6_matches_sequential_oracle():
    vqe, _, _ = build_phase2_continuity_vqe(6)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    assert descriptor_set.workload_id == _COUNTED_HYBRID_CONTINUITY_Q6_WORKLOAD_ID

    parameters = build_initial_parameters(descriptor_set.parameter_count)
    hybrid = execute_partitioned_density_channel_native_hybrid(
        descriptor_set, parameters
    )
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    metrics = build_density_comparison_metrics(hybrid.density_matrix, reference)

    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert hybrid.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
    assert hybrid.rho_is_valid is True
    assert hybrid.runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID
    assert hybrid.requested_runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID

    summary = _hybrid_partition_route_summary(hybrid)
    assert summary["partition_count"] == _EXPECTED_Q6_HYBRID_PARTITION_COUNT
    assert summary["runtime_class_counts"] == _EXPECTED_Q6_RUNTIME_CLASS_COUNTS
    assert summary["route_reason_counts"] == _EXPECTED_Q6_ROUTE_REASON_COUNTS

    classes = [rec.partition_runtime_class for rec in hybrid.partitions]
    assert any(c == "phase31_channel_native" for c in classes)
    assert any(
        c in ("phase3_unitary_island_fused", "phase3_supported_unfused") for c in classes
    )

    reasons = [rec.partition_route_reason for rec in hybrid.partitions]
    assert all(r in _FROZEN_HYBRID_ROUTE_REASONS for r in reasons)


# --- P31-S07-E01: unsupported-by-both must not be absorbed by hybrid routing ---


def test_phase31_hybrid_rejects_unsupported_gate_not_routed_quietly():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=2,
        source_type="microcase_builder",
        workload_id="hybrid_negative_rx_relaxed_surface",
        operation_specs=[
            {
                "kind": "gate",
                "name": "RX",
                "target_qbit": 0,
                "param_count": 1,
            },
            {
                "kind": "noise",
                "name": "phase_damping",
                "target_qbit": 0,
                "source_gate_index": 0,
                "fixed_value": _noise_value("phase_damping"),
                "param_count": 0,
            },
        ],
        strict_phase3_support=False,
    )
    ds = build_partition_descriptor_set(surface)
    parameters = build_initial_parameters(ds.parameter_count)
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        execute_partitioned_density_channel_native_hybrid(ds, parameters)
    assert excinfo.value.first_unsupported_condition == "gate_name"


# --- Non-counted support smoke for later performance slice (P31-S09); not Task 3 closure ---


def test_phase31_hybrid_structured_pair_repeat_q8_dense_smoke():
    descriptor_set = build_phase31_structured_descriptor_set(
        "phase31_pair_repeat",
        qbit_num=8,
        noise_pattern="dense",
        seed=20260318,
    )
    parameters = build_initial_parameters(descriptor_set.parameter_count)
    hybrid = execute_partitioned_density_channel_native_hybrid(descriptor_set, parameters)
    reference = execute_sequential_density_reference(descriptor_set, parameters)
    metrics = build_density_comparison_metrics(hybrid.density_matrix, reference)

    assert hybrid.runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    reasons = [rec.partition_route_reason for rec in hybrid.partitions]
    assert all(r in _FROZEN_HYBRID_ROUTE_REASONS for r in reasons)
