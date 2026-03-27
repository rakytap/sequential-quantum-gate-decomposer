"""Phase 3 partitioned density runtime (public API re-exports)."""

from __future__ import annotations

from squander.partitioning.noisy_runtime_core import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_FUSED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
    PHASE3_FUSION_KIND_NOISE_BOUNDARY,
    PHASE3_FUSION_KIND_UNITARY_ISLAND,
    PHASE3_RUNTIME_PATH_BASELINE,
    PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
    PHASE3_RUNTIME_PATH_SEQUENTIAL_REFERENCE,
    PHASE3_RUNTIME_VALIDITY_TOL,
    NoisyRuntimeExecutionResult,
    NoisyRuntimeFusedRegionRecord,
    NoisyRuntimePartitionRecord,
    _build_runtime_circuit,
    _validate_runtime_operation_alignment,
    build_runtime_audit_record,
    execute_partitioned_density,
    execute_partitioned_density_fused,
    execute_sequential_density_reference,
    runtime_partition_audit_dict,
    validate_runtime_request,
)
from squander.partitioning.noisy_runtime_errors import NoisyRuntimeValidationError

__all__ = [
    "PHASE3_FUSION_CLASS_DEFERRED",
    "PHASE3_FUSION_CLASS_FUSED",
    "PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED",
    "PHASE3_FUSION_KIND_NOISE_BOUNDARY",
    "PHASE3_FUSION_KIND_UNITARY_ISLAND",
    "PHASE3_RUNTIME_PATH_BASELINE",
    "PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS",
    "PHASE3_RUNTIME_PATH_SEQUENTIAL_REFERENCE",
    "PHASE3_RUNTIME_VALIDITY_TOL",
    "NoisyRuntimeExecutionResult",
    "NoisyRuntimeFusedRegionRecord",
    "NoisyRuntimePartitionRecord",
    "NoisyRuntimeValidationError",
    "_build_runtime_circuit",
    "_validate_runtime_operation_alignment",
    "build_runtime_audit_record",
    "execute_partitioned_density",
    "execute_partitioned_density_fused",
    "execute_sequential_density_reference",
    "runtime_partition_audit_dict",
    "validate_runtime_request",
]
