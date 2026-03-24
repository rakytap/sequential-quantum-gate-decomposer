from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from squander.density_matrix import DensityMatrix, NoisyCircuit
from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    SUPPORTED_GATE_NAMES,
    SUPPORTED_NOISE_NAMES,
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorMember,
    NoisyPartitionDescriptorSet,
    validate_partition_descriptor_set,
)

PHASE3_RUNTIME_SCHEMA_VERSION = "phase3_partitioned_density_runtime_v1"
PHASE3_RUNTIME_PATH_BASELINE = "partitioned_density_descriptor_baseline"
PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS = (
    "partitioned_density_descriptor_fused_unitary_islands"
)
PHASE3_RUNTIME_PATH_SEQUENTIAL_REFERENCE = "sequential_density_descriptor_reference"
PHASE3_RUNTIME_VALIDITY_TOL = 1e-10

PHASE3_FUSION_KIND_UNITARY_ISLAND = "unitary_island"
PHASE3_FUSION_KIND_NOISE_BOUNDARY = "noise_boundary"

PHASE3_FUSION_CLASS_FUSED = "actually_fused"
PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED = "supported_but_unfused"
PHASE3_FUSION_CLASS_DEFERRED = "deferred_or_unsupported_candidate"


class NoisyRuntimeValidationError(ValueError):
    """Structured runtime validation error for the Phase 3 partitioned runtime."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        workload_id: str,
        descriptor_schema_version: str,
        runtime_path: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.first_unsupported_condition = first_unsupported_condition
        self.failure_stage = failure_stage
        self.source_type = source_type
        self.requested_mode = requested_mode
        self.workload_id = workload_id
        self.descriptor_schema_version = descriptor_schema_version
        self.runtime_path = runtime_path
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "unsupported_category": self.category,
            "first_unsupported_condition": self.first_unsupported_condition,
            "failure_stage": self.failure_stage,
            "source_type": self.source_type,
            "requested_mode": self.requested_mode,
            "workload_id": self.workload_id,
            "descriptor_schema_version": self.descriptor_schema_version,
            "runtime_path": self.runtime_path,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NoisyRuntimePartitionRecord:
    partition_index: int
    canonical_operation_indices: tuple[int, ...]
    local_to_global_qbits: tuple[int, ...]
    global_to_local_qbits: tuple[tuple[int, int], ...]
    requires_remap: bool
    parameter_routing: tuple[tuple[int, int, int], ...]
    partition_parameter_count: int
    member_count: int
    gate_count: int
    noise_count: int
    operation_names: tuple[str, ...]
    operation_kinds: tuple[str, ...]
    runtime_circuit_qbit_num: int
    runtime_circuit_parameter_count: int

    @property
    def partition_qubit_span(self) -> tuple[int, ...]:
        return self.local_to_global_qbits

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_index": self.partition_index,
            "member_count": self.member_count,
            "gate_count": self.gate_count,
            "noise_count": self.noise_count,
            "canonical_operation_indices": list(self.canonical_operation_indices),
            "partition_qubit_span": list(self.partition_qubit_span),
            "local_to_global_qbits": list(self.local_to_global_qbits),
            "global_to_local_qbits": [
                {"global_qbit": global_qbit, "local_qbit": local_qbit}
                for global_qbit, local_qbit in self.global_to_local_qbits
            ],
            "requires_remap": self.requires_remap,
            "partition_parameter_count": self.partition_parameter_count,
            "parameter_routing": [
                {
                    "global_param_start": global_param_start,
                    "local_param_start": local_param_start,
                    "param_count": param_count,
                }
                for global_param_start, local_param_start, param_count in self.parameter_routing
            ],
            "operation_names": list(self.operation_names),
            "operation_kinds": list(self.operation_kinds),
            "runtime_circuit_qbit_num": self.runtime_circuit_qbit_num,
            "runtime_circuit_parameter_count": self.runtime_circuit_parameter_count,
        }


@dataclass(frozen=True)
class NoisyRuntimeFusedRegionRecord:
    partition_index: int
    candidate_kind: str
    classification: str
    reason: str
    partition_member_indices: tuple[int, ...]
    canonical_operation_indices: tuple[int, ...]
    operation_names: tuple[str, ...]
    global_target_qbits: tuple[int, ...]
    local_target_qbits: tuple[int, ...]
    member_count: int
    gate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_index": self.partition_index,
            "candidate_kind": self.candidate_kind,
            "classification": self.classification,
            "reason": self.reason,
            "partition_member_indices": list(self.partition_member_indices),
            "canonical_operation_indices": list(self.canonical_operation_indices),
            "operation_names": list(self.operation_names),
            "global_target_qbits": list(self.global_target_qbits),
            "local_target_qbits": list(self.local_target_qbits),
            "member_count": self.member_count,
            "gate_count": self.gate_count,
        }


@dataclass(frozen=True)
class NoisyRuntimeExecutionResult:
    runtime_schema_version: str
    planner_schema_version: str
    descriptor_schema_version: str
    requested_mode: str
    source_type: str
    workload_id: str
    qbit_num: int
    parameter_count: int
    runtime_path: str
    fallback_used: bool
    exact_output_present: bool
    density_matrix: DensityMatrix
    partitions: tuple[NoisyRuntimePartitionRecord, ...]
    fused_regions: tuple[NoisyRuntimeFusedRegionRecord, ...]
    runtime_ms: float
    peak_rss_kb: int

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "workload_id": self.workload_id,
        }

    @property
    def partition_count(self) -> int:
        return len(self.partitions)

    @property
    def descriptor_member_count(self) -> int:
        return sum(partition.member_count for partition in self.partitions)

    @property
    def gate_count(self) -> int:
        return sum(partition.gate_count for partition in self.partitions)

    @property
    def noise_count(self) -> int:
        return sum(partition.noise_count for partition in self.partitions)

    @property
    def max_partition_span(self) -> int:
        return max(
            (len(partition.partition_qubit_span) for partition in self.partitions),
            default=0,
        )

    @property
    def partition_member_counts(self) -> tuple[int, ...]:
        return tuple(partition.member_count for partition in self.partitions)

    @property
    def remapped_partition_count(self) -> int:
        return sum(partition.requires_remap for partition in self.partitions)

    @property
    def parameter_routing_segment_count(self) -> int:
        return sum(len(partition.parameter_routing) for partition in self.partitions)

    @property
    def fused_region_count(self) -> int:
        return sum(
            region.classification == PHASE3_FUSION_CLASS_FUSED
            for region in self.fused_regions
        )

    @property
    def supported_unfused_region_count(self) -> int:
        return sum(
            region.classification == PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
            for region in self.fused_regions
        )

    @property
    def deferred_region_count(self) -> int:
        return sum(
            region.classification == PHASE3_FUSION_CLASS_DEFERRED
            for region in self.fused_regions
        )

    @property
    def fused_gate_count(self) -> int:
        return sum(
            region.gate_count
            for region in self.fused_regions
            if region.classification == PHASE3_FUSION_CLASS_FUSED
        )

    @property
    def supported_unfused_gate_count(self) -> int:
        return sum(
            region.gate_count
            for region in self.fused_regions
            if region.classification == PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
        )

    @property
    def actual_fused_execution(self) -> bool:
        return self.fused_region_count > 0

    @property
    def trace(self) -> complex:
        return complex(self.density_matrix.trace())

    @property
    def trace_deviation(self) -> float:
        return float(abs(self.trace - 1.0))

    @property
    def rho_is_valid(self) -> bool:
        return bool(self.density_matrix.is_valid(tol=PHASE3_RUNTIME_VALIDITY_TOL))

    @property
    def purity(self) -> float:
        return float(self.density_matrix.purity())

    def density_matrix_numpy(self) -> np.ndarray:
        return np.asarray(self.density_matrix.to_numpy())

    def build_exact_output_record(
        self, *, include_density_matrix: bool = False
    ) -> dict[str, Any]:
        record = {
            "matrix_included": include_density_matrix,
            "shape": [self.density_matrix.dim, self.density_matrix.dim],
            "trace_real": float(np.real(self.trace)),
            "trace_imag": float(np.imag(self.trace)),
            "trace_deviation": self.trace_deviation,
            "rho_is_valid": self.rho_is_valid,
            "rho_is_valid_tol": PHASE3_RUNTIME_VALIDITY_TOL,
            "purity": self.purity,
            "density_real": None,
            "density_imag": None,
        }
        if include_density_matrix:
            rho_np = self.density_matrix_numpy()
            record["density_real"] = np.real(rho_np).tolist()
            record["density_imag"] = np.imag(rho_np).tolist()
        return record

    def to_dict(self, *, include_density_matrix: bool = False) -> dict[str, Any]:
        return {
            "runtime_schema_version": self.runtime_schema_version,
            "planner_schema_version": self.planner_schema_version,
            "descriptor_schema_version": self.descriptor_schema_version,
            "requested_mode": self.requested_mode,
            "source_type": self.source_type,
            "workload_id": self.workload_id,
            "provenance": self.provenance,
            "qbit_num": self.qbit_num,
            "parameter_count": self.parameter_count,
            "runtime_path": self.runtime_path,
            "summary": {
                "qbit_num": self.qbit_num,
                "parameter_count": self.parameter_count,
                "runtime_path": self.runtime_path,
                "fallback_used": self.fallback_used,
                "exact_output_present": self.exact_output_present,
                "partition_count": self.partition_count,
                "descriptor_member_count": self.descriptor_member_count,
                "gate_count": self.gate_count,
                "noise_count": self.noise_count,
                "max_partition_span": self.max_partition_span,
                "partition_member_counts": list(self.partition_member_counts),
                "remapped_partition_count": self.remapped_partition_count,
                "parameter_routing_segment_count": self.parameter_routing_segment_count,
                "fused_region_count": self.fused_region_count,
                "supported_unfused_region_count": self.supported_unfused_region_count,
                "deferred_region_count": self.deferred_region_count,
                "fused_gate_count": self.fused_gate_count,
                "supported_unfused_gate_count": self.supported_unfused_gate_count,
                "actual_fused_execution": self.actual_fused_execution,
                "runtime_ms": self.runtime_ms,
                "peak_rss_kb": self.peak_rss_kb,
                "density_trace_real": float(np.real(self.trace)),
                "density_trace_imag": float(np.imag(self.trace)),
                "trace_deviation": self.trace_deviation,
                "rho_is_valid": self.rho_is_valid,
                "rho_is_valid_tol": PHASE3_RUNTIME_VALIDITY_TOL,
                "purity": self.purity,
            },
            "exact_output": self.build_exact_output_record(
                include_density_matrix=include_density_matrix
            ),
            "partitions": [partition.to_dict() for partition in self.partitions],
            "fused_regions": [region.to_dict() for region in self.fused_regions],
        }


def _runtime_error(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    category: str,
    first_unsupported_condition: str,
    failure_stage: str,
    runtime_path: str,
    reason: str,
) -> NoisyRuntimeValidationError:
    return NoisyRuntimeValidationError(
        category=category,
        first_unsupported_condition=first_unsupported_condition,
        failure_stage=failure_stage,
        source_type=descriptor_set.source_type,
        requested_mode=descriptor_set.requested_mode,
        workload_id=descriptor_set.workload_id,
        descriptor_schema_version=descriptor_set.schema_version,
        runtime_path=runtime_path,
        reason=reason,
    )


def _coerce_parameter_vector(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: Iterable[float],
    *,
    runtime_path: str,
) -> np.ndarray:
    parameter_vector = np.asarray(list(parameters), dtype=np.float64)
    if parameter_vector.ndim != 1:
        raise _runtime_error(
            descriptor_set,
            category="runtime_request",
            first_unsupported_condition="parameter_vector",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Partitioned runtime requires a one-dimensional parameter vector",
        )
    if parameter_vector.size != descriptor_set.parameter_count:
        raise _runtime_error(
            descriptor_set,
            category="runtime_request",
            first_unsupported_condition="parameter_count",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime workload '{}' requires {} parameters, got {}".format(
                    descriptor_set.workload_id,
                    descriptor_set.parameter_count,
                    parameter_vector.size,
                )
            ),
        )
    return parameter_vector


def _validate_supported_member(
    descriptor_set: NoisyPartitionDescriptorSet,
    member: NoisyPartitionDescriptorMember,
    *,
    runtime_path: str,
) -> None:
    if member.kind == "gate":
        if member.name not in SUPPORTED_GATE_NAMES:
            raise _runtime_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="gate_name",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime supports only gate families {}, got '{}'".format(
                        sorted(SUPPORTED_GATE_NAMES), member.name
                    )
                ),
            )
        if member.name == "U3" and member.target_qbit is None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Partitioned runtime requires target_qbit for U3 operations",
            )
        if member.name == "CNOT" and (
            member.target_qbit is None or member.control_qbit is None
        ):
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="control_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Partitioned runtime requires both target_qbit and control_qbit for CNOT",
            )
        if member.fixed_value is not None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="fixed_value",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Partitioned runtime gate members must not carry fixed_value payloads",
            )
        return

    if member.kind == "noise":
        if member.name not in SUPPORTED_NOISE_NAMES:
            raise _runtime_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="noise_name",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime supports only local-noise families {}, got '{}'".format(
                        sorted(SUPPORTED_NOISE_NAMES), member.name
                    )
                ),
            )
        if member.target_qbit is None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Partitioned runtime requires target_qbit for local noise operations",
            )
        if member.param_count == 0 and member.fixed_value is None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="fixed_value",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime requires fixed_value on zero-parameter noise "
                    "operations for workload '{}'".format(descriptor_set.workload_id)
                ),
            )
        return

    raise _runtime_error(
        descriptor_set,
        category="unsupported_runtime_operation",
        first_unsupported_condition="operation_kind",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason="Partitioned runtime does not support descriptor kind '{}'".format(
            member.kind
        ),
    )


def validate_runtime_request(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: Iterable[float],
    *,
    runtime_path: str = PHASE3_RUNTIME_PATH_BASELINE,
) -> tuple[NoisyPartitionDescriptorSet, np.ndarray]:
    validated_descriptor_set = validate_partition_descriptor_set(descriptor_set)
    parameter_vector = _coerce_parameter_vector(
        validated_descriptor_set, parameters, runtime_path=runtime_path
    )
    if validated_descriptor_set.requested_mode != PARTITIONED_DENSITY_MODE:
        raise _runtime_error(
            validated_descriptor_set,
            category="runtime_request",
            first_unsupported_condition="requested_mode",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime supports only '{}' requests, got '{}'".format(
                    PARTITIONED_DENSITY_MODE, validated_descriptor_set.requested_mode
                )
            ),
        )
    for partition in validated_descriptor_set.partitions:
        for member in partition.members:
            _validate_supported_member(
                validated_descriptor_set, member, runtime_path=runtime_path
            )
    return validated_descriptor_set, parameter_vector


def _append_member_to_circuit(
    descriptor_set: NoisyPartitionDescriptorSet,
    circuit: NoisyCircuit,
    member: NoisyPartitionDescriptorMember,
    *,
    runtime_path: str,
) -> None:
    _validate_supported_member(descriptor_set, member, runtime_path=runtime_path)
    if member.kind == "gate":
        if member.name == "U3":
            circuit.add_U3(member.target_qbit)
            return
        if member.name == "CNOT":
            circuit.add_CNOT(member.target_qbit, member.control_qbit)
            return
    elif member.kind == "noise":
        if member.name == "local_depolarizing":
            if member.param_count == 0:
                circuit.add_local_depolarizing(member.target_qbit, member.fixed_value)
            else:
                circuit.add_local_depolarizing(member.target_qbit)
            return
        if member.name == "amplitude_damping":
            if member.param_count == 0:
                circuit.add_amplitude_damping(member.target_qbit, member.fixed_value)
            else:
                circuit.add_amplitude_damping(member.target_qbit)
            return
        if member.name == "phase_damping":
            if member.param_count == 0:
                circuit.add_phase_damping(member.target_qbit, member.fixed_value)
            else:
                circuit.add_phase_damping(member.target_qbit)
            return
    raise _runtime_error(
        descriptor_set,
        category="unsupported_runtime_operation",
        first_unsupported_condition="operation_name",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason="Partitioned runtime cannot lower operation '{}'".format(member.name),
    )


def _build_runtime_circuit(
    descriptor_set: NoisyPartitionDescriptorSet,
    members: Iterable[NoisyPartitionDescriptorMember],
    *,
    qbit_num: int,
    runtime_path: str,
) -> tuple[NoisyCircuit, tuple[NoisyPartitionDescriptorMember, ...]]:
    ordered_members = tuple(members)
    circuit = NoisyCircuit(qbit_num)
    for member in ordered_members:
        _append_member_to_circuit(
            descriptor_set, circuit, member, runtime_path=runtime_path
        )
    return circuit, ordered_members


def _normalize_runtime_operation_name(name: str) -> str:
    aliases = {
        "LocalDepolarizing": "local_depolarizing",
        "AmplitudeDamping": "amplitude_damping",
        "PhaseDamping": "phase_damping",
    }
    return aliases.get(name, name)


def _validate_runtime_circuit_shape(
    descriptor_set: NoisyPartitionDescriptorSet,
    circuit: NoisyCircuit,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    *,
    runtime_path: str,
    expected_param_start_attr: str,
) -> None:
    operation_info = list(circuit.get_operation_info())
    if len(operation_info) != len(members):
        raise _runtime_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="operation_count",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime built {} operations for workload '{}' but the "
                "descriptor members contain {}".format(
                    len(operation_info), descriptor_set.workload_id, len(members)
                )
            ),
        )
    for info, member in zip(operation_info, members):
        expected_param_start = getattr(member, expected_param_start_attr)
        if (
            _normalize_runtime_operation_name(info.name) != member.name
            or info.is_unitary != member.is_unitary
            or info.param_count != member.param_count
            or info.param_start != expected_param_start
        ):
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="operation_info",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime operation info diverged from the descriptor "
                    "contract for workload '{}' at canonical operation {}".format(
                        descriptor_set.workload_id, member.canonical_operation_index
                    )
                ),
            )
    expected_parameter_count = sum(member.param_count for member in members)
    if circuit.parameter_num != expected_parameter_count:
        raise _runtime_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="runtime_parameter_count",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime circuit expected {} parameters but built {}".format(
                    expected_parameter_count, circuit.parameter_num
                )
            ),
        )


def _validate_runtime_member_sequence(
    descriptor_set: NoisyPartitionDescriptorSet,
    circuit: NoisyCircuit,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    *,
    runtime_path: str,
) -> None:
    operation_info = list(circuit.get_operation_info())
    if len(operation_info) != len(members):
        raise _runtime_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="operation_count",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime built {} operations for workload '{}' but the "
                "segment members contain {}".format(
                    len(operation_info), descriptor_set.workload_id, len(members)
                )
            ),
        )
    expected_param_start = 0
    for info, member in zip(operation_info, members):
        if (
            _normalize_runtime_operation_name(info.name) != member.name
            or info.is_unitary != member.is_unitary
            or info.param_count != member.param_count
            or info.param_start != expected_param_start
        ):
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="operation_info",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime segment diverged from the descriptor contract "
                    "for workload '{}' at canonical operation {}".format(
                        descriptor_set.workload_id, member.canonical_operation_index
                    )
                ),
            )
        expected_param_start += member.param_count
    if circuit.parameter_num != expected_param_start:
        raise _runtime_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="runtime_parameter_count",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime segment expected {} parameters but built {}".format(
                    expected_param_start, circuit.parameter_num
                )
            ),
        )


def _build_partition_parameter_vector(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    parameter_vector: np.ndarray,
    *,
    runtime_path: str,
) -> np.ndarray:
    local_parameter_vector = np.zeros(
        partition.partition_parameter_count, dtype=np.float64
    )
    for (
        global_param_start,
        local_param_start,
        param_count,
    ) in partition.parameter_routing:
        global_stop = global_param_start + param_count
        local_stop = local_param_start + param_count
        if (
            global_stop > parameter_vector.size
            or local_stop > local_parameter_vector.size
        ):
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="parameter_routing",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime partition {} has out-of-range parameter_routing".format(
                        partition.partition_index
                    )
                ),
            )
        local_parameter_vector[local_param_start:local_stop] = parameter_vector[
            global_param_start:global_stop
        ]
    return local_parameter_vector


def _build_partition_record(
    partition: NoisyPartitionDescriptor, *, runtime_circuit: NoisyCircuit
) -> NoisyRuntimePartitionRecord:
    return NoisyRuntimePartitionRecord(
        partition_index=partition.partition_index,
        canonical_operation_indices=partition.canonical_operation_indices,
        local_to_global_qbits=partition.local_to_global_qbits,
        global_to_local_qbits=partition.global_to_local_qbits,
        requires_remap=partition.requires_remap,
        parameter_routing=partition.parameter_routing,
        partition_parameter_count=partition.partition_parameter_count,
        member_count=partition.member_count,
        gate_count=partition.gate_count,
        noise_count=partition.noise_count,
        operation_names=tuple(member.name for member in partition.members),
        operation_kinds=tuple(member.kind for member in partition.members),
        runtime_circuit_qbit_num=runtime_circuit.qbit_num,
        runtime_circuit_parameter_count=runtime_circuit.parameter_num,
    )


def _segment_parameter_vector(
    descriptor_set: NoisyPartitionDescriptorSet,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    local_parameter_vector: np.ndarray,
    *,
    runtime_path: str,
) -> np.ndarray:
    segment_parameter_count = sum(member.param_count for member in members)
    segment_parameters = np.zeros(segment_parameter_count, dtype=np.float64)
    cursor = 0
    for member in members:
        local_start = member.local_param_start
        local_stop = local_start + member.param_count
        if local_stop > local_parameter_vector.size:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="parameter_routing",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Partitioned runtime segment for workload '{}' references "
                    "out-of-range local parameters at canonical operation {}".format(
                        descriptor_set.workload_id, member.canonical_operation_index
                    )
                ),
            )
        if member.param_count:
            segment_parameters[cursor : cursor + member.param_count] = (
                local_parameter_vector[local_start:local_stop]
            )
        cursor += member.param_count
    return segment_parameters


def _execute_member_sequence(
    descriptor_set: NoisyPartitionDescriptorSet,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    local_parameter_vector: np.ndarray,
    rho: DensityMatrix,
    *,
    runtime_path: str,
) -> None:
    if not members:
        return
    segment_circuit, ordered_members = _build_runtime_circuit(
        descriptor_set,
        members,
        qbit_num=descriptor_set.qbit_num,
        runtime_path=runtime_path,
    )
    _validate_runtime_member_sequence(
        descriptor_set,
        segment_circuit,
        ordered_members,
        runtime_path=runtime_path,
    )
    segment_parameters = _segment_parameter_vector(
        descriptor_set,
        ordered_members,
        local_parameter_vector,
        runtime_path=runtime_path,
    )
    try:
        segment_circuit.apply_to(segment_parameters, rho)
    except Exception as exc:
        raise _runtime_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="segment_execution",
            failure_stage="runtime_execution",
            runtime_path=runtime_path,
            reason=(
                "Partitioned runtime failed while executing a member segment of workload "
                "'{}': {}".format(descriptor_set.workload_id, exc)
            ),
        ) from exc


def _iter_member_segments(
    members: tuple[NoisyPartitionDescriptorMember, ...],
) -> tuple[tuple[bool, tuple[NoisyPartitionDescriptorMember, ...]], ...]:
    if not members:
        return tuple()
    segments: list[tuple[bool, tuple[NoisyPartitionDescriptorMember, ...]]] = []
    current_is_unitary = members[0].is_unitary
    current_members: list[NoisyPartitionDescriptorMember] = [members[0]]
    for member in members[1:]:
        if member.is_unitary == current_is_unitary:
            current_members.append(member)
            continue
        segments.append((current_is_unitary, tuple(current_members)))
        current_is_unitary = member.is_unitary
        current_members = [member]
    segments.append((current_is_unitary, tuple(current_members)))
    return tuple(segments)


def _unique_local_qbits(
    members: tuple[NoisyPartitionDescriptorMember, ...],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                local_qbit
                for member in members
                for local_qbit in member.local_qubit_support
            }
        )
    )


def _u3_unitary(theta_over_2: float, phi: float, lam: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta_over_2), -np.exp(1j * lam) * np.sin(theta_over_2)],
            [
                np.exp(1j * phi) * np.sin(theta_over_2),
                np.exp(1j * (phi + lam)) * np.cos(theta_over_2),
            ],
        ],
        dtype=np.complex128,
    )


def _embed_single_qubit_gate(
    gate_matrix: np.ndarray, *, total_kernel_qbits: int, kernel_target_qbit: int
) -> np.ndarray:
    dim = 1 << total_kernel_qbits
    embedded = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        input_bit = (basis >> kernel_target_qbit) & 1
        base_state = basis & ~(1 << kernel_target_qbit)
        for output_bit in (0, 1):
            output_state = base_state | (output_bit << kernel_target_qbit)
            embedded[output_state, basis] = gate_matrix[output_bit, input_bit]
    return embedded


def _embed_cnot_gate(
    *, total_kernel_qbits: int, kernel_control_qbit: int, kernel_target_qbit: int
) -> np.ndarray:
    dim = 1 << total_kernel_qbits
    embedded = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        output_state = basis
        if (basis >> kernel_control_qbit) & 1:
            output_state ^= 1 << kernel_target_qbit
        embedded[output_state, basis] = 1.0
    return embedded


def _build_gate_matrix_for_member(
    descriptor_set: NoisyPartitionDescriptorSet,
    member: NoisyPartitionDescriptorMember,
    local_parameter_vector: np.ndarray,
    *,
    local_qbit_to_kernel_index: Mapping[int, int],
    total_kernel_qbits: int,
    runtime_path: str,
) -> np.ndarray:
    if member.name == "U3":
        if member.local_target_qbit is None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Fused partitioned runtime requires local_target_qbit for U3 members",
            )
        local_start = member.local_param_start
        local_stop = local_start + member.param_count
        if local_stop > local_parameter_vector.size:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="parameter_routing",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Fused partitioned runtime has out-of-range parameters for workload "
                    "'{}' at canonical operation {}".format(
                        descriptor_set.workload_id, member.canonical_operation_index
                    )
                ),
            )
        theta, phi, lam = local_parameter_vector[local_start:local_stop]
        return _embed_single_qubit_gate(
            _u3_unitary(float(theta), float(phi), float(lam)),
            total_kernel_qbits=total_kernel_qbits,
            kernel_target_qbit=local_qbit_to_kernel_index[member.local_target_qbit],
        )
    if member.name == "CNOT":
        if member.local_target_qbit is None or member.local_control_qbit is None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_control_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Fused partitioned runtime requires both local qubits for CNOT members",
            )
        return _embed_cnot_gate(
            total_kernel_qbits=total_kernel_qbits,
            kernel_control_qbit=local_qbit_to_kernel_index[member.local_control_qbit],
            kernel_target_qbit=local_qbit_to_kernel_index[member.local_target_qbit],
        )
    raise _runtime_error(
        descriptor_set,
        category="unsupported_runtime_operation",
        first_unsupported_condition="fusion_gate_name",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason="Fused partitioned runtime cannot build a fused kernel for '{}'".format(
            member.name
        ),
    )


def _build_fused_kernel(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    local_parameter_vector: np.ndarray,
    *,
    runtime_path: str,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    active_local_qbits = _unique_local_qbits(members)
    if not active_local_qbits:
        raise _runtime_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="local_qubit_support",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Fused partitioned runtime requires non-empty local_qubit_support",
        )
    if len(active_local_qbits) > 2:
        raise _runtime_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="fusion_qubit_span",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Fused partitioned runtime supports only unitary islands on up to 2 "
                "qubits, got span {} for workload '{}'".format(
                    len(active_local_qbits), descriptor_set.workload_id
                )
            ),
        )
    local_qbit_to_kernel_index = {
        local_qbit: kernel_index
        for kernel_index, local_qbit in enumerate(active_local_qbits)
    }
    kernel_dim = 1 << len(active_local_qbits)
    fused_kernel = np.eye(kernel_dim, dtype=np.complex128)
    for member in members:
        if not member.is_unitary:
            raise _runtime_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="fusion_member_kind",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Fused partitioned runtime only supports unitary members inside one fused island",
            )
        gate_matrix = _build_gate_matrix_for_member(
            descriptor_set,
            member,
            local_parameter_vector,
            local_qbit_to_kernel_index=local_qbit_to_kernel_index,
            total_kernel_qbits=len(active_local_qbits),
            runtime_path=runtime_path,
        )
        fused_kernel = gate_matrix @ fused_kernel
    global_target_qbits = tuple(
        partition.local_to_global_qbits[local_qbit] for local_qbit in active_local_qbits
    )
    return fused_kernel, active_local_qbits, global_target_qbits


def _build_unitary_region_record(
    partition: NoisyPartitionDescriptor,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    *,
    classification: str,
    reason: str,
) -> NoisyRuntimeFusedRegionRecord:
    active_local_qbits = _unique_local_qbits(members)
    global_target_qbits = tuple(
        partition.local_to_global_qbits[local_qbit] for local_qbit in active_local_qbits
    )
    return NoisyRuntimeFusedRegionRecord(
        partition_index=partition.partition_index,
        candidate_kind=PHASE3_FUSION_KIND_UNITARY_ISLAND,
        classification=classification,
        reason=reason,
        partition_member_indices=tuple(
            member.partition_member_index for member in members
        ),
        canonical_operation_indices=tuple(
            member.canonical_operation_index for member in members
        ),
        operation_names=tuple(member.name for member in members),
        global_target_qbits=global_target_qbits,
        local_target_qbits=active_local_qbits,
        member_count=len(members),
        gate_count=sum(member.is_unitary for member in members),
    )


def _build_noise_boundary_record(
    partition: NoisyPartitionDescriptor,
    left_members: tuple[NoisyPartitionDescriptorMember, ...],
    boundary_members: tuple[NoisyPartitionDescriptorMember, ...],
    right_members: tuple[NoisyPartitionDescriptorMember, ...],
) -> NoisyRuntimeFusedRegionRecord:
    relevant_members = left_members + boundary_members + right_members
    active_local_qbits = _unique_local_qbits(relevant_members)
    global_target_qbits = tuple(
        partition.local_to_global_qbits[local_qbit] for local_qbit in active_local_qbits
    )
    return NoisyRuntimeFusedRegionRecord(
        partition_index=partition.partition_index,
        candidate_kind=PHASE3_FUSION_KIND_NOISE_BOUNDARY,
        classification=PHASE3_FUSION_CLASS_DEFERRED,
        reason="explicit_noise_boundary",
        partition_member_indices=tuple(
            member.partition_member_index for member in relevant_members
        ),
        canonical_operation_indices=tuple(
            member.canonical_operation_index for member in relevant_members
        ),
        operation_names=tuple(member.name for member in relevant_members),
        global_target_qbits=global_target_qbits,
        local_target_qbits=active_local_qbits,
        member_count=len(relevant_members),
        gate_count=sum(member.is_unitary for member in relevant_members),
    )


def _execute_partition_with_optional_fusion(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    local_parameter_vector: np.ndarray,
    rho: DensityMatrix,
    *,
    runtime_path: str,
    allow_fusion: bool,
) -> tuple[NoisyRuntimeFusedRegionRecord, ...]:
    segments = _iter_member_segments(partition.members)
    fused_regions: list[NoisyRuntimeFusedRegionRecord] = []
    for segment_index, (is_unitary, segment_members) in enumerate(segments):
        if is_unitary:
            eligible_for_fusion = len(segment_members) >= 2
            if allow_fusion and eligible_for_fusion:
                try:
                    fused_kernel, active_local_qbits, global_target_qbits = (
                        _build_fused_kernel(
                            descriptor_set,
                            partition,
                            segment_members,
                            local_parameter_vector,
                            runtime_path=runtime_path,
                        )
                    )
                except NoisyRuntimeValidationError as exc:
                    if exc.first_unsupported_condition != "fusion_qubit_span":
                        raise
                    _execute_member_sequence(
                        descriptor_set,
                        segment_members,
                        local_parameter_vector,
                        rho,
                        runtime_path=runtime_path,
                    )
                    fused_regions.append(
                        _build_unitary_region_record(
                            partition,
                            segment_members,
                            classification=PHASE3_FUSION_CLASS_DEFERRED,
                            reason="fusion_qubit_span",
                        )
                    )
                    continue
                try:
                    rho.apply_local_unitary(
                        np.asarray(fused_kernel, dtype=np.complex128),
                        list(global_target_qbits),
                    )
                except Exception as exc:
                    raise _runtime_error(
                        descriptor_set,
                        category="unsupported_runtime_execution",
                        first_unsupported_condition="fused_partition_execution",
                        failure_stage="runtime_execution",
                        runtime_path=runtime_path,
                        reason=(
                            "Fused partitioned runtime failed while executing partition {} "
                            "of workload '{}' on local span {}: {}".format(
                                partition.partition_index,
                                descriptor_set.workload_id,
                                list(active_local_qbits),
                                exc,
                            )
                        ),
                    ) from exc
                fused_regions.append(
                    _build_unitary_region_record(
                        partition,
                        segment_members,
                        classification=PHASE3_FUSION_CLASS_FUSED,
                        reason="eligible_unitary_island",
                    )
                )
            else:
                _execute_member_sequence(
                    descriptor_set,
                    segment_members,
                    local_parameter_vector,
                    rho,
                    runtime_path=runtime_path,
                )
                fused_regions.append(
                    _build_unitary_region_record(
                        partition,
                        segment_members,
                        classification=PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
                        reason=(
                            "fusion_disabled"
                            if eligible_for_fusion
                            else "singleton_unitary_region"
                        ),
                    )
                )
            continue

        _execute_member_sequence(
            descriptor_set,
            segment_members,
            local_parameter_vector,
            rho,
            runtime_path=runtime_path,
        )
        if segment_index == 0 or segment_index == len(segments) - 1:
            continue
        left_is_unitary, left_members = segments[segment_index - 1]
        right_is_unitary, right_members = segments[segment_index + 1]
        if left_is_unitary and right_is_unitary:
            fused_regions.append(
                _build_noise_boundary_record(
                    partition, left_members, segment_members, right_members
                )
            )
    return tuple(fused_regions)


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def execute_partitioned_density(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: Iterable[float],
    *,
    runtime_path: str = PHASE3_RUNTIME_PATH_BASELINE,
    allow_fusion: bool = False,
) -> NoisyRuntimeExecutionResult:
    requested_runtime_path = runtime_path
    if allow_fusion and requested_runtime_path == PHASE3_RUNTIME_PATH_BASELINE:
        requested_runtime_path = PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS
    validated_descriptor_set, parameter_vector = validate_runtime_request(
        descriptor_set, parameters, runtime_path=requested_runtime_path
    )
    result_start = time.perf_counter()
    rho = DensityMatrix(validated_descriptor_set.qbit_num)
    partition_records: list[NoisyRuntimePartitionRecord] = []
    fused_regions: list[NoisyRuntimeFusedRegionRecord] = []
    for partition in validated_descriptor_set.partitions:
        partition_circuit, ordered_members = _build_runtime_circuit(
            validated_descriptor_set,
            partition.members,
            qbit_num=validated_descriptor_set.qbit_num,
            runtime_path=requested_runtime_path,
        )
        _validate_runtime_circuit_shape(
            validated_descriptor_set,
            partition_circuit,
            ordered_members,
            runtime_path=requested_runtime_path,
            expected_param_start_attr="local_param_start",
        )
        local_parameter_vector = _build_partition_parameter_vector(
            validated_descriptor_set,
            partition,
            parameter_vector,
            runtime_path=requested_runtime_path,
        )
        fused_regions.extend(
            _execute_partition_with_optional_fusion(
                validated_descriptor_set,
                partition,
                local_parameter_vector,
                rho,
                runtime_path=requested_runtime_path,
                allow_fusion=allow_fusion,
            )
        )
        partition_records.append(
            _build_partition_record(partition, runtime_circuit=partition_circuit)
        )
    actual_runtime_path = (
        requested_runtime_path
        if any(
            region.classification == PHASE3_FUSION_CLASS_FUSED
            for region in fused_regions
        )
        else PHASE3_RUNTIME_PATH_BASELINE
    )
    return NoisyRuntimeExecutionResult(
        runtime_schema_version=PHASE3_RUNTIME_SCHEMA_VERSION,
        planner_schema_version=validated_descriptor_set.planner_schema_version,
        descriptor_schema_version=validated_descriptor_set.schema_version,
        requested_mode=validated_descriptor_set.requested_mode,
        source_type=validated_descriptor_set.source_type,
        workload_id=validated_descriptor_set.workload_id,
        qbit_num=validated_descriptor_set.qbit_num,
        parameter_count=validated_descriptor_set.parameter_count,
        runtime_path=actual_runtime_path,
        fallback_used=False,
        exact_output_present=True,
        density_matrix=rho.clone(),
        partitions=tuple(partition_records),
        fused_regions=tuple(fused_regions),
        runtime_ms=(time.perf_counter() - result_start) * 1000.0,
        peak_rss_kb=_peak_rss_kb(),
    )


def execute_partitioned_density_fused(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: Iterable[float],
) -> NoisyRuntimeExecutionResult:
    return execute_partitioned_density(
        descriptor_set,
        parameters,
        runtime_path=PHASE3_RUNTIME_PATH_FUSED_UNITARY_ISLANDS,
        allow_fusion=True,
    )


def execute_sequential_density_reference(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: Iterable[float],
    *,
    runtime_path: str = PHASE3_RUNTIME_PATH_SEQUENTIAL_REFERENCE,
) -> DensityMatrix:
    validated_descriptor_set, parameter_vector = validate_runtime_request(
        descriptor_set, parameters, runtime_path=runtime_path
    )
    flattened_members = tuple(
        member
        for partition in validated_descriptor_set.partitions
        for member in partition.members
    )
    circuit, ordered_members = _build_runtime_circuit(
        validated_descriptor_set,
        flattened_members,
        qbit_num=validated_descriptor_set.qbit_num,
        runtime_path=runtime_path,
    )
    _validate_runtime_circuit_shape(
        validated_descriptor_set,
        circuit,
        ordered_members,
        runtime_path=runtime_path,
        expected_param_start_attr="param_start",
    )
    rho = DensityMatrix(validated_descriptor_set.qbit_num)
    try:
        circuit.apply_to(parameter_vector, rho)
    except Exception as exc:
        raise _runtime_error(
            validated_descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="sequential_reference_execution",
            failure_stage="runtime_execution",
            runtime_path=runtime_path,
            reason=(
                "Sequential reference execution failed for workload '{}': {}".format(
                    validated_descriptor_set.workload_id,
                    exc,
                )
            ),
        ) from exc
    return rho


def build_runtime_audit_record(
    result: NoisyRuntimeExecutionResult,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = result.to_dict(include_density_matrix=False)
    return {
        "runtime_schema_version": payload["runtime_schema_version"],
        "planner_schema_version": payload["planner_schema_version"],
        "descriptor_schema_version": payload["descriptor_schema_version"],
        "requested_mode": payload["requested_mode"],
        "runtime_path": payload["runtime_path"],
        "provenance": payload["provenance"],
        "summary": payload["summary"],
        "exact_output": payload["exact_output"],
        "partitions": payload["partitions"],
        "fused_regions": payload["fused_regions"],
        "metadata": dict(metadata) if metadata is not None else {},
    }
