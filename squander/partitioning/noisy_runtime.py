from __future__ import annotations

from dataclasses import dataclass
import resource
import time
from typing import Any, Iterable, Mapping

import numpy as np

from squander.density_matrix import DensityMatrix, NoisyCircuit
from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    SUPPORTED_PHASE3_GATE_NAMES,
    SUPPORTED_PHASE3_NOISE_NAMES,
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorMember,
    NoisyPartitionDescriptorSet,
    validate_partition_descriptor_set,
)

PHASE3_RUNTIME_SCHEMA_VERSION = "phase3_partitioned_density_runtime_v1"
PHASE3_RUNTIME_PATH_BASELINE = "partitioned_density_descriptor_baseline"
PHASE3_RUNTIME_PATH_SEQUENTIAL_REFERENCE = "sequential_density_descriptor_reference"
PHASE3_RUNTIME_VALIDITY_TOL = 1e-10


class NoisyRuntimeValidationError(ValueError):
    """Structured runtime validation error for Phase 3 Task 3."""

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
    operation_classes: tuple[str, ...]
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
            "operation_classes": list(self.operation_classes),
            "runtime_circuit_qbit_num": self.runtime_circuit_qbit_num,
            "runtime_circuit_parameter_count": self.runtime_circuit_parameter_count,
        }


@dataclass(frozen=True)
class NoisyRuntimeExecutionResult:
    runtime_schema_version: str
    planner_schema_version: str
    descriptor_schema_version: str
    requested_mode: str
    source_type: str
    entry_route: str
    workload_family: str
    workload_id: str
    qbit_num: int
    parameter_count: int
    runtime_path: str
    fallback_used: bool
    exact_output_present: bool
    density_matrix: DensityMatrix
    partitions: tuple[NoisyRuntimePartitionRecord, ...]
    runtime_ms: float
    peak_rss_kb: int

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "entry_route": self.entry_route,
            "workload_family": self.workload_family,
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
        return max((len(partition.partition_qubit_span) for partition in self.partitions), default=0)

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
            "entry_route": self.entry_route,
            "workload_family": self.workload_family,
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
            reason="Task 3 runtime requires a one-dimensional parameter vector",
        )
    if parameter_vector.size != descriptor_set.parameter_count:
        raise _runtime_error(
            descriptor_set,
            category="runtime_request",
            first_unsupported_condition="parameter_count",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Task 3 runtime workload '{}' requires {} parameters, got {}".format(
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
        if member.name not in SUPPORTED_PHASE3_GATE_NAMES:
            raise _runtime_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="gate_name",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Task 3 runtime supports only gate families {}, got '{}'".format(
                        sorted(SUPPORTED_PHASE3_GATE_NAMES), member.name
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
                reason="Task 3 runtime requires target_qbit for U3 operations",
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
                reason="Task 3 runtime requires both target_qbit and control_qbit for CNOT",
            )
        if member.fixed_value is not None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="fixed_value",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Task 3 gate members must not carry fixed_value payloads",
            )
        return

    if member.kind == "noise":
        if member.name not in SUPPORTED_PHASE3_NOISE_NAMES:
            raise _runtime_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="noise_name",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Task 3 runtime supports only local-noise families {}, got '{}'".format(
                        sorted(SUPPORTED_PHASE3_NOISE_NAMES), member.name
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
                reason="Task 3 runtime requires target_qbit for local noise operations",
            )
        if member.param_count == 0 and member.fixed_value is None:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="fixed_value",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Task 3 runtime requires fixed_value on zero-parameter noise "
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
        reason="Task 3 runtime does not support descriptor kind '{}'".format(member.kind),
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
                "Task 3 runtime supports only '{}' requests, got '{}'".format(
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
        reason="Task 3 runtime cannot lower operation '{}'".format(member.name),
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
                "Task 3 runtime built {} operations for workload '{}' but the "
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
                    "Task 3 runtime operation info diverged from the descriptor "
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
                "Task 3 runtime circuit expected {} parameters but built {}".format(
                    expected_parameter_count, circuit.parameter_num
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
    for global_param_start, local_param_start, param_count in partition.parameter_routing:
        global_stop = global_param_start + param_count
        local_stop = local_param_start + param_count
        if global_stop > parameter_vector.size or local_stop > local_parameter_vector.size:
            raise _runtime_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="parameter_routing",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Task 3 runtime partition {} has out-of-range parameter_routing".format(
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
        operation_classes=tuple(member.operation_class for member in partition.members),
        runtime_circuit_qbit_num=runtime_circuit.qbit_num,
        runtime_circuit_parameter_count=runtime_circuit.parameter_num,
    )


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def execute_partitioned_density(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: Iterable[float],
    *,
    runtime_path: str = PHASE3_RUNTIME_PATH_BASELINE,
) -> NoisyRuntimeExecutionResult:
    validated_descriptor_set, parameter_vector = validate_runtime_request(
        descriptor_set, parameters, runtime_path=runtime_path
    )
    result_start = time.perf_counter()
    rho = DensityMatrix(validated_descriptor_set.qbit_num)
    partition_records: list[NoisyRuntimePartitionRecord] = []
    for partition in validated_descriptor_set.partitions:
        partition_circuit, ordered_members = _build_runtime_circuit(
            validated_descriptor_set,
            partition.members,
            qbit_num=validated_descriptor_set.qbit_num,
            runtime_path=runtime_path,
        )
        _validate_runtime_circuit_shape(
            validated_descriptor_set,
            partition_circuit,
            ordered_members,
            runtime_path=runtime_path,
            expected_param_start_attr="local_param_start",
        )
        local_parameter_vector = _build_partition_parameter_vector(
            validated_descriptor_set,
            partition,
            parameter_vector,
            runtime_path=runtime_path,
        )
        try:
            partition_circuit.apply_to(local_parameter_vector, rho)
        except Exception as exc:
            raise _runtime_error(
                validated_descriptor_set,
                category="unsupported_runtime_execution",
                first_unsupported_condition="partition_execution",
                failure_stage="runtime_execution",
                runtime_path=runtime_path,
                reason=(
                    "Task 3 runtime failed while executing partition {} of workload "
                    "'{}': {}".format(
                        partition.partition_index,
                        validated_descriptor_set.workload_id,
                        exc,
                    )
                ),
            ) from exc
        partition_records.append(
            _build_partition_record(partition, runtime_circuit=partition_circuit)
        )
    return NoisyRuntimeExecutionResult(
        runtime_schema_version=PHASE3_RUNTIME_SCHEMA_VERSION,
        planner_schema_version=validated_descriptor_set.planner_schema_version,
        descriptor_schema_version=validated_descriptor_set.schema_version,
        requested_mode=validated_descriptor_set.requested_mode,
        source_type=validated_descriptor_set.source_type,
        entry_route=validated_descriptor_set.entry_route,
        workload_family=validated_descriptor_set.workload_family,
        workload_id=validated_descriptor_set.workload_id,
        qbit_num=validated_descriptor_set.qbit_num,
        parameter_count=validated_descriptor_set.parameter_count,
        runtime_path=runtime_path,
        fallback_used=False,
        exact_output_present=True,
        density_matrix=rho.clone(),
        partitions=tuple(partition_records),
        runtime_ms=(time.perf_counter() - result_start) * 1000.0,
        peak_rss_kb=_peak_rss_kb(),
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
                "Task 3 sequential reference execution failed for workload '{}': {}".format(
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
        "metadata": dict(metadata) if metadata is not None else {},
    }
