from __future__ import annotations

from typing import Any, Iterable, Mapping

from squander.partitioning.noisy_planner_validation import (
    build_phase3_continuity_planner_surface,
    preflight_planner_request,
)
from squander.partitioning.noisy_types import (
    DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
    PARTITIONED_DENSITY_MODE,
    PLANNER_OP_KIND_GATE,
    PLANNER_OP_KIND_NOISE,
    CanonicalNoisyPlannerOperation,
    CanonicalNoisyPlannerSurface,
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorMember,
    NoisyPartitionDescriptorSet,
)

class NoisyDescriptorValidationError(ValueError):
    """Structured descriptor-generation validation error for the Phase 3 partition descriptor surface."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        workload_id: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.first_unsupported_condition = first_unsupported_condition
        self.failure_stage = failure_stage
        self.source_type = source_type
        self.requested_mode = requested_mode
        self.workload_id = workload_id
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "unsupported_category": self.category,
            "first_unsupported_condition": self.first_unsupported_condition,
            "failure_stage": self.failure_stage,
            "source_type": self.source_type,
            "requested_mode": self.requested_mode,
            "workload_id": self.workload_id,
            "reason": self.reason,
        }

def descriptor_member_to_dict(
    descriptor_set: NoisyPartitionDescriptorSet,
    member: NoisyPartitionDescriptorMember,
) -> dict[str, Any]:
    op = descriptor_set.canonical_operation_for(member)
    return {
        "partition_member_index": member.partition_member_index,
        "canonical_operation_index": member.canonical_operation_index,
        "kind": op.kind,
        "name": op.name,
        "is_unitary": op.is_unitary,
        "source_gate_index": op.source_gate_index,
        "target_qbit": op.target_qbit,
        "control_qbit": op.control_qbit,
        "qubit_support": list(op.qubit_support),
        "local_target_qbit": member.local_target_qbit,
        "local_control_qbit": member.local_control_qbit,
        "local_qubit_support": list(member.local_qubit_support),
        "param_count": op.param_count,
        "param_start": op.param_start,
        "local_param_start": member.local_param_start,
        "fixed_value": op.fixed_value,
    }


def descriptor_partition_to_dict(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
) -> dict[str, Any]:
    routing = descriptor_set.partition_parameter_routing(partition)
    return {
        "partition_index": partition.partition_index,
        "member_count": partition.member_count,
        "gate_count": descriptor_set.partition_gate_count(partition),
        "noise_count": descriptor_set.partition_noise_count(partition),
        "canonical_operation_indices": list(
            descriptor_set.partition_canonical_operation_indices(partition)
        ),
        # partition_qubit_span duplicates local_to_global_qbits for frozen JSON compatibility.
        "partition_qubit_span": list(partition.partition_qubit_span),
        "local_to_global_qbits": list(partition.local_to_global_qbits),
        "global_to_local_qbits": [
            {"global_qbit": global_qbit, "local_qbit": local_qbit}
            for global_qbit, local_qbit in partition.global_to_local_qbits
        ],
        "requires_remap": partition.requires_remap,
        "partition_parameter_count": descriptor_set.partition_parameter_count(partition),
        "parameter_routing": [
            {
                "global_param_start": global_param_start,
                "local_param_start": local_param_start,
                "param_count": param_count,
            }
            for global_param_start, local_param_start, param_count in routing
        ],
        "members": [m.to_dict(descriptor_set) for m in partition.members],
    }

def _validate_descriptor_request(
    surface: CanonicalNoisyPlannerSurface, *, max_partition_qubits: int
) -> None:
    if max_partition_qubits <= 0:
        raise NoisyDescriptorValidationError(
            category="descriptor_request",
            first_unsupported_condition="max_partition_qubits",
            failure_stage="descriptor_generation_preflight",
            source_type=surface.source_type,
            requested_mode=surface.requested_mode,
            workload_id=surface.workload_id,
            reason="Partition descriptor generation requires max_partition_qubits > 0",
        )
    if surface.max_qubit_span > max_partition_qubits:
        raise NoisyDescriptorValidationError(
            category="partition_span",
            first_unsupported_condition="max_partition_qubits",
            failure_stage="descriptor_generation_preflight",
            source_type=surface.source_type,
            requested_mode=surface.requested_mode,
            workload_id=surface.workload_id,
            reason=(
                "Partition descriptor generation requires max_partition_qubits >= "
                "the canonical max_qubit_span {} for workload '{}'".format(
                    surface.max_qubit_span, surface.workload_id
                )
            ),
        )


##
# @brief Partition the operations of a canonical planner surface into partitions.
# The partition is done by the following algorithm (greedy algorithm):
# 1. Start with an empty partition.
# 2. Add the first operation to the partition.
# 3. For each subsequent operation, add it to the partition if it can be added without exceeding the maximum number of qubits allowed in a partition.
# 4. If the next operation cannot be added to the partition without exceeding the maximum number of qubits allowed in a partition, start a new partition.
# 5. Repeat steps 2-4 until all operations have been added to a partition.
# @param surface: A canonical planner surface.
# @param max_partition_qubits: The maximum number of qubits allowed in a partition.
# @return A tuple of tuples of operations, where each tuple is a partition.
def _partition_surface_operations(
    surface: CanonicalNoisyPlannerSurface, *, max_partition_qubits: int
) -> tuple[tuple[CanonicalNoisyPlannerOperation, ...], ...]:
    partitions: list[tuple[CanonicalNoisyPlannerOperation, ...]] = []
    current_partition: list[CanonicalNoisyPlannerOperation] = []
    current_support: set[int] = set()

    for operation in surface.operations:
        operation_support = set(operation.qubit_support)
        expanded_support = current_support | operation_support
        if current_partition and len(expanded_support) > max_partition_qubits:
            partitions.append(tuple(current_partition))
            current_partition = []
            current_support = set()
        current_partition.append(operation)
        current_support |= operation_support

    if current_partition:
        partitions.append(tuple(current_partition))

    return tuple(partitions)


def _build_descriptor_partition(
    operations: tuple[CanonicalNoisyPlannerOperation, ...], *, partition_index: int
) -> NoisyPartitionDescriptor:
    local_to_global_qbits = tuple(
        sorted({qbit for operation in operations for qbit in operation.qubit_support})
    )
    global_to_local = {
        global_qbit: local_qbit
        for local_qbit, global_qbit in enumerate(local_to_global_qbits)
    }

    members: list[NoisyPartitionDescriptorMember] = []
    local_param_start = 0

    for partition_member_index, operation in enumerate(operations):
        member = NoisyPartitionDescriptorMember(
            partition_member_index=partition_member_index,
            canonical_operation_index=operation.index,
            local_target_qbit=(
                None
                if operation.target_qbit is None
                else global_to_local[operation.target_qbit]
            ),
            local_control_qbit=(
                None
                if operation.control_qbit is None
                else global_to_local[operation.control_qbit]
            ),
            local_qubit_support=tuple(
                global_to_local[qbit] for qbit in operation.qubit_support
            ),
            local_param_start=local_param_start,
        )
        members.append(member)
        local_param_start += operation.param_count

    return NoisyPartitionDescriptor(
        partition_index=partition_index,
        local_to_global_qbits=local_to_global_qbits,
        members=tuple(members),
    )


def build_partition_descriptor_set(
    surface: CanonicalNoisyPlannerSurface,
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
) -> NoisyPartitionDescriptorSet:
    _validate_descriptor_request(surface, max_partition_qubits=max_partition_qubits)
    partitions = tuple(
        _build_descriptor_partition(
            partition_operations, partition_index=partition_index
        )
        for partition_index, partition_operations in enumerate(
            _partition_surface_operations(
                surface, max_partition_qubits=max_partition_qubits
            )
        )
    )
    descriptor_set = NoisyPartitionDescriptorSet(
        requested_mode=surface.requested_mode,
        source_type=surface.source_type,
        workload_id=surface.workload_id,
        qbit_num=surface.qbit_num,
        parameter_count=surface.parameter_count,
        max_partition_qubits=max_partition_qubits,
        operations=surface.operations,
        partitions=partitions,
    )
    return validate_partition_descriptor_set_against_surface(surface, descriptor_set)


def build_phase3_continuity_partition_descriptor_set(
    vqe,
    *,
    workload_id: str | None = None,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
) -> NoisyPartitionDescriptorSet:
    surface = build_phase3_continuity_planner_surface(
        vqe,
        workload_id=workload_id,
        requested_mode=requested_mode,
    )
    return build_partition_descriptor_set(
        surface, max_partition_qubits=max_partition_qubits
    )


def preflight_descriptor_request(
    *,
    source_type: str,
    workload_id: str,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    bridge_metadata: Mapping[str, Any] | None = None,
    operation_specs: Iterable[Mapping[str, Any]] | None = None,
    qbit_num: int | None = None,
    legacy_circuit: Any | None = None,
    density_noise: Iterable[Mapping[str, Any]] | None = None,
    strict_phase3_support: bool = True,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
) -> NoisyPartitionDescriptorSet:
    surface = preflight_planner_request(
        source_type=source_type,
        workload_id=workload_id,
        requested_mode=requested_mode,
        bridge_metadata=bridge_metadata,
        operation_specs=operation_specs,
        qbit_num=qbit_num,
        legacy_circuit=legacy_circuit,
        density_noise=density_noise,
        strict_phase3_support=strict_phase3_support,
    )
    return build_partition_descriptor_set(
        surface, max_partition_qubits=max_partition_qubits
    )


def _descriptor_error(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    category: str,
    first_unsupported_condition: str,
    failure_stage: str,
    reason: str,
) -> NoisyDescriptorValidationError:
    return NoisyDescriptorValidationError(
        category=category,
        first_unsupported_condition=first_unsupported_condition,
        failure_stage=failure_stage,
        source_type=descriptor_set.source_type,
        requested_mode=descriptor_set.requested_mode,
        workload_id=descriptor_set.workload_id,
        reason=reason,
    )


def validate_partition_descriptor_set(
    descriptor_set: NoisyPartitionDescriptorSet,
) -> NoisyPartitionDescriptorSet:
    expected_partition_index = 0
    expected_canonical_index = 0

    if len(descriptor_set.operations) != descriptor_set.descriptor_member_count:
        raise _descriptor_error(
            descriptor_set,
            category="dropped_operations",
            first_unsupported_condition="operations_table_size",
            failure_stage="descriptor_validation",
            reason=(
                "Descriptor operations table length {} does not match "
                "descriptor_member_count {}".format(
                    len(descriptor_set.operations),
                    descriptor_set.descriptor_member_count,
                )
            ),
        )

    for partition in descriptor_set.partitions:
        if partition.partition_index != expected_partition_index:
            raise _descriptor_error(
                descriptor_set,
                category="reordering_across_noise_boundaries",
                first_unsupported_condition="partition_index",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptors require contiguous partition indices starting "
                    "at 0; got partition_index {} while expecting {}".format(
                        partition.partition_index, expected_partition_index
                    )
                ),
            )
        if not partition.members:
            raise _descriptor_error(
                descriptor_set,
                category="descriptor_request",
                first_unsupported_condition="empty_partition",
                failure_stage="descriptor_validation",
                reason="Partition descriptors do not support empty partitions",
            )

        if len(set(partition.local_to_global_qbits)) != len(
            partition.local_to_global_qbits
        ):
            raise _descriptor_error(
                descriptor_set,
                category="incomplete_remapping",
                first_unsupported_condition="duplicate_global_qbit",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor partition {} contains duplicate entries in "
                    "local_to_global_qbits".format(partition.partition_index)
                ),
            )

        global_to_local = {
            global_qbit: local_qbit
            for local_qbit, global_qbit in enumerate(partition.local_to_global_qbits)
        }
        expected_parameter_routing: list[tuple[int, int, int]] = []
        expected_local_param_start = 0

        for member_index, member in enumerate(partition.members):
            if member.partition_member_index != member_index:
                raise _descriptor_error(
                    descriptor_set,
                    category="reordering_across_noise_boundaries",
                    first_unsupported_condition="partition_member_index",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} requires contiguous "
                        "partition_member_index values".format(
                            partition.partition_index
                        )
                    ),
                )
            if member.canonical_operation_index != expected_canonical_index:
                raise _descriptor_error(
                    descriptor_set,
                    category="reordering_across_noise_boundaries",
                    first_unsupported_condition="canonical_operation_index",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptors must preserve contiguous canonical "
                        "operation coverage; partition {} member {} expected canonical "
                        "index {} but got {}".format(
                            partition.partition_index,
                            member_index,
                            expected_canonical_index,
                            member.canonical_operation_index,
                        )
                    ),
                )
            op = descriptor_set.operations[member.canonical_operation_index]
            expected_canonical_index += 1

            expected_local_qubit_support = tuple(
                global_to_local[qbit] for qbit in op.qubit_support
            )
            if member.local_qubit_support != expected_local_qubit_support:
                raise _descriptor_error(
                    descriptor_set,
                    category="incomplete_remapping",
                    first_unsupported_condition="local_qubit_support",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has inconsistent local_qubit_support "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            expected_local_target = (
                None
                if op.target_qbit is None
                else global_to_local[op.target_qbit]
            )
            if member.local_target_qbit != expected_local_target:
                raise _descriptor_error(
                    descriptor_set,
                    category="incomplete_remapping",
                    first_unsupported_condition="local_target_qbit",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has inconsistent local_target_qbit "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            expected_local_control = (
                None
                if op.control_qbit is None
                else global_to_local[op.control_qbit]
            )
            if member.local_control_qbit != expected_local_control:
                raise _descriptor_error(
                    descriptor_set,
                    category="incomplete_remapping",
                    first_unsupported_condition="local_control_qbit",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has inconsistent local_control_qbit "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            if member.local_param_start != expected_local_param_start:
                raise _descriptor_error(
                    descriptor_set,
                    category="ambiguous_parameter_routing",
                    first_unsupported_condition="local_param_start",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has non-contiguous local_param_start "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            if op.param_count:
                expected_parameter_routing.append(
                    (op.param_start, member.local_param_start, op.param_count)
                )
            expected_local_param_start += op.param_count

        derived_routing = descriptor_set.partition_parameter_routing(partition)
        if derived_routing != tuple(expected_parameter_routing):
            raise _descriptor_error(
                descriptor_set,
                category="ambiguous_parameter_routing",
                first_unsupported_condition="parameter_routing",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor partition {} must keep parameter_routing aligned "
                    "with ordered member parameter metadata".format(
                        partition.partition_index
                    )
                ),
            )
        expected_partition_index += 1

    if expected_canonical_index != len(descriptor_set.operations):
        raise _descriptor_error(
            descriptor_set,
            category="dropped_operations",
            first_unsupported_condition="descriptor_member_count",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor coverage ended at canonical index {} but "
                "operations table has length {}".format(
                    expected_canonical_index, len(descriptor_set.operations)
                )
            ),
        )

    return descriptor_set


def _descriptor_operation_mismatch_category(
    expected: CanonicalNoisyPlannerOperation,
    actual: CanonicalNoisyPlannerOperation,
) -> tuple[str, str]:
    """Map first differing planner field to historical unsupported categories."""
    for field in (
        "kind",
        "name",
        "is_unitary",
        "source_gate_index",
        "target_qbit",
        "control_qbit",
        "param_count",
        "param_start",
        "fixed_value",
    ):
        if getattr(actual, field) != getattr(expected, field):
            category = (
                "hidden_noise_placement"
                if expected.kind == PLANNER_OP_KIND_NOISE
                or actual.kind == PLANNER_OP_KIND_NOISE
                else "dropped_operations"
            )
            return category, field
    if actual.qubit_support != expected.qubit_support:
        category = (
            "hidden_noise_placement"
            if expected.kind == PLANNER_OP_KIND_NOISE
            or actual.kind == PLANNER_OP_KIND_NOISE
            else "dropped_operations"
        )
        return category, "qubit_support"
    return "dropped_operations", "operation_mismatch"


def validate_partition_descriptor_set_against_surface(
    surface: CanonicalNoisyPlannerSurface,
    descriptor_set: NoisyPartitionDescriptorSet,
) -> NoisyPartitionDescriptorSet:
    validated = validate_partition_descriptor_set(descriptor_set)
    if (
        validated.requested_mode != surface.requested_mode
        or validated.source_type != surface.source_type
        or validated.workload_id != surface.workload_id
        or validated.qbit_num != surface.qbit_num
        or validated.parameter_count != surface.parameter_count
    ):
        raise _descriptor_error(
            validated,
            category="descriptor_request",
            first_unsupported_condition="provenance_mismatch",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor provenance and size metadata must match the "
                "canonical planner surface for workload '{}'".format(
                    surface.workload_id
                )
            ),
        )
    if validated.descriptor_member_count != surface.operation_count:
        raise _descriptor_error(
            validated,
            category="dropped_operations",
            first_unsupported_condition="operation_count",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor workload '{}' records {} descriptor members but "
                "the canonical surface contains {} operations".format(
                    surface.workload_id,
                    validated.descriptor_member_count,
                    surface.operation_count,
                )
            ),
        )
    if validated.noise_count != surface.noise_count:
        raise _descriptor_error(
            validated,
            category="hidden_noise_placement",
            first_unsupported_condition="noise_count",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor workload '{}' records {} noise operations but "
                "the canonical surface contains {}".format(
                    surface.workload_id,
                    validated.noise_count,
                    surface.noise_count,
                )
            ),
        )

    if validated.operations != surface.operations:
        if len(validated.operations) != len(surface.operations):
            raise _descriptor_error(
                validated,
                category="dropped_operations",
                first_unsupported_condition="operation_count",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor workload '{}' operations table length {} does not "
                    "match canonical surface length {}".format(
                        surface.workload_id,
                        len(validated.operations),
                        len(surface.operations),
                    )
                ),
            )
        for index, (desc_op, surf_op) in enumerate(
            zip(validated.operations, surface.operations)
        ):
            if desc_op == surf_op:
                continue
            category, first_cond = _descriptor_operation_mismatch_category(
                surf_op, desc_op
            )
            raise _descriptor_error(
                validated,
                category=category,
                first_unsupported_condition=first_cond,
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor canonical operation {} mismatches the "
                    "planner surface".format(index)
                ),
            )

    return validated


def build_descriptor_audit_record(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = validate_partition_descriptor_set(descriptor_set).to_dict()
    return {
        "requested_mode": payload["requested_mode"],
        "provenance": payload["provenance"],
        "summary": {
            "qbit_num": payload["qbit_num"],
            "parameter_count": payload["parameter_count"],
            "partition_count": payload["partition_count"],
            "descriptor_member_count": payload["descriptor_member_count"],
            "gate_count": payload["gate_count"],
            "noise_count": payload["noise_count"],
            "max_partition_qubits": payload["max_partition_qubits"],
            "max_partition_span": payload["max_partition_span"],
            "partition_member_counts": payload["partition_member_counts"],
            "remapped_partition_count": sum(
                partition["requires_remap"] for partition in payload["partitions"]
            ),
            "parameter_routing_segment_count": sum(
                len(partition["parameter_routing"])
                for partition in payload["partitions"]
            ),
        },
        "partitions": payload["partitions"],
        "metadata": dict(metadata) if metadata is not None else {},
    }
