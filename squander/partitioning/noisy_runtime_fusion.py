from __future__ import annotations

from typing import Mapping

import numpy as np

from squander.density_matrix import DensityMatrix
from squander.partitioning.noisy_runtime_core import (
    PHASE3_FUSION_CLASS_DEFERRED,
    PHASE3_FUSION_CLASS_FUSED,
    PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED,
    PHASE3_FUSION_KIND_NOISE_BOUNDARY,
    PHASE3_FUSION_KIND_UNITARY_ISLAND,
    NoisyRuntimeFusedRegionRecord,
    _execute_member_sequence,
)
from squander.partitioning.noisy_runtime_errors import (
    NoisyRuntimeValidationError,
    runtime_validation_error,
)
from squander.partitioning.noisy_types import (
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorMember,
    NoisyPartitionDescriptorSet,
)


def _iter_member_segments(
    descriptor_set: NoisyPartitionDescriptorSet,
    members: tuple[NoisyPartitionDescriptorMember, ...],
) -> tuple[tuple[bool, tuple[NoisyPartitionDescriptorMember, ...]], ...]:
    if not members:
        return tuple()
    segments: list[tuple[bool, tuple[NoisyPartitionDescriptorMember, ...]]] = []
    current_is_unitary = descriptor_set.canonical_operation_for(members[0]).is_unitary
    current_members: list[NoisyPartitionDescriptorMember] = [members[0]]
    for member in members[1:]:
        if (
            descriptor_set.canonical_operation_for(member).is_unitary
            == current_is_unitary
        ):
            current_members.append(member)
            continue
        segments.append((current_is_unitary, tuple(current_members)))
        current_is_unitary = descriptor_set.canonical_operation_for(member).is_unitary
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


def _u3_unitary_from_local_parameter_slice(
    local_parameter_vector: np.ndarray, local_start: int, local_stop: int
) -> np.ndarray:
    theta, phi, lam = local_parameter_vector[local_start:local_stop]
    return _u3_unitary(float(theta), float(phi), float(lam))


def _kernel_indices_for_fused_cnot(
    *,
    local_qbit_to_kernel_index: Mapping[int, int],
    local_target_qbit: int,
    local_control_qbit: int,
) -> tuple[int, int]:
    """Map descriptor CNOT wires to _embed_cnot_gate(kernel_control, kernel_target)."""
    return (
        local_qbit_to_kernel_index[local_control_qbit],
        local_qbit_to_kernel_index[local_target_qbit],
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
    op = descriptor_set.canonical_operation_for(member)
    if op.name == "U3":
        if member.local_target_qbit is None:
            raise runtime_validation_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Fused partitioned runtime requires local_target_qbit for U3 members",
            )
        local_start = member.local_param_start
        local_stop = local_start + op.param_count
        if local_stop > local_parameter_vector.size:
            raise runtime_validation_error(
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
        return _embed_single_qubit_gate(
            _u3_unitary_from_local_parameter_slice(
                local_parameter_vector, local_start, local_stop
            ),
            total_kernel_qbits=total_kernel_qbits,
            kernel_target_qbit=local_qbit_to_kernel_index[member.local_target_qbit],
        )
    if op.name == "CNOT":
        if member.local_target_qbit is None or member.local_control_qbit is None:
            raise runtime_validation_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_control_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Fused partitioned runtime requires both local qubits for CNOT members",
            )
        k_ctl, k_tgt = _kernel_indices_for_fused_cnot(
            local_qbit_to_kernel_index=local_qbit_to_kernel_index,
            local_target_qbit=member.local_target_qbit,
            local_control_qbit=member.local_control_qbit,
        )
        return _embed_cnot_gate(
            total_kernel_qbits=total_kernel_qbits,
            kernel_control_qbit=k_ctl,
            kernel_target_qbit=k_tgt,
        )
    raise runtime_validation_error(
        descriptor_set,
        category="unsupported_runtime_operation",
        first_unsupported_condition="fusion_gate_name",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason="Fused partitioned runtime cannot build a fused kernel for '{}'".format(
            op.name
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
        raise runtime_validation_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="local_qubit_support",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Fused partitioned runtime requires non-empty local_qubit_support",
        )
    if len(active_local_qbits) > 2:
        raise runtime_validation_error(
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
        op = descriptor_set.canonical_operation_for(member)
        if not op.is_unitary:
            raise runtime_validation_error(
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
    descriptor_set: NoisyPartitionDescriptorSet,
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
        operation_names=tuple(
            descriptor_set.canonical_operation_for(m).name for m in members
        ),
        global_target_qbits=global_target_qbits,
        local_target_qbits=active_local_qbits,
        member_count=len(members),
        gate_count=sum(
            descriptor_set.canonical_operation_for(m).is_unitary for m in members
        ),
    )


def _build_noise_boundary_record(
    descriptor_set: NoisyPartitionDescriptorSet,
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
        operation_names=tuple(
            descriptor_set.canonical_operation_for(m).name for m in relevant_members
        ),
        global_target_qbits=global_target_qbits,
        local_target_qbits=active_local_qbits,
        member_count=len(relevant_members),
        gate_count=sum(
            descriptor_set.canonical_operation_for(m).is_unitary
            for m in relevant_members
        ),
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
    segments = _iter_member_segments(descriptor_set, partition.members)
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
                            descriptor_set,
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
                    raise runtime_validation_error(
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
                        descriptor_set,
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
                        descriptor_set,
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
                    descriptor_set,
                    partition,
                    left_members,
                    segment_members,
                    right_members,
                )
            )
    return tuple(fused_regions)
