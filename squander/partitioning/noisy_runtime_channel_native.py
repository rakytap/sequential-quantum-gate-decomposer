"""Phase 3.1 channel-native fused motifs (Kraus composition, slice v1)."""

from __future__ import annotations

import numpy as np

from squander.density_matrix import DensityMatrix
from squander.partitioning.noisy_runtime_core import (
    PHASE3_FUSION_CLASS_FUSED,
    PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF,
    NoisyRuntimeFusedRegionRecord,
    _segment_parameter_vector,
)
from squander.partitioning.noisy_runtime_errors import runtime_validation_error
from squander.partitioning.noisy_types import (
    PLANNER_OP_KIND_GATE,
    PLANNER_OP_KIND_NOISE,
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorMember,
    NoisyPartitionDescriptorSet,
)

# P31-ADR-008 style (representation-level equality residuals)
_PHASE31_KRAUS_COMPLETENESS_TOL = 1e-10
_PHASE31_CHOI_POSITIVITY_FLOOR = -1e-12

_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kraus_operator_dim_for_support_qubit_count(
    support_qubit_count: int,
    *,
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
) -> int:
    """Hilbert-space dimension d = 2**n for bounded local support n in {1, 2}."""
    if support_qubit_count == 1:
        return 2
    if support_qubit_count == 2:
        return 4
    raise runtime_validation_error(
        descriptor_set,
        category="unsupported_runtime_execution",
        first_unsupported_condition="channel_native_representation",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason=(
            "Channel-native Kraus operator dimension supports local support "
            "qubit count 1 or 2 only, got {}".format(support_qubit_count)
        ),
    )


def _validate_kraus_bundle_shape(
    bundle: np.ndarray,
    *,
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
    label: str = "kraus_bundle",
) -> int:
    """Require rank-3 (K, d, d) with square Kraus matrices and d in {2, 4}."""
    if bundle.ndim != 3:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native {} must be rank-3 (kraus_count, d, d), got ndim={}".format(
                    label, bundle.ndim
                )
            ),
        )
    if bundle.shape[0] < 1:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native {} must have at least one Kraus operator".format(
                label
            ),
        )
    d = int(bundle.shape[1])
    if bundle.shape[2] != d:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native {} Kraus matrices must be square, got shape {}".format(
                    label, bundle.shape
                )
            ),
        )
    if d not in (2, 4):
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native {} supports operator dimension d in {{2, 4}} only, got d={}".format(
                    label, d
                )
            ),
        )
    return d


def _identity_kraus_bundle_for_support_qubit_count(
    support_qubit_count: int,
    *,
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
) -> np.ndarray:
    d = _kraus_operator_dim_for_support_qubit_count(
        support_qubit_count,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
    )
    bundle = np.array([np.eye(d, dtype=np.complex128)])
    _validate_kraus_bundle_shape(
        bundle,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
        label="identity kraus bundle",
    )
    return bundle


def _clamp_noise_rate_to_unit_interval(x: float) -> float:
    """Match LocalDepolarizingOp / AmplitudeDampingOp / PhaseDampingOp parametric clamp in noise_operation.cpp."""
    v = float(x)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


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


def _compose_kraus_bundles(
    acc: np.ndarray,
    nxt: np.ndarray,
    *,
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
) -> np.ndarray:
    """Φ_nxt ∘ Φ_acc has Kraus operators N_b A_a (apply acc first, then nxt)."""
    d_acc = _validate_kraus_bundle_shape(
        acc,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
        label="compose acc",
    )
    d_nxt = _validate_kraus_bundle_shape(
        nxt,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
        label="compose nxt",
    )
    if d_acc != d_nxt:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native Kraus bundle dimension mismatch in composition: "
                "d_acc={} d_nxt={}".format(d_acc, d_nxt)
            ),
        )
    na = acc.shape[0]
    nb = nxt.shape[0]
    d = d_acc
    out = np.empty((na * nb, d, d), dtype=np.complex128)
    idx = 0
    for b in range(nb):
        for a in range(na):
            out[idx] = nxt[b] @ acc[a]
            idx += 1
    return out


def _kraus_bundle_u3(theta: float, phi: float, lam: float) -> np.ndarray:
    u = _u3_unitary(float(theta), float(phi), float(lam))
    return np.array([u], dtype=np.complex128)


def _kraus_bundle_local_depolarizing(p: float) -> np.ndarray:
    """Match LocalDepolarizingOp in noise_operation.cpp (identity + Pauli weights)."""
    p = float(p)
    s0 = np.sqrt(max(0.0, 1.0 - 0.75 * p))
    sp = np.sqrt(max(0.0, p)) / 2.0
    return np.stack(
        [
            s0 * np.eye(2, dtype=np.complex128),
            sp * _PAULI_X,
            sp * _PAULI_Y,
            sp * _PAULI_Z,
        ]
    )


def _kraus_bundle_amplitude_damping(gamma: float) -> np.ndarray:
    g = float(gamma)
    sqrt_1mg = np.sqrt(max(0.0, 1.0 - g))
    sqrt_g = np.sqrt(max(0.0, g))
    k0 = np.array([[1, 0], [0, sqrt_1mg]], dtype=np.complex128)
    k1 = np.array([[0, sqrt_g], [0, 0]], dtype=np.complex128)
    return np.stack([k0, k1])


def _kraus_bundle_phase_damping(lam: float) -> np.ndarray:
    """Kraus form equivalent to PhaseDampingOp::apply_phase_damping (diag unchanged)."""
    lmb = float(lam)
    a = np.sqrt(max(0.0, 1.0 - lmb))
    b = np.sqrt(max(0.0, lmb))
    k0 = np.array([[1, 0], [0, a]], dtype=np.complex128)
    k1 = np.array([[0, 0], [0, b]], dtype=np.complex128)
    return np.stack([k0, k1])


def _check_kraus_bundle_invariants(
    bundle: np.ndarray,
    *,
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
) -> None:
    acc = np.zeros((2, 2), dtype=np.complex128)
    for k in range(bundle.shape[0]):
        kj = bundle[k]
        acc += kj.conj().T @ kj
    res = float(np.linalg.norm(acc - np.eye(2, dtype=np.complex128), ord="fro"))
    if res > _PHASE31_KRAUS_COMPLETENESS_TOL:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_invariant_failure",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native Kraus completeness residual {:.3e} exceeds tol {:.3e}".format(
                    res, _PHASE31_KRAUS_COMPLETENESS_TOL
                )
            ),
        )
    d = 2
    choi = np.zeros((d * d, d * d), dtype=np.complex128)
    for k in range(bundle.shape[0]):
        v = np.reshape(bundle[k], (d * d,), order="C")
        choi += np.outer(v, v.conj())
    choi = (choi + choi.conj().T) / 2.0
    min_eig = float(np.min(np.real(np.linalg.eigvalsh(choi))))
    if min_eig < _PHASE31_CHOI_POSITIVITY_FLOOR:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_invariant_failure",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native Choi minimum eigenvalue {:.3e} below floor {:.3e}".format(
                    min_eig, _PHASE31_CHOI_POSITIVITY_FLOOR
                )
            ),
        )


def _apply_kraus_bundle(
    bundle: np.ndarray,
    rho: DensityMatrix,
    *,
    qbit_num: int,
) -> DensityMatrix:
    if qbit_num != 1:
        raise NotImplementedError(
            "Slice v1 supports only qbit_num == 1 workloads for channel-native apply"
        )
    rho_np = np.asarray(rho.to_numpy(), dtype=np.complex128)
    out = np.zeros((2, 2), dtype=np.complex128)
    for k in range(bundle.shape[0]):
        kj = bundle[k]
        out += kj @ rho_np @ kj.conj().T
    return DensityMatrix.from_numpy(out)


def _validate_whole_partition_motif(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    *,
    runtime_path: str,
) -> tuple[int, ...]:
    members = partition.members
    if not members:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_support_surface",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native slice requires non-empty partition members",
        )
    allowed = frozenset(
        {"U3", "CNOT", "local_depolarizing", "amplitude_damping", "phase_damping"}
    )
    noise_seen = False
    local_support: set[int] = set()
    for member in members:
        op = descriptor_set.canonical_operation_for(member)
        if op.name not in allowed:
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="channel_native_support_surface",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native slice does not support operation '{}'".format(
                    op.name
                ),
            )
        if op.kind == PLANNER_OP_KIND_NOISE:
            noise_seen = True
        elif op.kind != PLANNER_OP_KIND_GATE:
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="channel_native_support_surface",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native slice expected gate or noise kind",
            )
        local_support.update(member.local_qubit_support)
    if not noise_seen:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_noise_presence",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native counted motif requires at least one noise operation",
        )
    if len(local_support) > 2:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_qubit_span",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native slice supports at most two local qubits in the motif",
        )
    for member in members:
        for lb in member.local_qubit_support:
            if lb not in local_support:
                raise runtime_validation_error(
                    descriptor_set,
                    category="unsupported_runtime_operation",
                    first_unsupported_condition="channel_native_support_surface",
                    failure_stage="runtime_preflight",
                    runtime_path=runtime_path,
                    reason="Member local qubit not contained in motif support",
                )
    if descriptor_set.qbit_num != 1 or len(local_support) != 1:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_qubit_span",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Slice v1 channel-native fusion is implemented only for single-qubit workloads",
        )
    if tuple(sorted(local_support)) != (0,):
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_qubit_span",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Single-qubit slice requires local motif support {{0}}",
        )
    return tuple(sorted(local_support))


def _member_to_kraus_bundle(
    descriptor_set: NoisyPartitionDescriptorSet,
    member: NoisyPartitionDescriptorMember,
    segment_parameters: np.ndarray,
    *,
    runtime_path: str,
) -> np.ndarray:
    op = descriptor_set.canonical_operation_for(member)
    if op.kind == PLANNER_OP_KIND_GATE:
        if op.name == "U3":
            if member.local_target_qbit is None:
                raise runtime_validation_error(
                    descriptor_set,
                    category="descriptor_to_runtime_mismatch",
                    first_unsupported_condition="local_target_qbit",
                    failure_stage="runtime_preflight",
                    runtime_path=runtime_path,
                    reason="Channel-native requires local_target_qbit for U3",
                )
            local_start = member.local_param_start
            local_stop = local_start + op.param_count
            theta, phi, lam = segment_parameters[local_start:local_stop]
            return _kraus_bundle_u3(float(theta), float(phi), float(lam))
        if op.name == "CNOT":
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="channel_native_support_surface",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Slice v1 channel-native does not support CNOT motifs yet",
            )
    if op.kind == PLANNER_OP_KIND_NOISE:
        local_start = member.local_param_start
        local_stop = local_start + op.param_count
        if op.name == "local_depolarizing":
            if op.param_count == 0:
                p = _clamp_noise_rate_to_unit_interval(float(op.fixed_value))
            else:
                if local_stop > segment_parameters.size:
                    raise runtime_validation_error(
                        descriptor_set,
                        category="descriptor_to_runtime_mismatch",
                        first_unsupported_condition="parameter_routing",
                        failure_stage="runtime_preflight",
                        runtime_path=runtime_path,
                        reason="Channel-native noise parameters out of range",
                    )
                p = _clamp_noise_rate_to_unit_interval(
                    float(segment_parameters[local_start])
                )
            return _kraus_bundle_local_depolarizing(p)
        if op.name == "amplitude_damping":
            if op.param_count == 0:
                g = _clamp_noise_rate_to_unit_interval(float(op.fixed_value))
            else:
                if local_stop > segment_parameters.size:
                    raise runtime_validation_error(
                        descriptor_set,
                        category="descriptor_to_runtime_mismatch",
                        first_unsupported_condition="parameter_routing",
                        failure_stage="runtime_preflight",
                        runtime_path=runtime_path,
                        reason="Channel-native noise parameters out of range",
                    )
                g = _clamp_noise_rate_to_unit_interval(
                    float(segment_parameters[local_start])
                )
            return _kraus_bundle_amplitude_damping(g)
        if op.name == "phase_damping":
            if op.param_count == 0:
                lam = _clamp_noise_rate_to_unit_interval(float(op.fixed_value))
            else:
                if local_stop > segment_parameters.size:
                    raise runtime_validation_error(
                        descriptor_set,
                        category="descriptor_to_runtime_mismatch",
                        first_unsupported_condition="parameter_routing",
                        failure_stage="runtime_preflight",
                        runtime_path=runtime_path,
                        reason="Channel-native noise parameters out of range",
                    )
                lam = _clamp_noise_rate_to_unit_interval(
                    float(segment_parameters[local_start])
                )
            return _kraus_bundle_phase_damping(lam)
    raise runtime_validation_error(
        descriptor_set,
        category="unsupported_runtime_operation",
        first_unsupported_condition="channel_native_support_surface",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason="Channel-native cannot lower operation '{}'".format(op.name),
    )


def _global_targets_for_local_support(
    partition: NoisyPartitionDescriptor,
    local_support: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(partition.local_to_global_qbits[lq] for lq in local_support)


def _build_channel_native_region_record(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    members: tuple[NoisyPartitionDescriptorMember, ...],
    *,
    kraus_count: int,
) -> NoisyRuntimeFusedRegionRecord:
    active_local_qbits = tuple(
        sorted({lq for m in members for lq in m.local_qubit_support})
    )
    global_target_qbits = tuple(
        partition.local_to_global_qbits[lq] for lq in active_local_qbits
    )
    return NoisyRuntimeFusedRegionRecord(
        partition_index=partition.partition_index,
        candidate_kind=PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF,
        classification=PHASE3_FUSION_CLASS_FUSED,
        reason="channel_native_motif_kraus_count_{}".format(kraus_count),
        partition_member_indices=tuple(m.partition_member_index for m in members),
        canonical_operation_indices=tuple(m.canonical_operation_index for m in members),
        operation_names=tuple(
            descriptor_set.canonical_operation_for(m).name for m in members
        ),
        global_target_qbits=global_target_qbits,
        local_target_qbits=active_local_qbits,
        member_count=len(members),
        gate_count=sum(
            1
            for m in members
            if descriptor_set.canonical_operation_for(m).kind == PLANNER_OP_KIND_GATE
        ),
    )


def execute_partition_channel_native(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    local_parameter_vector: np.ndarray,
    rho: DensityMatrix,
    *,
    runtime_path: str,
) -> tuple[tuple[NoisyRuntimeFusedRegionRecord, ...], DensityMatrix]:
    members = partition.members
    local_support = _validate_whole_partition_motif(
        descriptor_set, partition, runtime_path=runtime_path
    )
    segment_parameters = _segment_parameter_vector(
        descriptor_set,
        members,
        local_parameter_vector,
        runtime_path=runtime_path,
    )
    kraus_bundle = _identity_kraus_bundle_for_support_qubit_count(
        len(local_support),
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
    )
    for member in members:
        step = _member_to_kraus_bundle(
            descriptor_set,
            member,
            segment_parameters,
            runtime_path=runtime_path,
        )
        kraus_bundle = _compose_kraus_bundles(
            kraus_bundle,
            step,
            descriptor_set=descriptor_set,
            runtime_path=runtime_path,
        )
    _check_kraus_bundle_invariants(
        kraus_bundle,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
    )
    _ = _global_targets_for_local_support(partition, local_support)
    try:
        rho_out = _apply_kraus_bundle(
            kraus_bundle,
            rho,
            qbit_num=descriptor_set.qbit_num,
        )
    except NotImplementedError as exc:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_runtime_execution",
            failure_stage="runtime_execution",
            runtime_path=runtime_path,
            reason=str(exc),
        ) from exc
    except Exception as exc:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_runtime_execution",
            failure_stage="runtime_execution",
            runtime_path=runtime_path,
            reason="Channel-native apply failed: {}".format(exc),
        ) from exc
    rec = _build_channel_native_region_record(
        descriptor_set,
        partition,
        members,
        kraus_count=int(kraus_bundle.shape[0]),
    )
    return (rec,), rho_out
