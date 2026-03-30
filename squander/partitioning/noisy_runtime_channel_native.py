"""Phase 3.1 channel-native fused motifs (Kraus composition, slice v1)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from squander.density_matrix import DensityMatrix
from squander.partitioning.noisy_runtime_core import (
    PHASE3_FUSION_CLASS_FUSED,
    PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF,
    NoisyRuntimeFusedRegionRecord,
    _segment_parameter_vector,
)
from squander.partitioning.noisy_runtime_errors import runtime_validation_error
from squander.partitioning.noisy_runtime_fusion import (
    _embed_cnot_gate,
    _embed_single_qubit_gate,
    _kernel_indices_for_fused_cnot,
)
from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError
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
    d = _validate_kraus_bundle_shape(
        bundle,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
        label="invariant check",
    )
    acc = np.zeros((d, d), dtype=np.complex128)
    for k in range(bundle.shape[0]):
        kj = bundle[k]
        acc += kj.conj().T @ kj
    res = float(np.linalg.norm(acc - np.eye(d, dtype=np.complex128), ord="fro"))
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


def _embed_two_qubit_operator_on_globals(
    op: np.ndarray,
    *,
    qbit_num: int,
    g0: int,
    g1: int,
) -> np.ndarray:
    """Embed a 4×4 operator on global qubits ``g0`` (local index 0) and ``g1`` (local index 1).

    Basis convention matches ``_embed_single_qubit_gate``: global state index bit ``k`` is qubit ``k``.
    Subsystem row/column index is ``b_g0 + 2 * b_g1``.
    """
    dim = 1 << qbit_num
    mask_all = dim - 1
    rest_mask = mask_all & ~((1 << g0) | (1 << g1))
    embedded = np.zeros((dim, dim), dtype=np.complex128)
    for r in range(dim):
        for c in range(dim):
            if (r & rest_mask) != (c & rest_mask):
                continue
            sub_r = ((r >> g0) & 1) + 2 * ((r >> g1) & 1)
            sub_c = ((c >> g0) & 1) + 2 * ((c >> g1) & 1)
            embedded[r, c] = op[sub_r, sub_c]
    return embedded


def _apply_kraus_bundle(
    bundle: np.ndarray,
    rho: DensityMatrix,
    *,
    qbit_num: int,
    local_support: tuple[int, ...],
    global_target_qbits: tuple[int, ...],
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
) -> DensityMatrix:
    d = _validate_kraus_bundle_shape(
        bundle,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
        label="apply",
    )
    if len(local_support) != len(global_target_qbits):
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native apply local_support length {} does not match "
                "global_target_qbits length {}".format(
                    len(local_support), len(global_target_qbits)
                )
            ),
        )
    if len(local_support) > 2:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native apply supports at most two local qubits, got {}".format(
                    len(local_support)
                )
            ),
        )
    if (1 << len(local_support)) != d:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native apply bundle dimension d={} inconsistent with "
                "local support width {}".format(d, len(local_support))
            ),
        )
    for g in global_target_qbits:
        if g < 0 or g >= qbit_num:
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_execution",
                first_unsupported_condition="channel_native_representation",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Channel-native apply global target qubit {} out of range for qbit_num={}".format(
                        g, qbit_num
                    )
                ),
            )
    if d == 4 and global_target_qbits[0] == global_target_qbits[1]:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native apply requires two distinct global qubits for a 2-qubit bundle",
        )

    rho_np = np.asarray(rho.to_numpy(), dtype=np.complex128)
    dim = 1 << qbit_num
    if rho_np.shape != (dim, dim):
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native apply density matrix shape {} does not match qbit_num={}".format(
                    rho_np.shape, qbit_num
                )
            ),
        )

    if (
        qbit_num == 1
        and len(local_support) == 1
        and d == 2
        and global_target_qbits == (0,)
    ):
        out = np.zeros((2, 2), dtype=np.complex128)
        for k in range(bundle.shape[0]):
            kj = bundle[k]
            out += kj @ rho_np @ kj.conj().T
        return DensityMatrix.from_numpy(out)

    out = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(bundle.shape[0]):
        kj = bundle[k]
        if d == 2:
            k_full = _embed_single_qubit_gate(
                kj,
                total_kernel_qbits=qbit_num,
                kernel_target_qbit=global_target_qbits[0],
            )
        else:
            k_full = _embed_two_qubit_operator_on_globals(
                kj,
                qbit_num=qbit_num,
                g0=global_target_qbits[0],
                g1=global_target_qbits[1],
            )
        out += k_full @ rho_np @ k_full.conj().T
    return DensityMatrix.from_numpy(out)


_CHANNEL_NATIVE_SLICE_ALLOWED_OPS = frozenset(
    {"U3", "CNOT", "local_depolarizing", "amplitude_damping", "phase_damping"}
)


def _scan_channel_native_whole_partition_motif(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
) -> tuple[Literal["eligible"], tuple[int, ...]] | tuple[Literal["route"], str]:
    """Return eligible local support, or a frozen hybrid route reason (no exceptions)."""
    members = partition.members
    if not members:
        return ("route", "channel_native_support_surface")
    noise_seen = False
    local_support: set[int] = set()
    for member in members:
        op = descriptor_set.canonical_operation_for(member)
        if op.name not in _CHANNEL_NATIVE_SLICE_ALLOWED_OPS:
            return ("route", "channel_native_support_surface")
        if op.kind == PLANNER_OP_KIND_NOISE:
            noise_seen = True
        elif op.kind != PLANNER_OP_KIND_GATE:
            return ("route", "channel_native_support_surface")
        local_support.update(member.local_qubit_support)
    if not noise_seen:
        return ("route", "pure_unitary_partition")
    if len(local_support) > 2:
        return ("route", "channel_native_qubit_span")
    for member in members:
        for lb in member.local_qubit_support:
            if lb not in local_support:
                return ("route", "channel_native_support_surface")
    return ("eligible", tuple(sorted(local_support)))


def classify_partition_channel_native_route(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    *,
    runtime_path: str,
) -> tuple[bool, tuple[int, ...] | None, str]:
    """Preflight classification for hybrid routing (strict execution unchanged).

    Returns ``(eligible, local_support_or_none, route_reason)``. When ``eligible``
    is True, ``route_reason`` is ``eligible_channel_native_motif`` and
    ``local_support`` is the motif support. When ``eligible`` is False, the
    partition should be executed via the shipped Phase 3 path with
    ``partition_route_reason`` set to the returned frozen reason string.
    """
    del runtime_path  # reserved for future diagnostics parity with strict path
    scan = _scan_channel_native_whole_partition_motif(descriptor_set, partition)
    if scan[0] == "eligible":
        return (True, scan[1], "eligible_channel_native_motif")
    return (False, None, scan[1])


def _validate_whole_partition_motif(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    *,
    runtime_path: str,
) -> tuple[int, ...]:
    scan = _scan_channel_native_whole_partition_motif(descriptor_set, partition)
    if scan[0] == "eligible":
        return scan[1]
    route_reason = scan[1]
    if route_reason == "pure_unitary_partition":
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_noise_presence",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native counted motif requires at least one noise operation",
        )
    if route_reason == "channel_native_qubit_span":
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_operation",
            first_unsupported_condition="channel_native_qubit_span",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native slice supports at most two local qubits in the motif",
        )
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
    allowed = _CHANNEL_NATIVE_SLICE_ALLOWED_OPS
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
        if op.kind != PLANNER_OP_KIND_GATE and op.kind != PLANNER_OP_KIND_NOISE:
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_operation",
                first_unsupported_condition="channel_native_support_surface",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native slice expected gate or noise kind",
            )
    raise runtime_validation_error(
        descriptor_set,
        category="unsupported_runtime_operation",
        first_unsupported_condition="channel_native_support_surface",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason="Member local qubit not contained in motif support",
    )


def _local_qbit_to_kernel_index(local_support: tuple[int, ...]) -> dict[int, int]:
    return {lq: idx for idx, lq in enumerate(local_support)}


def _kraus_bundle_cnot_for_local_support(
    member: NoisyPartitionDescriptorMember,
    *,
    local_support: tuple[int, ...],
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
) -> np.ndarray:
    if len(local_support) != 2:
        raise runtime_validation_error(
            descriptor_set,
            category="unsupported_runtime_execution",
            first_unsupported_condition="channel_native_representation",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason=(
                "Channel-native CNOT lowering requires local support width 2, got {}".format(
                    len(local_support)
                )
            ),
        )
    if member.local_control_qbit is None or member.local_target_qbit is None:
        raise runtime_validation_error(
            descriptor_set,
            category="descriptor_to_runtime_mismatch",
            first_unsupported_condition="local_control_qbit",
            failure_stage="runtime_preflight",
            runtime_path=runtime_path,
            reason="Channel-native CNOT requires local control and target qubits",
        )
    for wire in (member.local_control_qbit, member.local_target_qbit):
        if wire not in local_support:
            raise runtime_validation_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native CNOT wire {} not in motif local_support".format(
                    wire
                ),
            )
    lmap = _local_qbit_to_kernel_index(local_support)
    k_ctl, k_tgt = _kernel_indices_for_fused_cnot(
        local_qbit_to_kernel_index=lmap,
        local_target_qbit=member.local_target_qbit,
        local_control_qbit=member.local_control_qbit,
    )
    u = _embed_cnot_gate(
        total_kernel_qbits=2,
        kernel_control_qbit=k_ctl,
        kernel_target_qbit=k_tgt,
    )
    return np.array([u], dtype=np.complex128)


def _lift_single_qubit_kraus_bundle_to_local_support(
    bundle_1q: np.ndarray,
    *,
    target_local_qbit: int | None,
    local_support: tuple[int, ...],
    descriptor_set: NoisyPartitionDescriptorSet,
    runtime_path: str,
    label: str = "single-qubit kraus bundle",
) -> np.ndarray:
    n = len(local_support)
    if n == 1:
        d = _validate_kraus_bundle_shape(
            bundle_1q,
            descriptor_set=descriptor_set,
            runtime_path=runtime_path,
            label=label,
        )
        if d != 2:
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_execution",
                first_unsupported_condition="channel_native_representation",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native {} on 1-qubit support must have d=2".format(
                    label
                ),
            )
        return bundle_1q
    if n == 2:
        if target_local_qbit is None:
            raise runtime_validation_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason=(
                    "Channel-native {} lift to 2-qubit support requires "
                    "local_target_qbit".format(label)
                ),
            )
        if target_local_qbit not in local_support:
            raise runtime_validation_error(
                descriptor_set,
                category="descriptor_to_runtime_mismatch",
                first_unsupported_condition="local_target_qbit",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native {} target local qubit {} not in support".format(
                    label, target_local_qbit
                ),
            )
        d = _validate_kraus_bundle_shape(
            bundle_1q,
            descriptor_set=descriptor_set,
            runtime_path=runtime_path,
            label=label,
        )
        if d != 2:
            raise runtime_validation_error(
                descriptor_set,
                category="unsupported_runtime_execution",
                first_unsupported_condition="channel_native_representation",
                failure_stage="runtime_preflight",
                runtime_path=runtime_path,
                reason="Channel-native {} before lift must have d=2".format(label),
            )
        kernel_idx = local_support.index(target_local_qbit)
        kcnt = bundle_1q.shape[0]
        out = np.empty((kcnt, 4, 4), dtype=np.complex128)
        for i in range(kcnt):
            out[i] = _embed_single_qubit_gate(
                bundle_1q[i],
                total_kernel_qbits=2,
                kernel_target_qbit=kernel_idx,
            )
        return out
    raise runtime_validation_error(
        descriptor_set,
        category="unsupported_runtime_execution",
        first_unsupported_condition="channel_native_representation",
        failure_stage="runtime_preflight",
        runtime_path=runtime_path,
        reason=(
            "Channel-native single-qubit bundle lift supports width 1 or 2 only, got {}".format(
                n
            )
        ),
    )


def _member_to_kraus_bundle(
    descriptor_set: NoisyPartitionDescriptorSet,
    member: NoisyPartitionDescriptorMember,
    segment_parameters: np.ndarray,
    *,
    local_support: tuple[int, ...],
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
            base = _kraus_bundle_u3(float(theta), float(phi), float(lam))
            return _lift_single_qubit_kraus_bundle_to_local_support(
                base,
                target_local_qbit=member.local_target_qbit,
                local_support=local_support,
                descriptor_set=descriptor_set,
                runtime_path=runtime_path,
                label="U3 kraus bundle",
            )
        if op.name == "CNOT":
            return _kraus_bundle_cnot_for_local_support(
                member,
                local_support=local_support,
                descriptor_set=descriptor_set,
                runtime_path=runtime_path,
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
            base = _kraus_bundle_local_depolarizing(p)
            return _lift_single_qubit_kraus_bundle_to_local_support(
                base,
                target_local_qbit=member.local_target_qbit,
                local_support=local_support,
                descriptor_set=descriptor_set,
                runtime_path=runtime_path,
                label="local_depolarizing kraus bundle",
            )
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
            base = _kraus_bundle_amplitude_damping(g)
            return _lift_single_qubit_kraus_bundle_to_local_support(
                base,
                target_local_qbit=member.local_target_qbit,
                local_support=local_support,
                descriptor_set=descriptor_set,
                runtime_path=runtime_path,
                label="amplitude_damping kraus bundle",
            )
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
            base = _kraus_bundle_phase_damping(lam)
            return _lift_single_qubit_kraus_bundle_to_local_support(
                base,
                target_local_qbit=member.local_target_qbit,
                local_support=local_support,
                descriptor_set=descriptor_set,
                runtime_path=runtime_path,
                label="phase_damping kraus bundle",
            )
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
            local_support=local_support,
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
    global_target_qbits = _global_targets_for_local_support(partition, local_support)
    try:
        rho_out = _apply_kraus_bundle(
            kraus_bundle,
            rho,
            qbit_num=descriptor_set.qbit_num,
            local_support=local_support,
            global_target_qbits=global_target_qbits,
            descriptor_set=descriptor_set,
            runtime_path=runtime_path,
        )
    except NoisyRuntimeValidationError:
        raise
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
