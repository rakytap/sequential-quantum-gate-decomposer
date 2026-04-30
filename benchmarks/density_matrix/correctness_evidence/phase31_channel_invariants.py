"""Kraus-bundle invariant metrics for Phase 3.1 strict microcase correctness records."""

from __future__ import annotations

from typing import Any

import numpy as np

from squander.partitioning.noisy_runtime import PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
from squander.partitioning.noisy_runtime_channel_native import (
    _compose_kraus_bundles,
    _identity_kraus_bundle_for_support_qubit_count,
    _member_to_kraus_bundle,
)
from squander.partitioning.noisy_runtime_core import (
    _build_partition_parameter_vector,
    _segment_parameter_vector,
)
from squander.partitioning.noisy_types import NoisyPartitionDescriptorSet

_KRAUS_COMPLETENESS_TOL = 1e-10
_CHOI_POSITIVITY_FLOOR = -1e-12


def _primary_partition(descriptor_set: NoisyPartitionDescriptorSet):
    for partition in descriptor_set.partitions:
        if len(partition.local_to_global_qbits) == descriptor_set.qbit_num:
            return partition
    return descriptor_set.partitions[0]


def build_fused_kraus_bundle_strict_microcase(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: np.ndarray,
) -> np.ndarray:
    """Fused channel-native Kraus bundle for the primary partition (matches pytest substrate)."""
    partition = _primary_partition(descriptor_set)
    local_vec = _build_partition_parameter_vector(
        descriptor_set,
        partition,
        parameters,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    seg = _segment_parameter_vector(
        descriptor_set,
        partition.members,
        local_vec,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    local_support = tuple(range(len(partition.local_to_global_qbits)))
    steps = [
        _member_to_kraus_bundle(
            descriptor_set,
            m,
            seg,
            local_support=local_support,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
        for m in partition.members
    ]
    acc = _identity_kraus_bundle_for_support_qubit_count(
        len(local_support),
        descriptor_set=descriptor_set,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    for step in steps:
        acc = _compose_kraus_bundles(
            acc,
            step,
            descriptor_set=descriptor_set,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
    return acc


def measure_kraus_bundle_invariants(bundle: np.ndarray) -> dict[str, Any]:
    """Scalar completeness / Choi checks aligned with ``_check_kraus_bundle_invariants``."""
    _k, d, _ = bundle.shape
    acc = np.zeros((d, d), dtype=np.complex128)
    for idx in range(bundle.shape[0]):
        kj = bundle[idx]
        acc += kj.conj().T @ kj
    completeness_residual = float(
        np.linalg.norm(acc - np.eye(d, dtype=np.complex128), ord="fro")
    )
    choi = np.zeros((d * d, d * d), dtype=np.complex128)
    for idx in range(bundle.shape[0]):
        vec = np.reshape(bundle[idx], (d * d,), order="C")
        choi += np.outer(vec, vec.conj())
    choi = (choi + choi.conj().T) / 2.0
    min_eig = float(np.min(np.real(np.linalg.eigvalsh(choi))))
    return {
        "kraus_operator_count": int(bundle.shape[0]),
        "support_dimension": int(d),
        "completeness_residual_frobenius": completeness_residual,
        "completeness_tolerance": _KRAUS_COMPLETENESS_TOL,
        "completeness_pass": completeness_residual <= _KRAUS_COMPLETENESS_TOL,
        "choi_min_eigenvalue_real": min_eig,
        "choi_positivity_floor": _CHOI_POSITIVITY_FLOOR,
        "choi_pass": min_eig >= _CHOI_POSITIVITY_FLOOR,
        "invariant_slice_pass": (
            completeness_residual <= _KRAUS_COMPLETENESS_TOL
            and min_eig >= _CHOI_POSITIVITY_FLOOR
        ),
    }


def build_strict_microcase_channel_invariants_slice(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: np.ndarray,
) -> dict[str, Any]:
    bundle = build_fused_kraus_bundle_strict_microcase(descriptor_set, parameters)
    metrics = measure_kraus_bundle_invariants(bundle)
    return {
        "runtime_path": PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        "representation_primary": "kraus_bundle",
        **metrics,
    }
