"""Phase 3.1 channel-native representation substrate (helpers), not public 2q runtime.

These tests import private helpers from ``noisy_runtime_channel_native`` to lock
bundle shape, invariants, and support-aware application semantics.

Still deferred (other engineering tasks): CNOT lowering (``P31-S04-E02``), public
positive admission of wider channel-native workloads (``P31-S05-E01``), and counted
end-to-end 2q sequential-reference gates (``P31-S06-E01``).

The first-slice public regression file
``test_partitioned_channel_native_phase31_slice.py`` remains the mandatory 1q
runtime bar; this module stays helper-oriented.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander.density_matrix import DensityMatrix
from squander.partitioning.noisy_runtime import PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
from squander.partitioning.noisy_runtime_channel_native import (
    _apply_kraus_bundle,
    _check_kraus_bundle_invariants,
    _kraus_bundle_phase_damping,
)
from squander.partitioning.noisy_runtime_fusion import _embed_single_qubit_gate
from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError
from tests.partitioning.fixtures.runtime import PHASE3_RUNTIME_DENSITY_TOL
from tests.partitioning.fixtures.workloads import build_phase31_microcase_descriptor_set


def _descriptor_set_for_invariant_tests():
    return build_phase31_microcase_descriptor_set(
        "phase31_microcase_1q_u3_local_noise_chain"
    )


def test_phase31_channel_native_invariants_accept_2q_identity_bundle():
    descriptor_set = _descriptor_set_for_invariant_tests()
    bundle = np.array([np.eye(4, dtype=np.complex128)])
    _check_kraus_bundle_invariants(
        bundle,
        descriptor_set=descriptor_set,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )


def test_phase31_channel_native_invariants_accept_kron_two_unitaries():
    descriptor_set = _descriptor_set_for_invariant_tests()
    u1 = np.eye(2, dtype=np.complex128)
    u2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    bundle = np.array([np.kron(u1, u2)])
    _check_kraus_bundle_invariants(
        bundle,
        descriptor_set=descriptor_set,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )


def test_phase31_channel_native_invariants_reject_non_trace_preserving_4x4():
    descriptor_set = _descriptor_set_for_invariant_tests()
    bundle = np.array([2.0 * np.eye(4, dtype=np.complex128)])
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        _check_kraus_bundle_invariants(
            bundle,
            descriptor_set=descriptor_set,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
    assert excinfo.value.first_unsupported_condition == "channel_native_invariant_failure"


def test_phase31_channel_native_apply_helper_embeds_1q_bundle_into_2q_density():
    descriptor_set = _descriptor_set_for_invariant_tests()
    bell = np.zeros((4, 4), dtype=np.complex128)
    bell[0, 0] = bell[3, 3] = bell[0, 3] = bell[3, 0] = 0.5
    rho = DensityMatrix.from_numpy(bell)
    bundle = _kraus_bundle_phase_damping(0.25)
    rho_np = np.asarray(rho.to_numpy(), dtype=np.complex128)
    ref_out = np.zeros((4, 4), dtype=np.complex128)
    for k in range(bundle.shape[0]):
        k_full = _embed_single_qubit_gate(
            bundle[k], total_kernel_qbits=2, kernel_target_qbit=0
        )
        ref_out += k_full @ rho_np @ k_full.conj().T
    out = _apply_kraus_bundle(
        bundle,
        rho,
        qbit_num=2,
        local_support=(0,),
        global_target_qbits=(0,),
        descriptor_set=descriptor_set,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    got = np.asarray(out.to_numpy(), dtype=np.complex128)
    assert float(np.linalg.norm(got - ref_out, ord="fro")) <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase31_channel_native_apply_helper_identity_2q_leaves_2q_density():
    descriptor_set = _descriptor_set_for_invariant_tests()
    rng = np.random.default_rng(0)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi = psi / np.linalg.norm(psi)
    rho = DensityMatrix.from_numpy(np.outer(psi, psi.conj()).astype(np.complex128))
    bundle = np.array([np.eye(4, dtype=np.complex128)])
    out = _apply_kraus_bundle(
        bundle,
        rho,
        qbit_num=2,
        local_support=(0, 1),
        global_target_qbits=(0, 1),
        descriptor_set=descriptor_set,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    assert float(
        np.linalg.norm(
            np.asarray(out.to_numpy()) - np.asarray(rho.to_numpy()), ord="fro"
        )
    ) <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase31_channel_native_apply_helper_2q_kron_matches_product_embed():
    """Kraus matrix for U0 on global 0 and U1 on global 1 matches product embed.

    ``_embed_two_qubit_operator_on_globals`` uses subsystem index ``b_g0 + 2*b_g1``
    (g0 LSB). NumPy ``np.kron(A, B)`` places the first factor on the more
    significant index bit, so the 4×4 operator is ``np.kron(U1, U0)``.
    """
    descriptor_set = _descriptor_set_for_invariant_tests()
    u0 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    u1 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    bundle = np.array([np.kron(u1, u0)])
    rng = np.random.default_rng(1)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi = psi / np.linalg.norm(psi)
    rho = DensityMatrix.from_numpy(np.outer(psi, psi.conj()).astype(np.complex128))
    rho_np = np.asarray(rho.to_numpy(), dtype=np.complex128)
    k_ref = _embed_single_qubit_gate(
        u0, total_kernel_qbits=2, kernel_target_qbit=0
    ) @ _embed_single_qubit_gate(u1, total_kernel_qbits=2, kernel_target_qbit=1)
    ref_out = k_ref @ rho_np @ k_ref.conj().T
    out = _apply_kraus_bundle(
        bundle,
        rho,
        qbit_num=2,
        local_support=(0, 1),
        global_target_qbits=(0, 1),
        descriptor_set=descriptor_set,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    got = np.asarray(out.to_numpy(), dtype=np.complex128)
    assert float(np.linalg.norm(got - ref_out, ord="fro")) <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase31_channel_native_apply_helper_rejects_support_above_2q():
    descriptor_set = _descriptor_set_for_invariant_tests()
    bundle = np.array([np.eye(2, dtype=np.complex128)])
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        _apply_kraus_bundle(
            bundle,
            DensityMatrix(3),
            qbit_num=3,
            local_support=(0, 1, 2),
            global_target_qbits=(0, 1, 2),
            descriptor_set=descriptor_set,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
    assert excinfo.value.first_unsupported_condition == "channel_native_representation"


def test_phase31_channel_native_apply_helper_rejects_4x4_bundle_with_1q_support():
    descriptor_set = _descriptor_set_for_invariant_tests()
    bundle = np.array([np.eye(4, dtype=np.complex128)])
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        _apply_kraus_bundle(
            bundle,
            DensityMatrix(2),
            qbit_num=2,
            local_support=(0,),
            global_target_qbits=(0,),
            descriptor_set=descriptor_set,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
    assert excinfo.value.first_unsupported_condition == "channel_native_representation"
