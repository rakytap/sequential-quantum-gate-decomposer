"""Phase 3.1 channel-native second slice: helper substrate, public runtime, boundaries.

**Helper / E01–E02 substrate:** imports private helpers from
``noisy_runtime_channel_native`` for bundle shape, invariants, support-aware
application, and (``P31-S04-E02``) CNOT / 2-local Kraus composition vs. the
sequential reference.

**Public runtime (``P31-S05-E01``):** ``execute_partitioned_density_channel_native``
admits bounded mixed motifs per partition via local-support validation in
``_validate_whole_partition_motif`` (no whole-workload single-qubit gate).

**Public boundary + audit (``P31-S05-E02``):** negative matrix for span, noise
presence, and non-channel-native gates on surfaces built with
``strict_phase3_support=False``; fused-region audit fields on positives.

The first-slice file ``test_partitioned_channel_native_phase31_slice.py`` is the
unchanged 1q public regression anchor; counted end-to-end 2q gates remain
``P31-S06-E01``.
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
from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
)
from squander.partitioning.noisy_runtime import (
    PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF,
    PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    execute_partitioned_density_channel_native,
    execute_sequential_density_reference,
)
from squander.partitioning.noisy_runtime_channel_native import (
    _apply_kraus_bundle,
    _check_kraus_bundle_invariants,
    _compose_kraus_bundles,
    _identity_kraus_bundle_for_support_qubit_count,
    _kraus_bundle_phase_damping,
    _member_to_kraus_bundle,
)
from squander.partitioning.noisy_runtime_core import (
    _build_partition_parameter_vector,
    _segment_parameter_vector,
    execute_sequential_density_reference,
)
from squander.partitioning.noisy_runtime_fusion import (
    _embed_cnot_gate,
    _embed_single_qubit_gate,
    _kernel_indices_for_fused_cnot,
)
from squander.partitioning.noisy_types import (
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorSet,
)
from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError
from tests.partitioning.fixtures.runtime import (
    PHASE3_RUNTIME_DENSITY_TOL,
    build_density_comparison_metrics,
    build_initial_parameters,
)
from tests.partitioning.fixtures.workloads import (
    _cnot,
    _noise_value,
    _u3,
    build_microcase_surface,
    build_phase31_microcase_descriptor_set,
)


def _descriptor_set_for_invariant_tests():
    return build_phase31_microcase_descriptor_set(
        "phase31_microcase_1q_u3_local_noise_chain"
    )


def _find_partition_with_span(
    descriptor_set: NoisyPartitionDescriptorSet,
    wanted_locals: tuple[int, ...] = (0, 1),
) -> NoisyPartitionDescriptor:
    for partition in descriptor_set.partitions:
        if partition.local_to_global_qbits == wanted_locals:
            return partition
    raise AssertionError(
        "no partition with local_to_global_qbits == {}".format(wanted_locals)
    )


def _local_qbit_to_kernel_index(local_support: tuple[int, ...]) -> dict[int, int]:
    return {lq: idx for idx, lq in enumerate(local_support)}


def _compose_kraus_step_sequence(
    steps: list[np.ndarray],
    *,
    descriptor_set: NoisyPartitionDescriptorSet,
    support_qubit_count: int,
    runtime_path: str,
) -> np.ndarray:
    acc = _identity_kraus_bundle_for_support_qubit_count(
        support_qubit_count,
        descriptor_set=descriptor_set,
        runtime_path=runtime_path,
    )
    for step in steps:
        acc = _compose_kraus_bundles(
            acc,
            step,
            descriptor_set=descriptor_set,
            runtime_path=runtime_path,
        )
    return acc


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


def test_phase31_channel_native_cnot_lowering_matches_embed_cnot_oracle():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_microcase_2q_cnot_local_noise_pair"
    )
    partition = _find_partition_with_span(ds)
    cnot_member = next(
        m
        for m in partition.members
        if ds.canonical_operation_for(m).name == "CNOT"
    )
    params = build_initial_parameters(ds.parameter_count)
    local_vec = _build_partition_parameter_vector(
        ds,
        partition,
        params,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    seg = _segment_parameter_vector(
        ds,
        partition.members,
        local_vec,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    local_support = tuple(range(len(partition.local_to_global_qbits)))
    got = _member_to_kraus_bundle(
        ds,
        cnot_member,
        seg,
        local_support=local_support,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    lmap = _local_qbit_to_kernel_index(local_support)
    k_ctl, k_tgt = _kernel_indices_for_fused_cnot(
        local_qbit_to_kernel_index=lmap,
        local_target_qbit=cnot_member.local_target_qbit,
        local_control_qbit=cnot_member.local_control_qbit,
    )
    expected = np.array(
        [
            _embed_cnot_gate(
                total_kernel_qbits=2,
                kernel_control_qbit=k_ctl,
                kernel_target_qbit=k_tgt,
            )
        ],
        dtype=np.complex128,
    )
    assert got.shape == expected.shape
    assert (
        float(np.linalg.norm(got[0] - expected[0], ord="fro"))
        <= PHASE3_RUNTIME_DENSITY_TOL
    )


def test_phase31_channel_native_cnot_both_orientations_lower_consistently():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_microcase_2q_multi_noise_entangler_chain"
    )
    partition = _find_partition_with_span(ds)
    params = build_initial_parameters(ds.parameter_count)
    local_vec = _build_partition_parameter_vector(
        ds,
        partition,
        params,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    seg = _segment_parameter_vector(
        ds,
        partition.members,
        local_vec,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    local_support = tuple(range(len(partition.local_to_global_qbits)))
    lmap = _local_qbit_to_kernel_index(local_support)
    cnot_members = [
        m
        for m in partition.members
        if ds.canonical_operation_for(m).name == "CNOT"
    ]
    assert len(cnot_members) == 2
    for m in cnot_members:
        got = _member_to_kraus_bundle(
            ds,
            m,
            seg,
            local_support=local_support,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
        k_ctl, k_tgt = _kernel_indices_for_fused_cnot(
            local_qbit_to_kernel_index=lmap,
            local_target_qbit=m.local_target_qbit,
            local_control_qbit=m.local_control_qbit,
        )
        expected = np.array(
            [
                _embed_cnot_gate(
                    total_kernel_qbits=2,
                    kernel_control_qbit=k_ctl,
                    kernel_target_qbit=k_tgt,
                )
            ],
            dtype=np.complex128,
        )
        assert (
            float(np.linalg.norm(got[0] - expected[0], ord="fro"))
            <= PHASE3_RUNTIME_DENSITY_TOL
        )


def test_phase31_channel_native_ordered_fusion_matches_sequential_on_2q_cnot_microcase():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_microcase_2q_cnot_local_noise_pair"
    )
    partition = _find_partition_with_span(ds)
    params = build_initial_parameters(ds.parameter_count)
    local_vec = _build_partition_parameter_vector(
        ds,
        partition,
        params,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    seg = _segment_parameter_vector(
        ds,
        partition.members,
        local_vec,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    local_support = tuple(range(len(partition.local_to_global_qbits)))
    global_targets = partition.local_to_global_qbits
    steps = [
        _member_to_kraus_bundle(
            ds,
            m,
            seg,
            local_support=local_support,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
        for m in partition.members
    ]
    acc = _identity_kraus_bundle_for_support_qubit_count(
        len(local_support),
        descriptor_set=ds,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    for step in steps:
        acc = _compose_kraus_bundles(
            acc,
            step,
            descriptor_set=ds,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
        _check_kraus_bundle_invariants(
            acc,
            descriptor_set=ds,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
    rho0 = DensityMatrix(2)
    rho_out = _apply_kraus_bundle(
        acc,
        rho0,
        qbit_num=2,
        local_support=local_support,
        global_target_qbits=global_targets,
        descriptor_set=ds,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    reference = execute_sequential_density_reference(ds, params)
    metrics = build_density_comparison_metrics(rho_out, reference)
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase31_channel_native_descriptor_order_matters_for_fusion():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_microcase_2q_cnot_local_noise_pair"
    )
    partition = _find_partition_with_span(ds)
    params = build_initial_parameters(ds.parameter_count)
    local_vec = _build_partition_parameter_vector(
        ds,
        partition,
        params,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    seg = _segment_parameter_vector(
        ds,
        partition.members,
        local_vec,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    local_support = tuple(range(len(partition.local_to_global_qbits)))
    global_targets = partition.local_to_global_qbits
    steps = [
        _member_to_kraus_bundle(
            ds,
            m,
            seg,
            local_support=local_support,
            runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
        )
        for m in partition.members
    ]
    acc_ordered = _compose_kraus_step_sequence(
        steps,
        descriptor_set=ds,
        support_qubit_count=len(local_support),
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    acc_reversed = _compose_kraus_step_sequence(
        list(reversed(steps)),
        descriptor_set=ds,
        support_qubit_count=len(local_support),
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    rho0 = DensityMatrix(2)
    out_ordered = _apply_kraus_bundle(
        acc_ordered,
        rho0,
        qbit_num=2,
        local_support=local_support,
        global_target_qbits=global_targets,
        descriptor_set=ds,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    out_reversed = _apply_kraus_bundle(
        acc_reversed,
        rho0,
        qbit_num=2,
        local_support=local_support,
        global_target_qbits=global_targets,
        descriptor_set=ds,
        runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE,
    )
    reference = execute_sequential_density_reference(ds, params)
    assert (
        build_density_comparison_metrics(out_ordered, reference)["frobenius_norm_diff"]
        <= PHASE3_RUNTIME_DENSITY_TOL
    )
    fro_rev = float(
        np.linalg.norm(
            np.asarray(out_ordered.to_numpy(), dtype=np.complex128)
            - np.asarray(out_reversed.to_numpy(), dtype=np.complex128),
            ord="fro",
        )
    )
    assert fro_rev > 10.0 * PHASE3_RUNTIME_DENSITY_TOL


# --- Public runtime (P31-S05-E01) + fused-region audit (P31-S05-E02) ---


def test_phase31_local_support_q4_smoke_partitions_two_disjoint_pairs():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_local_support_q4_spectator_embedding_smoke"
    )
    assert [p.local_to_global_qbits for p in ds.partitions] == [(0, 1), (2, 3)]


def _assert_channel_native_motif_fused_region(fr):
    assert fr.candidate_kind == PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF
    assert fr.partition_member_indices == (0, 1, 2, 3, 4, 5)
    assert fr.operation_names == (
        "U3",
        "U3",
        "CNOT",
        "amplitude_damping",
        "phase_damping",
        "U3",
    )


def test_phase31_channel_native_public_2q_cnot_microcase_matches_sequential():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_microcase_2q_cnot_local_noise_pair"
    )
    parameters = build_initial_parameters(ds.parameter_count)
    reference = execute_sequential_density_reference(ds, parameters)
    result = execute_partitioned_density_channel_native(ds, parameters)

    assert result.runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
    assert result.fused_region_count >= 1
    motif_regions = [
        fr
        for fr in result.fused_regions
        if fr.candidate_kind == PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF
    ]
    assert len(motif_regions) >= 1
    _assert_channel_native_motif_fused_region(motif_regions[0])
    assert motif_regions[0].global_target_qbits == (0, 1)
    assert motif_regions[0].local_target_qbits == (0, 1)

    metrics = build_density_comparison_metrics(result.density_matrix, reference)
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


def test_phase31_channel_native_public_4q_smoke_matches_sequential():
    ds = build_phase31_microcase_descriptor_set(
        "phase31_local_support_q4_spectator_embedding_smoke"
    )
    parameters = build_initial_parameters(ds.parameter_count)
    reference = execute_sequential_density_reference(ds, parameters)
    result = execute_partitioned_density_channel_native(ds, parameters)

    assert result.runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE
    assert result.fused_region_count == 2
    motif_regions = [
        fr
        for fr in result.fused_regions
        if fr.candidate_kind == PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF
    ]
    assert len(motif_regions) == 2
    by_global = {fr.global_target_qbits: fr for fr in motif_regions}
    assert set(by_global) == {(0, 1), (2, 3)}
    _assert_channel_native_motif_fused_region(by_global[(0, 1)])
    _assert_channel_native_motif_fused_region(by_global[(2, 3)])
    assert by_global[(0, 1)].local_target_qbits == (0, 1)
    assert by_global[(2, 3)].local_target_qbits == (0, 1)

    metrics = build_density_comparison_metrics(result.density_matrix, reference)
    assert metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
    assert metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL


# --- Public boundary (P31-S05-E02) ---


def test_phase31_channel_native_public_rejects_motif_span_over_two_local_qubits():
    surface = build_microcase_surface("microcase_3q_mixed_local_noise_sequence")
    ds = build_partition_descriptor_set(surface, max_partition_qubits=3)
    parameters = build_initial_parameters(ds.parameter_count)
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        execute_partitioned_density_channel_native(ds, parameters)
    assert excinfo.value.first_unsupported_condition == "channel_native_qubit_span"


def test_phase31_channel_native_public_rejects_partition_without_noise():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=2,
        source_type="microcase_builder",
        workload_id="second_slice_2q_unitary_only_channel_native_negative",
        operation_specs=[
            _u3(0),
            _u3(1),
            _cnot(1, 0),
            _u3(0),
        ],
    )
    ds = build_partition_descriptor_set(surface)
    parameters = build_initial_parameters(ds.parameter_count)
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        execute_partitioned_density_channel_native(ds, parameters)
    assert excinfo.value.first_unsupported_condition == "channel_native_noise_presence"


def test_phase31_channel_native_public_rejects_non_phase3_gate_on_relaxed_surface():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=2,
        source_type="microcase_builder",
        workload_id="second_slice_rx_on_relaxed_surface_channel_native_negative",
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
        execute_partitioned_density_channel_native(ds, parameters)
    assert excinfo.value.first_unsupported_condition == "channel_native_support_surface"


def test_phase31_channel_native_public_rejects_unsupported_noise_on_relaxed_surface():
    surface = build_canonical_planner_surface_from_operation_specs(
        qbit_num=1,
        source_type="microcase_builder",
        workload_id="second_slice_bit_flip_on_relaxed_surface_channel_native_negative",
        operation_specs=[
            _u3(0),
            {
                "kind": "noise",
                "name": "bit_flip",
                "target_qbit": 0,
                "source_gate_index": 0,
                "fixed_value": 0.1,
                "param_count": 0,
            },
        ],
        strict_phase3_support=False,
    )
    ds = build_partition_descriptor_set(surface)
    parameters = build_initial_parameters(ds.parameter_count)
    with pytest.raises(NoisyRuntimeValidationError) as excinfo:
        execute_partitioned_density_channel_native(ds, parameters)
    assert excinfo.value.first_unsupported_condition == "channel_native_support_surface"
