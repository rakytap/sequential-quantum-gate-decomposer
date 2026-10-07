"""M-F5a task-1: density interop harness (module-level, no new energy API)."""

from __future__ import annotations

import numpy as np
import pytest

from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base as VariationalQuantumEigensolver,
)
import squander.VQA.qgd_Variational_Quantum_Eigensolver_Base_Wrapper as vqe_wrapper_ext

import tests.VQE.test_VQE as vqe_test_module

# Host-pinned at cdcfe6b1 on rocky-local (ET-2); not retuned without Reviewer (N-17).
ET2_GOLDEN_ENERGY_CDCFE6B1 = -0.7583303034656004


@pytest.fixture
def density_vqe_4q():
    vqe, _ = vqe_test_module.Test_VQE()._build_density_backend_vqe(4)
    return vqe


def test_harness_symbols_live_on_extension_module_only(density_vqe_4q):
    for name in (
        "harness_density_lower_ns",
        "harness_density_set_timer_flag",
        "harness_density_subtimes_ns",
    ):
        assert hasattr(vqe_wrapper_ext, name)
        assert not hasattr(density_vqe_4q, name)


def test_harness_lower_ns_returns_positive_int64(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)
    elapsed_ns = vqe_wrapper_ext.harness_density_lower_ns(density_vqe_4q, parameters)
    assert isinstance(elapsed_ns, int)
    assert elapsed_ns > 0


def test_harness_subtimes_returns_six_int64s(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)
    vqe_wrapper_ext.harness_density_set_timer_flag(density_vqe_4q, True)
    vqe_wrapper_ext.harness_density_lower_ns(density_vqe_4q, parameters)
    subtimes = vqe_wrapper_ext.harness_density_subtimes_ns(density_vqe_4q)
    assert len(subtimes) == 6
    assert all(isinstance(value, int) for value in subtimes)
    assert sum(subtimes) > 0


def test_density_energy_bit_identical_with_timer_flag(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)

    vqe_wrapper_ext.harness_density_set_timer_flag(density_vqe_4q, False)
    energy_flag_off = float(density_vqe_4q.Optimization_Problem(parameters))

    vqe_wrapper_ext.harness_density_set_timer_flag(density_vqe_4q, True)
    energy_flag_on = float(density_vqe_4q.Optimization_Problem(parameters))

    assert energy_flag_off == energy_flag_on
    assert energy_flag_off == ET2_GOLDEN_ENERGY_CDCFE6B1


def test_harness_neither_path_returns_energy(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)
    lower_ns = vqe_wrapper_ext.harness_density_lower_ns(density_vqe_4q, parameters)
    subtimes = vqe_wrapper_ext.harness_density_subtimes_ns(density_vqe_4q)
    assert not isinstance(lower_ns, float)
    assert not any(isinstance(value, float) for value in subtimes)


def test_task2_evqe_6q_cell_pins_timer_identity_and_aer_oracle():
    pytest.importorskip("qiskit_aer")
    from benchmarks.density_matrix.interop_profile.interop_lane import build_task_evaluator

    vqe, hamiltonian = build_task_evaluator(6)
    parameters = np.linspace(0.05, 0.05 * 30, 30, dtype=np.float64)

    bridge = vqe.describe_density_bridge()
    assert vqe.get_Parameter_Num() == 30
    assert bridge["operation_count"] == 18
    assert bridge["gate_count"] == 15
    assert bridge["noise_count"] == 3
    assert int(hamiltonian.nnz) == 224

    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, False)
    energy_flag_off = float(vqe.Optimization_Problem(parameters))
    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, True)
    energy_flag_on = float(vqe.Optimization_Problem(parameters))
    assert energy_flag_off == energy_flag_on

    for name in (
        "harness_density_lower_ns",
        "harness_density_set_timer_flag",
        "harness_density_subtimes_ns",
    ):
        assert not hasattr(vqe, name)

    vqe.set_Optimized_Parameters(parameters)
    tester = vqe_test_module.Test_VQE()
    aer_real, aer_imag = tester._get_density_backend_aer_reference(vqe, hamiltonian)
    bound = 1e-12 + 1e-5 * abs(aer_real)
    assert abs(energy_flag_off - aer_real) <= bound
    assert abs(aer_imag) <= 1e-12


def test_task3_evqe_8q_cell_pins_timer_identity_and_aer_oracle():
    pytest.importorskip("qiskit_aer")
    from benchmarks.density_matrix.interop_profile.interop_lane import build_task_evaluator

    vqe, hamiltonian = build_task_evaluator(8)
    parameters = np.linspace(0.05, 0.05 * 42, 42, dtype=np.float64)

    bridge = vqe.describe_density_bridge()
    assert vqe.get_Parameter_Num() == 42
    assert bridge["operation_count"] == 24
    assert bridge["gate_count"] == 21
    assert bridge["noise_count"] == 3
    assert int(hamiltonian.nnz) == 1152

    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, False)
    energy_flag_off = float(vqe.Optimization_Problem(parameters))
    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, True)
    energy_flag_on = float(vqe.Optimization_Problem(parameters))
    assert energy_flag_off == energy_flag_on

    for name in (
        "harness_density_lower_ns",
        "harness_density_set_timer_flag",
        "harness_density_subtimes_ns",
    ):
        assert not hasattr(vqe, name)

    vqe.set_Optimized_Parameters(parameters)
    tester = vqe_test_module.Test_VQE()
    aer_real, aer_imag = tester._get_density_backend_aer_reference(vqe, hamiltonian)
    bound = 1e-12 + 1e-5 * abs(aer_real)
    assert abs(energy_flag_off - aer_real) <= bound
    assert abs(aer_imag) <= 1e-12


def test_task4_width4_attribution_descriptor_pins():
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_width4_attribution_anchor,
    )

    vqe, descriptor_set, bridge = build_width4_attribution_anchor()
    assert vqe.get_Parameter_Num() == 18
    assert bridge["operation_count"] == 12
    assert bridge["gate_count"] == 9
    assert bridge["noise_count"] == 3
    assert descriptor_set.workload_id == "phase2_xxz_hea_q4_continuity"


@pytest.mark.parametrize(
    "route_id",
    ("R-base", "R-fused", "R-hybrid"),
)
def test_task4_attribution_route_sample_has_orchestration_and_apply(route_id: str):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_route_row,
        build_width4_attribution_anchor,
    )
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters

    expected_primitive_calls = {"R-base": 8, "R-fused": 8, "R-hybrid": 5}

    vqe, descriptor_set, _bridge = build_width4_attribution_anchor()
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    row = build_route_row(route_id, descriptor_set, parameters, sample_count=2)
    assert row["orchestration"]["mean_ns"] >= 0
    assert row["apply_component"]["mean_ns"] > 0
    assert row["throughput"]["divisor"] == 3072
    assert row["throughput"]["mean_ns_per_op"] == pytest.approx(
        row["apply_component"]["mean_ns"] / 3072, rel=1e-12
    )
    assert row["throughput"]["upper_bound_95_ns_per_op"] == pytest.approx(
        row["apply_component"]["upper_bound_95_ns"] / 3072, rel=1e-12
    )
    assert "mean_O" not in row
    assert "overhead" not in row
    for sample in row["samples"]:
        wall_ns = int(sample["orchestration_ns"]) + int(sample["apply_component_ns"])
        assert int(sample["apply_component_ns"]) < wall_ns
        assert int(sample["apply_primitive_calls"]) > 0
    assert {sample["apply_primitive_calls"] for sample in row["samples"]} == {
        expected_primitive_calls[route_id]
    }
    executor = lane.ROUTE_TABLE[route_id]["executor"]
    for _ in range(2):
        sample, result = lane._time_route_sample(
            route_id, executor, descriptor_set, parameters
        )
        assert sample["apply_component_ns"] < int(result.runtime_ms * 1_000_000)
    if route_id == "R-hybrid":
        assert row["partition_runtime_classes"] == [
            "phase31_channel_native",
            "phase3_unitary_island_fused",
            "phase31_channel_native",
            "phase3_unitary_island_fused",
            "phase3_unitary_island_fused",
        ]
        assert "the executed class" not in row["apply_label"]


def test_task4_r_strict_handback_on_continuity_anchor():
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_route_row,
        build_width4_attribution_anchor,
    )
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters

    vqe, descriptor_set, _bridge = build_width4_attribution_anchor()
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    with pytest.raises(ValueError, match="attribution route handback for R-strict"):
        build_route_row("R-strict", descriptor_set, parameters, sample_count=1)
