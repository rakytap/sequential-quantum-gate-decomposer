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
