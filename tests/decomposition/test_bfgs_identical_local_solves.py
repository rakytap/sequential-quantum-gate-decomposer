"""Exact restart-result reuse must preserve native BFGS outputs and quality."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander import N_Qubit_Decomposition_custom
from squander.gates.qgd_Circuit import qgd_Circuit


def fit(target, structure, parameters, reuse, cost=3):
    optimizer = N_Qubit_Decomposition_custom(
        target.conj().T.copy(),
        config={
            "parallel": 0,
            "use_float": False,
            "random_seed": 20261008,
            "max_iteration_loops_bfgs": 8,
            "max_inner_iterations_bfgs": 1000,
            "max_outer_iterations": 1,
            "bfgs_reuse_identical_local_solves": reuse,
        },
        accelerator_num=0,
    )
    optimizer.set_Gate_Structure(structure)
    optimizer.set_Cost_Function_Variant(cost)
    optimizer.set_Optimizer("BFGS")
    optimizer.set_Optimization_Tolerance(1e-14)
    optimizer.set_Optimization_Blocks(10000)
    optimizer.set_Optimized_Parameters(parameters.copy())
    optimizer.set_Verbose(2)
    optimizer.Start_Decomposition()
    fitted = np.asarray(optimizer.get_Optimized_Parameters_Double()).copy()
    matrix = np.asarray(optimizer.get_Circuit().get_Matrix(fitted)).copy()
    return fitted, matrix, optimizer.get_Decomposition_Error()


@pytest.mark.parametrize("width", [2, 3, 4])
def test_hs_identical_restart_reuse_preserves_unreachable_stationary_point(width, capfd):
    source = QuantumCircuit(width)
    source.x(0)
    target = np.asarray(Operator(source).data)
    structure = qgd_Circuit(width)
    structure.add_RZ(0)
    parameters = np.zeros(1)

    original = fit(target, structure, parameters, False)
    original_log = capfd.readouterr().out
    reused = fit(target, structure, parameters, True)
    reused_log = capfd.readouterr().out

    np.testing.assert_array_equal(reused[0], original[0])
    np.testing.assert_array_equal(reused[1], original[1])
    assert reused[2] == original[2]
    assert "BFGS: reused identical local solve" not in original_log
    assert "BFGS: reused identical local solve" in reused_log
    # X times any RZ has exactly zero trace, hence an identically constant HS
    # objective and zero gradient. Every repeat is truly bitwise identical.
    # Caching must not fabricate acceptance or alter the original solve.
    assert reused[2] == 1.0
    assert reused[2] > 1e-10


def test_stateful_correction_objective_does_not_reuse_restart_results(capfd):
    source = QuantumCircuit(2)
    source.cp(0.37, 0, 1)
    structure = qgd_Circuit(2)
    structure.add_U3(0)
    structure.add_U3(1)
    fit(np.asarray(Operator(source).data), structure, np.zeros(6), True, cost=4)
    assert "BFGS: reused identical local solve" not in capfd.readouterr().out
