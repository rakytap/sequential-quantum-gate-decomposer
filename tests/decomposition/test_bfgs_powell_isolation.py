"""BFGS optimizer state must remain isolated across widths and threads."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from squander import Circuit, N_Qubit_Decomposition_custom, Qiskit_IO


def _fit(width):
    circuit = Circuit(width)
    for control in range(width - 1):
        circuit.add_U3(control + 1)
        circuit.add_U3(control)
        circuit.add_CNOT(control + 1, control)
    for wire in range(width):
        circuit.add_U3(wire)
    count = 6 * (width - 1) + 3 * width
    angles = np.linspace(0.17, 0.71, count, dtype=np.float64)
    target = Operator(Qiskit_IO.get_Qiskit_Circuit(circuit, angles)).data
    initial = angles + 0.025 * np.cos(np.arange(count, dtype=np.float64))
    config = dict(parallel=0, use_float=False, float32=False, random_seed=20260926,
                  max_outer_iterations=1, max_inner_iterations_bfgs2=500,
                  use_basin_hopping=True, use_differential_evolution=False,
                  use_dual_annealing=False, tolerance=1e-14,
                  optimization_tolerance=1e-10, synthesis_acceptance_tolerance=1e-10)
    optimizer = N_Qubit_Decomposition_custom(target.conj().T.copy(), config=config)
    optimizer.set_Verbose(0)
    optimizer.set_Optimizer('BFGS2')
    optimizer.set_Cost_Function_Variant(3)
    optimizer.set_Gate_Structure(circuit)
    optimizer.set_Optimization_Blocks(1000)
    optimizer.set_Iteration_Loops({width: 0})
    optimizer.set_Optimized_Parameters(initial)
    assert optimizer.Optimization_Problem(initial) > 1e-6
    optimizer.Start_Decomposition()
    fitted = optimizer.get_Optimized_Parameters_Double()
    result = Operator(Qiskit_IO.get_Qiskit_Circuit(optimizer.get_Circuit(), fitted)).data
    overlap = np.vdot(target, result)
    phase = overlap / abs(overlap)
    chord = np.linalg.norm(result - phase * target, 'fro') ** 2 / len(target)
    assert chord * (1 - chord / 4) < 1e-10
    assert optimizer.get_Decomposition_Error() < 1e-10
    return fitted, result


@pytest.mark.parametrize('widths', [(1, 4, 2, 3, 1), (4, 1, 3, 2, 4)])
def test_bfgs_powell_cross_width_calls_preserve_results(widths):
    first_parameters, first_matrix = _fit(widths[0])
    for width in widths[1:-1]:
        _fit(width)
    last_parameters, last_matrix = _fit(widths[-1])
    np.testing.assert_array_equal(last_parameters, first_parameters)
    np.testing.assert_array_equal(last_matrix, first_matrix)


def test_bfgs_powell_concurrent_calls_preserve_results():
    widths = (1, 4, 2, 3)
    references = {width: _fit(width) for width in widths}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_fit, widths * 2))
    for width, (parameters, matrix) in zip(widths * 2, results):
        np.testing.assert_array_equal(parameters, references[width][0])
        np.testing.assert_array_equal(matrix, references[width][1])
