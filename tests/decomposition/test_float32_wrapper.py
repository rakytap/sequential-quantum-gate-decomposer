import numpy as np

from squander import (
    N_Qubit_Decomposition,
    N_Qubit_Decomposition_adaptive,
    N_Qubit_Decomposition_custom,
    N_Qubit_Decomposition_Tabu_Search,
    N_Qubit_Decomposition_Tree_Search,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit


def _identity(n=2, dtype=np.complex64):
    return np.eye(1 << n, dtype=dtype)


def test_decomposition_wrappers_accept_complex64_with_use_float():
    config = {"use_float": True}
    classes = (
        N_Qubit_Decomposition,
        N_Qubit_Decomposition_adaptive,
        N_Qubit_Decomposition_custom,
        N_Qubit_Decomposition_Tree_Search,
        N_Qubit_Decomposition_Tabu_Search,
    )

    for cls in classes:
        decomp = cls(_identity(dtype=np.complex64), config=config)
        assert decomp.get_Qbit_Num() == 2


def test_use_float_routes_complex128_input_to_float_path():
    decomp = N_Qubit_Decomposition(_identity(dtype=np.complex128), config={"use_float": True})
    params = np.array([], dtype=np.float64)

    unitary = np.asarray(decomp.get_Unitary())
    matrix = np.asarray(decomp.get_Matrix(params))

    assert unitary.dtype == np.complex64
    assert matrix.dtype == np.complex64


def test_use_float_returns_float32_unitary_and_matrix():
    decomp = N_Qubit_Decomposition(_identity(dtype=np.complex64), config={"use_float": True})
    params = np.array([], dtype=np.float32)

    unitary = np.asarray(decomp.get_Unitary())
    matrix = np.asarray(decomp.get_Matrix(params))

    assert unitary.dtype == np.complex64
    assert matrix.dtype == np.complex64
    np.testing.assert_allclose(matrix, np.eye(4, dtype=np.complex64), atol=1e-6)


def test_set_unitary_accepts_complex64_and_keeps_float_getter():
    decomp = N_Qubit_Decomposition_adaptive(_identity(dtype=np.complex64), config={"use_float": True})
    replacement = _identity(dtype=np.complex64)

    assert decomp.set_Unitary(replacement) == 0
    unitary = np.asarray(decomp.get_Unitary())

    assert unitary.dtype == np.complex64
    np.testing.assert_allclose(unitary, replacement, atol=1e-6)


def test_float32_parameters_work_for_cost_grad_and_batch():
    decomp = N_Qubit_Decomposition(_identity(dtype=np.complex64), config={"use_float": True})
    params = np.array([], dtype=np.float32)
    batch = np.zeros((2, 0), dtype=np.float32)

    assert np.isclose(decomp.Optimization_Problem(params), 0.0, atol=1e-6)

    grad = np.asarray(decomp.Optimization_Problem_Grad(params))
    assert grad.dtype == np.float32
    assert grad.size == 0

    cost, combined_grad = decomp.Optimization_Problem_Combined(params)
    assert np.isclose(cost, 0.0, atol=1e-6)
    assert np.asarray(combined_grad).dtype == np.float32

    batched = np.asarray(decomp.Optimization_Problem_Batch(batch))
    assert batched.dtype == np.float32
    np.testing.assert_allclose(batched, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_float32_combined_handles_nested_gate_blocks_repeatedly():
    qbit_num = 4
    inner = Circuit(qbit_num)
    parameters = []
    for qbit in range(qbit_num):
        inner.add_U3(qbit)
        parameters.extend([0.1, 0.2, 0.3])
    inner.add_CNOT(1, 0)
    for qbit in range(qbit_num):
        inner.add_U3(qbit)
        parameters.extend([0.1, 0.2, 0.3])

    outer = Circuit(qbit_num)
    outer.add_Circuit(inner)
    parameters = np.asarray(parameters, dtype=np.float32)

    decomp = N_Qubit_Decomposition_custom(
        _identity(qbit_num, dtype=np.complex64),
        config={"use_float": True, "parallel": 0},
    )
    decomp.set_Gate_Structure(outer)

    for _ in range(20):
        cost, grad = decomp.Optimization_Problem_Combined(parameters)
        assert np.isfinite(cost)
        assert np.asarray(grad).dtype == np.float32


def test_nested_gate_block_combined_gradient_matches_finite_difference():
    qbit_num = 3
    inner = Circuit(qbit_num)
    inner.add_U3(0)
    inner.add_U3(1)
    inner.add_CNOT(1, 0)
    inner.add_U3(2)
    inner.add_CNOT(2, 1)
    inner.add_U3(0)

    outer = Circuit(qbit_num)
    outer.add_U3(2)
    outer.add_Circuit(inner)
    outer.add_U3(1)

    parameters = np.linspace(0.1, 1.7, 18, dtype=np.float64)
    decomp = N_Qubit_Decomposition_custom(
        _identity(qbit_num, dtype=np.complex128),
        config={"parallel": 0},
    )
    decomp.set_Gate_Structure(outer)

    _, grad = decomp.Optimization_Problem_Combined(parameters)
    grad = np.asarray(grad)

    eps = 1e-7
    finite_diff = np.empty_like(parameters)
    for idx in range(parameters.size):
        params_plus = parameters.copy()
        params_minus = parameters.copy()
        params_plus[idx] += eps
        params_minus[idx] -= eps
        finite_diff[idx] = (
            decomp.Optimization_Problem(params_plus)
            - decomp.Optimization_Problem(params_minus)
        ) / (2 * eps)

    np.testing.assert_allclose(grad, finite_diff, rtol=1e-5, atol=1e-6)
