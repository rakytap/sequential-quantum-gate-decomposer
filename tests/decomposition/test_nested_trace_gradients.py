"""Nested trace gradients agree with matrix Jacobians and independent differences."""
import numpy as np
import pytest

from squander import N_Qubit_Decomposition_custom
from squander.gates.qgd_Circuit import qgd_Circuit


def nested_circuit(width, include_empty=True):
    circuit = qgd_Circuit(width)
    circuit.add_U3(0)
    if include_empty:
        circuit.add_Circuit(qgd_Circuit(width))
    for layer in range(3):
        a, b = layer % width, (layer + 1) % width
        local = qgd_Circuit(width)
        local.add_U3(a)
        local.add_U3(b)
        block = qgd_Circuit(width)
        block.add_Circuit(local)
        block.add_CNOT(b, a)
        block.add_RZ(b)
        block.add_CRY(a, b)
        circuit.add_Circuit(block)
    circuit.add_U3(width - 1)
    return circuit


def objective(circuit, parameters, target):
    product = np.asarray(circuit.get_Matrix(parameters)) @ target.conj().T
    trace = np.trace(product)
    return 1.0 - abs(trace / product.shape[0]) ** 2


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("use_float", [False, True])
def test_nested_trace_gradient_matches_matrix_jacobian_and_finite_differences(width, use_float):
    circuit = nested_circuit(width)
    rng = np.random.default_rng(20261008 + width)
    target_parameters = rng.uniform(-1, 1, circuit.get_Parameter_Num())
    target = np.asarray(circuit.get_Matrix(target_parameters)).copy()
    optimizer = N_Qubit_Decomposition_custom(
        target.conj().T.copy(),
        config={"parallel": 0, "use_float": use_float, "float32": use_float},
        accelerator_num=0,
    )
    optimizer.set_Gate_Structure(circuit)
    optimizer.set_Cost_Function_Variant(3)
    # The matrix-Jacobian API rejects a zero-parameter BLOCK_OPERATION.
    # Use the same physical circuit with that identity removed as a separate
    # reference, while retaining empty blocks in the evaluator under test.
    reference_circuit = nested_circuit(width, include_empty=False)
    reference = N_Qubit_Decomposition_custom(
        target.conj().T.copy(),
        config={"parallel": 0, "use_float": use_float, "float32": use_float},
        accelerator_num=0,
    )
    reference.set_Gate_Structure(reference_circuit)
    reference.set_Cost_Function_Variant(3)
    for _ in range(3):
        parameters = rng.uniform(-1, 1, circuit.get_Parameter_Num())
        saved_parameters = parameters.copy()
        saved_matrix = np.asarray(circuit.get_Matrix(parameters)).copy()
        cost, gradient = optimizer.Optimization_Problem_Combined(parameters)
        np.testing.assert_array_equal(reference_circuit.get_Matrix(parameters), saved_matrix)
        product, jacobian = reference.Optimization_Problem_Combined_Unitary(parameters)
        trace = np.trace(product)
        trace_derivatives = np.trace(jacobian, axis1=1, axis2=2)
        expected = -2 * np.real(np.conj(trace) * trace_derivatives) / product.shape[0] ** 2
        accuracy = 3e-6 if use_float else 2e-12
        np.testing.assert_allclose(gradient, expected, rtol=accuracy, atol=accuracy)
        assert abs(cost - objective(circuit, parameters, target)) < accuracy

        differences = np.empty(len(parameters))
        step = 5e-7
        for index in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step
            minus[index] -= step
            differences[index] = (objective(circuit, plus, target) -
                                  objective(circuit, minus, target)) / (2 * step)
        np.testing.assert_allclose(gradient, differences,
                                   rtol=3e-6 if use_float else 2e-8,
                                   atol=3e-6 if use_float else 2e-8)
        np.testing.assert_array_equal(parameters, saved_parameters)
        np.testing.assert_array_equal(circuit.get_Matrix(parameters), saved_matrix)


def test_empty_nested_trace_circuit_is_identity_with_no_parameters():
    circuit = qgd_Circuit(3)
    circuit.add_Circuit(qgd_Circuit(3))
    optimizer = N_Qubit_Decomposition_custom(
        np.eye(8, dtype=np.complex128), config={"parallel": 0, "use_float": False},
        accelerator_num=0,
    )
    optimizer.set_Gate_Structure(circuit)
    optimizer.set_Cost_Function_Variant(3)
    cost, gradient = optimizer.Optimization_Problem_Combined(np.empty(0))
    assert cost == 0
    assert gradient.shape == (0,)
