"""Certify chronological inversion without relying on optimizer convergence."""

import ctypes

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander import N_Qubit_Decomposition_Tree_Search  # Load libqgd.


def inverse_word(width, word, targets, controls, parameters):
    function = ctypes.CDLL('libqgd.so').qgd_osr_reverse_logical_factor_word
    ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ctypes.c_int, ctypes.c_int, ip, ctypes.c_int,
                         ip, ip, dp, ctypes.c_int]
    function.restype = ctypes.c_int
    return function(width, len(word), word.ctypes.data_as(ip), len(targets),
                    targets.ctypes.data_as(ip), controls.ctypes.data_as(ip),
                    parameters.ctypes.data_as(dp), len(parameters))


def matrix(width, word, targets, controls, parameters):
    circuit = QuantumCircuit(width)
    for index, digit in enumerate(word):
        target, control = int(targets[digit]), int(controls[digit])
        for offset, wire in ((0, target), (3, control)):
            theta, phi, lam = parameters[6 * index + offset:6 * index + offset + 3]
            circuit.u(2 * theta, phi, lam, wire)
        circuit.cx(control, target)
    for wire in range(width):
        theta, phi, lam = parameters[6 * len(word) + 3 * wire:6 * len(word) + 3 * wire + 3]
        circuit.u(2 * theta, phi, lam, wire)
    return Operator(circuit).data


@pytest.mark.parametrize('width', [2, 3, 4])
@pytest.mark.parametrize('count', [0, 1, 4, 13])
def test_native_factor_inversion_preserves_the_matrix_and_physical_word(width, count):
    rng = np.random.default_rng(28000 + width * 100 + count)
    controls = np.asarray(list(range(width - 1)), dtype=np.int32)
    targets = controls + 1
    word = rng.integers(0, len(targets), size=count, dtype=np.int32)
    parameters = rng.uniform(-2 * np.pi, 2 * np.pi, size=6 * count + 3 * width)
    original_word = word.copy()
    expected = matrix(width, word, targets, controls, parameters).conj().T
    assert inverse_word(width, word, targets, controls, parameters) == 0
    np.testing.assert_array_equal(word, original_word[::-1])
    actual = matrix(width, word, targets, controls, parameters)
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, phase / abs(phase) * expected, atol=3e-13, rtol=0)


@pytest.mark.parametrize('theta', [0.0, np.pi / 2, np.pi])
def test_native_factor_inversion_handles_u3_coordinate_poles(theta):
    targets, controls = np.asarray([1], dtype=np.int32), np.asarray([0], dtype=np.int32)
    word = np.asarray([0, 0], dtype=np.int32)
    parameters = np.tile([theta, 0.21, -0.53], 6).astype(float)
    expected = matrix(2, word, targets, controls, parameters).conj().T
    assert inverse_word(2, word, targets, controls, parameters) == 0
    actual = matrix(2, word, targets, controls, parameters)
    phase = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, phase / abs(phase) * expected, atol=3e-13, rtol=0)


def test_native_factor_inversion_rejects_invalid_inputs_without_mutating_buffers():
    targets, controls = np.asarray([1], dtype=np.int32), np.asarray([0], dtype=np.int32)
    word = np.asarray([1], dtype=np.int32)
    parameters = np.linspace(0, 1, 12)
    before = parameters.copy()
    assert inverse_word(2, word, targets, controls, parameters) == -1
    np.testing.assert_array_equal(parameters, before)
    word[0] = 0
    parameters[0] = np.nan
    before = parameters.copy()
    assert inverse_word(2, word, targets, controls, parameters) == -1
    np.testing.assert_array_equal(parameters, before)
