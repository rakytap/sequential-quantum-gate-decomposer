"""Independent controls for native matrix-derived causal OSR factors."""

import ctypes
import itertools
import re

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander import N_Qubit_Decomposition_Tree_Search  # Loads the native core.


TOLERANCE = 1e-10


def _target(width, root, pair):
    active = [q for q in range(width) if q != root]
    neighbors = active[1:] + [root]
    circuit = QuantumCircuit(width)
    for i, (leaf, neighbor) in enumerate(zip(active, neighbors)):
        circuit.u(0.31 + 0.13 * i, -0.19, 0.47, leaf)
        if pair:
            circuit.cry(0.73 + 0.11 * i, leaf, neighbor)
        else:
            circuit.cx(leaf, neighbor)
    for i, wire in enumerate(active):
        circuit.u(0.27 + 0.07 * i, 0.39, -0.61, wire)
    for a, b in zip(active, active[1:]):
        circuit.cx(a, b)
    circuit.u(0.43, -0.23, 0.57, root)
    return np.asarray(Operator(circuit).data), active, neighbors


def _decode(target, root, order, neighbors, graph, budget=14, capacity=32):
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int)
    function = ctypes.CDLL("libqgd.so").qgd_osr_logical_factor
    function.argtypes = [double_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        int_pointer, int_pointer, ctypes.c_uint, ctypes.c_double,
                        ctypes.c_int, double_pointer, double_pointer, int_pointer,
                        int_pointer, double_pointer, ctypes.c_int, int_pointer,
                        int_pointer, double_pointer]
    function.restype = ctypes.c_int
    width = int(np.log2(target.shape[0]))
    dimension = 1 << width
    half = dimension // 2
    padded = np.zeros((dimension, dimension + 3), dtype=np.complex128)
    padded[:, :dimension] = target
    original = padded.copy()
    leaf = np.asarray(order, dtype=np.int32)
    neighbor = np.asarray(neighbors, dtype=np.int32)
    mask = sum(1 << (a * width + b) for a, b in {tuple(sorted(e)) for e in graph})
    child = np.full((half, half), np.nan, dtype=np.complex128)
    local = np.full((2, 2), np.nan, dtype=np.complex128)
    targets = np.zeros(32, dtype=np.int32)
    controls = np.zeros(32, dtype=np.int32)
    matrices = np.zeros((32, 2, 2), dtype=np.complex128)
    count, cnots, error = ctypes.c_int(), ctypes.c_int(), ctypes.c_double()
    status = function(padded.ctypes.data_as(double_pointer), width, padded.shape[1], root,
                      leaf.ctypes.data_as(int_pointer), neighbor.ctypes.data_as(int_pointer),
                      mask, TOLERANCE, budget, child.ctypes.data_as(double_pointer),
                      local.ctypes.data_as(double_pointer), targets.ctypes.data_as(int_pointer),
                      controls.ctypes.data_as(int_pointer), matrices.ctypes.data_as(double_pointer),
                      capacity, ctypes.byref(count), ctypes.byref(cnots), ctypes.byref(error))
    np.testing.assert_equal(padded, original)
    return status, child, local, targets, controls, matrices, count.value, cnots.value, error.value


@pytest.mark.parametrize("width,root", [(n, q) for n in (2, 3, 4) for q in range(n)])
@pytest.mark.parametrize("pair", [False, True])
def test_native_logical_factor_reconstructs_all_wire_orders(width, root, pair):
    target, order, neighbors = _target(width, root, pair)
    graph = list(itertools.combinations(range(width), 2))
    result = _decode(target, root, order, neighbors, graph)
    status, child, local, targets, controls, matrices, count, cnots, error = result
    assert status == 1
    assert np.isfinite(error) and error <= TOLERANCE
    prefix = QuantumCircuit(width)
    for i in range(count):
        if controls[i] < 0:
            prefix.unitary(matrices[i], [int(targets[i])])
        else:
            prefix.cx(int(controls[i]), int(targets[i]))
    assert prefix.count_ops().get("cx", 0) == cnots
    assert cnots <= (2 if pair else 1) * (width - 1)
    reconstructed = prefix.copy()
    reconstructed.unitary(child, order)
    reconstructed.unitary(local, [root])
    actual = np.asarray(Operator(reconstructed).data)
    overlap = np.vdot(actual, target)
    phase = overlap / abs(overlap)
    chord = np.linalg.norm(target - phase * actual) ** 2 / len(target)
    assert chord * (1 - chord / 4) <= TOLERANCE
    np.testing.assert_allclose(child.conj().T @ child, np.eye(len(child)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(local.conj().T @ local, np.eye(2), atol=1e-12, rtol=0)


def test_native_logical_factor_rejects_missing_couplers_and_prefix_budget():
    target, order, neighbors = _target(4, 3, True)
    graph = list(itertools.combinations(range(4), 2))
    good = _decode(target, 3, order, neighbors, graph)
    assert good[0] == 1 and good[7] > 0
    assert _decode(target, 3, order, neighbors, [], budget=14)[0] == 0
    assert _decode(target, 3, order, neighbors, graph, budget=good[7] - 1)[0] == 0
    assert _decode(target, 3, order, neighbors, graph, capacity=0)[0] == -1


@pytest.mark.parametrize("invalid", ["nonunitary", "nonfinite", "duplicate_leaf"])
def test_native_logical_factor_rejects_invalid_targets_and_causal_orders(invalid):
    target, order, neighbors = _target(3, 2, True)
    if invalid == "nonunitary":
        target *= 1.01
    elif invalid == "nonfinite":
        target[0, 0] = np.nan
    else:
        order = [order[0], order[0]]
    graph = list(itertools.combinations(range(3), 2))
    assert _decode(target, 2, order, neighbors, graph)[0] == 0


def test_native_logical_factor_deduplicates_identical_spectator_models(capfd):
    from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization

    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    for wire in range(4):
        circuit.u(0.31 + 0.07 * wire, -0.19, 0.47, wire)
    target = np.asarray(Operator(circuit).data)
    config = {"strategy": "TreeSearch", "use_osr": True, "use_graph_search": True,
              "synthesis_acceptance_tolerance": TOLERANCE, "tolerance": 1e-14,
              "random_seed": 20260926, "parallel": 0, "use_float": False,
              "optimizer": "BFGS2", "tree_level_max": 1,
              "osr_geometric_completion_motifs": True,
              "osr_certified_terminal_completion": True, "osr_native_path_trace": True,
              "max_inner_iterations_bfgs2": 1000, "max_iteration_loops": 8}
    optimizer = qgd_Wide_Circuit_Optimization(config=config)
    solutions = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=list(itertools.combinations(range(4), 2)))
    assert solutions
    trace = capfd.readouterr().out
    models = re.findall(r"OSR logical factor models: width=4,.*?factors=(\d+), unique=(\d+), duplicate_factors=(\d+)", trace)
    assert models
    assert any(int(duplicates) > 0 and int(unique) < int(factors)
               for factors, unique, duplicates in models)
    from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit
    for gates, parameters in solutions:
        actual_circuit = get_Qiskit_Circuit(gates, parameters)
        assert actual_circuit.count_ops().get("cx", 0) == 1
        actual = np.asarray(Operator(actual_circuit).data)
        overlap = np.vdot(actual, target)
        chord = np.linalg.norm(target - overlap / abs(overlap) * actual) ** 2 / len(target)
        assert chord * (1 - chord / 4) <= TOLERANCE
