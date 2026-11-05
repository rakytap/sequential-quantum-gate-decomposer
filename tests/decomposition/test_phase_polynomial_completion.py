"""Independent matrix/topology controls for certified native parity motifs."""
import ctypes
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import squander


def _native(name):
    try:
        return getattr(ctypes.CDLL(None), name)
    except AttributeError:
        return getattr(ctypes.CDLL(str(Path(squander.__file__).parent / "libqgd.so")), name)


def _infidelity(target, actual):
    overlap = np.vdot(target, actual)
    chord = np.linalg.norm(actual - overlap / abs(overlap) * target) ** 2 / len(target)
    return float(chord * (1 - chord / 4))


def _pointer(array, kind=ctypes.c_int):
    return array.ctypes.data_as(ctypes.POINTER(kind))


def _matcher(width, rows, truth, edges, budget):
    call = _native("qgd_osr_phase_parity_network")
    ip = ctypes.POINTER(ctypes.c_int)
    call.argtypes = [ctypes.c_int, ip, ip, ctypes.c_int, ip, ip,
                     ctypes.c_int, ip, ip, ctypes.POINTER(ctypes.c_ulonglong)]
    call.restype = ctypes.c_int
    arrays = [np.asarray(value, dtype=np.int32) for value in
              (rows, truth, [edge[0] for edge in edges], [edge[1] for edge in edges])]
    word = np.zeros(max(1, budget), dtype=np.int32)
    coefficients = np.zeros(1 << width, dtype=np.int32)
    statistics = np.zeros(3, dtype=np.uint64)
    count = call(width, _pointer(arrays[0]), _pointer(arrays[1]), budget,
                 _pointer(arrays[2]), _pointer(arrays[3]), len(edges),
                 _pointer(word), _pointer(coefficients), _pointer(statistics, ctypes.c_ulonglong))
    return count, word, coefficients


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("shape", ["all_to_all", "line", "star"])
def test_parity_motifs_reconstruct_without_source_word_seeding(width, shape):
    rng = np.random.default_rng(20261008 + width)
    pairs = ([(a, b) for a in range(width) for b in range(a + 1, width)]
             if shape == "all_to_all" else
             [(a, a + 1) for a in range(width - 1)] if shape == "line" else
             [(0, b) for b in range(1, width)])
    edges = pairs + [(b, a) for a, b in pairs]
    for budget in range(9):
        source = QuantumCircuit(width)
        rows = [1 << bit for bit in range(width)]
        truth = np.zeros(1 << width, dtype=np.int32)

        def phase(wire, value):
            source.p(float(value * np.pi / 4), wire)
            for x in range(1 << width):
                truth[x] = (truth[x] + value * (bin(rows[wire] & x).count("1") % 2)) % 8

        for wire in range(width):
            phase(wire, int(rng.integers(8)))
        for _ in range(budget):
            control, target = edges[int(rng.integers(len(edges)))]
            source.cx(control, target)
            rows[target] ^= rows[control]
            phase(target, int(rng.integers(8)))
        count, word, coefficients = _matcher(width, rows, truth, edges, budget)
        assert 0 <= count <= budget
        result = QuantumCircuit(width)
        current = [1 << bit for bit in range(width)]
        applied = set()

        def phases():
            for wire, mask in enumerate(current):
                if mask not in applied:
                    result.p(float(coefficients[mask] * np.pi / 4), wire)
                    applied.add(mask)

        phases()
        for edge in word[:count]:
            control, target = edges[int(edge)]
            result.cx(control, target)
            current[target] ^= current[control]
            phases()
        assert current == rows
        assert all(mask in applied for mask in range(1, 1 << width) if coefficients[mask])
        assert _infidelity(Operator(source).data, Operator(result).data) < 1e-24
        assert int(result.count_ops().get("cx", 0)) == count


def _factor(target, width, root, pairs, budget, tolerance=1e-10):
    call = _native("qgd_osr_phase_polynomial_factor")
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    call.argtypes = [dp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                     ctypes.c_double, ctypes.c_int, ctypes.c_int, ip, ip, dp, dp]
    call.restype = ctypes.c_int
    matrix = np.ascontiguousarray(target, dtype=np.complex128)
    capacity = 512
    controls = np.zeros(capacity, dtype=np.int32)
    targets = np.zeros(capacity, dtype=np.int32)
    gates = np.zeros((capacity, 2, 2), dtype=np.complex128)
    error = ctypes.c_double(-1)
    couplers = sum(1 << (min(a, b) * width + max(a, b)) for a, b in pairs)
    size = call(_pointer(matrix, ctypes.c_double), width, 1 << width, root, couplers,
                tolerance, budget, capacity, _pointer(controls), _pointer(targets),
                _pointer(gates, ctypes.c_double), ctypes.byref(error))
    if size < 0:
        return size, None, error.value
    circuit = QuantumCircuit(width)
    for index in range(size):
        if controls[index] < 0:
            circuit.unitary(gates[index], [int(targets[index])])
        else:
            circuit.cx(int(controls[index]), int(targets[index]))
    return size, circuit, error.value


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("shape", ["line", "star"])
@pytest.mark.parametrize("move_root", [False, True])
def test_phase_factor_certifies_local_frames_and_boundary_permutations(width, shape, move_root):
    rng = np.random.default_rng(1103 + width)
    source = QuantumCircuit(width)
    for wire in range(width):
        source.u(*rng.normal(size=3), wire)
    pairs = ([(a, a + 1) for a in range(width - 1)] if shape == "line" else
             [(0, b) for b in range(1, width)])
    for control, target in pairs:
        source.cx(control, target)
        source.p(np.pi / 4, target)
    for control, target in reversed(pairs):
        source.cx(control, target)
    if move_root:
        source.cx(0, 1)
        source.cx(1, 0)
        source.cx(0, 1)
    for wire in range(width):
        source.u(*rng.normal(size=3), wire)
    target = Operator(source).data
    budget = int(source.count_ops().get("cx", 0))
    size, result, error = _factor(target, width, 0, pairs, budget)
    assert size >= 0
    assert error <= 1e-10
    assert _infidelity(target, Operator(result).data) <= 1e-10
    assert int(result.count_ops().get("cx", 0)) <= budget
    allowed = {frozenset(pair) for pair in pairs}
    for item in result.data:
        if item.operation.name == "cx":
            assert frozenset(result.find_bit(q).index for q in item.qubits) in allowed


@pytest.mark.parametrize("width", [3, 4])
def test_phase_factor_rejects_unsupported_and_nonunitary_targets(width):
    rng = np.random.default_rng(943 + width)
    dimension = 1 << width
    dense = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    unitary, _ = np.linalg.qr(dense)
    pairs = [(a, b) for a in range(width) for b in range(a + 1, width)]
    assert _factor(unitary, width, 0, pairs, 6)[0] < 0
    assert _factor(2 * np.eye(dimension), width, 0, pairs, 6)[0] < 0
