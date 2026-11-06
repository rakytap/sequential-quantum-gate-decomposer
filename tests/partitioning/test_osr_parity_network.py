"""Independent GF(2), topology and matrix checks for native parity motifs."""

import ctypes
from collections import deque

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander import N_Qubit_Decomposition_Tree_Search  # Load the native core.


def networks(width, required, edges, budget):
    function = ctypes.CDLL("libqgd.so").qgd_osr_parity_networks
    pointer = ctypes.POINTER(ctypes.c_uint)
    function.argtypes = [
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_int,
        pointer, pointer, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = ctypes.c_int
    couplers = sum(1 << (a * width + b) for a, b in edges)
    minimum, count = ctypes.c_int(), ctypes.c_int()
    assert function(width, required, couplers, budget, None, None, 0,
                    ctypes.byref(minimum), ctypes.byref(count)) == 0
    if count.value == 0:
        return minimum.value, []
    rows = np.zeros(count.value * width, dtype=np.uint32)
    moves = np.zeros(count.value * minimum.value * 2, dtype=np.uint32)
    assert function(width, required, couplers, budget,
                    rows.ctypes.data_as(pointer), moves.ctypes.data_as(pointer),
                    count.value, ctypes.byref(minimum), ctypes.byref(count)) == 0
    return minimum.value, [
        (tuple(int(x) for x in rows[i * width:(i + 1) * width]),
         moves[i * minimum.value * 2:(i + 1) * minimum.value * 2].reshape(-1, 2))
        for i in range(count.value)
    ]


def exhaustive_minimum(width, required, edges):
    initial = tuple(1 << q for q in range(width))
    queue = deque([(initial, 0, 0)])
    visited = {(initial, 0)}
    best, endings = None, set()
    while queue:
        rows, seen, depth = queue.popleft()
        if best is not None and depth > best:
            break
        if seen == required and rows[0] == 1 and all(not row & 1 for row in rows[1:]):
            best = depth
            endings.add(rows)
            continue
        for a, b in edges:
            for control, target in ((a, b), (b, a)):
                changed = list(rows)
                changed[target] ^= changed[control]
                updated = seen
                if changed[target] & 1 and changed[target] > 1:
                    updated |= (1 << ((changed[target] >> 1) - 1)) & required
                state = (tuple(changed), updated)
                if state not in visited:
                    visited.add(state)
                    queue.append((state[0], updated, depth + 1))
    return best, endings


@pytest.mark.parametrize("width,required", [
    (2, 0), (2, 1), (3, 1), (3, 3), (3, 7),
    (4, 1), (4, 3), (4, 7), (4, 17), (4, 65), (4, 127),
])
def test_native_parity_networks_reconstruct_the_phase_program(width, required):
    edges = [(a, b) for a in range(width) for b in range(a + 1, width)]
    minimum, emitted = networks(width, required, edges, 14)
    assert emitted
    assert len({ending for ending, _ in emitted}) == len(emitted)
    if width <= 3:
        exact_minimum, exact_endings = exhaustive_minimum(width, required, edges)
        assert minimum == exact_minimum
        assert {ending for ending, _ in emitted} == exact_endings
    if minimum > 0:
        assert networks(width, required, edges, minimum - 1)[1] == []
    angles = {mask: 0.13 + 0.07 * mask
              for mask in range(1, 1 << (width - 1))
              if required & (1 << (mask - 1))}
    for ending, moves in emitted:
        circuit = QuantumCircuit(width)
        rows = [1 << q for q in range(width)]
        seen = set()
        for control, target in moves:
            control, target = int(control), int(target)
            assert tuple(sorted((control, target))) in edges
            circuit.cx(control, target)
            rows[target] ^= rows[control]
            parity = rows[target]
            if parity & 1 and parity > 1:
                mask = parity >> 1
                if mask in angles and mask not in seen:
                    circuit.rz(angles[mask], target)
                    seen.add(mask)
        assert seen == set(angles)
        assert tuple(rows) == ending
        assert ending[0] == 1 and all(not row & 1 for row in ending[1:])
        dimension = 1 << width
        expected = np.zeros((dimension, dimension), dtype=complex)
        for column in range(dimension):
            output = sum((bin(row & column).count("1") % 2) << q
                         for q, row in enumerate(ending))
            phase = sum(angle * (-1 if bin((1 | (mask << 1)) & column).count("1") % 2 else 1)
                        for mask, angle in angles.items())
            expected[output, column] = np.exp(-0.5j * phase)
        np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("edges", [((0, 1), (1, 2)), ((0, 1), (0, 2))])
def test_native_parity_networks_keep_the_actual_topology(edges):
    minimum, emitted = networks(3, 7, edges, 14)
    exact_minimum, exact_endings = exhaustive_minimum(3, 7, edges)
    assert minimum == exact_minimum
    assert {ending for ending, _ in emitted} == exact_endings
    for _, moves in emitted:
        assert all(tuple(sorted((int(a), int(b)))) in edges for a, b in moves)


def test_native_parity_networks_do_not_invent_a_disconnected_edge():
    assert networks(3, 7, ((0, 1),), 14)[1] == []
