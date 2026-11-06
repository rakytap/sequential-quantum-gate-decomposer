"""Exact Pauli-normalizer checks and complete single-qubit Clifford tables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sympy

if __package__:
    from .symbolic_gates import complete_clifford, full_clifford, make_gate
else:
    from symbolic_gates import complete_clifford, full_clifford, make_gate


def _zero(value):
    if value == 0:
        return True
    value = sympy.sympify(value)
    if not value.free_symbols:
        value = sympy.expand_complex(value)
    return sympy.simplify(value) == 0


def _signed_pauli(matrix):
    """Recognize a Hermitian signed Pauli without enumerating 4**n labels."""
    size = matrix.rows
    nonzero = [row for row in range(size) if not _zero(matrix[row, 0])]
    if len(nonzero) != 1:
        return None
    xmask = nonzero[0]
    first = matrix[xmask, 0]
    zmask = 0
    for q in range(size.bit_length()-1):
        ratio = sympy.simplify(matrix[xmask ^ (1 << q), 1 << q] / first)
        if ratio == -1:
            zmask |= 1 << q
        elif ratio != 1:
            return None
    for col in range(size):
        for row in range(size):
            expected = first * (-1 if bin(zmask & col).count("1") % 2 else 1) if row == col ^ xmask else 0
            if not _zero(matrix[row, col]-expected):
                return None
    phase = sympy.simplify(first / sympy.I**bin(xmask & zmask).count("1"))
    if phase not in (sympy.S.One, sympy.S.NegativeOne):
        return None
    label = "".join("IXZY"[((xmask >> q) & 1) + 2*((zmask >> q) & 1)]
                    for q in reversed(range(size.bit_length()-1)))
    return int(phase), label


def pauli_action(matrix):
    """Return images of X_i, Z_i under U P U†, or None if not provably Clifford.

    The returned sequence is (X_0, Z_0, X_1, Z_1, ...); labels use Qiskit's
    most-significant-wire-first notation. No floating-point tolerance is used.
    """
    return _pauli_action_cached(sympy.ImmutableMatrix(matrix))


@lru_cache(maxsize=512)
def _pauli_action_cached(matrix):
    size = matrix.rows
    if matrix.cols != size or size < 2 or size & (size-1):
        raise ValueError("Clifford analysis requires a square qubit unitary")
    if any(not _zero(value) for value in matrix.H*matrix-sympy.eye(size)):
        return None
    action = []
    for q in range(size.bit_length()-1):
        for axis in ("X", "Z"):
            generator = sympy.zeros(size)
            for col in range(size):
                row = col ^ (1 << q) if axis == "X" else col
                generator[row, col] = -1 if axis == "Z" and col & (1 << q) else 1
            image = _signed_pauli(matrix*generator*matrix.H)
            if image is None:
                return None
            action.append(image)
    return tuple(action)


def is_clifford(matrix):
    return pauli_action(matrix) is not None


def _exact_phase(left, right):
    pivot = next(i for i, value in enumerate(right) if not _zero(value))
    phase = sympy.simplify(left[pivot]/right[pivot])
    if any(not _zero(a-phase*b) for a, b in zip(left, right)):
        raise ValueError("Matrices are not equal up to global phase")
    for exponent in range(8):
        if _zero(phase-sympy.exp(sympy.I*sympy.pi*exponent/4)):
            return exponent
    raise ValueError("Clifford phase is not an eighth root of unity")


@dataclass(frozen=True)
class CliffordAnalysis:
    representatives: tuple
    # All 192 names map to (projective index, exact eighth-root phase exponent).
    elements: tuple
    multiplication: tuple
    multiplication_phases: tuple
    inverse_indices: tuple


@lru_cache(maxsize=1)
def analyze_single_qubit_cliffords():
    """Validate all 192 original variants and derive the 24x24 exact group table.

    multiplication[i][j] is the representative of U_i @ U_j. Its accompanying
    phase k satisfies U_i @ U_j = exp(i*pi*k/4) U_result. Thus a phase discarded
    for classification can still be retained when constructing controlled gates.
    """
    names = tuple(full_clifford)
    matrices = tuple(make_gate(name) for name in names)
    actions = tuple(pauli_action(matrix) for matrix in matrices)
    if len(names) != 24 or None in actions or len(set(actions)) != 24:
        raise ValueError("The original representatives do not span 24 Cliffords")
    lookup = {action: i for i, action in enumerate(actions)}
    elements = []
    physical = set()
    for name in complete_clifford:
        matrix = make_gate(name)
        index = lookup.get(pauli_action(matrix))
        if index is None:
            raise ValueError(f"Not a Clifford: {name}")
        phase = _exact_phase(matrix, matrices[index])
        elements.append((name, index, phase))
        physical.add((index, phase))
    if len(elements) != 192 or len(physical) != 192:
        raise ValueError("The original phase-resolved catalog is not complete")
    table, phases = [], []
    for left in matrices:
        row, phase_row = [], []
        for right in matrices:
            product = left*right
            index = lookup[pauli_action(product)]
            row.append(index)
            phase_row.append(_exact_phase(product, matrices[index]))
        table.append(tuple(row))
        phases.append(tuple(phase_row))
    identity = lookup[pauli_action(sympy.eye(2))]
    inverses = tuple(next(j for j in range(24) if table[i][j] == identity) for i in range(24))
    return CliffordAnalysis(names, tuple(elements), tuple(table), tuple(phases), inverses)
