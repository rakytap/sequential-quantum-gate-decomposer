import pytest
import numpy as np
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.synthesis.PartAM_utils import group_into_two_qubit_blocks


def _count_gates_by_qubit_num(circuit, qubit_num):
    """Count gates with exactly `qubit_num` involved qubits, recursing into blocks."""
    count = 0
    for gate in circuit.get_Gates():
        if isinstance(gate, Circuit):
            count += _count_gates_by_qubit_num(gate, qubit_num)
        else:
            if len(gate.get_Involved_Qbits()) == qubit_num:
                count += 1
    return count


def _get_params(circuit, seed=42):
    np.random.seed(seed)
    return np.random.uniform(0, 2 * np.pi, circuit.get_Parameter_Num())


# ============================================================================
# Structure tests
# ============================================================================

def test_all_top_level_elements_are_circuits():
    c = Circuit(3)
    c.add_H(0)
    c.add_CNOT(0, 1)
    c.add_RZ(1)
    c.add_CNOT(1, 2)

    result = group_into_two_qubit_blocks(c)
    for gate in result.get_Gates():
        assert isinstance(gate, Circuit)


def test_each_block_has_exactly_one_two_qubit_gate():
    c = Circuit(3)
    c.add_H(0)
    c.add_CNOT(0, 1)
    c.add_RZ(1)
    c.add_CNOT(1, 2)
    c.add_H(2)

    result = group_into_two_qubit_blocks(c)
    for block in result.get_Gates():
        two_qubit_count = sum(
            1 for g in block.get_Gates()
            if len(g.get_Involved_Qbits()) == 2
        )
        assert two_qubit_count == 1


def test_block_count_equals_two_qubit_gate_count():
    c = Circuit(4)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CNOT(2, 3)

    result = group_into_two_qubit_blocks(c)
    assert len(result.get_Gates()) == 3


def test_only_2qubit_gates_each_block_has_one_gate():
    """With no single-qubit gates, each block contains exactly the 2-qubit gate."""
    c = Circuit(3)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)

    result = group_into_two_qubit_blocks(c)
    for block in result.get_Gates():
        assert len(block.get_Gates()) == 1


# ============================================================================
# Gate count preservation tests
# ============================================================================

def test_total_single_qubit_gate_count_preserved():
    c = Circuit(3)
    c.add_H(0)
    c.add_RZ(1)
    c.add_CNOT(0, 1)
    c.add_RZ(0)
    c.add_CNOT(1, 2)
    c.add_H(2)

    result = group_into_two_qubit_blocks(c)
    assert _count_gates_by_qubit_num(result, 1) == _count_gates_by_qubit_num(c, 1)


def test_total_two_qubit_gate_count_preserved():
    c = Circuit(4)
    c.add_H(0)
    c.add_CNOT(0, 1)
    c.add_H(2)
    c.add_CNOT(1, 2)
    c.add_CNOT(2, 3)

    result = group_into_two_qubit_blocks(c)
    assert _count_gates_by_qubit_num(result, 2) == _count_gates_by_qubit_num(c, 2)


# ============================================================================
# Block membership tests
# ============================================================================

def test_leading_single_qubit_gates_in_first_block():
    """Single-qubit gates before the first 2-qubit gate go into the first block."""
    c = Circuit(2)
    c.add_H(0)
    c.add_H(1)
    c.add_CNOT(0, 1)

    result = group_into_two_qubit_blocks(c)
    blocks = result.get_Gates()
    assert len(blocks) == 1
    assert len(blocks[0].get_Gates()) == 3  # H(0) + H(1) + CNOT


def test_trailing_single_qubit_gates_in_last_block():
    """Single-qubit gates after the last 2-qubit gate on a qubit go into that last block."""
    c = Circuit(2)
    c.add_CNOT(0, 1)
    c.add_H(0)
    c.add_RZ(1)

    result = group_into_two_qubit_blocks(c)
    blocks = result.get_Gates()
    assert len(blocks) == 1
    assert len(blocks[0].get_Gates()) == 3  # CNOT + H(0) + RZ(1)


def test_interleaved_single_qubit_gates_split_correctly():
    """Single-qubit gates between two 2-qubit gates go to the next block."""
    c = Circuit(3)
    c.add_CNOT(0, 1)   # block 0
    c.add_H(0)          # -> block 1 (next 2-qubit gate involving q0)
    c.add_RZ(1)         # -> block 1 (next 2-qubit gate involving q1)
    c.add_CNOT(0, 1)   # block 1

    result = group_into_two_qubit_blocks(c)
    blocks = result.get_Gates()
    assert len(blocks) == 2
    assert len(blocks[0].get_Gates()) == 1  # only CNOT
    assert len(blocks[1].get_Gates()) == 3  # H + RZ + CNOT


# ============================================================================
# Correctness (unitary equivalence) tests
# ============================================================================

def test_unitary_equivalence_cnot_chain():
    c = Circuit(3)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CNOT(0, 2)

    result = group_into_two_qubit_blocks(c)
    params = _get_params(c)
    assert np.allclose(c.get_Matrix(params), result.get_Matrix(params), atol=1e-10)


def test_unitary_equivalence_with_single_qubit_gates():
    c = Circuit(3)
    c.add_H(0)
    c.add_RZ(1)
    c.add_CNOT(0, 1)
    c.add_H(1)
    c.add_CNOT(1, 2)
    c.add_H(2)

    result = group_into_two_qubit_blocks(c)
    params = _get_params(c)
    assert np.allclose(c.get_Matrix(params), result.get_Matrix(params), atol=1e-10)


@pytest.mark.parametrize("N", [2, 3, 4])
def test_unitary_equivalence_parametric(N):
    c = Circuit(N)
    c.add_RZ(0)
    c.add_CNOT(0, 1)
    c.add_RY(1)
    if N > 2:
        c.add_RZ(2)
        c.add_CNOT(1, 2)
    if N > 3:
        c.add_RZ(3)
        c.add_CNOT(2, 3)

    result = group_into_two_qubit_blocks(c)
    params = _get_params(c)
    assert np.allclose(c.get_Matrix(params), result.get_Matrix(params), atol=1e-10)


# ============================================================================
# Edge cases
# ============================================================================

def test_empty_circuit_returns_empty():
    c = Circuit(3)
    result = group_into_two_qubit_blocks(c)
    assert len(result.get_Gates()) == 0


def test_single_two_qubit_gate_no_singles():
    c = Circuit(2)
    c.add_CNOT(0, 1)

    result = group_into_two_qubit_blocks(c)
    blocks = result.get_Gates()
    assert len(blocks) == 1
    assert isinstance(blocks[0], Circuit)
    assert len(blocks[0].get_Gates()) == 1
