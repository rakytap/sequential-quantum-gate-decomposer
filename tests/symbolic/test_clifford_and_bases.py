import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import BASIS_SETS, get_basis, make_basis
from symbolic_clifford import analyze_single_qubit_cliffords, is_clifford, pauli_action


def test_full_phase_resolved_clifford_group_and_composition_table():
    analysis = analyze_single_qubit_cliffords()
    assert len(analysis.representatives) == 24
    assert len(analysis.elements) == 192
    assert len({(index, phase) for _, index, phase in analysis.elements}) == 192
    identity = next(i for i, name in enumerate(analysis.representatives) if name == "I")
    for i, row in enumerate(analysis.multiplication):
        assert sorted(row) == list(range(24))
        assert row[analysis.inverse_indices[i]] == identity
    # Associativity is independent of the matrix normalizer implementation.
    table = analysis.multiplication
    for a in range(24):
        for b in range(24):
            for c in range(24):
                assert table[table[a][b]][c] == table[a][table[b][c]]
                phase_left = analysis.multiplication_phases[a][b] + analysis.multiplication_phases[table[a][b]][c]
                phase_right = analysis.multiplication_phases[b][c] + analysis.multiplication_phases[a][table[b][c]]
                assert (phase_left-phase_right) % 8 == 0


@pytest.mark.parametrize("name", ["H", "S", "Sx", "ECR", "CX", "CZ", "DCX", "iSWAP", "FSWAP"])
def test_exact_pauli_normalizer_recognizes_known_cliffords(name):
    assert is_clifford(gates.make_gate(name))


@pytest.mark.parametrize("name", ["T", "SSWAP", "SiSWAP", "B", "CCX", "CSWAP"])
def test_exact_pauli_normalizer_rejects_known_noncliffords(name):
    assert not is_clifford(gates.make_gate(name))


def test_pauli_wire_and_sign_conventions():
    assert pauli_action(gates.make_gate("CX")) == ((1,"XX"),(1,"IZ"),(1,"XI"),(1,"ZZ"))
    assert pauli_action(gates.make_gate("H")) == ((1,"Z"),(1,"X"))
    assert pauli_action(gates.make_gate("T")) is None
    assert pauli_action(sp.zeros(2)) is None


def test_presets_have_one_entangler_and_honest_universality_labels():
    assert len(BASIS_SETS) >= 20
    for basis in BASIS_SETS.values():
        assert gates.gate_definition(basis.entangler).num_qubits == 2
        assert all(gates.gate_definition(name).num_qubits == 1 for name in basis.single_qubit_gates)
        assert sum(gates.gate_definition(name).num_qubits == 2 for name in basis.gates) == 1
    assert get_basis("ibm_ecr").exact_arbitrary_angles
    assert not get_basis("clifford_t_cx").exact_arbitrary_angles
    assert get_basis("clifford_only").universality == "clifford"


def test_basis_cost_counts_entanglers_and_rejects_other_primitives():
    basis = get_basis("cx_u")
    assert basis.entangler_count([("UGate",0,0,0,(0,)), ("CNOT",(0,1)), ("CX",(1,0))]) == 2
    with pytest.raises(ValueError, match="not in basis"):
        basis.entangler_count([("CZ",(0,1))])
    for nonentangler in ["SWAP", "XX", "YY", "ZZ"]:
        with pytest.raises(ValueError, match="entangling"):
            make_basis("invalid", ("U3",), nonentangler)


@pytest.mark.parametrize("name,denominator", [("S", 4), ("T", 8)])
def test_original_parameter_cost_prefers_named_gates_to_constant_u3(name, denominator):
    pytest.importorskip("deap")
    from sympy_param import AngleScale, CircuitBuilder, ParamIndex, get_gatecost, make_angle
    zero = make_angle(ParamIndex(-1), AngleScale(0, 1))
    phase = make_angle(ParamIndex(-1), AngleScale(1, denominator))
    named = CircuitBuilder().add_gate((name, (0,), ()))
    parameterized = CircuitBuilder().add_gate(("U3", (0,), (zero, zero, phase)))
    assert get_gatecost(named, []) < get_gatecost(parameterized, [])
    assert gates.compile_gates(1, parameterized.to_sympy([])) == gates.make_gate(name)
