import copy
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_clifford_t_global import _encode, _step, _multiply, _inverse, exact_word_ball, solve_global_clifford_t
from symbolic_clifford_search import group_data, solve_clifford_alphabet, verify_shortest_path_certificate


def test_integer_ring_arithmetic_matches_exact_primitive_matrices():
    identity = _encode(sp.eye(2))
    state, matrix = identity, sp.eye(2)
    for name in ('H', 'T', 'S', 'Tdg', 'H', 'T', 'H', 'S', 'Tdg'):
        primitive = gates.make_gate(name)
        state = _step(state, name)
        matrix = (primitive*matrix).applyfunc(lambda x:sp.simplify(sp.expand_complex(x)))
        assert state == _encode(matrix)
        assert _multiply(_encode(primitive), _encode(matrix.H)) == _encode((primitive*matrix.H).applyfunc(sp.expand))
        assert _multiply(state, _inverse(state)) == identity


def test_exhaustive_ball_contains_nonclifford_prefixes_and_all_192_witnesses():
    ball = exact_word_ball()
    assert len(ball) == 628
    assert _encode(gates.make_gate('T')) in ball
    result = solve_global_clifford_t()
    assert len(result['solutions']) == 192
    assert max(map(len, result['solutions'].values())) == 14
    for state, word in result['solutions'].items():
        actual = _encode(sp.eye(2))
        for name in word: actual = _step(actual, name)
        assert actual == _encode(group_data()[1][state])


def test_compiler_exposes_global_proof_and_rejects_forged_certificates():
    solution = solve_clifford_alphabet(('H', 'S', 'T', 'Tdg'))
    assert solution['global_local_optimality']
    assert verify_shortest_path_certificate(solution)
    changed = copy.deepcopy(solution)
    changed['exhaustive_word_certificate']['radius'] = 6
    with pytest.raises(ValueError): verify_shortest_path_certificate(changed)
    changed = copy.deepcopy(solution)
    changed.pop('exhaustive_word_certificate')
    with pytest.raises(ValueError): verify_shortest_path_certificate(changed)
