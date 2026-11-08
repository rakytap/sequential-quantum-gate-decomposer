import sympy as sp
from symbolic_gates import make_controlled,apply_to
from symbolic_reflection_involution import reflection,five_arbitrary_matrices
from symbolic_controlled_chart import _zero


def test_five_unrestricted_reflection_word_with_angle_free():
    theta=sp.Symbol('reflection_angle',real=True)
    word=five_arbitrary_matrices(theta)
    assert [wires for matrix,wires in word]==[(0,1),(1,2),(0,2),(0,1),(1,2)]
    actual=sp.eye(8)
    for matrix,wires in word:
        assert _zero(matrix.H*matrix-sp.eye(4))
        actual=apply_to(actual,3,matrix,wires)
    assert _zero(actual-make_controlled(reflection(theta),2))
