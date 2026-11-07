import sympy as sp
import pytest
from symbolic_exchange_invariants import (verify_sqrt_iswap_iff_reduction,
                                         sqrt_iswap_two_pulse_condition)
from symbolic_controlled_rank_bound import (verify_arbitrary_interaction_source,
                                            weyl_class_cost_counterexamples)


def test_exact_invariant_iff_algebra():
    assert verify_sqrt_iswap_iff_reduction()


@pytest.mark.parametrize('coordinates,inside',[
    ((0,0,0),True),((sp.pi/8,sp.pi/8,0),True),
    ((sp.pi/4,0,0),True),((sp.pi/4,sp.pi/8,sp.pi/8),True),
    ((sp.pi/4,sp.pi/4,sp.pi/4),False),
    ((sp.pi/8,sp.pi/8,sp.pi/8),False),
    ((sp.pi/4,sp.pi/6,-sp.pi/12),True)])
def test_invariant_condition_including_closed_boundary(coordinates,inside):
    x,y,z=coordinates
    g1=(sp.cos(2*x)*sp.cos(2*y)*sp.cos(2*z)+sp.I*sp.sin(2*x)*sp.sin(2*y)*sp.sin(2*z))**2
    g2=sum(sp.cos(4*t) for t in coordinates)
    assert bool(sqrt_iswap_two_pulse_condition(sp.simplify(g1),sp.simplify(g2))) is inside


def test_three_unrestricted_interactions_are_an_exact_fredkin_fermionic_upper_bound():
    assert verify_arbitrary_interaction_source()


def test_controlled_cost_cannot_be_classified_by_relative_weyl_class():
    examples=weyl_class_cost_counterexamples()
    assert [r['minimum_cx'] for r in examples]==[0,1,2]
    assert [r['singleton_cut_ranks'] for r in examples]==[[1,1,1],[2,2,1],[2,2,2]]
