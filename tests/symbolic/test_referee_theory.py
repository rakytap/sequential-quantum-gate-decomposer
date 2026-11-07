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


def test_balanced_normal_form_polynomial_components():
    from symbolic_balanced_involution import verify_normal_form_polynomials
    assert verify_normal_form_polynomials()


@pytest.mark.parametrize('edge,coordinates',[
    ('alpha_zero',lambda t:(0,t)),('alpha_half_pi',lambda t:(sp.pi/2,t)),
    ('beta_zero',lambda t:(t,0)),('beta_half_pi',lambda t:(t,sp.pi/2))])
def test_entire_rank_two_boundary_has_a_literal_three_cnot_word(edge,coordinates):
    from symbolic_balanced_involution import normal_form,boundary_source
    from symbolic_gates import make_controlled
    from symbolic_identity import prove_circuit_identity
    t=sp.Symbol('balanced_theta',real=True)
    assert prove_circuit_identity(make_controlled(normal_form(*coordinates(t)),2),boundary_source(edge,t)) is True


@pytest.mark.parametrize('a,b,minimum',[(0,0,1),(0,sp.pi/2,2),(sp.pi/2,0,2),(sp.pi/2,sp.pi/2,1),
                                    (0,sp.pi/4,3),(sp.pi/4,0,3),(sp.pi/4,sp.pi/4,4)])
def test_balanced_classification_includes_every_rank_stratum(a,b,minimum):
    from symbolic_balanced_involution import normal_form,balanced_controlled_cost
    assert balanced_controlled_cost(normal_form(a,b).applyfunc(sp.simplify))['minimum_cx']==minimum


def test_whole_balanced_normal_form_has_a_four_cnot_construction():
    from symbolic_balanced_involution import normal_form,controlled_normal_form_source
    from symbolic_gates import make_controlled
    from symbolic_identity import prove_circuit_identity
    a,b=sp.symbols('balanced_alpha balanced_beta',real=True)
    assert prove_circuit_identity(make_controlled(normal_form(a,b),2),controlled_normal_form_source(a,b)) is True
