import sympy as sp

from symbolic_contract import parameter_score
from symbolic_cost_proof import negative_identity_certificate,proposed_nonnegative_parameter_score,legacy_parameter_score,operation_count_bound


def test_unrestricted_legacy_objective_has_an_exact_negative_identity_cycle():
    certificate=negative_identity_certificate()
    assert certificate["status"]=="proved_no_global_minimum"
    assert sp.Rational(certificate["strict_score_upper_bound"])<0
    assert certificate["identity_verification"]=="exact_free_real_U3_inverse_identity"


def test_approved_repair_preserves_ordinary_pi_costs_and_excludes_negative_cycles():
    word=(("U3",sp.pi/2,sp.pi/4,-sp.pi/8,(0,)),("CX",(0,1)))
    assert parameter_score(word)==legacy_parameter_score(word)
    angle=1/sp.log(sp.Rational(2049,2048))
    pair=(("U3",angle,0,0,(0,)),("U3",-angle,0,0,(0,)))
    assert proposed_nonnegative_parameter_score(pair)==sp.Rational(1003,250)
    assert legacy_parameter_score(pair)==sp.sympify(negative_identity_certificate()["exact_score"])
    assert parameter_score(pair)>0


def test_nonnegative_objective_gives_a_finite_word_length_bound():
    assert operation_count_bound(2,sp.Rational(11621,500))==7
    assert operation_count_bound(0,0)==0
