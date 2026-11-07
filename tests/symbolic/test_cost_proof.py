import sympy as sp

from symbolic_contract import parameter_score
from symbolic_cost_proof import negative_identity_certificate,proposed_nonnegative_parameter_score


def test_unrestricted_legacy_objective_has_an_exact_negative_identity_cycle():
    certificate=negative_identity_certificate()
    assert certificate["status"]=="proved_no_global_minimum"
    assert sp.Rational(certificate["strict_score_upper_bound"])<0
    assert certificate["identity_verification"]=="exact_free_real_U3_inverse_identity"


def test_proposed_repair_is_opt_in_and_does_not_change_ordinary_pi_costs():
    word=(("U3",sp.pi/2,sp.pi/4,-sp.pi/8,(0,)),("CX",(0,1)))
    assert proposed_nonnegative_parameter_score(word)==parameter_score(word)
    angle=1/sp.log(sp.Rational(2049,2048))
    pair=(("U3",angle,0,0,(0,)),("U3",-angle,0,0,(0,)))
    assert proposed_nonnegative_parameter_score(pair)==sp.Rational(1003,250)
    assert parameter_score(pair)==sp.sympify(negative_identity_certificate()["exact_score"])
