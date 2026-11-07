"""Exact obstruction to global minimization of the unrestricted legacy score.

The proposed nonnegative score is opt-in here. No compiler objective changes
until its policy has been selected and compatibility has been audited.
"""

import sympy as sp

if __package__:
    from .symbolic_contract import parameter_score,encode_sequence
    from .symbolic_identity import prove_circuit_identity
else:
    from symbolic_contract import parameter_score,encode_sequence
    from symbolic_identity import prove_circuit_identity


def negative_identity_certificate():
    theta=sp.Symbol("cost_theta",real=True)
    identity=(("U3",theta,0,0,(0,)),("U3",-theta,0,0,(0,)))
    if prove_circuit_identity(sp.eye(2),identity) is not True:raise ArithmeticError("Inverse identity failed")
    x=sp.Symbol("cost_x",positive=True)
    difference=sp.log(1+x)-x
    derivative=sp.simplify(sp.diff(difference,x))
    if difference.subs(x,0)!=0 or derivative.is_negative is not True:raise ArithmeticError("Log bound failed")
    denominator=sp.log(sp.Rational(2049,2048))
    angle=1/denominator
    word=tuple((op[0],angle if op[1]==theta else -angle,*op[2:]) for op in identity)
    score=parameter_score(word)
    expected=sp.Rational(1003,250)+2*sp.log(denominator,2)/5
    if sp.simplify(score-expected)!=0:raise ArithmeticError("Legacy cycle score changed")
    # 0 < log(1+2^-11) < 2^-11. Monotonicity of log gives this
    # STRICT rational upper bound for the identity pair's score.
    bound=sp.Rational(1003,250)-sp.Rational(22,5)
    assert bound==sp.Rational(-97,250)
    return dict(status="proved_no_global_minimum",scope="unrestricted real-angle legacy parameter score",
        operations=encode_sequence(word),identity_verification="exact_free_real_U3_inverse_identity",
        exact_score=str(score),strict_score_upper_bound=str(bound),
        log_inequality_derivative=str(derivative),
        consequence="Appending arbitrarily many exact identity pairs preserves both phase policies and entangler count while the score tends to minus infinity")


def proposed_nonnegative_parameter_score(sequence):
    """Concrete proposed repair; this does not change the active objective."""
    total=sp.S.Zero
    for operation in sequence:
        total+=1<<len(operation[-1])
        for parameter in operation[1:-1]:
            denominator=sp.simplify(parameter).as_numer_denom()[1]
            total+=(sp.log(sp.Max(1,sp.Abs(denominator)),2)+sp.Rational(1,100))/5
    return total
