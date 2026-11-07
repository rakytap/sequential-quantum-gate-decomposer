"""Historical objective obstruction and the approved nonnegative repair."""

import sympy as sp
from functools import lru_cache

if __package__:
    from .symbolic_contract import parameter_score,encode_sequence,circuit_cost,validate_operations
    from . import symbolic_gates as gates
    from .symbolic_identity import prove_circuit_identity
else:
    from symbolic_contract import parameter_score,encode_sequence,circuit_cost,validate_operations
    import symbolic_gates as gates
    from symbolic_identity import prove_circuit_identity


def legacy_parameter_score(sequence):
    total=sp.S.Zero
    for operation in sequence:
        total+=1<<len(operation[-1])
        for parameter in operation[1:-1]:
            denominator=sp.simplify(parameter).as_numer_denom()[1]
            total+=(sp.log(sp.Abs(denominator),2)+sp.Rational(1,100))/5
    return total


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
    score=legacy_parameter_score(word)
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
    """Compatibility alias for the repair approved after the original audit."""
    return parameter_score(sequence)


def operation_count_bound(entanglers,score):
    """Every potentially cheaper word has at most this many local gates.

    Each native entangler costs at least four and each local at least two.
    This bounds word length; it does not solve the continuous angle problem.
    """
    return sp.floor((sp.sympify(score)-4*entanglers)/2)


@lru_cache(maxsize=1024)
def _support_witnesses(target,phase_exact):
    """Exact sample witnesses establish whole-family wire support.

    A nonzero specialization proves a family cannot omit that wire. It does
    not assert the same lower bound at every parameter specialization.
    """
    width=target.rows.bit_length()-1
    symbols=sorted(target.free_symbols,key=str)
    witnesses={}
    literal=None
    for value in (sp.S.Zero,sp.pi/3,sp.pi/5):
        substitutions={p:value for p in symbols}
        matrix=target.subs(substitutions)
        for wire in range(width):
            if wire in witnesses:continue
            for axis in ('X','Z'):
                pauli=gates.apply_to(sp.eye(target.rows),width,gates.make_gate(axis),(wire,))
                commutator=matrix*pauli-pauli*matrix
                found=next(((index,sp.simplify(entry)) for index,entry in enumerate(commutator)
                            if sp.simplify(entry).is_zero is False),None)
                if found:
                    witnesses[wire]=dict(axis=axis,entry=found[0],nonzero=str(found[1]),
                        substitutions={str(p):str(v) for p,v in substitutions.items()})
                    break
        if phase_exact and literal is None:
            for index,entry in enumerate(matrix-sp.eye(target.rows)):
                entry=sp.simplify(entry)
                if entry.is_zero is False:
                    literal=dict(entry=index,nonzero=str(entry),
                        substitutions={str(p):str(v) for p,v in substitutions.items()})
                    break
        if len(witnesses)==width and (not phase_exact or literal is not None):break
        if not symbols:break
    return witnesses,literal


def secondary_score_floor_certificate(target,result,basis):
    """A global secondary minimum when a proved primitive/support floor meets.

    The matrix identity remains the construction certificate's responsibility.
    At E>0 the bound relaxes every local gate, so it needs no support claim.
    """
    if result.get('status')!='verified_symbolic':return None
    target=sp.ImmutableMatrix(target)
    cost=result['cost'];E=cost['entanglers']
    support,literal=({},None) if E else _support_witnesses(target,result['phase_exact'])
    local_bound=max(len(support),int(literal is not None))
    def arity(name):
        definition=gates.gate_definition(name)
        return len(definition.parameters)-len(definition.defaults)
    native_floor=4+sp.Rational(arity(basis.entangler),500)
    local_floor=2+sp.Rational(min(arity(name) for name in basis.single_qubit_gates),500)
    bound=E*native_floor+local_bound*local_floor
    if sp.simplify(sp.sympify(cost['parameter_score'])-bound)!=0:return None
    symbols={str(p):p for p in target.free_symbols}
    sequence=tuple((op['gate'],*(sp.sympify(p,locals=symbols) for p in op['parameters']),tuple(op['wires']))
                   for op in result['operations'])
    sequence=validate_operations(sequence,basis,target.rows.bit_length()-1)
    if circuit_cost(sequence).as_dict()!=cost:raise ValueError('Construction cost differs from its score-floor witness')
    return dict(method='nonnegative_primitive_score_and_wire_support_floor',
        score_lower_bound=str(bound),matched_score=cost['parameter_score'],
        entangler_count=E,local_count_lower_bound=local_bound,
        native_primitive_floor=str(native_floor),local_primitive_floor=str(local_floor),
        wire_support_witnesses={str(w):v for w,v in support.items()},literal_nonidentity_witness=literal,
        secondary_minimum_at_fixed_entangler_count=True,
        global_lexicographic_minimum=(E==0 or result.get('entangler_optimal') is True),
        scope='whole_symbolic_family' if target.free_symbols else 'fixed_target',
        specialization_caveat='A specialized parameter value can admit a cheaper circuit',
        objective_version=2,requires_verified_target_construction=True)
