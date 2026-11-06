"""Exact exponential-polynomial circuit identities without numerical sampling.

Exponent arguments are expanded into rational multiples of common symbolic
terms. Those exponentials become formal Laurent variables. A zero rational
function in independent variables is a sufficient identity certificate; no
converse or hidden phase relation is assumed.
"""

import math
from functools import lru_cache
import sympy as sp

if __package__:
    from .symbolic_gates import make_gate, apply_to
else:
    from symbolic_gates import make_gate, apply_to


def exponential_forms(expressions,*,return_converter=False):
    """Return exact rational forms and their inverse exponential substitution.

    The formal variables are shared by every supplied expression. They are
    used for algebra, not numerical samples or assumptions about independence
    of the original gate parameters.
    """
    rewritten=[sp.expand(entry.rewrite(sp.exp)) for entry in expressions]
    denominators={}
    def terms(argument):
        return sp.Add.make_args(sp.expand(argument))
    for entry in rewritten:
        for exponential in entry.atoms(sp.exp):
            for term in terms(exponential.args[0]):
                if not term.free_symbols:
                    continue
                coefficient,base=term.as_coeff_Mul()
                if not coefficient.is_Rational:
                    raise ValueError("Unsupported non-rational exponential coefficient")
                old=denominators.get(base,1)
                denominator=int(coefficient.q)
                denominators[base]=old*denominator//math.gcd(old,denominator)
    symbols={base:sp.Dummy(f"e{index}") for index,base in enumerate(denominators)}
    def exponential_value(exponential):
        result=sp.S.One
        for term in terms(exponential.args[0]):
            if not term.free_symbols:
                result*=sp.simplify(sp.expand_complex(sp.exp(term)))
            else:
                coefficient,base=term.as_coeff_Mul()
                power=coefficient*denominators[base]
                if power.is_Integer is not True:
                    raise ValueError("The exponential context lacks this denominator")
                result*=symbols[base]**int(power)
        return result
    def convert(expression):
        entry=sp.expand(sp.sympify(expression).rewrite(sp.exp))
        replacements={e:exponential_value(e) for e in entry.atoms(sp.exp)}
        return sp.cancel(sp.expand(entry.xreplace(replacements)))
    transformed=[convert(entry) for entry in rewritten]
    inverse={symbols[base]:sp.exp(base/denominator) for base,denominator in denominators.items()}
    if return_converter:return tuple(transformed),inverse,convert
    return tuple(transformed),inverse


@lru_cache(maxsize=8192)
def _zero_coefficient(coefficient):
    if coefficient==0:return True
    if coefficient.is_Rational:return False
    simplified=sp.simplify(coefficient)
    if simplified==0:return True
    if not coefficient.free_symbols:
        # Root-of-unity products beyond eighth roots do not always simplify
        # in exponential form. Their real/imaginary algebra is still exact.
        return sp.simplify(sp.expand_complex(simplified))==0
    return False


def zero_form(expression,variables=()):
    """Prove a rational expression zero, including its exact coefficients."""
    expression=sp.cancel(sp.expand(expression))
    if expression==0:return True
    numerator=sp.fraction(expression)[0]
    if variables:
        coefficients=sp.Poly(numerator,*variables).coeffs()
    else:
        coefficients=[numerator]
    return all(_zero_coefficient(coefficient) for coefficient in coefficients)


def prove_circuit_identity(target, sequence, factor=1):
    target=sp.ImmutableMatrix(target)
    factor=sp.sympify(factor)
    matrices=[make_gate(op[0],*op[1:-1]) for op in sequence]
    expressions=[entry for matrix in [target,*matrices] for entry in matrix]+[factor]
    try:transformed,inverse=exponential_forms(expressions)
    except ValueError:return None
    position=0
    def restore(matrix):
        nonlocal position
        result=sp.ImmutableMatrix(matrix.rows,matrix.cols,transformed[position:position+len(matrix)])
        position+=len(matrix)
        return result
    target_form=restore(target)
    native=[restore(matrix) for matrix in matrices]
    phase=transformed[-1]
    width=target.rows.bit_length()-1
    actual=sp.eye(target.rows)
    for matrix,operation in zip(native,sequence):
        actual=apply_to(actual,width,matrix,operation[-1]).applyfunc(lambda x:sp.cancel(sp.expand(x)))
    return all(zero_form(a-phase*b,tuple(inverse)) for a,b in zip(target_form,actual))
