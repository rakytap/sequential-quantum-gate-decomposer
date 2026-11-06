"""Exact exponential-polynomial circuit identities without numerical sampling.

Exponent arguments are expanded into rational multiples of common symbolic
terms. Those exponentials become formal Laurent variables. A zero rational
function in independent variables is a sufficient identity certificate; no
converse or hidden phase relation is assumed.
"""

import math
import sympy as sp

if __package__:
    from .symbolic_gates import make_gate, apply_to
else:
    from symbolic_gates import make_gate, apply_to


def prove_circuit_identity(target, sequence, factor=1):
    target=sp.ImmutableMatrix(target)
    factor=sp.sympify(factor)
    matrices=[make_gate(op[0],*op[1:-1]) for op in sequence]
    expressions=[entry for matrix in [target,*matrices] for entry in matrix]+[factor]
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
                    return None
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
                result*=symbols[base]**int(coefficient*denominators[base])
        return result
    transformed=[]
    for entry in rewritten:
        replacements={e:exponential_value(e) for e in entry.atoms(sp.exp)}
        transformed.append(sp.cancel(sp.expand(entry.xreplace(replacements))))
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
    def zero(expression):
        expression=sp.cancel(sp.expand(expression))
        if expression==0:return True
        # Eighth/sixteenth-root constants may have equivalent nested radical
        # forms. Normalize their coefficients, without sampling the variables.
        numerator=sp.fraction(expression)[0]
        if symbols:
            coefficients=sp.Poly(numerator,*symbols.values()).coeffs()
        else:
            coefficients=[numerator]
        return all(sp.simplify(coefficient)==0 for coefficient in coefficients)
    return all(zero(a-phase*b) for a,b in zip(target_form,actual))
