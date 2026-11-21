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
        expanded=sp.simplify(sp.expand_complex(simplified))
        if expanded==0:return True
        # Propose a remaining algebraic-zero check numerically, then decide it
        # in an exact number field. A small numerical value never accepts an
        # identity; nonzero rational perturbations remain nonzero in the field.
        if expanded.is_algebraic is True:
            try:
                if abs(complex(expanded.evalf(50)))<1e-25:
                    return sp.to_number_field(expanded).coeffs()==[0]
            except (ValueError,TypeError,NotImplementedError):pass
        return False
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


@lru_cache(maxsize=32)
def _cyclotomic_domain(denominator):
    root=sp.exp(sp.I*sp.pi/denominator)
    variable=sp.Dummy("cyclotomic_root")
    # exp(i*pi/d) is a primitive 2d-th root. Supplying its known minimal
    # polynomial avoids rediscovering a high-degree radical representation.
    polynomial=sp.Poly(sp.cyclotomic_poly(2*denominator,variable),variable,domain=sp.QQ)
    domain=sp.QQ.algebraic_field((polynomial,root))
    return domain,domain.unit


def _cyclotomic_values(entries):
    """Exact rational/root-of-unity coefficients, declining other constants.

    A single cyclotomic generator avoids repeatedly discovering relations
    between many equivalent nested radicals. Every conversion is algebraic;
    neither floating evaluation nor tolerance enters the verifier.
    """
    rewritten=tuple(sp.sympify(entry).rewrite(sp.exp) for entry in entries)
    denominator=8
    for expression in rewritten:
        for exponential in expression.atoms(sp.exp):
            angle=sp.simplify(exponential.args[0]/(sp.I*sp.pi))
            if angle.is_Rational is not True:return None
            denominator=math.lcm(denominator,int(angle.q))
    radicals=[(sp.sqrt(2),4,False,sp.S.One),(sp.sqrt(3),6,False,sp.S.One)]
    upper=sp.sqrt(2)
    for power in range(3,7):
        radicals.extend([(sp.sqrt(2+upper),2**power,False,sp.S.One),
                         (sp.sqrt(2-upper),2**power,True,sp.S.One),
                         (sp.sqrt((2+upper)/4),2**power,False,sp.Rational(1,2)),
                         (sp.sqrt((2-upper)/4),2**power,True,sp.Rational(1,2))])
        upper=sp.sqrt(2+upper)
    present=[(value,period,sine,scale) for value,period,sine,scale in radicals if any(expr.has(value) for expr in rewritten)]
    has_sqrt_six=any(expr.has(sp.sqrt(6)) for expr in rewritten)
    if has_sqrt_six:present.extend(radicals[:2])
    for _,period,_,_ in present:denominator=math.lcm(denominator,period)
    if denominator>256:return None
    domain,root=_cyclotomic_domain(denominator)
    imaginary=root**(denominator//2)
    constants={sp.I:imaginary}
    for value,period,sine,scale in present:
        phase=root**(denominator//period)
        inverse=domain.one/phase
        constants[value]=domain.convert(scale)*(-imaginary*(phase-inverse) if sine else phase+inverse)
    if has_sqrt_six:constants[sp.sqrt(6)]=constants[sp.sqrt(2)]*constants[sp.sqrt(3)]
    cache={}
    def convert(expression):
        if expression in cache:return cache[expression]
        if expression.is_Rational:result=domain.convert(expression)
        elif expression in constants:result=constants[expression]
        elif expression.func==sp.exp:
            angle=sp.simplify(expression.args[0]/(sp.I*sp.pi))
            result=root**int(angle*denominator)
        elif expression.is_Add:
            result=sum((convert(term) for term in expression.args),domain.zero)
        elif expression.is_Mul:
            result=domain.one
            for term in expression.args:result*=convert(term)
        elif expression.is_Pow and expression.exp.is_Integer:
            result=convert(expression.base)**int(expression.exp)
        else:raise ValueError("Not a supported cyclotomic expression")
        cache[expression]=result
        return result
    try:return domain,dict(zip(entries,map(convert,rewritten)))
    except (ValueError,ZeroDivisionError):return None


def _algebraic_circuit_matrix(target,matrices,sequence,phase=1):
    """Multiply fixed circuits in one exact number field, with sparse rows.

    Repeated generic expression simplification after every embedded gate is
    unnecessary here. A non-algebraic coefficient declines this fast path.
    """
    if __package__:
        from . import symbolic_gates as gates
    else:
        import symbolic_gates as gates
    entries=tuple(dict.fromkeys([entry for matrix in [target,*matrices] for entry in matrix]+[phase]))
    converted=_cyclotomic_values(entries)
    if converted is None:return None
    domain,values=converted
    width=target.rows.bit_length()-1
    size=target.rows
    actual=[{row:domain.one} for row in range(size)]
    for matrix,operation in zip(matrices,sequence):
        wires=operation[-1]
        positions=wires if gates.little_endian else tuple(width-1-wire for wire in wires)
        mask=sum(1<<wire for wire in positions)
        offsets=[sum(((index>>bit)&1)<<wire for bit,wire in enumerate(positions)) for index in range(matrix.rows)]
        coefficients=[[(column,values[matrix[row,column]]) for column in range(matrix.cols)
                       if values[matrix[row,column]]!=domain.zero] for row in range(matrix.rows)]
        for base in range(size):
            if base&mask:continue
            indices=[base|offset for offset in offsets]
            previous=[actual[index] for index in indices]
            for index,terms in zip(indices,coefficients):
                result={}
                for local,coefficient in terms:
                    for column,value in previous[local].items():
                        result[column]=result.get(column,domain.zero)+coefficient*value
                actual[index]={column:value for column,value in result.items() if value!=domain.zero}
    return domain,values,actual


def _algebraic_circuit_identity(target,matrices,sequence,phase):
    data=_algebraic_circuit_matrix(target,matrices,sequence,phase)
    if data is None:return None
    domain,values,actual=data
    factor=values[phase]
    return all(values[target[row,column]]==factor*actual[row].get(column,domain.zero)
               for row in range(target.rows) for column in range(target.cols))


def infer_algebraic_circuit_scalar(target,sequence):
    """Infer and prove a fixed circuit's scalar in one exact number field.

    None declines the algebraic domain; False proves that no scalar works.
    A returned expression is checked against every matrix entry and its
    exact unit norm before it can replace a slower symbolic matrix product.
    """
    target=sp.ImmutableMatrix(target)
    matrices=[make_gate(op[0],*op[1:-1]) for op in sequence]
    if target.free_symbols or any(matrix.free_symbols for matrix in matrices):return None
    data=_algebraic_circuit_matrix(target,matrices,sequence)
    if data is None:return None
    domain,values,actual=data
    pivot=next(((row,column,value) for row,entries in enumerate(actual)
                for column,value in entries.items() if value!=domain.zero),None)
    if pivot is None:return False
    row,column,value=pivot
    factor=values[target[row,column]]/value
    if not all(values[target[row,column]]==factor*actual[row].get(column,domain.zero)
               for row in range(target.rows) for column in range(target.cols)):return False
    expression=domain.to_sympy(factor)
    if domain.from_sympy(expression*sp.conjugate(expression))!=domain.one:return False
    return expression


def prove_circuit_identity(target, sequence, factor=1):
    target=sp.ImmutableMatrix(target)
    factor=sp.sympify(factor)
    matrices=[make_gate(op[0],*op[1:-1]) for op in sequence]
    expressions=[entry for matrix in [target,*matrices] for entry in matrix]+[factor]
    if not any(expression.free_symbols for expression in expressions):
        algebraic=_algebraic_circuit_identity(target,matrices,sequence,factor)
        if algebraic is not None:return algebraic
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
    if not inverse:
        algebraic=_algebraic_circuit_identity(target_form,native,sequence,phase)
        if algebraic is not None:return algebraic
    width=target.rows.bit_length()-1
    actual=sp.eye(target.rows)
    for matrix,operation in zip(native,sequence):
        actual=apply_to(actual,width,matrix,operation[-1]).applyfunc(lambda x:sp.cancel(sp.expand(x)))
    return all(zero_form(a-phase*b,tuple(inverse)) for a,b in zip(target_form,actual))
