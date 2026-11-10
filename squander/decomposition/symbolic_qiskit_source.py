"""Recover exact expressions from Qiskit's analytical gate definitions.

This is an attributed construction source, not a new synthesis discovery.
Recovered constants are proposals until the complete symbolic identity is
proved by the caller; numerical fitting alone is never a certificate.
"""

from fractions import Fraction
import math
import sympy as sp


def _exact_constant(value):
    number=float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite SDK constant")
    if abs(number-round(number))<1e-12:
        return sp.Integer(round(number))
    ratio=Fraction(number/math.pi).limit_denominator(64)
    if abs(float(ratio)*math.pi-number)<1e-12:
        return sp.pi*sp.Rational(ratio.numerator,ratio.denominator)
    rational=Fraction(number).limit_denominator(1024)
    if abs(float(rational)-number)<1e-12:
        return sp.Rational(rational.numerator,rational.denominator)
    raise ValueError("SDK angle has no supported exact proposal")


def analytical_definition(name,parameters):
    from qiskit.circuit import Parameter, library
    if __package__:
        from .symbolic_gates import _QISKIT_NAMES
    else:
        from symbolic_gates import _QISKIT_NAMES
    classes=[key for key,value in _QISKIT_NAMES.items() if value==name]
    if not classes:
        raise ValueError("No matching SDK gate class")
    symbols=sorted(set().union(*(sp.sympify(p).free_symbols for p in parameters)),key=str)
    sdk_symbols={str(symbol):Parameter(str(symbol)) for symbol in symbols}
    def to_sdk(expression):
        expression=sp.sympify(expression)
        if expression in symbols:return sdk_symbols[str(expression)]
        # Register constructors share this path with angle expressions.
        # Qiskit rejects float control counts even when they are integral.
        if expression.is_Integer:return int(expression)
        if not expression.free_symbols:return float(expression)
        if expression.is_Add:return sum(to_sdk(x) for x in expression.args)
        if expression.is_Mul:
            result=1
            for x in expression.args:result*=to_sdk(x)
            return result
        if expression.is_Pow:return to_sdk(expression.base)**to_sdk(expression.exp)
        raise ValueError("Unsupported symbolic SDK parameter expression")
    gate=getattr(library,classes[0])(*(to_sdk(p) for p in parameters))
    canonical={"u":"U3","u1":"P","u2":"U2","id":"I","x":"X","y":"Y","z":"Z", "h":"H",
               "s":"S","sdg":"Sdg","t":"T","tdg":"Tdg","sx":"Sx","sxdg":"Sxdg",
               "rx":"Rx","ry":"Ry","rz":"Rz","p":"P","cx":"CX"}
    real_symbols={str(symbol):symbol for symbol in symbols}
    def from_sdk(value):
        if isinstance(value,(float,int)):
            return _exact_constant(value) if isinstance(value,float) else sp.Integer(value)
        expression=value.sympify() if hasattr(value,"sympify") else sp.sympify(value)
        expression=expression.xreplace({symbol:real_symbols[str(symbol)] for symbol in expression.free_symbols})
        return expression.xreplace({atom:_exact_constant(atom) for atom in expression.atoms(sp.Float)})
    output=[]
    phase=sp.S.Zero
    def expand(operation,wires,depth=0):
        nonlocal phase
        if depth>30:raise ValueError("Cyclic SDK analytical definition")
        if operation.name in canonical:
            output.append((canonical[operation.name],*(from_sdk(p) for p in operation.params),tuple(wires)))
            return
        definition=operation.definition
        if definition is None:raise ValueError(f"SDK operation {operation.name} lacks an analytical definition")
        phase+=from_sdk(definition.global_phase)
        for instruction in definition.data:
            expand(instruction.operation,[wires[definition.find_bit(q).index] for q in instruction.qubits],depth+1)
    expand(gate,range(gate.num_qubits))
    return tuple(output),sp.exp(sp.I*sp.simplify(phase))
