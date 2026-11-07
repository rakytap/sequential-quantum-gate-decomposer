"""Shared exact scalar expressions for constructive symbolic circuits.

The bindings are a directed acyclic graph, not additional independent gate
parameters. Expanding repeated spectral projectors into every Euler angle is
exponential in size. Keeping their arithmetic shared also makes zero-radius
branches lazy: an unused division by zero is never evaluated.

This module deliberately does not assign the historical parameter score to
the binding names. That would score aliases instead of the actual angles.
"""

from functools import lru_cache
import sympy as sp


class ExpressionProgram:
    def __init__(self, inputs=(), prefix="_sq"):
        self.inputs = tuple(sorted(set(inputs), key=str))
        names = {str(value) for value in self.inputs}
        while any(name.startswith(prefix) for name in names):
            prefix += "_"
        self.prefix = prefix
        self.bindings = []
        self._cache = {}

    def bind(self, expression, *, real=True):
        expression = sp.sympify(expression)
        if expression.is_Atom:
            return expression
        key = expression, real
        if key not in self._cache:
            symbol = sp.Symbol(self.prefix + str(len(self.bindings)),
                               **({"real": True} if real else {}))
            self._cache[key] = symbol
            self.bindings.append((symbol, expression))
        return self._cache[key]

    def matrix(self, matrix, *, real=True):
        return matrix.applyfunc(lambda value: self.bind(value, real=real))

    def choose(self, first, condition, second):
        return self.bind(sp.Piecewise((first, condition), (second, True), evaluate=False))

    def vector_choice(self, vectors, conditions):
        return sp.Matrix([self.bind(sp.Piecewise(
            *((vector[i], condition) for vector, condition in zip(vectors, conditions)),
            evaluate=False)) for i in range(len(vectors[0]))])

    def encode(self):
        return dict(format="exact_scalar_dag_v1", inputs=[str(value) for value in self.inputs],
                    bindings=[dict(symbol=str(symbol), real=symbol.is_real is True,
                                   expression=str(expression))
                              for symbol, expression in self.bindings])


def decode_program(data, inputs):
    """Reject unknown inputs, forward references, and redefined symbols."""
    inputs = tuple(sorted(set(inputs), key=str))
    if data.get("format") != "exact_scalar_dag_v1" or data.get("inputs") != list(map(str, inputs)):
        raise ValueError("Expression program inputs or format do not match")
    environment = {str(symbol): symbol for symbol in inputs}
    bindings = {}
    for row in data["bindings"]:
        name = row["symbol"]
        if not isinstance(name, str) or not name.isidentifier() or name in environment:
            raise ValueError("Invalid or repeated expression binding")
        if not isinstance(row["real"], bool):
            raise ValueError("Invalid expression domain")
        expression = sp.sympify(row["expression"], locals=environment)
        if expression.atoms(sp.Float) or not expression.free_symbols <= set(environment.values()):
            raise ValueError("Nonexact expression or unbound/forward symbol")
        symbol = sp.Symbol(name, **({"real": True} if row["real"] else {}))
        environment[name] = symbol
        bindings[symbol] = expression
    return environment, bindings


def evaluate_program(data, inputs, outputs, substitutions, *, digits=None):
    """Evaluate only dependencies of the requested outputs.

    Exact evaluation is the default. ``digits`` is a numerical convenience for
    drawings/regressions, not a circuit certificate. Conditions still retain
    their explicit branches; numerical equality uses the requested precision.
    """
    environment, bindings = decode_program(data, inputs)
    substitutions = {environment[str(key)]: sp.sympify(value)
                     for key, value in substitutions.items()}
    if set(substitutions) != set(inputs):
        raise ValueError("Supply every original parameter exactly once")
    if any(value.free_symbols for value in substitutions.values()):
        raise ValueError("Evaluation requires concrete original parameters")

    @lru_cache(maxsize=None)
    def visit(value):
        if value in substitutions:
            return substitutions[value] if digits is None else substitutions[value].evalf(digits)
        if value in bindings:
            result = visit(bindings[value])
            if digits is not None and value.is_real is True:
                if abs(complex(sp.im(result))) >= 10.0 ** (-digits // 2):
                    raise ArithmeticError("Non-real value in a real expression binding")
                result = sp.re(result)
            return result
        if value.is_Atom:
            return value
        if isinstance(value, sp.Piecewise):
            for expression, condition in value.args:
                if truth(condition):
                    return visit(expression)
            raise ArithmeticError("No expression branch applies")
        if value.func in (sp.And, sp.Or, sp.Not) or getattr(value, "is_Relational", False):
            return sp.true if truth(value) else sp.false
        result = value.func(*(visit(argument) for argument in value.args))
        return result if digits is None else result.evalf(digits)

    def truth(condition):
        if condition is sp.true:
            return True
        if condition is sp.false:
            return False
        if condition.func is sp.And:
            return all(truth(argument) for argument in condition.args)
        if condition.func is sp.Or:
            return any(truth(argument) for argument in condition.args)
        if condition.func is sp.Not:
            return not truth(condition.args[0])
        left, right = (visit(argument) for argument in condition.args)
        delta = left - right
        if digits is None:
            delta = sp.simplify(delta)
        elif condition.func in (sp.Eq, sp.Ne):
            # Only the opt-in numerical evaluator has a roundoff threshold.
            zero = abs(complex(delta)) < 10.0 ** (-digits // 2)
            return zero if condition.func is sp.Eq else not zero
        answer = condition.func(delta, 0)
        if answer not in (sp.true, sp.false):
            raise ArithmeticError("Cannot decide exact expression branch: " + str(answer))
        return answer is sp.true

    return tuple(visit(sp.sympify(value, locals=environment)) for value in outputs)


def compile_numeric_program(data, inputs, outputs, *, equality_tolerance=1e-10):
    """Compile a trusted exact DAG for reusable double-precision binding.

    Bindings remain lazy, so unused singular branches are never evaluated.
    This evaluator is an interoperability path, not an identity certificate.
    The program's existing exact verifier supplies the family guarantee.
    """
    import cmath
    import math
    from types import SimpleNamespace
    from sympy.printing.pycode import PythonCodePrinter
    inputs=tuple(sorted(set(inputs),key=str))
    environment,bindings=decode_program(data,inputs)
    indices={symbol:i for i,symbol in enumerate(bindings)}
    arguments={symbol:'_p'+str(i) for i,symbol in enumerate(inputs)}

    class Printer(PythonCodePrinter):
        def _print_Symbol(self,value):
            if value in indices:return '_get('+str(indices[value])+')'
            if value in arguments:return arguments[value]
            raise ValueError('Unknown numerical program symbol')
        def _print_re(self,value):return '('+self._print(value.args[0])+').real'
        def _print_im(self,value):return '('+self._print(value.args[0])+').imag'
        def _print_conjugate(self,value):return '('+self._print(value.args[0])+').conjugate()'
        def _print_Equality(self,value):return '_close('+','.join(map(self._print,value.args))+')'
        def _print_Unequality(self,value):return '(not _close('+','.join(map(self._print,value.args))+'))'
    printer=Printer()
    codes=[compile(printer.doprint(expression),'<exact_scalar_dag>','eval') for expression in bindings.values()]
    output_codes=[compile(printer.doprint(sp.sympify(value,locals=environment)),
                          '<exact_scalar_output>','eval') for value in outputs]
    real_flags=[symbol.is_real is True for symbol in bindings]
    def real(value):
        value=complex(value)
        if abs(value.imag)>1e-7:raise ArithmeticError('Numerical binding left the real chart')
        return value.real
    functions={name:getattr(math,name) for name in dir(math) if not name.startswith('_')}
    functions.update(sqrt=cmath.sqrt,exp=cmath.exp,sin=cmath.sin,cos=cmath.cos,
                     atan2=lambda y,x:math.atan2(real(y),real(x)),floor=lambda x:math.floor(real(x)))
    common=dict(math=SimpleNamespace(**functions),abs=abs,min=min,max=max,
                _close=lambda a,b:abs(a-b)<equality_tolerance)
    def evaluate(substitutions):
        supplied={str(key):value for key,value in substitutions.items()}
        if set(supplied)!=set(map(str,inputs)):raise ValueError('Supply every original parameter exactly once')
        namespace=dict(common,**{arguments[symbol]:real(supplied[str(symbol)]) for symbol in inputs})
        cache={}
        def get(index):
            if index not in cache:
                value=eval(codes[index],{'__builtins__':{}},namespace)
                cache[index]=real(value) if real_flags[index] else value
            return cache[index]
        namespace['_get']=get
        return tuple(eval(code,{'__builtins__':{}},namespace) for code in output_codes)
    return evaluate
