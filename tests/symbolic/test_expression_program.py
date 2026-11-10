import copy

import pytest
import sympy as sp

from symbolic_expression_program import ExpressionProgram, decode_program, evaluate_program


def test_lazy_shared_expressions_have_no_extra_parameters_or_eager_pole():
    x = sp.Symbol('x', real=True)
    program = ExpressionProgram((x,))
    reciprocal = program.bind(1/x)
    safe = program.choose(7, sp.Eq(x, 0, evaluate=False), reciprocal)
    assert program.bind(1/x) == reciprocal
    encoded = program.encode()
    assert evaluate_program(encoded, (x,), [safe], {x: 0}) == (7,)
    assert evaluate_program(encoded, (x,), [safe], {x: sp.Rational(2, 3)}) == (sp.Rational(3, 2),)
    assert encoded['inputs'] == ['x']
    for mutation in ('forward', 'redefined', 'float', 'input'):
        changed = copy.deepcopy(encoded)
        if mutation == 'forward': changed['bindings'][0]['expression'] = 'unbound+1'
        elif mutation == 'redefined': changed['bindings'][1]['symbol'] = changed['bindings'][0]['symbol']
        elif mutation == 'float': changed['bindings'][0]['expression'] = '0.125'
        else: changed['inputs'] = ['y']
        with pytest.raises(ValueError):
            decode_program(changed, (x,))


def test_original_binding_names_cannot_capture_user_parameters():
    x = sp.Symbol('_sq0', real=True)
    program = ExpressionProgram((x,))
    answer = program.bind(2*x)
    assert answer != x
    assert evaluate_program(program.encode(), (x,), [answer], {x: 3}) == (6,)
    with pytest.raises(ValueError):
        evaluate_program(program.encode(), (x,), [answer], {})
