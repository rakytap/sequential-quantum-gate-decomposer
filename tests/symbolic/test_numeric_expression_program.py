import sympy as sp
import pytest
from symbolic_expression_program import ExpressionProgram,compile_numeric_program


def test_numeric_dag_is_lazy_at_an_unused_singular_branch():
    x=sp.Symbol('x',real=True)
    p=ExpressionProgram((x,))
    inverse=p.bind(1/x)
    result=p.choose(7,sp.Eq(x,0),inverse)
    evaluate=compile_numeric_program(p.encode(),(x,),(str(result),))
    assert evaluate({x:0})==(7,)
    assert evaluate({x:2})==(.5,)
    with pytest.raises(ValueError):evaluate({})


def test_numeric_dag_keeps_complex_scalar_and_real_outputs_separate():
    x=sp.Symbol('x',real=True)
    p=ExpressionProgram((x,))
    scalar=p.bind(sp.exp(sp.I*x),real=False)
    angle=p.bind(sp.atan2(sp.im(scalar),sp.re(scalar)))
    evaluate=compile_numeric_program(p.encode(),(x,),(str(scalar),str(angle)))
    a,b=evaluate({x:.3})
    assert complex(a)==pytest.approx(complex(sp.exp(.3*sp.I)))
    assert b==pytest.approx(.3)
