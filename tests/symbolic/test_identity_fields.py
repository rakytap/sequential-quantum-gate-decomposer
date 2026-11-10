import sympy as sp

import symbolic_gates as gates
from symbolic_identity import _cyclotomic_values,prove_circuit_identity


def test_exact_field_multiplication_keeps_operand_order_and_literal_phase():
    word=(("H",(0,)),("CX",(0,2)),("Ry",sp.pi/8,(1,)),
          ("CX",(1,0)),("Rz",sp.pi/4,(2,)))
    target=gates.compile_gates(3,word)
    assert prove_circuit_identity(target,word)
    assert not prove_circuit_identity(target,word,sp.I)
    reversed_control=word[:1]+(("CX",(2,0)),)+word[2:]
    assert not prove_circuit_identity(target,reversed_control)
    changed=target.copy()
    changed[0,0]+=sp.Rational(1,10**30)
    assert not prove_circuit_identity(changed,word)


def test_cyclotomic_constants_are_exact_and_unsupported_constants_decline():
    entries=(sp.sqrt(2),sp.sqrt(3),sp.sqrt(2+sp.sqrt(2)),sp.sqrt(2-sp.sqrt(2)),
             sp.exp(sp.I*sp.pi/16),1/sp.sqrt(2))
    domain,values=_cyclotomic_values(entries)
    assert values[sp.sqrt(2)]**2==domain.convert(2)
    assert values[sp.sqrt(3)]**2==domain.convert(3)
    assert values[entries[2]]**2==domain.convert(2)+values[sp.sqrt(2)]
    assert values[entries[3]]**2==domain.convert(2)-values[sp.sqrt(2)]
    assert values[entries[4]]**16==domain.convert(-1)
    assert values[entries[5]]*values[sp.sqrt(2)]==domain.one
    assert _cyclotomic_values((sp.pi,)) is None
