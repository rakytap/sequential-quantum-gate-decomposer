import copy

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_sqrt_iswap_phase import (synthesize_controlled_phase,
    verify_controlled_phase_certificate,verify_phase_chart_lemma)


@pytest.mark.parametrize("basis_name",["sqrt_iswap_u","google_sqrt_iswap"])
@pytest.mark.parametrize("exact",[False,True])
def test_free_real_controlled_phase_chart_is_proved_and_replayable(basis_name,exact):
    theta=sp.Symbol("theta",real=True)
    basis=get_basis(basis_name)
    result=synthesize_controlled_phase(theta,basis,phase_exact=exact)
    assert verify_phase_chart_lemma()
    assert result["cost"]["entanglers"]==2
    assert result["cost"]["single_qubit_gates"]==(6 if exact else 5)
    assert verify_controlled_phase_certificate(gates.make_gate("CP",theta),result,basis)
    changed=copy.deepcopy(result)
    changed["operations"][0]["parameters"][0]="0"
    assert not verify_controlled_phase_certificate(gates.make_gate("CP",theta),changed,basis)
    assert not verify_controlled_phase_certificate(gates.make_gate("CP",theta+sp.Rational(1,100)),result,basis)
    # These check specialization/serialization. The all-angle polynomial
    # and chart proof above, rather than these samples, certify the family.
    for angle in (0,sp.pi,-sp.pi,2*sp.pi,-3*sp.pi,sp.pi/3,sp.Rational(-9,4)):
        operations=tuple((op["gate"],*(sp.sympify(p,locals={"theta":theta}).subs(theta,angle)
                                     for p in op["parameters"]),tuple(op["wires"])) for op in result["operations"])
        actual=np.eye(4,dtype=complex)
        for op in operations:
            primitive=gates.make_gate(op[0],*(p.evalf(22) for p in op[1:-1]))
            embedded=gates.apply_to(sp.eye(4),2,primitive,op[-1])
            actual=np.asarray(embedded.evalf(22),complex)@actual
        phase=1 if exact else complex(sp.sympify(result["virtual_gp"]["factor"],locals={"theta":theta}).subs(theta,angle).evalf(22))
        expected=np.asarray(gates.make_gate("CP",angle).evalf(22),complex)
        assert np.max(np.abs(phase*actual-expected))<5e-14


def test_nonreal_angle_and_unrelated_native_basis_are_rejected():
    with pytest.raises(ValueError):
        synthesize_controlled_phase(sp.I,get_basis("sqrt_iswap_u"),phase_exact=False)
    assert synthesize_controlled_phase(sp.Symbol("theta",real=True),get_basis("cx_u"),phase_exact=True) is None


def test_optional_search_deadline_preserves_an_already_proved_incumbent(monkeypatch):
    import symbolic_synthesis as synthesis
    from symbolic_archive import SymbolicDeadline
    def expired(*args,**kwargs):raise SymbolicDeadline()
    monkeypatch.setattr(synthesis,"_synthesize_existing",expired)
    theta=sp.Symbol("theta",real=True)
    result=synthesis.synthesize("CP",(theta,),"sqrt_iswap_u",phase_exact=True)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==2
    assert verify_controlled_phase_certificate(gates.make_gate("CP",theta),result,get_basis("sqrt_iswap_u"))
