import copy

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_controlled_chart import (synthesize_controlled_chart,
    verify_controlled_chart_certificate,verify_controlled_chart_lemma)


@pytest.mark.parametrize("basis_name",["berkeley_b","sqrt_iswap_u","google_sqrt_iswap"])
@pytest.mark.parametrize("exact",[False,True])
def test_controlled_su2_chart_keeps_a_full_cu_family_and_phase_contract(basis_name,exact):
    parameters=sp.symbols("theta phi lbda gamma",real=True)
    target=gates.make_gate("CU",*parameters)
    basis=get_basis(basis_name)
    result=synthesize_controlled_chart(target,basis,phase_exact=exact)
    assert verify_controlled_chart_lemma()
    assert result["cost"]["entanglers"]==2
    assert verify_controlled_chart_certificate(target,result,basis)
    legacy=copy.deepcopy(result)
    legacy["controlled_chart_certificate"].pop("additional_frames",None)
    assert verify_controlled_chart_certificate(target,legacy,basis)
    changed=copy.deepcopy(result)
    previous=changed["operations"][0]["parameters"][0]
    changed["operations"][0]["parameters"][0]="("+previous+")+pi"
    assert not verify_controlled_chart_certificate(target,changed,basis)
    changed=copy.deepcopy(result)
    changed["local_cost_optimal"]=True
    assert not verify_controlled_chart_certificate(target,changed,basis)
    symbols=dict(zip(map(str,parameters),parameters))
    for values in ((0,0,0,0),(0,sp.pi,sp.pi,0),(sp.pi,0,0,0),
                   (sp.pi/3,-sp.pi/5,sp.pi/7,sp.pi/11),
                   (sp.Rational(-7,3),sp.Rational(5,4),sp.Rational(-4,3),sp.Rational(2,5))):
        sub=dict(zip(parameters,values))
        actual=np.eye(4,dtype=complex)
        for op in result["operations"]:
            angles=[sp.sympify(p,locals=symbols).subs(sub).evalf(22) for p in op["parameters"]]
            assert not any(p.has(sp.nan,sp.zoo) for p in angles)
            matrix=gates.apply_to(sp.eye(4),2,gates.make_gate(op["gate"],*angles),tuple(op["wires"]))
            actual=np.asarray(matrix.evalf(22),complex)@actual
        phase=1 if exact else complex(sp.sympify(result["virtual_gp"]["factor"],locals=symbols).subs(sub).evalf(22))
        expected=np.asarray(target.subs(sub).evalf(22),complex)
        assert np.max(np.abs(phase*actual-expected))<5e-13


def test_nonunitary_controlled_blocks_and_osr_four_target_are_not_certified():
    basis=get_basis("sqrt_iswap_u")
    target=sp.Matrix(gates.make_gate("CP",sp.pi/7));target[0,0]+=sp.Rational(1,10**20)
    assert synthesize_controlled_chart(target,basis,phase_exact=False) is None
    assert synthesize_controlled_chart(gates.make_gate("SWAP"),basis,phase_exact=False) is None


def test_continuous_native_ms_control_axis_is_proved_before_it_is_used():
    phi0,phi1,theta=sp.symbols("phi0 phi1 theta",real=True)
    target=gates.make_gate("IonQMS",phi0,phi1,theta)
    basis=get_basis("sqrt_iswap_u")
    frames=((("H",),("Rz",2*sp.pi*phi0)),)
    result=synthesize_controlled_chart(target,basis,phase_exact=False,additional_frames=frames)
    assert result["cost"]["entanglers"]==2
    assert verify_controlled_chart_certificate(target,result,basis)
    changed=copy.deepcopy(result)
    changed["controlled_chart_certificate"]["additional_frames"][0][1][1]="pi/7"
    assert not verify_controlled_chart_certificate(target,changed,basis)


def test_opaque_matrix_entry_point_uses_the_native_osr_chart():
    from symbolic_synthesis import synthesize_matrix
    theta=sp.Symbol("theta",real=True)
    target=gates.make_gate("CRX",theta)
    result=synthesize_matrix(target,"sqrt_iswap_u",phase_exact=False,seconds=1)
    assert result["status"]=="verified_symbolic" and result["cost"]["entanglers"]==2
    assert verify_controlled_chart_certificate(target,result,get_basis("sqrt_iswap_u"))
