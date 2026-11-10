import copy

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_matchgate_chart import synthesize_matchgate_chart, verify_matchgate_certificate, verify_matchgate_chart_lemma


@pytest.mark.parametrize("basis_name",["berkeley_b","sqrt_iswap_u","google_sqrt_iswap"])
@pytest.mark.parametrize("exact",[False,True])
def test_free_czs_matchgate_chart_has_six_locals_and_replay(basis_name,exact):
    parameters=sp.symbols("theta phi gamma",real=True)
    target=gates.make_gate("CZS",*parameters)
    basis=get_basis(basis_name)
    result=synthesize_matchgate_chart(target,basis,phase_exact=exact)
    assert verify_matchgate_chart_lemma()
    assert result["cost"]["entanglers"]==2
    assert result["cost"]["single_qubit_gates"]==6
    assert verify_matchgate_certificate(target,result,basis)
    changed=copy.deepcopy(result)
    changed["matchgate_certificate"]["odd_su2_parts"][0]="1"
    assert not verify_matchgate_certificate(target,changed,basis)
    symbols=dict(zip(map(str,parameters),parameters))
    # These numerical checks exercise executable branch expressions, after
    # the unrestricted exact polar/native lemmas have proved the family.
    for values in ((0,0,0),(sp.pi/2,0,0),(0,0,sp.pi),
                   (sp.pi/3,sp.pi/5,sp.pi/7),(-sp.pi/2,sp.pi,-sp.pi)):
        sub=dict(zip(parameters,values))
        actual=np.eye(4,dtype=complex)
        for op in result["operations"]:
            angles=[sp.sympify(p,locals=symbols).subs(sub).evalf(22) for p in op["parameters"]]
            assert not any(p.has(sp.nan,sp.zoo) for p in angles)
            matrix=gates.apply_to(sp.eye(4),2,gates.make_gate(op["gate"],*angles),tuple(op["wires"]))
            actual=np.asarray(matrix.evalf(22),complex)@actual
        phase=1 if exact else complex(sp.sympify(result["virtual_gp"]["factor"],locals=symbols).subs(sub).evalf(22))
        assert np.max(np.abs(phase*actual-np.asarray(target.subs(sub).evalf(22),complex)))<5e-13


def test_chart_rejects_a_determinant_mismatch_and_nonunitary_perturbation():
    basis=get_basis("sqrt_iswap_u")
    assert synthesize_matchgate_chart(gates.make_gate("CP",sp.pi/7),basis,phase_exact=False) is None
    target=sp.Matrix(gates.make_gate("CZS",sp.pi/3,sp.pi/5,sp.pi/7))
    target[1,1]+=sp.Rational(1,10**20)
    assert synthesize_matchgate_chart(target,basis,phase_exact=False) is None
