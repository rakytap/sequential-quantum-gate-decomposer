import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_identity import prove_circuit_identity
from symbolic_native_fixed import synthesize_fixed_native
from symbolic_bases import get_basis
from symbolic_sqrt_iswap_face import _face_angles,_face_word,verify_face_lemma,verify_syc_face_motif


@pytest.mark.parametrize("y,z",[(0,0),(sp.pi/4,0),(0,sp.pi/4),(-sp.pi/4,0)])
def test_face_endpoints_have_exact_finite_rotation_charts(y,z):
    assert verify_face_lemma()
    angles=_face_angles(y,z)
    assert not any(angle.has(sp.nan,sp.zoo) for angle in angles)
    assert prove_circuit_identity(gates.make_gate("Canonical",sp.pi/4,y,z),_face_word(*angles))


def test_face_constructor_rejects_an_unproved_domain():
    with pytest.raises(ValueError):_face_angles(sp.pi/3,0)
    with pytest.raises(ValueError):_face_angles(sp.Symbol("y",real=True),0)


@pytest.mark.parametrize("basis_name",["sqrt_iswap_u","google_sqrt_iswap"])
@pytest.mark.parametrize("exact",[False,True])
def test_syc_native_chart_executes_the_certified_angles(basis_name,exact):
    assert verify_syc_face_motif()
    result=synthesize_fixed_native(gates.make_gate("SYC"),get_basis(basis_name),phase_exact=exact)
    actual=np.eye(4,dtype=complex)
    for op in result["operations"]:
        angles=[sp.sympify(value).evalf(22) for value in op["parameters"]]
        matrix=gates.apply_to(sp.eye(4),2,gates.make_gate(op["gate"],*angles),tuple(op["wires"]))
        actual=np.asarray(matrix.evalf(22),complex)@actual
    phase=1 if exact else complex(sp.sympify(result["virtual_gp"]["factor"]).evalf(22))
    assert np.max(np.abs(phase*actual-np.asarray(gates.make_gate("SYC").evalf(22),complex)))<5e-13
    assert result["entangler_optimal"]
