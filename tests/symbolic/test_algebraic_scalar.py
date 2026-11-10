import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_contract import verify_circuit
from symbolic_identity import infer_algebraic_circuit_scalar


def test_scalar_inference_checks_all_entries_and_unit_norm_in_the_number_field():
    word=(("U3",sp.pi/8,sp.pi/4,-sp.pi/4,(0,)),("CX",(0,2)),("U3",-sp.pi/8,0,0,(2,)))
    actual=gates.compile_gates(3,word)
    phase=sp.exp(sp.I*sp.pi/8)
    assert sp.simplify(infer_algebraic_circuit_scalar(phase*actual,word)-phase)==0
    wrong=sp.MutableDenseMatrix(phase*actual)
    wrong[7,7]+=sp.Rational(1,10**20)
    assert infer_algebraic_circuit_scalar(wrong,word) is False
    assert infer_algebraic_circuit_scalar(2*actual,word) is False


def test_fixed_projective_verification_does_not_build_a_generic_symbolic_matrix(monkeypatch):
    import symbolic_contract as contract
    def forbidden(*args,**kwargs):pytest.fail("generic compile_gates should not run")
    monkeypatch.setattr(contract,"compile_gates",forbidden)
    result=verify_circuit(gates.make_gate("X"),(("U3",sp.pi,0,sp.pi,(0,)),),get_basis("cx_u"),phase_exact=False)
    assert result["status"]=="verified_symbolic"
    assert result["virtual_gp"]["factor"]=="1"


def test_free_angle_declines_fixed_field_scalar_inference():
    theta=sp.Symbol("theta",real=True)
    assert infer_algebraic_circuit_scalar(gates.make_gate("Rx",theta),(("U3",theta,-sp.pi/2,sp.pi/2,(0,)),)) is None


def test_normalized_dyadic_radicals_share_the_same_exact_number_field():
    from symbolic_identity import _cyclotomic_values
    entries=(sp.sqrt(sp.Rational(1,2)+sp.sqrt(2)/4),
             sp.sqrt(sp.Rational(1,2)-sp.sqrt(2)/4),sp.sqrt(6))
    domain,values=_cyclotomic_values(entries)
    assert values[entries[0]]==domain.from_sympy(sp.cos(sp.pi/8))
    assert values[entries[1]]==domain.from_sympy(sp.sin(sp.pi/8))
    assert values[entries[2]]**2==domain.convert(6)
