import copy

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_archive import bounded_synthesis
from symbolic_bases import get_basis
from symbolic_rotation_vector import verify_rotation_vector_certificate,verify_rotation_vector_lemma


@pytest.mark.parametrize("basis",["cx_u","cx_rz_ry","ibm_cx","ionq_aria","iqm_cz","cx_h_phase"])
@pytest.mark.parametrize("exact",[False,True])
def test_rotation_vector_preserves_three_free_real_parameters_and_replayable_proof(basis,exact):
    vector=sp.symbols("vx vy vz",real=True)
    result=bounded_synthesis("RV",vector,basis,phase_exact=exact,seconds=60)
    assert result["status"]=="verified_symbolic",result
    assert result["cost"]["entanglers"]==0 and result["entangler_optimal"]
    assert not result["local_cost_optimal"]
    assert verify_rotation_vector_certificate(gates.make_gate("RV",*vector),result,get_basis(basis))
    if basis=="cx_u":assert result["cost"]["single_qubit_gates"]==(2 if exact else 1)
    changed=copy.deepcopy(result)
    previous=changed["operations"][0]["parameters"][0]
    changed["operations"][0]["parameters"][0]="("+previous+")+pi"
    assert not verify_rotation_vector_certificate(gates.make_gate("RV",*vector),changed,get_basis(basis))


@pytest.mark.parametrize("binding",[(0,0,0),(0,0,sp.pi),(sp.pi,0,0),(0,sp.pi,0),
                                    (0,0,-2*sp.pi),(sp.pi/2,sp.pi/3,-sp.pi/4)])
def test_rotation_vector_chart_specializations_include_both_zero_amplitude_branches(binding):
    vector=sp.symbols("vx vy vz",real=True)
    result=bounded_synthesis("RV",vector,"cx_u",phase_exact=True,seconds=60)
    substitution=dict(zip(vector,binding))
    operations=tuple((op["gate"],*(sp.sympify(value,locals=dict(zip(("vx","vy","vz"),vector))).subs(substitution)
                                   for value in op["parameters"]),tuple(op["wires"])) for op in result["operations"])
    assert not any(sp.sympify(value).has(sp.nan,sp.zoo) for op in operations for value in op[1:-1])
    actual=gates.compile_gates(1,operations)
    target=gates.make_gate("RV",*binding)
    error=np.max(np.abs(np.asarray((actual-target).evalf(25),dtype=complex)))
    assert error<1e-20


def test_rotation_vector_lemma_and_discrete_obstruction_are_distinct():
    assert verify_rotation_vector_lemma()
    vector=sp.symbols("vx vy vz",real=True)
    result=bounded_synthesis("RV",vector,"clifford_t_cx",phase_exact=True,seconds=30)
    assert result["status"]=="proved_impossible"
    assert result["obstruction"]=="discrete_alphabet_nonconstant_family"
