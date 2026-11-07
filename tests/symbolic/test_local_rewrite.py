import copy

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_contract import verify_circuit
from symbolic_local_rewrite import optimize_local_result, verify_local_rewrite_certificate


@pytest.mark.parametrize("exact",[False,True])
def test_exact_local_block_rewrite_tracks_scalar_and_replays_its_proof(exact):
    basis=get_basis("cx_u")
    word=(("U3",sp.pi/2,0,sp.pi,(0,)),("U3",0,0,sp.pi/4,(0,)),
          ("U3",sp.pi/2,0,sp.pi,(0,)),("U3",0,0,sp.pi/3,(0,)))
    target=gates.compile_gates(1,word)
    parent=verify_circuit(target,word,basis,phase_exact=exact,scalar_factor=1)
    result=optimize_local_result(target,parent,basis)
    assert result["cost"]["single_qubit_gates"]<4
    assert verify_local_rewrite_certificate(target,result,basis)
    assert result["virtual_gp"] is None if exact else result["virtual_gp"]["physical"] is False
    tampered=copy.deepcopy(result)
    tampered["operations"][0]["parameters"][0]="pi/13"
    assert not verify_local_rewrite_certificate(target,tampered,basis)
    wrong=sp.Matrix(target);wrong[0,0]+=sp.Rational(1,10**20)
    assert not verify_local_rewrite_certificate(wrong,result,basis)


def test_disjoint_entanglers_do_not_block_one_wire_fusion_or_change_wire_order():
    basis=get_basis("cx_u")
    word=(("U3",sp.pi/2,0,sp.pi,(2,)),("CX",(0,1)),
          ("U3",0,0,sp.pi/4,(2,)),("CX",(1,0)),
          ("U3",sp.pi/2,0,sp.pi,(2,)),("U3",0,0,sp.pi/3,(2,)))
    target=gates.compile_gates(3,word)
    parent=verify_circuit(target,word,basis,phase_exact=False,scalar_factor=1)
    result=optimize_local_result(target,parent,basis)
    assert result["cost"]["entanglers"]==2
    assert result["cost"]["single_qubit_gates"]==1
    assert verify_local_rewrite_certificate(target,result,basis)
    assert [op["wires"] for op in result["operations"] if len(op["wires"])==2]==[[0,1],[1,0]]


def test_unproved_parent_and_physical_gp_are_rejected():
    basis=get_basis("cx_u")
    parent=dict(status="verified_symbolic",phase_exact=True,operations=[],virtual_gp=None)
    with pytest.raises(ValueError,match="parent identity"):
        optimize_local_result(gates.make_gate("X"),parent,basis)


def test_single_full_turn_is_removed_projectively_and_its_literal_scalar_is_charged():
    basis=get_basis("cx_rz_ry")
    word=(("Rz",-2*sp.pi,(0,)),)
    target=-sp.eye(2)
    parent=verify_circuit(target,word,basis,phase_exact=False,scalar_factor=1)
    result=optimize_local_result(target,parent,basis)
    assert result["operations"]==[] and result["virtual_gp"]["factor"]=="-1"
    assert verify_local_rewrite_certificate(target,result,basis)
    literal=verify_circuit(target,word,basis,phase_exact=True)
    exact=optimize_local_result(target,literal,basis)
    assert exact["cost"]["single_qubit_gates"]==1
    assert exact["virtual_gp"] is None


@pytest.mark.parametrize("exact",[False,True])
def test_phase_flips_around_a_nonclifford_native_rotation_need_one_local(exact):
    basis=get_basis("cz_rz_rx")
    word=(("Rz",-sp.pi,(0,)),("Rx",sp.pi/4,(0,)),("Rz",-sp.pi,(0,)))
    target=gates.compile_gates(1,word)
    parent=verify_circuit(target,word,basis,phase_exact=exact,scalar_factor=1)
    result=optimize_local_result(target,parent,basis)
    assert result["cost"]["single_qubit_gates"]==1
    assert result["operations"][0]["gate"]=="Rx"
    assert verify_local_rewrite_certificate(target,result,basis)
