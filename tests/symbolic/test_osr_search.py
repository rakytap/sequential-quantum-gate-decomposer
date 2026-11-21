import sympy as sp
import pytest

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_identity import prove_circuit_identity,exponential_forms
from symbolic_osr_search import osr_source_candidates,interaction_proposals
from symbolic_clifford_two_qubit import coset_search,clifford_source_candidate
from symbolic_synthesis import synthesize_matrix,synthesize


def test_shared_exponential_context_rejects_unrepresented_denominators():
    theta=sp.Symbol("theta",real=True)
    _,_,convert=exponential_forms((sp.exp(sp.I*theta),),return_converter=True)
    with pytest.raises(ValueError):convert(sp.exp(sp.I*theta/2))


@pytest.mark.parametrize("gate,basis",[("CP","zz_u"),("CRX","xx_rz_ry"),("CRY","zx_u")])
def test_osr_derives_a_native_pulse_from_an_opaque_parameterized_matrix(gate,basis):
    theta=sp.Symbol("theta",real=True)
    target=gates.make_gate(gate,theta)
    candidates,search=osr_source_candidates(target,get_basis(basis),seconds=15)
    assert candidates,search
    candidate=candidates[0]
    assert sum(len(op[-1])==2 for op in candidate["sequence"])==1
    assert prove_circuit_identity(target,candidate["sequence"],candidate["factor"])


def test_osr_peeling_separates_a_three_qubit_chain_without_a_gate_name():
    target=gates.compile_gates(3,[("CX",(0,1)),("CX",(1,2))])
    candidates,search=osr_source_candidates(target,get_basis("cx_u"),seconds=15)
    assert candidates,search
    candidate=candidates[0]
    assert sum(len(op[-1])==2 for op in candidate["sequence"])==2
    assert prove_circuit_identity(target,candidate["sequence"],candidate["factor"])


@pytest.mark.parametrize("entangler",["CX","CZ","iSWAP"])
def test_native_clifford_search_is_closed_over_twenty_local_cosets(entangler):
    result=coset_search(entangler)
    assert result["reachable_cosets"]==20
    assert max(result["distances"].values())==3


@pytest.mark.parametrize("gate,count",[("CX",2),("DCX",1),("SWAP",3),("iSWAP",1)])
def test_iswap_circuits_are_synthesized_as_whole_operators(gate,count):
    target=gates.make_gate(gate)
    candidate=clifford_source_candidate(target,get_basis("iswap_u"))
    assert candidate["discrete_minimum"]==count
    assert sum(len(op[-1])==2 for op in candidate["sequence"])==count
    assert prove_circuit_identity(target,candidate["sequence"],candidate["factor"])
    assert candidate["proof_scope"].endswith("Clifford_local_frames_only")


def test_opaque_three_qubit_chain_has_a_matching_global_osr_bound():
    target=gates.compile_gates(3,[("CX",(0,1)),("CX",(1,2))])
    result=synthesize_matrix(target,"cx_u",seconds=10)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==2 and result["entangler_optimal"]
    assert result["osr_certificate"]["lower_bound"]==2


def test_parameterized_native_optimality_keeps_its_special_angle_condition():
    theta=sp.Symbol("theta",real=True)
    result=synthesize_matrix(gates.make_gate("CP",theta),"zz_u",phase_exact=False,seconds=10)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==1
    assert not result["entangler_optimal"]
    assert result["conditional_entangler_optimality"]["condition"]!="True"


def test_native_coset_solution_beats_independent_cnot_rebasing():
    result=synthesize("SWAP",(),"iswap_u")
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==3
    assert result["construction_method"].startswith("native_Clifford")
    assert result["entangler_optimal"]
    assert result["entangler_optimality_certificate"]["lower_bound"]==3
