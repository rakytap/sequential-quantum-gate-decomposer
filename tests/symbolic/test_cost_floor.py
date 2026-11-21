import copy
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_contract import verify_circuit
from symbolic_cost_proof import secondary_score_floor_certificate


def test_free_family_single_native_gate_meets_global_score_floor():
    theta=sp.Symbol('theta',real=True)
    basis=get_basis('cx_rz_ry')
    target=gates.make_gate('Rz',theta)
    result=verify_circuit(target,(('Rz',theta,(0,)),),basis,phase_exact=True)
    certificate=secondary_score_floor_certificate(target,result,basis)
    assert certificate['global_lexicographic_minimum']
    assert certificate['score_lower_bound']=='1001/500'
    assert certificate['scope']=='whole_symbolic_family'


def test_wire_support_and_literal_scalar_are_distinct_lower_bounds():
    basis=get_basis('clifford_only')
    word=(('X',(0,)),('Z',(1,)))
    # This supplied custom alphabet is fixed, so the score floor is exact.
    from symbolic_bases import make_basis
    basis=make_basis('pauli',('X','Z'),'CZ')
    target=gates.compile_gates(2,word)
    result=verify_circuit(target,word,basis,phase_exact=True)
    certificate=secondary_score_floor_certificate(target,result,basis)
    assert certificate['local_count_lower_bound']==2
    assert certificate['global_lexicographic_minimum']


def test_native_pulse_floor_needs_its_separate_entangler_proof():
    basis=get_basis('cx_u');target=gates.make_gate('CX')
    result=verify_circuit(target,(('CX',(0,1)),),basis,phase_exact=True)
    certificate=secondary_score_floor_certificate(target,result,basis)
    assert certificate['secondary_minimum_at_fixed_entangler_count']
    assert not certificate['global_lexicographic_minimum']
    result['entangler_optimal']=True
    assert secondary_score_floor_certificate(target,result,basis)['global_lexicographic_minimum']
