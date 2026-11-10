import sympy as sp
import pytest
from symbolic_gates import make_gate,compile_gates,make_controlled,apply_to
from symbolic_identity import prove_circuit_identity

def test_one_tk2_dcx_is_literal():
    from symbolic_native_echoes import native_dcx_source
    word,phase=native_dcx_source('TK2')
    assert sum(len(op[-1])==2 for op in word)==1
    assert prove_circuit_identity(make_gate('DCX'),word,phase) is True


def test_two_sycamore_polynomial_axis_lemma():
    from symbolic_sycamore_phase import verify_sycamore_phase_lemma
    assert verify_sycamore_phase_lemma()


def test_sqiSW_polynomial_elimination_and_known_classes():
    from symbolic_exchange_invariants import sqrt_iswap_two_pulse_polynomial_condition
    a,b,W=sp.symbols('a b W',real=True)
    assert sp.expand((W**2-8*a)**2-64*(a*a+b*b))==W**4-16*a*W**2-64*b*b
    # Local, CNOT, SQiSW and SWAP classes, respectively.
    for g1,g2,expected in [(1,3,True),(0,1,True),(sp.Rational(1,4),1,True),(-1,-3,False)]:
        assert sqrt_iswap_two_pulse_polynomial_condition(g1,g2)==expected


@pytest.mark.parametrize('angle',[0,sp.pi,sp.pi/2,-sp.pi/2,sp.pi/4,-sp.pi/4,7*sp.pi,sp.Rational(2,7)])
def test_sycamore_phase_branches_and_signs_replay_in_actual_catalog_convention(angle):
    import numpy as np
    from symbolic_sycamore_phase import sycamore_phase_source
    word,factor=sycamore_phase_source(angle);actual=np.eye(4,dtype=complex)
    assert sum(op[0]=='SYC' for op in word)==2
    for op in word:
        small=np.array(make_gate(op[0],*op[1:-1]).evalf(18),complex)
        full=small if len(op[-1])==2 else np.kron(np.eye(2),small) if op[-1]==(0,) else np.kron(small,np.eye(2))
        actual=full@actual
    assert np.max(np.abs(actual-np.array(make_gate('CP',angle).evalf(18),complex)))<1e-12


def test_sycamore_certificate_rejects_changed_native_angle():
    from symbolic_bases import get_basis
    from symbolic_controlled_chart import synthesize_controlled_chart,verify_controlled_chart_certificate
    import copy
    target=make_gate('CS');basis=get_basis('google_syc')
    result=synthesize_controlled_chart(target,basis,phase_exact=True)
    assert result['cost']['entanglers']==2
    assert verify_controlled_chart_certificate(target,result,basis)
    changed=copy.deepcopy(result)
    changed['operations'][0]['parameters'][0]='1/7'
    assert not verify_controlled_chart_certificate(target,changed,basis)


def test_entire_trace_two_family_has_an_exact_eight_cnot_word():
    from symbolic_reflection_involution import reflection,controlled_reflection_source
    t=sp.Symbol('reflection_angle',real=True)
    word=controlled_reflection_source(t)
    assert sum(len(op[-1])==2 for op in word)==8
    assert prove_circuit_identity(make_controlled(reflection(t),2),word) is True


def test_trace_two_upper_bound_with_six_arbitrary_two_qubit_gates():
    from symbolic_reflection_involution import reflection,six_arbitrary_matrices
    t=sp.Symbol('reflection_angle',real=True);actual=sp.eye(8)
    for matrix,wires in six_arbitrary_matrices(t):actual=apply_to(actual,3,matrix,wires)
    target=make_controlled(reflection(t),2)
    from symbolic_controlled_chart import _zero
    assert _zero(actual-target)


def test_reflection_rank_strata_and_fredkin_endpoint():
    from symbolic_reflection_involution import reflection,reflection_schmidt_minor,singlet_frame
    t=sp.Symbol('reflection_angle',real=True)
    k=sp.sin(2*t)
    assert sp.trigsimp(sp.expand_trig(reflection_schmidt_minor(t)+k*k*(2-k*k)))==0
    L=singlet_frame();target=L*reflection(sp.pi/4)*L.H
    assert target==make_gate('SWAP')
    assert sp.trace(reflection(0))==2
