import copy
import json

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import BASIS_SETS, get_basis
from symbolic_entangler_proof import (
    native_profile, small_count_certificate, verify_small_count_certificate,
    verify_two_pulse_trace_lemmas,
)


def test_two_pulse_trace_and_every_native_profile_are_exact():
    assert verify_two_pulse_trace_lemmas()
    for name in dict.fromkeys(b.entangler for b in BASIS_SETS.values()):
        assert 2 <= native_profile(name)['crossing_rank_capacity'] <= 4
    assert native_profile('fSim')['spin_minimal_polynomial_degree_cap'] == 3


@pytest.mark.parametrize('basis', ['cx_u','cz_u','iswap_u','xx_rz_ry','zz_u',
                                    'zx_u','ionq_aria','ionq_forte'])
def test_three_pulse_exclusion_covers_fixed_and_variable_native_rotations(basis):
    target = gates.make_gate('Canonical', sp.pi/8, sp.pi/8, sp.pi/8)
    proof = small_count_certificate(target, get_basis(basis), 3)
    assert proof['lower_bound'] == 3
    assert proof['scope'] == 'fixed_target_minimum'
    assert proof['witness']['evidence'][-1]['method'] == 'real_two_pulse_spin_trace'
    assert verify_small_count_certificate(target, get_basis(basis), proof)
    assert verify_small_count_certificate(target, get_basis(basis), json.loads(json.dumps(proof)))


def test_two_pulse_exclusion_does_not_exclude_one_native_pulse():
    # SYC and sqrt-SWAP need not cancel with a second identical native pulse.
    # Their violation of a TWO-pulse invariant is not a three-pulse bound.
    for name, basis in [('SYC','google_syc'), ('SSWAP','sqrt_swap_u')]:
        target = gates.make_gate(name)
        proof = small_count_certificate(target, get_basis(basis), 1)
        assert proof is not None and proof['lower_bound'] == 1
        assert small_count_certificate(target, get_basis(basis), 3) is None


def test_unequal_native_class_proves_two_sqrt_swap_pulses_for_cnot():
    proof = small_count_certificate(gates.make_gate('CX'), get_basis('sqrt_swap_u'), 2)
    assert proof['lower_bound'] == 2
    assert any(e['method'] == 'unequal_Makhlin_invariant' for e in proof['witness']['evidence'])


def test_one_fsim_cannot_have_four_distinct_spin_eigenvalues():
    target = gates.make_gate('B')
    proof = small_count_certificate(target, get_basis('google_fsim'), 2)
    assert proof is not None
    assert any(e['method'] == 'spin_Krylov_degree_exceeds_native_cap'
               for e in proof['witness']['evidence'])


def test_family_witness_proves_uniform_budget_without_pointwise_claim():
    a,b,c = sp.symbols('a b c', real=True)
    target = gates.make_gate('Canonical', a,b,c)
    proof = small_count_certificate(target, get_basis('zz_u'), 3)
    assert proof['scope'] == 'minimum_uniform_family_budget'
    assert proof['substitutions']
    assert proof['specialization_caveat']
    # The same family also contains a local member, requiring no entanglers.
    assert small_count_certificate(target.subs({a:0,b:0,c:0}), get_basis('zz_u'), 1) is None
    for field, value in [('lower_bound',2), ('substitutions',{'a':'0','b':'0','c':'0'}),
                         ('local_cost_optimality',True), ('scope','fixed_target_minimum')]:
        changed = copy.deepcopy(proof)
        changed[field] = value
        assert not verify_small_count_certificate(target, get_basis('zz_u'), changed)


def test_exact_su2_determinant_parity_strengthens_the_relaxed_lower_bound():
    target = gates.make_gate('Canonical', sp.pi/8, sp.pi/8, sp.pi/8)
    basis = get_basis('cx_rz_ry')
    proof = small_count_certificate(target, basis, 4, phase_exact=True)
    assert proof['lower_bound'] == 4 and proof['exact_determinant_parity']
    assert small_count_certificate(target, basis, 4, phase_exact=False) is None


def test_native_phase_and_locals_do_not_change_the_lower_bound():
    local = sp.kronecker_product(gates.make_gate('U3',sp.pi/2,sp.pi/4,0),
                                 gates.make_gate('U3',sp.pi/2,0,sp.pi/4))
    target = sp.exp(sp.I*sp.pi/8)*local*gates.make_gate('SSWAP')
    proof = small_count_certificate(target, get_basis('cx_u'), 3)
    assert proof and proof['lower_bound'] == 3


def test_lower_bound_never_accepts_nonunitary_or_floating_input():
    basis = get_basis('cx_u')
    with pytest.raises(ValueError):
        small_count_certificate(2*sp.eye(4), basis, 1)
    with pytest.raises(ValueError):
        small_count_certificate(sp.Float(1.0)*gates.make_gate('CX'), basis, 1)
