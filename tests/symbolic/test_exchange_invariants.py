import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_exchange_invariants import (
    verify_exchange_obstruction_lemmas, exchange_three_pulse_lower_bound,
)


def test_exchange_sums_of_squares_and_reflection_relations_are_exact():
    assert verify_exchange_obstruction_lemmas()


@pytest.mark.parametrize('name', ['SWAP','SSWAP','SSWAPdg','SYC'])
def test_sqisw_obstruction_covers_nondiagonal_and_nonreal_invariant_targets(name):
    proof = exchange_three_pulse_lower_bound(gates.make_gate(name), 'SiSWAP')
    assert proof and proof['lower_bound'] == 3
    assert proof['construction_claim'] is False


@pytest.mark.parametrize('name', ['I4','CX','iSWAP','SiSWAP','B'])
def test_sqisw_two_pulse_region_never_gets_a_three_pulse_bound(name):
    target = sp.eye(4) if name == 'I4' else gates.make_gate(name)
    assert exchange_three_pulse_lower_bound(target, 'SiSWAP') is None


def test_sqrt_swap_two_pulse_relation_excludes_syc_but_not_cnot_or_swap():
    assert exchange_three_pulse_lower_bound(gates.make_gate('SYC'),'SSWAP')['lower_bound'] == 3
    for name in ('CX','SWAP','SSWAP'):
        assert exchange_three_pulse_lower_bound(gates.make_gate(name),'SSWAP') is None


def test_sqisw_lower_bound_is_invariant_under_physical_phase_and_local_frames():
    local = sp.kronecker_product(gates.make_gate('H'),gates.make_gate('S'))
    target = sp.exp(sp.I*sp.pi/8)*local*gates.make_gate('SSWAP')
    proof = exchange_three_pulse_lower_bound(target,'SiSWAP')
    assert proof and proof['lower_bound'] == 3
