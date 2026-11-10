import sympy as sp
import pytest
from symbolic_controlled_rank_bound import (four_cnot_source,verify_controlled_rank_lemmas,
                                           controlled_full_rank_lower_bound)
from symbolic_gates import make_gate
from symbolic_identity import prove_circuit_identity


def test_four_cnot_fredkin_source_is_literal_and_all_phase_lower_bound_matches():
    target=make_gate('fFredkin');word=four_cnot_source()
    assert sum(len(op[-1])==2 for op in word)==4
    assert prove_circuit_identity(target,word) is True
    assert verify_controlled_rank_lemmas()
    certificate=controlled_full_rank_lower_bound(target,'CX')
    assert certificate['lower_bound']==4
    assert certificate['relative_osr']['rank_lower_bound']==4
    assert certificate['surviving_three_gate_edge_words']==12
    assert [w['rank_lower_bound'] for w in certificate['cut_witnesses']]==[2,4,4]
    assert 'arbitrary U(2)' in certificate['assumptions']


@pytest.mark.parametrize('native',['CZ','ECR','Rzz','Rxx','Rzx'])
def test_lower_bound_also_relaxes_the_ising_pulse_angle(native):
    assert controlled_full_rank_lower_bound(make_gate('fFredkin'),native)['lower_bound']==4


def test_theorem_does_not_transfer_to_rank_four_native_pulses_or_toffoli():
    assert controlled_full_rank_lower_bound(make_gate('fFredkin'),'iSWAP') is None
    assert controlled_full_rank_lower_bound(make_gate('CCX'),'CX') is None
