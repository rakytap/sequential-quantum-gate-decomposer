import copy
import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_local_polar import polar_component,synthesize_local_polar,verify_local_polar_certificate,verify_local_polar_lemma
from symbolic_identity import prove_circuit_identity


@pytest.mark.parametrize('word',[(('X',(0,)),),(('T',(0,)),),(('Rz',2*sp.pi,(0,)),),
                               (('U3',sp.pi/3,sp.pi/5,sp.pi/7,(0,)),)])
def test_exact_polar_lemma_and_all_zero_charts(word):
    assert verify_local_polar_lemma()
    fused,angle,certificate=polar_component(word)
    actual=gates.compile_gates(1,fused)
    expected=gates.compile_gates(1,word)
    assert np.max(np.abs(np.asarray((sp.exp(sp.I*angle)*actual-expected).evalf(22),complex)))<5e-14
    assert certificate['chart'] in {'A=0','C=0','A!=0,C!=0'}


def test_polar_component_declines_nonunitary_or_free_symbol_blocks():
    with pytest.raises(ValueError):polar_component((('U3',sp.I,0,0,(0,)),))
    with pytest.raises(ValueError):polar_component((('Ry',sp.Symbol('theta',real=True),(0,)),))


def test_peres_native_two_axis_lowering_matches_the_sdk_local_count():
    basis=get_basis('xx_rz_ry');target=gates.make_gate('Peres')
    result=synthesize_local_polar('Peres',basis,phase_exact=False)
    assert result['cost']['entanglers']==5
    assert result['cost']['single_qubit_gates']==35
    assert verify_local_polar_certificate(target,result,basis)
    actual=np.eye(8,dtype=complex)
    for operation in result['operations']:
        primitive=gates.make_gate(operation['gate'],*(sp.sympify(p).evalf(22) for p in operation['parameters']))
        embedded=gates.apply_to(sp.eye(8),3,primitive,tuple(operation['wires']))
        actual=np.asarray(embedded.evalf(22),complex)@actual
    phase=complex(sp.sympify(result['virtual_gp']['factor']).evalf(22))
    assert np.max(np.abs(phase*actual-np.asarray(target,complex)))<5e-14


@pytest.mark.parametrize('name,basis,count',[('C3SX','iswap_u',40),('C4X','iswap_u',78)])
@pytest.mark.parametrize('exact',[False,True])
def test_complete_parent_and_component_proofs_replay_with_both_phase_policies(name,basis,count,exact):
    selected=get_basis(basis);target=gates.make_gate(name)
    result=synthesize_local_polar(name,selected,phase_exact=exact)
    assert result['cost']['single_qubit_gates']==count+int(exact)
    assert verify_local_polar_certificate(target,result,selected)
    changed=copy.deepcopy(result)
    changed['local_polar_certificate']['blocks'][0]['proof']['norms']=['0','0']
    assert not verify_local_polar_certificate(target,changed,selected)
    changed=copy.deepcopy(result);changed['local_cost_optimal']=True
    assert not verify_local_polar_certificate(target,changed,selected)
    width=target.rows.bit_length()-1
    actual=np.eye(target.rows,dtype=complex)
    for operation in result['operations']:
        parameters=[sp.sympify(p).evalf(22) for p in operation['parameters']]
        primitive=gates.make_gate(operation['gate'],*parameters)
        embedded=gates.apply_to(sp.eye(target.rows),width,primitive,tuple(operation['wires']))
        actual=np.asarray(embedded.evalf(22),complex)@actual
    phase=1 if exact else complex(sp.sympify(result['virtual_gp']['factor']).evalf(22))
    assert np.max(np.abs(phase*actual-np.asarray(target,complex)))<5e-14
