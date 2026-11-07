import copy
import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_basis_transfer import synthesize_template_transfer,verify_basis_transfer_certificate,commute_diagonal_locals
from symbolic_bases import get_basis
from symbolic_identity import prove_circuit_identity


def test_diagonal_commutation_preserves_free_angles_and_blocked_wire_order():
    a,b=sp.symbols('a b',real=True)
    word=(('Rz',a,(0,)),('CZ',(0,1)),('Rz',b,(0,)),('Rx',a,(1,)))
    output=commute_diagonal_locals(word,2)
    assert sum(op[0]=='Rz' for op in output)==1
    assert prove_circuit_identity(gates.compile_gates(2,word),output) is True


@pytest.mark.parametrize('name,count',[('cz_rz_rx',69),('ibm_cz',163)])
def test_c4x_parent_and_local_components_are_replayed_before_acceptance(name,count):
    basis=get_basis(name);target=gates.make_gate('C4X')
    result=synthesize_template_transfer('C4X',basis,phase_exact=False)
    assert result['cost']['entanglers']==36
    assert result['cost']['single_qubit_gates']==count
    assert verify_basis_transfer_certificate(target,result,basis)
    changed=copy.deepcopy(result);changed['basis_transfer_certificate']['parent_factor']='I'
    assert not verify_basis_transfer_certificate(target,changed,basis)
    changed=copy.deepcopy(result)
    operation=next(op for op in changed['operations'] if op['parameters'])
    operation['parameters'][0]='('+operation['parameters'][0]+')+1/100'
    assert not verify_basis_transfer_certificate(target,changed,basis)
    actual=np.eye(32,dtype=complex)
    for operation in result['operations']:
        parameters=[sp.sympify(p).evalf(22) for p in operation['parameters']]
        primitive=gates.make_gate(operation['gate'],*parameters)
        embedded=gates.apply_to(sp.eye(32),5,primitive,tuple(operation['wires']))
        actual=np.asarray(embedded.evalf(22),complex)@actual
    actual*=complex(sp.sympify(result['virtual_gp']['factor']).evalf(22))
    assert np.max(np.abs(actual-np.asarray(target,complex)))<5e-14
