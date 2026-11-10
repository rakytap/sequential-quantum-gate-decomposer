import sympy as sp
from symbolic_gates import make_controlled,apply_to
from symbolic_reflection_involution import reflection,five_arbitrary_matrices
from symbolic_controlled_chart import _zero
import pytest


@pytest.mark.parametrize('negative', [False, True])
def test_seven_cnot_reflection_word_is_literal_with_angle_free(negative):
    from symbolic_reflection_involution import seven_cnot_source
    from symbolic_gates import compile_gates
    theta = sp.Symbol('reflection_angle', real=True)
    word = seven_cnot_source(theta, negative)
    assert sum(g[0] == 'CX' for g in word) == 7
    assert not any(g[0] == 'GP' for g in word)
    assert sum(bool(sp.sympify(p).free_symbols) for g in word for p in g[1:-1]) == 2
    target = make_controlled((-1 if negative else 1)*reflection(theta), 2)
    assert _zero(compile_gates(3, word) - target)


def test_five_unrestricted_reflection_word_with_angle_free():
    theta=sp.Symbol('reflection_angle',real=True)
    word=five_arbitrary_matrices(theta)
    assert [wires for matrix,wires in word]==[(0,1),(1,2),(0,2),(0,1),(1,2)]
    actual=sp.eye(8)
    for matrix,wires in word:
        assert _zero(matrix.H*matrix-sp.eye(4))
        actual=apply_to(actual,3,matrix,wires)
    assert _zero(actual-make_controlled(reflection(theta),2))


@pytest.mark.parametrize('native,target,minimum',[
    ('SSWAP','CNOT',2),('SSWAP','CS',3),('SSWAP','CT',4),('SSWAP','B',3),
    ('SSWAP','SWAP',2),('SSWAP','SSWAPdg',3),
    ('SYC','CNOT',2),('SYC','CS',2),('SYC','B',2),('SYC','SWAP',3),
    ('SiSWAP','CNOT',2),('SiSWAP','B',2),('SiSWAP','SWAP',3)])
def test_exact_monodromy_counts_include_chirality_and_closed_boundaries(native,target,minimum):
    from symbolic_monodromy import certify_coordinate,fixed_depth_certificate,verify_depth_certificate
    from symbolic_gates import make_gate
    a=certify_coordinate(make_gate(native))['coordinate']
    b=certify_coordinate(make_gate(target))['coordinate']
    for count in range(minimum+1):
        result=fixed_depth_certificate(a,b,count)
        assert verify_depth_certificate(a,b,result)
        assert result['reachable']==(count==minimum)


def test_monodromy_rejects_a_changed_exact_primal_or_exclusion():
    from symbolic_monodromy import certify_coordinate,fixed_depth_certificate,verify_depth_certificate
    from symbolic_gates import make_gate
    import copy
    a=certify_coordinate(make_gate('SSWAP'))['coordinate']
    b=certify_coordinate(make_gate('B'))['coordinate']
    excluded=fixed_depth_certificate(a,b,2)
    assert not excluded['reachable']
    damaged=copy.deepcopy(excluded);damaged['branches'][0]['certificate']['weights']=[]
    assert not verify_depth_certificate(a,b,damaged)
    feasible=fixed_depth_certificate(a,b,3)
    assert feasible['reachable']
    damaged=copy.deepcopy(feasible);damaged['branches'][0]['certificate']['point'][0]='1234'
    assert not verify_depth_certificate(a,b,damaged)
