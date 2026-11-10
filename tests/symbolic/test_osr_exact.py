import sympy as sp
import pytest

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_identity import exponential_forms,zero_form
from symbolic_osr import (bipartitions,realign,schmidt_certificate,product_factors,
                          product_from_form,tensor_on_cut,profile,interaction_bound)


def exact_equal(left,right):
    forms,inverse=exponential_forms(tuple(left-right))
    return all(zero_form(x,tuple(inverse)) for x in forms)


def test_realignment_and_factorization_preserve_little_endian_wires_and_phase():
    theta=sp.Symbol("theta",real=True)
    left=gates.make_gate("Ry",theta)
    right=sp.I*gates.make_gate("H")
    matrix=tensor_on_cut(left,right,(1,),2)
    assert exact_equal(matrix,sp.kronecker_product(left,right))
    certificate=schmidt_certificate(matrix,(1,))
    assert certificate.rank_upper_bound==1
    a,b=product_factors(matrix,(1,))
    assert exact_equal(matrix,tensor_on_cut(a,b,(1,),2))
    forms,inverse=exponential_forms(tuple(matrix))
    pair=product_from_form(sp.ImmutableMatrix(4,4,forms),(1,),tuple(inverse))
    assert pair is not None


@pytest.mark.parametrize("gate,rank",[("CX",2),("iSWAP",4),("SWAP",4),("ECR",2),("SSWAP",4)])
def test_fixed_gate_ranks_have_exact_minor_certificates(gate,rank):
    certificate=schmidt_certificate(gates.make_gate(gate),(0,))
    assert certificate.rank_upper_bound==rank
    assert certificate.fixed_rank==rank


def test_free_parameter_rank_drops_are_conditions_not_uniform_minima():
    theta=sp.Symbol("theta",real=True)
    certificate=schmidt_certificate(gates.make_gate("Rzz",theta),(0,))
    assert certificate.rank_upper_bound==2
    assert certificate.condition!=sp.true
    assert schmidt_certificate(gates.make_gate("Rzz",0),(0,)).fixed_rank==1
    assert product_factors(gates.make_gate("Rzz",theta),(0,)) is None


def test_all_bipartitions_are_examined_for_a_three_qubit_opaque_operator():
    a,b,c,t=sp.symbols("a b c t",real=True)
    target=gates.apply_to(sp.eye(8),3,gates.make_gate("U3",a,b,c),(0,))
    target=gates.apply_to(target,3,gates.make_gate("Rzz",t),(1,2))
    certificates={row.cut:row for row in profile(target)}
    assert set(certificates)=={(0,),(1,),(2,)}
    assert certificates[(0,)].rank_upper_bound==1
    assert certificates[(1,)].rank_upper_bound==2
    assert certificates[(2,)].rank_upper_bound==2
    assert len(tuple(bipartitions(4)))==7


def test_osr_bound_uses_the_actual_native_rank_capacity():
    target=gates.make_gate("SWAP")
    assert interaction_bound(target,get_basis("cx_u"))["lower_bound"]==2
    assert interaction_bound(target,get_basis("iswap_u"))["lower_bound"]==1
    with pytest.raises(ValueError):realign(target,())
    with pytest.raises(ValueError):realign(target,(0,1))
