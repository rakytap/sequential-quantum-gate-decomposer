import itertools

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_identity import prove_circuit_identity
from symbolic_native_echoes import b_cartan_source,b_pauli_echo,sqrt_iswap_exchange_source


@pytest.mark.parametrize("axes",list(itertools.product("XYZ",repeat=2)))
def test_b_pauli_transport_keeps_an_unrestricted_angle(axes):
    theta=sp.Symbol("theta",real=True)
    word,factor=b_pauli_echo(axes,theta)
    assert sum(len(op[-1])==2 for op in word)==2
    assert prove_circuit_identity(gates.make_gate("PauliRotation","".join(reversed(axes)),theta),word,factor)


@pytest.mark.parametrize("positions",[(0,1),(0,2),(1,2)])
def test_two_b_echo_proves_each_entire_cartan_plane(positions):
    a,b=sp.symbols("a b",real=True)
    values=[0,0,0];values[positions[0]]=a;values[positions[1]]=b
    word,factor=b_cartan_source(values)
    assert sum(len(op[-1])==2 for op in word)==2
    assert prove_circuit_identity(gates.make_gate("Canonical",*values),word,factor)


@pytest.mark.parametrize("positions",[(0,1),(0,2),(1,2)])
@pytest.mark.parametrize("signs",list(itertools.product((-1,1),repeat=2)))
def test_two_b_boundary_family_is_literal_phase_exact_for_all_remaining_angles(positions,signs):
    theta=sp.Symbol("theta",real=True)
    values=[theta]*3
    for i,sign in zip(positions,signs):values[i]=sign*sp.pi/4
    word,factor=b_cartan_source(values)
    assert sum(len(op[-1])==2 for op in word)==2
    assert prove_circuit_identity(gates.make_gate("Canonical",*values),word,factor)


@pytest.mark.parametrize("positions",[(0,1),(0,2),(1,2)])
@pytest.mark.parametrize("sign",[-1,1])
def test_two_sqrt_iswap_exchange_echo_is_symbolic_without_angle_folding(positions,sign):
    theta=sp.Symbol("theta",real=True)
    values=[0,0,0];values[positions[0]]=theta;values[positions[1]]=sign*theta
    word,factor=sqrt_iswap_exchange_source(values)
    assert sum(len(op[-1])==2 for op in word)==2
    assert prove_circuit_identity(gates.make_gate("Canonical",*values),word,factor)
