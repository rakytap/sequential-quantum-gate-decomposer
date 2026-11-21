import itertools

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_identity import prove_circuit_identity
from symbolic_native_echoes import b_cartan_source,b_pauli_echo,sqrt_iswap_exchange_source,b_sqrt_swap_source,native_dcx_source


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


@pytest.mark.parametrize("name",["SSWAP","SSWAPdg"])
def test_two_b_square_root_swap_has_five_locals_with_a_full_exact_proof(name):
    word,factor=b_sqrt_swap_source(name=="SSWAPdg")
    assert sum(len(op[-1])==2 for op in word)==2
    assert sum(len(op[-1])==1 for op in word)==5
    assert prove_circuit_identity(gates.make_gate(name),word,factor)


@pytest.mark.parametrize("exact",[False,True])
def test_square_root_swap_compiler_preserves_two_b_pulses_and_charges_literal_phase(exact):
    from symbolic_synthesis import synthesize
    result=synthesize("SSWAPdg",(),"berkeley_b",phase_exact=exact,osr_seconds=2)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==2 and result["entangler_optimal"]
    assert result["cost"]["single_qubit_gates"]==(6 if exact else 5)
    assert not result["local_cost_optimal"]


def test_explicit_real_phase_does_not_need_a_root_of_unity_search():
    from symbolic_native_pauli import _phase
    angle=sp.acos(1/sp.sqrt(3))-sp.pi/2
    assert _phase(sp.exp(sp.I*angle))==angle
    assert _phase(sp.exp(sp.Symbol("theta",real=True))) is None


@pytest.mark.parametrize("native",["B","SiSWAP"])
def test_dcx_is_a_proved_clifford_frame_around_two_native_pulses(native):
    word,factor=native_dcx_source(native)
    assert sum(len(op[-1])==2 for op in word)==2
    assert prove_circuit_identity(gates.make_gate("DCX"),word,factor)
