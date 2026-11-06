import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_identity import prove_circuit_identity,zero_form
from symbolic_native_pauli import cartan_native_source,native_cartan_candidate
from symbolic_synthesis import synthesize,synthesize_matrix,_physical_scalar


@pytest.mark.parametrize("entangler",["CX","iSWAP"])
@pytest.mark.parametrize("count",[2,3])
def test_discovered_native_network_proves_a_free_cartan_family(entangler,count):
    a,b,c=sp.symbols("a b c",real=True)
    coordinates=(a,b,c if count==3 else 0)
    sequence,factor=cartan_native_source(*coordinates,entangler)
    assert sum(len(op[-1])==2 for op in sequence)==count
    assert prove_circuit_identity(gates.make_gate("Canonical",*coordinates),sequence,factor)


@pytest.mark.parametrize("gate,parameter_count",[("fSim",2),("PhasedFSim",5)])
def test_matrix_gauge_keeps_every_parameter_and_zero_amplitude_limit(gate,parameter_count):
    parameters=sp.symbols(f"p:{parameter_count}",real=True)
    target=gates.make_gate(gate,*parameters)
    candidate=native_cartan_candidate(target,get_basis("iswap_u"))
    assert candidate is not None
    assert sum(len(op[-1])==2 for op in candidate["sequence"])==3
    assert prove_circuit_identity(target,candidate["sequence"],candidate["factor"])
    for angle in (0,sp.pi/2):
        substitution={parameters[0]:angle}
        sequence=tuple((op[0],*(sp.sympify(p).subs(substitution) for p in op[1:-1]),op[-1]) for op in candidate["sequence"])
        assert prove_circuit_identity(target.subs(substitution),sequence,candidate["factor"].subs(substitution))


@pytest.mark.parametrize("basis,count",[("google_fsim",2),("tket_tk2",1),("zz_u",3),("zx_u",3)])
def test_variable_native_family_synthesizes_the_whole_cartan_operator(basis,count):
    a,b,c=sp.symbols("a b c",real=True)
    target=gates.make_gate("Canonical",a,b,c)
    candidate=native_cartan_candidate(target,get_basis(basis))
    assert sum(len(op[-1])==2 for op in candidate["sequence"])==count
    assert prove_circuit_identity(target,candidate["sequence"],candidate["factor"])


@pytest.mark.parametrize("gate,basis",[("SWAP","sqrt_swap_u"),("iSWAP","sqrt_iswap_u")])
def test_native_powers_have_matching_continuous_local_invariant_bounds(gate,basis):
    result=synthesize_matrix(gates.make_gate(gate),basis,seconds=10)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==2 and result["entangler_optimal"]
    assert result["entangler_optimality_certificate"]["lower_bound"]==2


@pytest.mark.parametrize("gate,basis,count",[("CCX","iswap_u",10),("CCX","zz_u",5),("CSWAP","zz_u",6)])
def test_two_wire_osr_windows_reduce_multiqubit_native_cost(gate,basis,count):
    result=synthesize(gate,(),basis,osr_seconds=5)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==count
    assert result["construction_method"]=="OSR_two_wire_window_resynthesis"
    assert result["window_replacements"]
    assert not result["entangler_optimal"]


def test_sixteenth_root_identity_is_proved_without_floating_angles():
    expression=-sp.exp(sp.I*sp.pi/8)+sp.sqrt(2)*sp.I*(1-sp.I)*sp.exp(-sp.I*sp.pi/8)/2
    assert zero_form(expression)
    assert not zero_form(expression+sp.Rational(1,10**20))


@pytest.mark.parametrize("angle",[-sp.pi/12,sp.pi/24,5*sp.pi/12])
def test_constant_phase_proposals_accept_only_an_exact_root_identity(angle):
    from symbolic_native_pauli import _phase
    root=sp.expand_complex(sp.exp(sp.I*angle))
    assert _phase(root)==angle
    perturbed=root+sp.Rational(1,10**15)
    assert _phase(perturbed)!=angle


def test_projectively_local_native_specialization_can_still_need_a_phase_resource():
    exact=synthesize("Rxx",(sp.pi,),"xx_rz_ry",phase_exact=True,osr_seconds=10)
    projective=synthesize("Rxx",(sp.pi,),"xx_rz_ry",phase_exact=False,osr_seconds=10)
    assert exact["status"]==projective["status"]=="verified_symbolic"
    assert exact["cost"]["entanglers"]==1 and exact["entangler_optimal"]
    assert projective["cost"]["entanglers"]==0 and projective["entangler_optimal"]
    assert exact["phase_resource_lower_bound"]["lower_bound"]==1
    word=_physical_scalar(sp.I,get_basis("xx_rz_ry"),2)
    assert sum(len(op[-1])==2 for op in word)==1
    assert prove_circuit_identity(sp.I*sp.eye(4),word)


@pytest.mark.parametrize("basis",["ionq_aria_full","quantinuum_zzmax"])
def test_three_wire_involution_square_provides_a_missing_exact_phase(basis):
    from symbolic_contract import verify_circuit
    factor=sp.exp(sp.I*sp.pi/4)
    word=_physical_scalar(factor,get_basis(basis),3)
    assert word and not any(op[0]=="GP" for op in word)
    assert verify_circuit(factor*sp.eye(8),word,get_basis(basis))["status"]=="verified_symbolic"
