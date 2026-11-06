import copy
from types import SimpleNamespace

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis,make_basis
from symbolic_contract import parameter_score,verify_circuit,retain_phase_results
from symbolic_feasibility import feasibility_obstruction,determinant_group
from symbolic_identity import prove_circuit_identity
from symbolic_clifford_search import solve_clifford_alphabet,verify_shortest_path_certificate,clifford_path,group_data
from symbolic_synthesis import synthesize,_physical_scalar
from symbolic_rules import cartan_source,native_cx,source_rule
from symbolic_uqc import decode_expression,import_native_result
from symbolic_qiskit_source import analytical_definition


def test_parameter_cost_matches_the_existing_implementation():
    pytest.importorskip("deap")
    from sympy_param import get_gatecost
    class Angle:
        def __init__(self,value):self.value=value
        def to_sympy(self,unused):return self.value
    theta=sp.Symbol("theta",real=True)
    sequence=[("U3",theta/2,sp.pi/4,0,(0,)),("CX",(0,1)),("Rz",-sp.pi/2,(1,))]
    old=SimpleNamespace(gates=[(op[0],op[-1],tuple(Angle(sp.sympify(p)) for p in op[1:-1])) for op in sequence])
    assert float(parameter_score(sequence))==pytest.approx(get_gatecost(old,[]),abs=1e-12)
    many=[("GPi2",sp.Rational(1,8),(0,))]*100
    assert len(str(parameter_score(many)))<100


def test_gp_is_never_a_physical_escape_hatch():
    theta=sp.Symbol("theta",real=True)
    basis=get_basis("cx_rz_ry")
    with pytest.raises(ValueError,match="GP"):
        verify_circuit(gates.make_gate("GP",theta,1),[("GP",theta,1,(0,))],basis)
    free=verify_circuit(gates.make_gate("P",theta),[("Rz",theta,(0,))],basis,
                        phase_exact=False,scalar_factor=sp.exp(sp.I*theta/2))
    assert free["status"]=="verified_symbolic" and free["virtual_gp"]["physical"] is False
    assert free["virtual_gp"]["cost"]==0
    assert verify_circuit(gates.make_gate("P",theta),[("Rz",theta,(0,))],basis)["status"]=="identity_not_proved"


def test_phase_preference_retains_projective_only_when_strictly_cheaper():
    exact=dict(status="verified_symbolic",cost=dict(entanglers=2,parameter_score="8",single_qubit_gates=2))
    free=copy.deepcopy(exact)
    assert retain_phase_results(exact,free)["projective"] is None
    free["cost"]["parameter_score"]="7"
    assert retain_phase_results(exact,free)["preferred"]=="projective"
    free["cost"]["entanglers"]=3
    assert retain_phase_results(exact,free)["projective"] is None


def test_determinants_use_the_embedded_register_width():
    assert determinant_group(get_basis("ibm_cx"),1)==4
    assert determinant_group(get_basis("ibm_cx"),3)==1
    target=gates.make_gate("CCX")
    obstruction=feasibility_obstruction(target,get_basis("ibm_cx"))
    assert obstruction["status"]=="proved_impossible"
    assert obstruction["determinant"]=="-1"
    assert feasibility_obstruction(target,get_basis("ibm_cx"),phase_exact=False) is None
    assert feasibility_obstruction(target,get_basis("cx_u")) is None


def test_determinant_witness_can_vary_parameters_independently():
    t,shift=sp.symbols("t global_shift",real=True)
    obstruction=feasibility_obstruction(gates.make_gate("FSWAPPow",t,shift),get_basis("xx_rz_ry"))
    assert obstruction["status"]=="proved_impossible"
    assert obstruction["scope"]=="generic_family"
    assert obstruction["exact_nonidentity_witness"]=={"global_shift":"0","t":"1/2"}


def test_continuous_and_fixed_discrete_obstructions_are_separate():
    theta=sp.Symbol("theta",real=True)
    row=feasibility_obstruction(gates.make_gate("Ry",theta),get_basis("clifford_t_cx"),phase_exact=False)
    assert row["obstruction"]=="discrete_alphabet_nonconstant_family"
    row=feasibility_obstruction(gates.make_gate("T"),get_basis("clifford_only"),phase_exact=False)
    assert row["obstruction"]=="not_a_clifford"
    row=feasibility_obstruction(gates.make_gate("C3X"),get_basis("clifford_t_cx"),phase_exact=False)
    assert row["obstruction"]=="determinant_and_clifford_t_field"


def test_hs_clifford_certificate_covers_all_192_phases_and_detects_tampering():
    result=solve_clifford_alphabet(("H","S"))
    assert len(result["distances"])==192 and result["global_local_optimality"]
    assert verify_shortest_path_certificate(result)
    for name,state in group_data()[3].items():
        actual=gates.compile_gates(1,clifford_path(result,state))
        assert all(sp.simplify(sp.expand_complex(a-b))==0 for a,b in zip(actual,gates.make_gate(name)))
    broken=copy.deepcopy(result)
    state=next(state for state in broken["distances"] if state!=group_data()[-1])
    broken["distances"][state]+=1
    with pytest.raises(ValueError):verify_shortest_path_certificate(broken)


def test_continuous_clifford_search_does_not_overstate_its_proof():
    result=solve_clifford_alphabet(("U3",))
    assert len(result["distances"])==192
    assert result["global_local_optimality"] is False
    assert result["proof_scope"]=="listed_clifford_preserving_words_only"


@pytest.mark.parametrize("basis",["cx_u","cx_rz_ry","ibm_cx","google_cz","ionq_aria","quantinuum_zzphase","aqt_rxx","iqm_cz","cx_h_phase"])
def test_parameterized_local_rules_are_exact_identities(basis):
    theta=sp.Symbol("theta",real=True)
    row=synthesize("Rx",(theta,),basis,phase_exact=False)
    assert row["status"]=="verified_symbolic",row
    assert row["cost"]["entanglers"]==0
    assert all(op["gate"]!="GP" for op in row["operations"])


def test_scalar_is_synthesized_using_only_allowed_gates_when_possible():
    theta=sp.Symbol("theta",real=True)
    row=synthesize("GP",(theta,1),"cx_u",phase_exact=True)
    assert row["status"]=="verified_symbolic" and len(row["operations"])==2
    assert all(op["gate"]=="U3" for op in row["operations"])
    assert synthesize("GP",(theta,1),"cx_rz_ry",phase_exact=True)["status"]=="proved_impossible"


def test_composed_phase_closure_is_replayable_and_rejects_a_changed_factor():
    from symbolic_contract import verify_phase_closure_certificate
    theta=sp.Symbol("theta",real=True)
    target=gates.make_gate("GP",theta,1)
    result=synthesize("GP",(theta,1),"cx_u",phase_exact=True)
    assert result["verification"]=="composition_of_exact_symbolic_identities"
    assert verify_phase_closure_certificate(target,result,get_basis("cx_u"))
    changed=copy.deepcopy(result)
    changed["phase_closure_certificate"]["factor"]="exp(2*I*theta)"
    assert not verify_phase_closure_certificate(target,changed,get_basis("cx_u"))


def test_exponential_proof_handles_free_parameters_and_rejects_an_error():
    theta,phi=sp.symbols("theta phi",real=True)
    sequence=[("Rz",phi,(0,)),("Ry",theta,(0,)),("Rz",-phi,(0,))]
    target=gates.compile_gates(1,sequence)
    assert prove_circuit_identity(target,sequence)
    assert not prove_circuit_identity(gates.make_gate("Rx",theta),sequence)
    assert prove_circuit_identity(gates.make_gate("T"),[("Rz",sp.pi/4,(0,))],sp.exp(sp.I*sp.pi/8))


def test_cartan_identity_has_distinct_literal_and_projective_counts():
    a,b,c=sp.symbols("a b c",real=True)
    target=gates.make_gate("Canonical",a,b,c)
    sequence,factor=cartan_source(a,b,c)
    assert sum(op[0]=="CX" for op in sequence)==3
    assert prove_circuit_identity(target,sequence,factor)
    assert not prove_circuit_identity(target,sequence)
    sequence,factor=cartan_source(a,b,c,even=True)
    assert sum(op[0]=="CX" for op in sequence)==4
    assert prove_circuit_identity(target,sequence,factor)


def test_givens_rule_preserves_the_signed_odd_parity_rotation():
    theta=sp.Symbol("theta",real=True)
    sequence,factor=source_rule("Givens",(theta,))
    assert prove_circuit_identity(gates.make_gate("Givens",theta),sequence,factor)


@pytest.mark.parametrize("entangler",["CX","CZ","ECR","iSWAP","SiSWAP","SSWAP","B","XX90","Rxx","Rzz","Rzx","SYC","IonQMS","ZZPhase"])
def test_every_native_cnot_frame_is_an_exact_identity(entangler):
    sequence,factor=native_cx(entangler)
    assert prove_circuit_identity(gates.make_gate("CX"),sequence,factor)


@pytest.mark.parametrize("exponent,expected_count",[(1,3),(2,2),(4,0),(7,3)])
def test_physical_phase_gadgets_charge_their_entanglers(exponent,expected_count):
    basis=get_basis("cx_rz_ry")
    factor=sp.exp(sp.I*sp.pi*exponent/4)
    sequence=_physical_scalar(factor,basis,2)
    row=verify_circuit(factor*sp.eye(4),sequence,basis)
    assert row["status"]=="verified_symbolic"
    assert row["cost"]["entanglers"]==expected_count


def test_custom_universality_labels_are_not_mathematical_evidence():
    # An incorrect caller label cannot make valid CS circuits impossible.
    basis=make_basis("custom",("H","S"),"CS",universality="clifford")
    assert feasibility_obstruction(gates.make_gate("CS"),basis,phase_exact=False) is None
    basis=make_basis("custom",("H","SqrtH"),"CX",universality="approximate_discrete")
    assert feasibility_obstruction(gates.make_gate("SqrtH"),basis,phase_exact=False) is None


def test_sdk_symbolic_register_constructor_keeps_integer_controls():
    pytest.importorskip("qiskit")
    theta=sp.Symbol("theta",real=True)
    sequence,factor=analytical_definition("MCPhase",(theta,2))
    assert prove_circuit_identity(gates.make_gate("MCPhase",theta,2),sequence,factor)
    assert not any(sp.sympify(p).atoms(sp.Float) for op in sequence for p in op[1:-1])


def test_uqc_expression_import_is_exact_and_whitelisted():
    theta=sp.Symbol("theta",real=True)
    node=dict(op="Times",args=[dict(op="Rational",args=[dict(op="Integer",value="1"),dict(op="Integer",value="2")]),dict(op="Symbol",name="p0")])
    assert decode_expression(node,{"p0":theta})==theta/2
    for invalid in [dict(op="Float",value="1.5"),dict(op="Symbol",name="__import__"),dict(op="Run",args=[])]:
        with pytest.raises(ValueError):decode_expression(invalid,{"p0":theta})
    result=dict(id="probe",status="completed",gates=[dict(tag=3,angle=node,wire=1),dict(tag=0,control=1,target=2)])
    assert import_native_result(dict(id="probe",num_qubits=2),result,variables={"p0":theta})==(("Rz",-theta/2,(1,)),("CX",(1,0)))
