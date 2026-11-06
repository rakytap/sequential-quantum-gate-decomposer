import copy
import importlib.util
from pathlib import Path

import pytest

from symbolic_references import reference_ledger


path=Path(__file__).resolve().parents[2]/"examples/decomposition/symbolic_archive_analysis.py"
spec=importlib.util.spec_from_file_location("symbolic_archive_analysis",path)
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_only_matching_fixed_basis_counts_receive_minimum_certificates():
    result=dict(status="verified_symbolic",cost=dict(entanglers=6),entangler_optimal=False,local_cost_optimal=False)
    row=dict(gate="CCX",basis="cx_u",target_scope="fixed_gate",phase_exact=result,
             projective=None,projective_omitted_as_no_better=True)
    rows=[copy.deepcopy(row),dict(copy.deepcopy(row),basis="xx_rz_ry"),
          dict(copy.deepcopy(row),target_scope="free_real_family")]
    baseline=dict(gate="CCX",basis="cx_u",status="verified_numeric",entanglers=6,input="CCXGate")
    report=analysis.annotate(dict(rows=rows),dict(rows=[baseline]),reference_ledger())
    first,variable_entangler,free_family=report["rows"]
    assert first["phase_exact"]["entangler_optimal"]
    assert not first["phase_exact"]["local_cost_optimal"]
    assert first["qiskit_fixed_target_comparison"]["delta_entanglers"]==0
    assert not first["qiskit_fixed_target_comparison"]["novelty_claim"]
    assert "entangler_lower_bound" not in variable_entangler
    assert not variable_entangler["phase_exact"]["entangler_optimal"]
    assert "qiskit_fixed_target_comparison" not in free_family
    bad=copy.deepcopy(row)
    bad["phase_exact"]["cost"]["entanglers"]=5
    with pytest.raises(ValueError,match="contradicts"):
        analysis.annotate(dict(rows=[bad]),dict(rows=[]),reference_ledger())


def test_uniform_literature_audit_exposes_verified_but_nonoptimal_families():
    def row(name,basis,count,scope="free_real_family"):
        return dict(gate=name,basis=basis,target_scope=scope,
                    phase_exact=dict(status="verified_symbolic",cost=dict(entanglers=count)),
                    projective=None,projective_omitted_as_no_better=True)
    report=analysis.annotate_uniform_literature_bounds(dict(rows=[
        row("Canonical","berkeley_b",6),row("PhasedFSim","sqrt_iswap_u",6),
        row("SWAP","berkeley_b",2,"fixed_gate"),row("Rz","sqrt_iswap_u",0)]))
    first,second,matched,one_wire=report["rows"]
    assert first["literature_uniform_upper_bound_comparison"]["modes"]["phase_exact"]["status"]=="exceeds_published_uniform_bound"
    assert second["literature_uniform_upper_bound_comparison"]["modes"]["projective"]["uniform_upper_bound"]==3
    assert matched["literature_uniform_upper_bound_comparison"]["modes"]["projective"]["status"]=="within_published_uniform_bound"
    assert not matched["literature_uniform_upper_bound_comparison"]["novelty_claim"]
    assert "literature_uniform_upper_bound_comparison" not in one_wire


def test_cartan_generic_cost_distinguishes_exact_su2_parity_and_special_angles():
    assert analysis.verify_cartan_trace_lemma()
    def row(basis,exact,projective,parameters):
        result=lambda count:dict(status="verified_symbolic",cost=dict(entanglers=count),entangler_optimal=False)
        return dict(gate="Canonical",basis=basis,parameters=parameters,
                    phase_exact=result(exact),projective=result(projective))
    rows=[row("cx_rz_ry",4,3,["a","b","c"]),row("cx_u",3,3,["a","b","c"]),
          row("cx_rz_ry",0,0,["0","0","0"]),
          row("cx_rz_ry",4,3,["pi/8","pi/8","pi/8"])]
    report=analysis.annotate_cartan_phase_bounds(dict(rows=rows))
    strict,unrestricted,identity,special=report["rows"]
    assert strict["phase_exact"]["conditional_entangler_optimality"]["count"]==4
    assert strict["projective"]["conditional_entangler_optimality"]["count"]==3
    assert unrestricted["phase_exact"]["conditional_entangler_optimality"]["count"]==3
    assert "conditional_entangler_optimality" not in identity["phase_exact"]
    assert special["phase_exact"]["entangler_optimal"]
    assert special["phase_exact"]["entangler_optimality_certificate"]["phase_parity"]["required_native_count"]=="even"


def test_targeted_reruns_preserve_each_source_revision_and_failed_attempt():
    old=dict(gate="MCPhase",basis="cx_u",parameters=["theta","2"],target_scope="register_fixture",
             phase_statuses=dict(phase_exact="compiler_error",projective="compiler_error"),
             phase_exact=dict(status="compiler_error",reason="float control count"),preferred=None)
    new=dict(old,phase_statuses=dict(phase_exact="verified_symbolic",projective="verified_symbolic"),
             phase_exact=dict(status="verified_symbolic",cost=dict(entanglers=0,parameter_score="0",single_qubit_gates=0)),preferred="phase_exact")
    base=dict(rows=[old],source_sha256=dict(adapter="old"),comparison_contract=dict(ancillas=0))
    supplement=dict(rows=[new],source_sha256=dict(adapter="fixed"),comparison_contract=dict(ancillas=0))
    merged=analysis.merge_synthesis_runs([base,supplement])
    assert merged["synthesis_runs"][1]["source_sha256"]["adapter"]=="fixed"
    assert merged["rows"][0]["synthesis_run_index"]==1
    assert merged["rows"][0]["previous_attempts"][0]["phase_exact"]["reason"]=="float control count"
    assert merged["summary"]["phase_exact"]==dict(verified_symbolic=1)
    assert "synthesis_run_index" not in old
    # Appending to an already merged archive must keep the older correction's
    # own source hashes, including the indices inside its attempt history.
    third=copy.deepcopy(supplement)
    third["source_sha256"]["adapter"]="OSR"
    merged_again=analysis.merge_synthesis_runs([merged,third])
    row=merged_again["rows"][0]
    assert len(merged_again["synthesis_runs"])==3
    assert row["synthesis_run_index"]==2
    assert [old["synthesis_run_index"] for old in row["previous_attempts"]]==[0,1]
    assert merged_again["synthesis_runs"][1]["source_sha256"]["adapter"]=="fixed"
    changed=copy.deepcopy(supplement)
    changed["rows"][0]["parameters"]=["phi","2"]
    with pytest.raises(ValueError,match="changes the target"):
        analysis.merge_synthesis_runs([base,changed])


def test_time_limited_rerun_keeps_a_verified_incumbent_and_its_source():
    good=dict(status="verified_symbolic",cost=dict(entanglers=2,parameter_score="8",single_qubit_gates=0))
    old=dict(gate="SWAP",basis="cx_u",phase_exact=good,projective=None,
             phase_statuses=dict(phase_exact="verified_symbolic",projective="verified_symbolic"),synthesis_run_index=0)
    new=dict(old,phase_exact=dict(status="timeout"),projective=dict(status="timeout"),
             phase_statuses=dict(phase_exact="timeout",projective="timeout"),synthesis_run_index=1)
    retained=analysis.retain_incumbents(old,new)
    assert retained["phase_exact"]["status"]=="verified_symbolic"
    assert retained["mode_source_run_indices"]==dict(phase_exact=0,projective=0)
    assert retained["last_attempt_phase_statuses"]["phase_exact"]=="timeout"
    assert retained["synthesis_run_index"]==1


def test_phase_policies_can_retain_different_best_runs():
    def result(entanglers):
        return dict(status="verified_symbolic",cost=dict(entanglers=entanglers,parameter_score="8",single_qubit_gates=0))
    old=dict(gate="SWAP",basis="cx_u",phase_exact=result(2),projective=result(1),
             phase_statuses=dict(phase_exact="verified_symbolic",projective="verified_symbolic"),synthesis_run_index=0)
    new=dict(old,phase_exact=result(3),projective=result(0),synthesis_run_index=1)
    retained=analysis.retain_incumbents(old,new)
    assert retained["phase_exact"]["cost"]["entanglers"]==2
    assert retained["projective"]["cost"]["entanglers"]==0
    assert retained["mode_source_run_indices"]==dict(phase_exact=0,projective=1)
    impossible=dict(new,phase_exact=dict(status="proved_impossible"))
    with pytest.raises(ValueError,match="contradicts"):
        analysis.retain_incumbents(old,impossible)


def test_a_verified_projective_identity_factor_is_also_an_exact_circuit():
    row=dict(gate="SWAP",basis="cx_u",phase_exact=dict(status="timeout"),
             projective=dict(status="verified_symbolic",cost=dict(entanglers=3,parameter_score="12",single_qubit_gates=0),
                             virtual_gp=dict(factor="1",physical=False,cost=0)),
             phase_statuses=dict(phase_exact="timeout",projective="verified_symbolic"),synthesis_run_index=0)
    retained=analysis.retain_incumbents(row,dict(row,synthesis_run_index=1))
    assert retained["phase_exact"]["status"]=="verified_symbolic"
    assert retained["phase_exact"]["virtual_gp"] is None
    assert retained["phase_exact"]["derived_from_projective_identity_factor"]
    assert retained["projective"] is None
