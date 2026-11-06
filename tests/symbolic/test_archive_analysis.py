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


def test_targeted_reruns_preserve_each_source_revision_and_failed_attempt():
    old=dict(gate="MCPhase",basis="cx_u",parameters=["theta","2"],target_scope="register_fixture",
             phase_statuses=dict(phase_exact="compiler_error",projective="compiler_error"),
             phase_exact=dict(status="compiler_error",reason="float control count"),preferred=None)
    new=dict(old,phase_statuses=dict(phase_exact="verified_symbolic",projective="verified_symbolic"),
             phase_exact=dict(status="verified_symbolic"),preferred="phase_exact")
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
