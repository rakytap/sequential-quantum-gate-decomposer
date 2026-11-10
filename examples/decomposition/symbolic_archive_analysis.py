#!/usr/bin/env python3
"""Attach matched fixed-target reference bounds to a symbolic archive.

The numerical SDK comparison concerns entangler upper bounds for fixed
targets only. It establishes neither a free-family identity nor novelty.
"""

import argparse
from collections import Counter
from functools import lru_cache
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import sympy as sp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"squander/decomposition"))
from symbolic_bases import BASIS_SETS
from symbolic_gates import make_gate,gate_definition
from symbolic_minimality import cnot_cost_certificate
from symbolic_contract import retain_phase_results
from symbolic_feasibility import determinant_group


def load(path):
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt") as stream:return json.load(stream)


def retain_incumbents(previous,replacement):
    """Keep the least-cost proved circuit per phase policy, with its source."""
    def options(row,mode):
        source=row.get("mode_source_run_indices",{})
        current=row.get("synthesis_run_index",0)
        result=row.get(mode)
        choices=[]
        if result is not None:
            choices.append((result,source.get(mode,current)))
        projective=row.get("projective")
        if (mode=="phase_exact" and projective and projective["status"]=="verified_symbolic"
                and projective.get("virtual_gp",{}).get("factor")=="1"):
            derived=copy.deepcopy(projective)
            derived.update(phase_exact=True,virtual_gp=None,derived_from_projective_identity_factor=True)
            choices.append((derived,source.get("projective",current)))
        exact=row.get("phase_exact")
        if mode=="projective" and exact and exact["status"]=="verified_symbolic":
            # Literal equality is also a projective construction, independently
            # of whether the separate projective search finished in its budget.
            derived=copy.deepcopy(exact)
            derived.update(phase_exact=False,virtual_gp=dict(factor="1",physical=False,cost=0),
                           entangler_optimal=False,local_cost_optimal=False)
            derived.pop("phase_resource_lower_bound",None)
            choices.append((derived,source.get("phase_exact",current)))
        return choices
    def cost(choice):
        result,run=choice
        value=result["cost"]
        score=value.get('parameter_score')
        return (value["entanglers"],score is None,0 if score is None else float(sp.sympify(score)),
                0 if score is None else value["single_qubit_gates"],
                -bool(result.get("entangler_optimal")),-bool(result.get("local_cost_optimal")),-run)
    selected={}
    provenance={}
    for mode in ("phase_exact","projective"):
        choices=options(previous,mode)+options(replacement,mode)
        good=[choice for choice in choices if choice[0]["status"]=="verified_symbolic"]
        impossible=[choice for choice in choices if choice[0]["status"]=="proved_impossible"]
        if good and impossible:
            raise ValueError(f"Verified construction contradicts an impossibility record: {replacement['gate']}/{replacement['basis']}/{mode}")
        if good:
            result,run=min(good,key=cost)
            result=copy.deepcopy(result)
            # A local rewrite does not invalidate a matching lower bound for
            # the same target, native basis and phase contract. Carry that
            # proof over independently of which construction won the score.
            count=result["cost"]["entanglers"]
            uncertain=[copy.deepcopy(source) for source,unused in good
                       if source['cost']['entanglers']==count
                       and source['cost'].get('parameter_score') is None and source!=result]
            if uncertain:
                result['uncompared_same_entangler_alternatives']=uncertain
            proof_keys=("osr_certificate","cnot_invariant_lower_bound",
                        "entangler_optimality_certificate","phase_resource_lower_bound")
            for source,unused in good:
                if source["cost"]["entanglers"]!=count:continue
                osr=source.get("osr_certificate",{})
                phase_independent=(osr.get("lower_bound")==count and osr.get("condition")=="True"
                                   or source.get("cnot_invariant_lower_bound")==count)
                if not result.get("entangler_optimal") and (source.get("entangler_optimal") or phase_independent):
                    result["entangler_optimal"]=True
                    for key in proof_keys:
                        if not source.get("entangler_optimal") and key not in {"osr_certificate","cnot_invariant_lower_bound"}:
                            continue
                        if key in source and (key!="phase_resource_lower_bound" or mode=="phase_exact"):
                            result.setdefault(key,copy.deepcopy(source[key]))
                    result["minimum_proof_retained_across_local_rewrite"]=True
        elif impossible:
            result,run=impossible[-1]
        else:
            result=replacement.get(mode) or dict(status=replacement["phase_statuses"][mode])
            run=replacement.get("mode_source_run_indices",{}).get(mode,replacement["synthesis_run_index"])
        selected[mode]=copy.deepcopy(result)
        provenance[mode]=run
    row=copy.deepcopy(replacement)
    row["raw_attempt"]=copy.deepcopy(replacement.get("raw_attempt") or
        {key:value for key,value in replacement.items() if key not in {"previous_attempts","raw_attempt"}})
    row["last_attempt_phase_statuses"]=copy.deepcopy(replacement["phase_statuses"])
    row["phase_statuses"]={mode:result["status"] for mode,result in selected.items()}
    row.update(retain_phase_results(selected["phase_exact"],selected["projective"]))
    row["mode_source_run_indices"]=provenance
    return row


def merge_synthesis_runs(reports):
    """Replace selected cells while keeping their implementation provenance."""
    combined=copy.deepcopy(reports[0])
    combined["synthesis_runs"]=[]
    normalized=[]
    for report in reports:
        offset=len(combined["synthesis_runs"])
        runs=report.get("synthesis_runs") or [{k:copy.deepcopy(v) for k,v in report.items() if k!="rows"}]
        combined["synthesis_runs"].extend(copy.deepcopy(runs))
        def remap(row):
            row=copy.deepcopy(row)
            local=row.get("synthesis_run_index",0)
            if not isinstance(local,int) or not 0<=local<len(runs):
                raise ValueError("Invalid source run index")
            row["synthesis_run_index"]=offset+local
            if "mode_source_run_indices" in row:
                if any(not isinstance(value,int) or not 0<=value<len(runs) for value in row["mode_source_run_indices"].values()):
                    raise ValueError("Invalid phase-policy source run index")
                row["mode_source_run_indices"]={mode:offset+value for mode,value in row["mode_source_run_indices"].items()}
            if "previous_attempts" in row:row["previous_attempts"]=[remap(old) for old in row["previous_attempts"]]
            if "raw_attempt" in row:row["raw_attempt"]=remap(row["raw_attempt"])
            return row
        normalized.append([remap(row) for row in report["rows"]])
    index={(row["gate"],row["basis"]):row for row in normalized[0]}
    for report,rows in zip(reports[1:],normalized[1:]):
        if report["comparison_contract"]!=combined["comparison_contract"]:
            raise ValueError("Supplement has a different comparison contract")
        for replacement in rows:
            key=replacement["gate"],replacement["basis"]
            if key not in index:raise ValueError("Supplement adds an unrequested cell")
            previous=index[key]
            if any(replacement[k]!=previous[k] for k in ("parameters","target_scope")):
                raise ValueError("Supplement changes the target")
            row=retain_incumbents(previous,replacement)
            row["previous_attempts"]=previous.get("previous_attempts",[])+[
                {k:v for k,v in previous.items() if k!="previous_attempts"}]
            index[key]=row
    combined["rows"]=[index[row["gate"],row["basis"]] for row in combined["rows"]]
    # A later exhaustive alphabet proof strengthens the old restricted-word
    # certificate independently of which circuits win the cost comparison.
    for run in combined["synthesis_runs"]:
        for key,certificate in run.get("clifford_certificates",{}).items():
            previous=combined.setdefault("clifford_certificates",{}).get(key)
            if previous is None or certificate.get("global_local_optimality") or not previous.get("global_local_optimality"):
                combined["clifford_certificates"][key]=copy.deepcopy(certificate)
    combined["mixed_synthesis_runs"]=len(combined["synthesis_runs"])>1
    combined["source_provenance_policy"]="Resolve mode_source_run_indices for retained phase results, falling back to synthesis_run_index; synthesis_run_index identifies the latest attempted run; root hashes describe the first run"
    combined["incumbent_policy"]="Retain the least-cost verified circuit for each phase policy; preserve raw outcomes in last_attempt_phase_statuses and every earlier attempt"
    combined["summary"]={mode:dict(Counter(row["phase_statuses"][mode] for row in combined["rows"]))
                          for mode in ("phase_exact","projective")}
    combined["summary"].update(cells=len(combined["rows"]),
        projective_retained_as_strictly_cheaper=sum(row["preferred"]=="projective" and row["phase_exact"]["status"]=="verified_symbolic" for row in combined["rows"]))
    combined["all_targets_solved"]=all(all(status in {"verified_symbolic","proved_impossible"}
        for status in row["phase_statuses"].values()) for row in combined["rows"])
    combined['secondary_scores_unresolved']=sum(
        result is not None and result.get('status')=='verified_symbolic' and result['cost'].get('parameter_score') is None
        for row in combined['rows'] for result in (row.get('phase_exact'),row.get('projective')))
    return combined


def annotate(report,qiskit,ledger):
    report=copy.deepcopy(report)
    # Each of these fixed primitives is locally equivalent to one CNOT.
    # Do not transfer this bound to a tunable interaction or arbitrary basis.
    equivalent={"CX","CNOT","CZ","ECR","XX90","ZZMax"}
    for entangler in equivalent:
        if cnot_cost_certificate(make_gate(entangler)).minimum!=1:
            raise ValueError(f"CNOT equivalence was not proved for {entangler}")
    minima={row["gate"]:row for row in ledger["records"]
            if row["basis"]=="cx_u" and row["parameter_domain"]=="fixed gate"
            and row["claim"]=="proved_minimum" and row["ancillas"]==0
            and row["topology"]=="all_to_all" and row["lower_bound"] is not None}
    sdk={(row["gate"],row["basis"]):row for row in qiskit["rows"]}
    statistics=Counter()
    for row in report["rows"]:
        if row["target_scope"]!="fixed_gate":continue
        minimum=minima.get("CX" if row["gate"]=="CNOT" else row["gate"])
        if minimum and BASIS_SETS[row["basis"]].entangler in equivalent:
            lower=minimum["lower_bound"]
            row["entangler_lower_bound"]=dict(count=lower,reference=minimum["id"],
                assumptions="Projective target, arbitrary U(2) locals, zero ancillas, all-to-all",
                transfer="Every native primitive is locally equivalent to one CNOT",
                exact_phase_upper_bound_transferred=False)
            for mode in ("phase_exact","projective"):
                result=row.get(mode)
                if result is None or result["status"]!="verified_symbolic":continue
                count=result["cost"]["entanglers"]
                if count<lower:raise ValueError(f"Circuit contradicts the lower bound: {row['gate']}/{row['basis']}")
                if count==lower:
                    result["entangler_optimal"]=True
                    result["entangler_optimality_certificate"]=row["entangler_lower_bound"]
                    statistics[mode+"_matched_entangler_minimum"]+=1
        baseline=sdk.get((row["gate"],row["basis"]))
        if not baseline or baseline["status"]!="verified_numeric":continue
        projective=row.get("projective")
        if row.get("projective_omitted_as_no_better"):
            # An exact circuit is also a valid projective upper bound.
            projective=row["phase_exact"]
        comparison=dict(status="not_compared",baseline_kind="Qiskit selected verified numerical upper bound",
            baseline_input=baseline.get("input"),baseline_entanglers=baseline["entanglers"],
            phase_policy="Projective; SDK scalar metadata allowed",symbolic_family_claim=False,novelty_claim=False)
        if projective and projective["status"]=="verified_symbolic":
            delta=projective["cost"]["entanglers"]-baseline["entanglers"]
            comparison.update(status="compared_fixed_target",delta_entanglers=delta)
            statistics["sdk_fixed_better" if delta<0 else "sdk_fixed_equal" if delta==0 else "sdk_fixed_worse"]+=1
        row["qiskit_fixed_target_comparison"]=comparison
    report["matched_reference_summary"]=dict(statistics)
    report["matched_reference_policy"]="Fixed targets only; matching a lower bound certifies entanglers, never the secondary local score"
    return annotate_cartan_phase_bounds(annotate_uniform_literature_bounds(report))


@lru_cache(maxsize=1)
def verify_cartan_trace_lemma():
    a,b,c=sp.symbols("a b c",real=True)
    matrix=make_gate("Canonical",a,b,c)
    yy=sp.kronecker_product(make_gate("Y"),make_gate("Y"))
    imaginary=sp.im(sp.trace(matrix*yy*matrix.T*yy))
    expected=4*sp.sin(2*a)*sp.sin(2*b)*sp.sin(2*c)
    return sp.simplify(matrix.det()-1)==0 and sp.trigsimp(sp.expand_complex(imaginary-expected))==0


def annotate_cartan_phase_bounds(report):
    """Certify generic three-CNOT cost and the strict SU(2) parity surcharge."""
    equivalent={"CX","CNOT","CZ","ECR","XX90","ZZMax"}
    locals_={name:sp.Symbol(name,real=True) for name in ("a","b","c")}
    statistics=Counter()
    for row in report["rows"]:
        if row["gate"]!="Canonical" or len(row.get("parameters",()))!=3:continue
        basis=BASIS_SETS[row["basis"]]
        if basis.entangler not in equivalent:continue
        parameters=tuple(sp.sympify(value,locals=locals_) for value in row["parameters"])
        if any(value.is_real is not True for value in parameters):continue
        condition=sp.Ne(sp.prod(sp.sin(2*value) for value in parameters),0)
        if condition==sp.false:continue
        if not verify_cartan_trace_lemma():raise ArithmeticError("Cartan trace lemma did not verify")
        parity=(determinant_group(basis,1)==1 and sp.simplify(make_gate(basis.entangler).det()+1)==0)
        for mode in ("phase_exact","projective"):
            result=row.get(mode)
            if not result or result["status"]!="verified_symbolic":continue
            count=4 if mode=="phase_exact" and parity else 3
            if result["cost"]["entanglers"]!=count:continue
            certificate=dict(count=count,condition=str(condition),
                source="CNOT spin-flip trace exclusion"+(" and exact determinant parity" if count==4 else ""),
                reference="https://arxiv.org/abs/quant-ph/0308045",
                statement="A nonreal normalized spin-flip trace excludes two CNOTs; every supplied native pulse costs one CNOT under relaxed locals")
            if count==4:
                certificate["phase_parity"]=dict(target_determinant="1",local_determinant="1",native_determinant="-1",required_native_count="even")
            if condition==sp.true:
                result["entangler_optimal"]=True
                result["entangler_optimality_certificate"]=certificate
            else:result["conditional_entangler_optimality"]=certificate
            statistics[mode+"_matched_conditioned_minimum"]+=1
    report["cartan_phase_bound_summary"]=dict(statistics)
    return report


def annotate_uniform_literature_bounds(report):
    """Audit analytical universal two-qubit bounds, including free families.

    A circuit below a uniform worst-case bound is not a literature win. These
    relaxed-local bounds expose known optimization gaps without inventing
    per-target minima or comparing a family with a numerical SDK binding.
    """
    bounds={"berkeley_b":(2,"https://arxiv.org/abs/quant-ph/0312193"),
            "sqrt_iswap_u":(3,"https://arxiv.org/abs/2105.06074"),
            "google_sqrt_iswap":(3,"https://arxiv.org/abs/2105.06074")}
    statistics=Counter()
    for row in report["rows"]:
        if row["basis"] not in bounds:continue
        definition=gate_definition(row["gate"])
        if definition.num_qubits!=2:
            # The archived Unitary register fixture is explicitly the 4x4 B.
            if (row["gate"]!="Unitary" or row["target_scope"]!="register_fixture"
                    or row.get("parameters")!=[str(make_gate("B"))]):continue
        upper,source=bounds[row["basis"]]
        modes={}
        for mode in ("phase_exact","projective"):
            result=row.get(mode)
            if mode=="projective" and row.get("projective_omitted_as_no_better"):result=row.get("phase_exact")
            if not result or result["status"]!="verified_symbolic":continue
            count=result["cost"]["entanglers"]
            status="exceeds_published_uniform_bound" if count>upper else "within_published_uniform_bound"
            modes[mode]=dict(status=status,construction_entanglers=count,uniform_upper_bound=upper)
            statistics[mode+"_"+status]+=1
        row["literature_uniform_upper_bound_comparison"]=dict(source=source,modes=modes,
            scope="Any two-qubit unitary with arbitrary locals; these presets can synthesize scalar phase locally",
            target_minimality_claim=False,novelty_claim=False)
    report["literature_uniform_bound_summary"]=dict(statistics)
    return report


def annotate_secondary_cost_floors(report):
    """Attach independent analytic score floors without altering circuits."""
    if report.get("comparison_contract",{}).get("parameter_score_policy",{}).get("version")!=2:
        report["secondary_cost_floor_summary"]=dict(status="requires_nonnegative_objective_v2")
        return report
    from symbolic_archive import symbolic_cases
    from symbolic_cost_proof import secondary_score_floor_certificate
    cases={name:parameters for name,parameters,scope in symbolic_cases()}
    targets={}
    statistics=Counter()
    for row in report["rows"]:
        row.pop("secondary_cost_floor_certificates",None)
        certificates={}
        for mode in ("phase_exact","projective"):
            result=row.get(mode)
            if mode=="projective" and row.get("projective_omitted_as_no_better"):
                exact=row.get("phase_exact")
                result=dict(exact,phase_exact=False) if exact else None
            if not result or result.get("status")!="verified_symbolic":continue
            if result["cost"]["entanglers"] and result["cost"]["single_qubit_gates"]:continue
            name=row["gate"]
            if name not in targets:targets[name]=make_gate(name,*cases[name])
            certificate=secondary_score_floor_certificate(targets[name],result,BASIS_SETS[row["basis"]])
            if certificate:
                certificates[mode]=certificate
                statistics[mode+"_secondary_minimum_at_fixed_E"]+=1
                if certificate["global_lexicographic_minimum"]:statistics[mode+"_global_lexicographic_minimum"]+=1
        if certificates:row["secondary_cost_floor_certificates"]=certificates
    report["secondary_cost_floor_summary"]=dict(statistics)
    report["secondary_cost_floor_source_sha256"]=hashlib.sha256((ROOT/"squander/decomposition/symbolic_cost_proof.py").read_bytes()).hexdigest()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--supplement",type=Path,action="append",default=[],
                        help="Replace selected cells with a later run; preserve per-row source hashes and previous attempts")
    parser.add_argument("--qiskit",type=Path,default=ROOT/"benchmarks/symbolic/reference_comparison_20261106.json.gz")
    args=parser.parse_args()
    ledger=ROOT/"squander/decomposition/symbolic_reference_ledger.json"
    report=merge_synthesis_runs([load(path) for path in [args.input,*args.supplement]])
    report=annotate(report,load(args.qiskit),load(ledger))
    report=annotate_secondary_cost_floors(report)
    report["reference_analysis_sha256"]={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__),args.input,*args.supplement,args.qiskit,ledger)}
    opener=gzip.open if args.output.suffix==".gz" else open
    with opener(args.output,"wt") as stream:json.dump(report,stream,separators=(",",":"),allow_nan=False)
    print(json.dumps(report["matched_reference_summary"],sort_keys=True))


if __name__=="__main__":main()
