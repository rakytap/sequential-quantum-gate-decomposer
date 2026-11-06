#!/usr/bin/env python3
"""Attach matched fixed-target reference bounds to a symbolic archive.

The numerical SDK comparison concerns entangler upper bounds for fixed
targets only. It establishes neither a free-family identity nor novelty.
"""

import argparse
from collections import Counter
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
from symbolic_gates import make_gate
from symbolic_minimality import cnot_cost_certificate
from symbolic_contract import retain_phase_results


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
        return (value["entanglers"],float(sp.sympify(value["parameter_score"])),value["single_qubit_gates"],
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
        elif impossible:
            result,run=impossible[-1]
        else:
            result=replacement.get(mode) or dict(status=replacement["phase_statuses"][mode])
            run=replacement.get("mode_source_run_indices",{}).get(mode,replacement["synthesis_run_index"])
        selected[mode]=copy.deepcopy(result)
        provenance[mode]=run
    row=copy.deepcopy(replacement)
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
    combined["mixed_synthesis_runs"]=len(combined["synthesis_runs"])>1
    combined["source_provenance_policy"]="Resolve mode_source_run_indices for retained phase results, falling back to synthesis_run_index; synthesis_run_index identifies the latest attempted run; root hashes describe the first run"
    combined["incumbent_policy"]="Retain the least-cost verified circuit for each phase policy; preserve raw outcomes in last_attempt_phase_statuses and every earlier attempt"
    combined["summary"]={mode:dict(Counter(row["phase_statuses"][mode] for row in combined["rows"]))
                          for mode in ("phase_exact","projective")}
    combined["summary"].update(cells=len(combined["rows"]),
        projective_retained_as_strictly_cheaper=sum(row["preferred"]=="projective" and row["phase_exact"]["status"]=="verified_symbolic" for row in combined["rows"]))
    combined["all_targets_solved"]=all(all(status in {"verified_symbolic","proved_impossible"}
        for status in row["phase_statuses"].values()) for row in combined["rows"])
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
    report["reference_analysis_sha256"]={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__),args.input,*args.supplement,args.qiskit,ledger)}
    opener=gzip.open if args.output.suffix==".gz" else open
    with opener(args.output,"wt") as stream:json.dump(report,stream,separators=(",",":"),allow_nan=False)
    print(json.dumps(report["matched_reference_summary"],sort_keys=True))


if __name__=="__main__":main()
