#!/usr/bin/env python3
"""Transfer proved fixed-target words across exact aliases/local frames.

Matrix keys propose aliases and single-wire Clifford conjugations. Each
resulting native circuit is reproved against its entire exact target.
"""

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"squander/decomposition"))
import sympy as sp
import symbolic_gates as gates
from symbolic_archive import comparison_contract,wall_time_limit,SymbolicDeadline
from symbolic_bases import get_basis
from symbolic_contract import retain_phase_results,resource_key,verify_circuit
from symbolic_synthesis import _local,_physical_scalar,_verify_phase_candidate
from symbolic_synthesis_archive import source_hashes,write_report


def load(path):
    with gzip.open(path,"rt") as stream:return json.load(stream)


def effective(row):
    return row.get("projective") or (row.get("phase_exact") if row.get("projective_omitted_as_no_better") else None)


def decode(result):
    return tuple((op["gate"],*(sp.sympify(p) for p in op["parameters"]),tuple(op["wires"]))
                 for op in result["operations"])


def matrix_key(matrix):
    return sp.ImmutableMatrix(matrix.applyfunc(lambda x:sp.simplify(sp.expand_complex(x))))


def proposals(rows):
    targets={row["gate"]:gates.make_gate(row["gate"]) for row in rows if row["target_scope"]=="fixed_gate"}
    by_matrix=defaultdict(list)
    for name,target in targets.items():by_matrix[matrix_key(target)].append(name)
    transfers=defaultdict(list)
    for name,target in targets.items():
        transfers[name].extend((other,None,None) for other in by_matrix[matrix_key(target)] if other!=name)
        width=target.rows.bit_length()-1
        for wire in range(width):
            for gate in ("H","S","Sdg"):
                frame=gates.apply_to(sp.eye(target.rows),width,gates.make_gate(gate),(wire,))
                transformed=matrix_key(frame*target*frame.H)
                transfers[name].extend((other,gate,wire) for other in by_matrix.get(transformed,()) if other!=name)
    lookup={(row["gate"],row["basis"]):row for row in rows}
    candidates=defaultdict(list)
    for source in rows:
        if source["target_scope"]!="fixed_gate":continue
        parent=effective(source)
        if not parent or parent["status"]!="verified_symbolic":continue
        basis=get_basis(source["basis"])
        word=decode(parent)
        factor=sp.sympify(parent["virtual_gp"]["factor"]) if parent.get("virtual_gp") else sp.S.One
        for name,gate,wire in transfers[source["gate"]]:
            row=lookup[name,source["basis"]]
            current=effective(row)
            if not current or current["status"]!="verified_symbolic":continue
            proposed,scalar=word,factor
            if gate:
                inverse={"H":"H","S":"Sdg","Sdg":"S"}[gate]
                before,f0=_local(inverse,(),basis)
                after,f1=_local(gate,(),basis)
                before=tuple((*op[:-1],(wire,)) for op in before)
                after=tuple((*op[:-1],(wire,)) for op in after)
                proposed=before+word+after
                scalar=sp.simplify(factor*f0*f1)
            if resource_key(proposed)>=resource_key(decode(current)):continue
            candidates[name,source["basis"]].append((proposed,scalar,source["gate"],gate,wire))
    return [(lookup[key],value) for key,value in sorted(candidates.items())]


def refine(job):
    row,candidates,seconds=job
    target=gates.make_gate(row["gate"])
    basis=get_basis(row["basis"])
    results={}
    for mode in ("phase_exact","projective"):
        start=time.monotonic()
        result=dict(status="proposal_not_proved",phase_exact=mode=="phase_exact")
        try:
            with wall_time_limit(seconds):
                for word,factor,source,gate,wire in sorted(candidates,key=lambda v:resource_key(v[0])):
                    if mode=="phase_exact":
                        try:
                            if factor!=1:word+=_physical_scalar(factor,basis,target.rows.bit_length()-1)
                        except ValueError:continue
                        candidate=_verify_phase_candidate(target,word,basis,True,factor)
                    else:candidate=verify_circuit(target,word,basis,phase_exact=False,scalar_factor=factor)
                    if candidate["status"]!="verified_symbolic":continue
                    candidate.update(construction_method="exact_fixed_target_alias_or_Clifford_frame_transfer",
                        entangler_optimal=False,local_cost_optimal=False,
                        transfer_proposal=dict(source_gate=source,frame=gate,wire=wire,
                            full_target_identity_reproved=True))
                    result=candidate
                    break
        except SymbolicDeadline:result=dict(status="timeout",phase_exact=mode=="phase_exact")
        results[mode]=dict(result,elapsed_seconds=time.monotonic()-start)
    return dict(gate=row["gate"],basis=row["basis"],parameters=row["parameters"],target_scope=row["target_scope"],
                phase_statuses={key:value["status"] for key,value in results.items()},
                **retain_phase_results(results["phase_exact"],results["projective"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seconds-per-mode",type=float,default=120)
    args=parser.parse_args()
    if args.workers<1 or args.seconds_per_mode<=0:parser.error("Positive worker count/budget required")
    source=load(args.input)
    if source["comparison_contract"]!=comparison_contract():parser.error("Archive contract differs")
    hashes=source_hashes();hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,versions=source.get("versions",{}),comparison_contract=comparison_contract(),
        budget_seconds_per_mode=args.seconds_per_mode,clifford_certificates={},rows=[],
        selection=dict(kind="exact_fixed_target_aliases_and_one_wire_Clifford_conjugations",
            archive=str(args.input),archive_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()))
    jobs=[(*job,args.seconds_per_mode) for job in proposals(source["rows"])]
    print(f"{len(jobs)} fixed-target transfer cells",flush=True)
    checkpoint=args.output.with_name(args.output.name+".partial.json.gz")
    with mp.get_context("fork").Pool(args.workers) as pool:
        for row in pool.imap_unordered(refine,jobs):
            report["rows"].append(row)
            if len(report["rows"])%10==0:
                write_report(checkpoint,report);print(len(report["rows"]),flush=True)
    report["rows"].sort(key=lambda r:(r["gate"].casefold(),r["basis"]))
    report["summary"]={mode:dict(Counter(r["phase_statuses"][mode] for r in report["rows"])) for mode in ("phase_exact","projective")}
    report["summary"]["cells"]=len(jobs)
    report["complete_selected_cells"]=len(report["rows"])==len(jobs)
    write_report(args.output,report);checkpoint.unlink(missing_ok=True)
    print(json.dumps(report["summary"],sort_keys=True))


if __name__=="__main__":main()
