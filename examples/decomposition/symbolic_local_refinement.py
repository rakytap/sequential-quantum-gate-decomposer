#!/usr/bin/env python3
"""Replay and optimize exact one-wire blocks in archived local-count gaps."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
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
from symbolic_archive import comparison_contract,wall_time_limit,SymbolicDeadline
from symbolic_bases import get_basis
from symbolic_contract import retain_phase_results
from symbolic_gates import make_gate
from symbolic_local_rewrite import optimize_local_result
from symbolic_sdk_refinement import projective_result,load
from symbolic_synthesis_archive import source_hashes,write_report


def refine(job):
    row,seconds=job
    results={}
    for mode in ("phase_exact","projective"):
        parent=copy.deepcopy(row.get(mode) if mode=="phase_exact" else projective_result(row))
        if parent is None or parent["status"]!="verified_symbolic":
            results[mode]=parent or dict(status=row["phase_statuses"][mode])
            continue
        if mode=="projective" and parent["phase_exact"]:
            parent.update(phase_exact=False,virtual_gp=dict(factor="1",physical=False,cost=0))
        start=time.monotonic()
        try:
            with wall_time_limit(seconds):
                result=optimize_local_result(make_gate(row["gate"]),parent,get_basis(row["basis"]))
        except SymbolicDeadline:
            result=dict(status="timeout",reason="Local block proof budget")
        except Exception as error:
            result=dict(status="compiler_error",error_type=type(error).__name__,reason=str(error))
        results[mode]=dict(result,phase_exact=mode=="phase_exact",elapsed_seconds=time.monotonic()-start)
    return dict(gate=row["gate"],parameters=row["parameters"],basis=row["basis"],target_scope=row["target_scope"],
                phase_statuses={m:r["status"] for m,r in results.items()},
                **retain_phase_results(results["phase_exact"],results["projective"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--baselines",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seconds-per-mode",type=float,default=60)
    args=parser.parse_args()
    if args.workers<1 or args.seconds_per_mode<=0:parser.error("Positive budget and workers required")
    previous,baselines=load(args.input),load(args.baselines)
    sdk={(r["gate"],r["basis"]):r for r in baselines["rows"] if r["status"]=="verified_numeric"}
    selected=[]
    for row in previous["rows"]:
        result=projective_result(row);base=sdk.get((row["gate"],row["basis"]))
        if row["target_scope"]!="fixed_gate" or not result or not base or result["status"]!="verified_symbolic":continue
        if (result["cost"]["entanglers"]==base["entanglers"] and result["cost"]["single_qubit_gates"]>
                sum(len(op["wires"])==1 for op in base["operations"])):selected.append(row)
    hashes=source_hashes();hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_sha256=hashes,comparison_contract=comparison_contract(),versions=baselines["versions"],
        selection=dict(input=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
            baselines=str(args.baselines),baseline_sha256=hashlib.sha256(args.baselines.read_bytes()).hexdigest()),
        budget_seconds_per_mode=args.seconds_per_mode,rows=[],clifford_certificates={})
    checkpoint=args.output.with_name(args.output.name+".partial.json.gz")
    with mp.get_context("fork").Pool(args.workers) as pool:
        for result in pool.imap_unordered(refine,[(row,args.seconds_per_mode) for row in selected]):
            report["rows"].append(result)
            if len(report["rows"])%10==0:
                write_report(checkpoint,report);print(f"{len(report['rows'])}/{len(selected)} local block cells",flush=True)
    report["rows"].sort(key=lambda r:(r["gate"].casefold(),r["basis"]))
    report["summary"]={m:dict(Counter(r["phase_statuses"][m] for r in report["rows"])) for m in ("phase_exact","projective")}
    report["summary"]["cells"]=len(report["rows"])
    report["complete_selected_cells"]=len(report["rows"])==len(selected)
    write_report(args.output,report);checkpoint.unlink(missing_ok=True)
    print(json.dumps(report["summary"],sort_keys=True))


if __name__=="__main__":main()
