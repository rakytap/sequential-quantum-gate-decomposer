#!/usr/bin/env python3
"""Expose newly improved archive circuits to the normal symbolic compiler."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"squander/decomposition"))
import sympy as sp


def load(path):
    with gzip.open(path,"rt") as stream:return json.load(stream)


def effective(row,mode):
    return row.get(mode) or (row.get("phase_exact") if mode=="projective" and row.get("projective_omitted_as_no_better") else None)


def key(result):
    if not result or result.get("status")!="verified_symbolic":return (float("inf"),)*3
    cost=result["cost"]
    return cost["entanglers"],float(sp.sympify(cost["parameter_score"])),cost["single_qubit_gates"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source,previous=load(args.archive),load(args.previous)
    if source["comparison_contract"]!=previous["comparison_contract"]:parser.error("Archive contracts differ")
    registry=json.loads(args.output.read_text())
    old={(row["gate"],row["basis"]):row for row in previous["rows"]}
    entries={(row["gate"],row["basis"]):row for row in registry["entries"]}
    added=0
    for row in source["rows"]:
        if row["target_scope"]!="fixed_gate":continue
        for mode in ("phase_exact","projective"):
            result=effective(row,mode)
            if key(result)>=key(effective(old[row["gate"],row["basis"]],mode)):continue
            # These constructors already replay their exact component
            # certificates through _native_chart_incumbent. Flattening them
            # into the full-matrix template registry discards that proof
            # structure and can make an otherwise fast replay expensive.
            if result.get("verification")=="composition_of_exact_fixed_native_identities":continue
            if not any(len(op["wires"])==2 for op in result["operations"]):continue
            entry=entries.setdefault((row["gate"],row["basis"]),dict(gate=row["gate"],basis=row["basis"],candidates=[]))
            candidate=dict(operations=result["operations"],
                factor=result["virtual_gp"]["factor"] if result.get("virtual_gp") else "1",source_mode=mode,
                source_run_index=row.get("mode_source_run_indices",{}).get(mode,row.get("synthesis_run_index",0)))
            if any(item["operations"]==candidate["operations"] and item["factor"]==candidate["factor"] for item in entry["candidates"]):continue
            entry["candidates"].append(candidate);added+=1
    registry["entries"]=[entries[key] for key in sorted(entries)]
    registry["refinement_source"]=dict(archive=str(args.archive),sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        comparison_base_sha256=hashlib.sha256(args.previous.read_bytes()).hexdigest(),
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        proof_policy="Native templates are full-identity checked again before compiler acceptance")
    args.output.write_text(json.dumps(registry,separators=(",",":"))+"\n")
    print(json.dumps(dict(entries=len(entries),added_candidates=added)))


if __name__=="__main__":main()
