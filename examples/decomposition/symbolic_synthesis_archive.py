#!/usr/bin/env python3
"""Checkpointed exact symbolic catalog archive with separate phase policies."""

import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"squander/decomposition"
sys.path.insert(0,str(SOURCE))
from symbolic_archive import symbolic_cases,archive_cell,comparison_contract
from symbolic_bases import BASIS_SETS
from symbolic_clifford_search import solve_clifford_alphabet,export_certificate


def source_hashes():
    paths=[Path(__file__),*sorted(SOURCE.glob("symbolic_*.py")),*sorted(SOURCE.glob("symbolic_*.json")),SOURCE/"sympy_gen.py"]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_report(path,report):
    temporary=path.with_name(path.name+".tmp")
    opener=gzip.open if path.suffix==".gz" else open
    with opener(temporary,"wt") as stream:json.dump(report,stream,allow_nan=False,separators=(",",":"))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--basis",action="append",default=[])
    parser.add_argument("--gate",action="append",default=[])
    parser.add_argument("--seconds-per-mode",type=float,default=10)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.seconds_per_mode<=0 or args.workers<1:parser.error("Positive time budget and worker count required")
    bases=list(BASIS_SETS) if not args.basis or "all" in args.basis else list(dict.fromkeys(args.basis))
    if set(bases)-set(BASIS_SETS):parser.error("Unknown basis")
    cases=list(symbolic_cases())
    if args.gate:
        requested=set(args.gate)
        cases=[case for case in cases if case[0] in requested]
        if requested-{c[0] for c in cases}:parser.error("Use canonical catalog gate names")
    hashes=source_hashes()
    checkpoint=args.output.with_name(args.output.name+".partial.json.gz")
    old=None
    if args.resume:
        candidate=checkpoint if checkpoint.exists() else args.output
        if candidate.exists():
            opener=gzip.open if candidate.suffix==".gz" else open
            with opener(candidate,"rt") as stream:old=json.load(stream)
            if old["source_sha256"]!=hashes or old["bases"]!=bases or old["case_names"]!=[c[0] for c in cases]:
                parser.error("Checkpoint implementation or selection differs; use a new output file")
            if old["budget_seconds_per_mode"]!=args.seconds_per_mode:parser.error("Checkpoint time budget differs")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True)
    versions={name:importlib.metadata.version(name) for name in ("sympy","numpy")}
    try:versions["qiskit"]=importlib.metadata.version("qiskit")
    except importlib.metadata.PackageNotFoundError:versions["qiskit"]=None
    report=old or dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),
        git_revision=git.stdout.strip(),source_sha256=hashes,versions=versions,comparison_contract=comparison_contract(),
        bases=bases,case_names=[c[0] for c in cases],budget_seconds_per_mode=args.seconds_per_mode)
    # Deliberately deferred: no Wolfram license purchase and no fabricated results.
    report["universalqcompiler"]=dict(status="deferred_by_user",runs=0,
        source_revision="c72cd84a537786756ae8490c07f3f599aa411323",
        reason="Licensed Wolfram kernel unavailable; user declined purchase",
        future_strategy="One native analytical run per target family, followed by separately labelled native-basis rebasing")
    report.setdefault("rows",[])
    report.setdefault("clifford_certificates",{})
    report["reference_datasets"]=[dict(path="benchmarks/symbolic/reference_comparison_20261106.json.gz",
        kind="numeric_sample_only",phase_resource="explicit_SDK_scalar_allowed",
        comparable_to_free_family_identity=False),dict(path="squander/decomposition/symbolic_reference_ledger.json",
        kind="curated_literature_bounds",requires_matching_assumptions=True)]
    # Fork workers share the exact group tables and primitive searches. These
    # reusable searches are setup costs, not charged against individual jobs.
    for basis in bases:
        local=BASIS_SETS[basis].single_qubit_gates
        key="|".join(local)
        solution=solve_clifford_alphabet(local)
        if key not in report["clifford_certificates"]:
            report["clifford_certificates"][key]=export_certificate(solution)
    completed={(r["gate"],r["basis"]) for r in report["rows"]}
    jobs=[(name,p,scope,basis,args.seconds_per_mode) for name,p,scope in cases for basis in bases
          if (name,basis) not in completed]
    total=len(cases)*len(bases)
    write_report(checkpoint,report)
    def accept(rows):
        for row in rows:
            report["rows"].append(row)
            count=len(report["rows"])
            if count%100==0:
                print(f"{count}/{total} gate/basis cells",flush=True)
                write_report(checkpoint,report)
    if args.workers==1:accept(map(archive_cell,jobs))
    else:
        if "fork" not in mp.get_all_start_methods():parser.error("Parallel archive requires fork; use --workers 1")
        # Resource caps avoid nested BLAS parallelism in optional SDK imports.
        for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
        with mp.get_context("fork").Pool(args.workers) as pool:
            accept(pool.imap_unordered(archive_cell,jobs,chunksize=1))
    order={name:i for i,(name,_,_) in enumerate(cases)}
    report["rows"].sort(key=lambda r:(order[r["gate"]],bases.index(r["basis"])))
    report["summary"]={mode:dict(Counter(row["phase_statuses"][mode] for row in report["rows"]))
                       for mode in ("phase_exact","projective")}
    report["summary"].update(cells=len(report["rows"]),
        projective_retained_as_strictly_cheaper=sum(r["preferred"]=="projective" and r["phase_exact"]["status"]=="verified_symbolic" for r in report["rows"]))
    report["complete_catalog_enumeration"]=len(report["rows"])==total
    report["all_targets_solved"]=all(all(v in {"verified_symbolic","proved_impossible"} for v in r["phase_statuses"].values()) for r in report["rows"])
    write_report(args.output,report)
    checkpoint.unlink(missing_ok=True)
    print(json.dumps(report["summary"],sort_keys=True))
    return int(any("compiler_error" in r["phase_statuses"].values() for r in report["rows"]))


if __name__=="__main__":raise SystemExit(main())
