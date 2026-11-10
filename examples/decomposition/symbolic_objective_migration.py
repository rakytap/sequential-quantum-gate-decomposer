#!/usr/bin/env python3
"""Audit unchanged retained scores and explicitly migrate an archive to v2.

Historical source snapshots and their comparison contracts remain intact.
Only a proof that every retained denominator magnitude is >=1 permits this
lossless migration; changed or undecided expressions abort before writing.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import sympy as sp
from symbolic_archive import comparison_contract, wall_time_limit, SymbolicDeadline
from symbolic_gates import GATE_DEFINITIONS

SYMBOLS={name:sp.Symbol(name,real=True) for definition in GATE_DEFINITIONS.values()
         for name in definition.parameters}
SYMBOLS.update({name:sp.Symbol(name,real=True) for name in ('theta','alpha','beta')})


def audit_parameter(text):
    try:
        with wall_time_limit(120):
            expression=sp.sympify(text,locals=SYMBOLS)
            denominator=sp.simplify(expression).as_numer_denom()[1]
            condition=sp.simplify(sp.Abs(denominator)>=1)
            return dict(parameter=text,denominator=str(denominator),unchanged=condition is sp.S.true,
                        condition=str(condition))
    except SymbolicDeadline:
        return dict(parameter=text,unchanged=False,condition='audit_timeout')


def retained_parameters(report):
    return sorted({p for row in report['rows'] for mode in ('phase_exact','projective')
                   for operation in (row.get(mode) or {}).get('operations',[])
                   for p in operation['parameters']})


def migrate(report, audit):
    parameters=retained_parameters(report)
    if sorted(item['parameter'] for item in audit)!=parameters or any(item.get('unchanged') is not True for item in audit):
        raise ValueError('Complete exact unchanged-score audit is required')
    previous=report['comparison_contract']
    if previous.get('objective') not in (
        ['native_two_qubit_operations','original_parameter_sensitive_score','single_qubit_operations'],
        comparison_contract()['objective']):
        raise ValueError('Unknown archive objective')
    report['comparison_contract']=comparison_contract()
    report['objective_migration']=dict(from_contract=previous,to_contract=comparison_contract(),
        unchanged_retained_scores=True,distinct_retained_parameters=len(parameters),
        proof='Every simplified retained parameter has denominator magnitude >=1',
        audit_sha256=hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        source_snapshots='Historical synthesis_runs retain their actual source hashes and historical contracts',
        rerun_required=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    raw=args.input.read_bytes()
    report=json.loads(gzip.decompress(raw) if args.input.suffix=='.gz' else raw)
    parameters=retained_parameters(report)
    with mp.Pool(args.workers) as pool:audit=list(pool.imap_unordered(audit_parameter,parameters,chunksize=1))
    audit.sort(key=lambda item:item['parameter'])
    evidence=dict(generated_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=hashlib.sha256(raw).hexdigest(),input=str(args.input),
        distinct_retained_parameters=len(parameters),changed_or_undecided=sum(not item['unchanged'] for item in audit),
        parameters=audit)
    args.audit_output.write_text(json.dumps(evidence,indent=2)+'\n')
    report=migrate(report,audit)
    report['objective_migration'].update(input_sha256=evidence['input_sha256'],audit_path=str(args.audit_output))
    encoded=json.dumps(report,allow_nan=False,separators=(',',':')).encode()
    temporary=args.output.with_name(args.output.name+'.tmp')
    temporary.write_bytes(gzip.compress(encoded) if args.output.suffix=='.gz' else encoded)
    temporary.replace(args.output)
    print(json.dumps({key:evidence[key] for key in ('distinct_retained_parameters','changed_or_undecided')}),flush=True)


if __name__=='__main__':main()
