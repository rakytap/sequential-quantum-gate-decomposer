#!/usr/bin/env python3
"""Exact minimum-count certificates for the previously sampled native columns."""
import argparse,gzip,hashlib,json,multiprocessing as mp,sys,time
from collections import Counter
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'squander/decomposition'))
import sympy as sp
from symbolic_archive import symbolic_cases,wall_time_limit,SymbolicDeadline
from symbolic_gates import make_gate
from symbolic_monodromy import certify_coordinate,verify_coordinate,fixed_depth_certificate,verify_depth_certificate


@lru_cache(maxsize=1)
def targets():return {n:make_gate(n,*p) for n,p,_ in symbolic_cases()}
def target_of(row):
    matrix=targets()[row['gate']]
    return matrix.subs({s:sp.sympify(row['substitutions'][str(s)]) for s in matrix.free_symbols})
@lru_cache(maxsize=2)
def native_of(basis):return make_gate('SYC' if basis=='google_syc' else 'SSWAP')
@lru_cache(maxsize=512)
def depth(native,target,count):return fixed_depth_certificate(native,target,count)


def run(job):
    source,maximum,seconds=job;started=time.perf_counter()
    row={k:source[k] for k in ('gate','basis','target_scope','substitutions','witness_scope')}
    try:
        with wall_time_limit(seconds):
            N=certify_coordinate(native_of(row['basis']));T=certify_coordinate(target_of(row))
            if T is None:return dict(row,status='coordinate_not_certified')
            row.update(native_coordinate=N,target_coordinate=T,depths=[])
            for count in range(maximum+1):
                record=depth(tuple(N['coordinate']),tuple(T['coordinate']),count)
                if 'reachable' not in record:return dict(row,status='exact_LP_certificate_not_reconstructed')
                row['depths'].append(record)
                if record['reachable']:
                    row.update(status='proved_native_class_minimum',minimum=count)
                    if source.get('entanglers') is not None and count>source['entanglers']:
                        raise ArithmeticError('Polytope result contradicts replayed numerical witness')
                    break
            else:row.update(status='proved_lower_bound',lower_bound=maximum+1)
    except SymbolicDeadline:row.update(status='proof_deadline',deadline_seconds=seconds)
    row['wall_seconds']=time.perf_counter()-started
    print(row['gate'],row['basis'],row['status'],row.get('minimum',row.get('lower_bound','')),flush=True)
    return row


def verify(row):
    if row['status'] not in ('proved_native_class_minimum','proved_lower_bound'):return False
    assert verify_coordinate(native_of(row['basis']),row['native_coordinate'])
    assert verify_coordinate(target_of(row),row['target_coordinate'])
    N=row['native_coordinate']['coordinate'];T=row['target_coordinate']['coordinate']
    for count,record in enumerate(row['depths']):
        assert record['count']==count and verify_depth_certificate(N,T,record)
        assert record['reachable']==(row['status']=='proved_native_class_minimum' and count==row['minimum'])
    assert len(row['depths'])==row.get('minimum',row.get('lower_bound',0)-1)+1
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-count',type=int,default=4)
    p.add_argument('--seconds',type=float,default=25);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:
        report=json.load(gzip.open(a.output,'rt'));verified=sum(verify(r) for r in report['rows'])
        print(json.dumps(dict(verified_records=verified,statuses=dict(Counter(r['status'] for r in report['rows'])))));return
    source=ROOT/'benchmarks/symbolic/native_column_gap_audit_20261107.json.gz'
    proposals=json.load(gzip.open(source,'rt'))
    with mp.get_context('fork').Pool(a.workers) as pool:rows=list(pool.imap_unordered(run,[(r,a.max_count,a.seconds) for r in proposals['rows']]))
    rows.sort(key=lambda r:(r['gate'],r['basis']))
    paths=[Path(__file__),ROOT/'squander/decomposition/symbolic_monodromy.py',ROOT/'squander/decomposition/symbolic_monodromy_qlr.json',ROOT/'squander/decomposition/symbolic_reflection_involution.py']
    report=dict(schema='native-monodromy-exact-count-audit-v1',rows=rows,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        sampled_target_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        resource_scope='Fixed native pulse, arbitrary U(2) locals, scalar phase; U3 and PhasedXZ alphabets generate U(2) without GP via exact phase-absorption identities',
        parameter_scope='Every family record refers only to its named exact substitution; a matching full-family construction can separately establish a minimum uniform budget',
        proof_policy='Exact complete Makhlin invariant equality and rational primal/Farkas witnesses. LP floats are proposals only; no numerical fit is an accepted proof.',
        source_reference='https://quantum-journal.org/papers/q-2020-03-26-247/',maximum_count=a.max_count,
        summary=dict(Counter(r['status'] for r in rows)))
    with gzip.open(a.output,'wt') as stream:json.dump(report,stream,separators=(',',':'))
    print(json.dumps(report['summary']))
if __name__=='__main__':main()
