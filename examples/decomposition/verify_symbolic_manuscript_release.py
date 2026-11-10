#!/usr/bin/env python3
"""Replay changed constructions and audit provenance of the frozen manuscript grid.

Unchanged construction payloads are compared with the immutable pre-manuscript
archive, rather than relabeled as newly replayed. This checks analytical proof
components, not a formalization of the accompanying human-readable theorem.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import sympy as sp
from symbolic_archive import symbolic_cases,wall_time_limit,SymbolicDeadline
from symbolic_bases import get_basis
from symbolic_gates import make_gate
from symbolic_contract import verify_circuit
from symbolic_sqrt_iswap_cartan import verify_sqrt_iswap_cartan_certificate
from symbolic_controlled_rank_bound import controlled_full_rank_lower_bound
from symbolic_controlled_chart import verify_controlled_chart_certificate


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def effective(row,mode):
    value=copy.deepcopy(row.get(mode))
    if mode=='projective' and row.get('projective_omitted_as_no_better'):
        value=copy.deepcopy(row['phase_exact'])
        if value['status']=='verified_symbolic':
            value.update(phase_exact=False,virtual_gp=dict(factor='1',physical=False,cost=0))
    return value


def signature(result):
    keys=('status','operations','cost','virtual_gp','expression_program','verification','phase_exact')
    return {key:result.get(key) for key in keys}


def replay(job):
    try:return _replay(job)
    except SymbolicDeadline as error:
        # multiprocessing.Pool transports Exception, not BaseException.
        raise TimeoutError('Changed-construction proof deadline: '+str(job[:3])) from error


def _replay(job):
    name,basis_name,mode,result,seconds=job
    start=time.perf_counter()
    with wall_time_limit(seconds):
        parameters=next(p for n,p,scope in symbolic_cases() if n==name)
        target=make_gate(name,*parameters);basis=get_basis(basis_name)
        if result['verification']=='exact_three_SQiSW_Pauli_tensor_chart':
            valid=verify_sqrt_iswap_cartan_certificate(target,result,basis)
        elif result['verification']=='exact_controlled_SU2_chart_and_native_phase_lemma':
            valid=verify_controlled_chart_certificate(target,result,basis)
        else:
            symbols={str(x):x for x in target.free_symbols}
            operations=tuple((op['gate'],*(sp.sympify(p,locals=symbols) for p in op['parameters']),tuple(op['wires']))
                             for op in result['operations'])
            factor=None if mode=='phase_exact' else sp.sympify(result['virtual_gp']['factor'],locals=symbols)
            proof=verify_circuit(target,operations,basis,phase_exact=mode=='phase_exact',scalar_factor=factor)
            valid=proof['status']=='verified_symbolic' and proof['cost']==result['cost']
        certificate=result.get('entangler_optimality_certificate',{})
        if certificate.get('method')=='controlled_full_Schmidt_rank_excludes_three_Ising_interactions':
            computed=controlled_full_rank_lower_bound(target,basis.entangler)
            valid=valid and json.loads(json.dumps(computed))==certificate
        if not valid:raise ArithmeticError('Changed identity/certificate replay failed: '+str((name,basis_name,mode)))
    return dict(gate=name,basis=basis_name,mode=mode,status='replayed',
                method=result['verification'],wall_seconds=time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz')
    parser.add_argument('--base',type=Path,default=ROOT/'benchmarks/symbolic/synthesis_reference_20261107_manuscript_base.json.gz')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/symbolic_manuscript/release-verification.json')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();archive=load(args.archive);base=load(args.base)
    index={(r['gate'],r['basis']):r for r in base['rows']}
    jobs=[];unchanged=0
    for row in archive['rows']:
        for mode in ('phase_exact','projective'):
            result=effective(row,mode);old=effective(index[row['gate'],row['basis']],mode)
            if signature(result)==signature(old):unchanged+=1
            else:
                if result['status']!='verified_symbolic':raise ArithmeticError('Unexpected changed feasibility record')
                jobs.append((row['gate'],row['basis'],mode,result,90))
    checks=[];cache={};resolved={};history={};revision_exceptions=[]
    for i,run in enumerate(archive['synthesis_runs']):
        for name,expected in run['source_sha256'].items():
            path=Path(name)
            relative=str(path.relative_to(ROOT)) if path.is_absolute() else name
            key=(run['git_revision'],relative)
            if key not in cache:
                process=subprocess.run(['git','show',run['git_revision']+':'+relative],cwd=ROOT,capture_output=True)
                cache[key]=hashlib.sha256(process.stdout).hexdigest() if process.returncode==0 else None
            if cache[key]!=expected:
                # Early exploratory runs recorded a HEAD preceding uncommitted
                # sources. Resolve their recorded content hashes explicitly;
                # never claim these bytes existed at the original revision.
                content_key=(relative,expected)
                if content_key not in resolved:
                    if relative not in history:
                        history[relative]=subprocess.check_output(['git','log','--all','--format=%H','--',relative],cwd=ROOT,text=True).splitlines()
                    for revision in history[relative]:
                        data=subprocess.check_output(['git','show',revision+':'+relative],cwd=ROOT)
                        if hashlib.sha256(data).hexdigest()==expected:
                            resolved[content_key]=revision;break
                    else:raise ArithmeticError('No committed source content matches archive hash: '+str(content_key))
                revision_exceptions.append(dict(run=i,path=relative,recorded_revision=run['git_revision'],
                                                content_sha256=expected,resolved_content_revision=resolved[content_key]))
        checks.append(dict(run=i,revision=run['git_revision'],source_blobs=len(run['source_sha256'])))
    report=dict(schema='symbolic-manuscript-release-verification-v1',archive_sha256=sha(args.archive),
                preserved_base_sha256=sha(args.base),phase_modes=2*len(archive['rows']),
                unchanged_construction_modes=unchanged,changed_construction_modes=len(jobs),
                source_checks=checks,distinct_revision_source_pairs=len(cache),all_source_checks_passed=True,
                source_policy='Verify recorded revision where possible; otherwise resolve exact recorded source hash from Git history and list the exception',
                recorded_revision_exceptions=revision_exceptions,
                retained_unresolved_scores=archive['secondary_scores_unresolved'],replays=[])
    with mp.get_context('fork').Pool(args.workers) as pool:
        for row in pool.imap_unordered(replay,jobs):
            report['replays'].append(row)
            print(row['gate'],row['basis'],row['mode'],'replayed',flush=True)
    report['all_changed_constructions_replayed']=len(report['replays'])==len(jobs)
    report['replay_methods']=dict(Counter(r['method'] for r in report['replays']))
    report['limits']=['Unchanged modes are payload comparisons, not new identity replays',
                      'Old native/exchange reports retain their immutable input and proof-attempt timeouts',
                      'Four-Ising theorem is analytical, with algebraic components checked by this release']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_checks','replays')},indent=2))


if __name__=='__main__':main()
