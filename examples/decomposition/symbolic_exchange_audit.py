#!/usr/bin/env python3
"""Attach replayed square-root-exchange lower bounds to verified circuits."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import sympy as sp
from symbolic_archive import symbolic_cases
from symbolic_bases import BASIS_SETS
from symbolic_gates import make_gate
from symbolic_entangler_proof import _specializations
from symbolic_exchange_invariants import exchange_three_pulse_lower_bound


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with gzip.open(path,'rt') as stream:
        return json.load(stream)


def write(path,data):
    with gzip.open(path,'wt') as stream:
        json.dump(data,stream,separators=(',',':'),allow_nan=False)


def witnesses(name,target):
    parameters=CASES[name]
    preferred={'Canonical':(sp.pi/8,)*3,'TK2':(sp.Rational(1,4),)*3,
               'SWAPPow':(sp.Rational(1,2),0),
               'fSim':(sp.pi/2,sp.pi/6),
               'PhasedFSim':(sp.pi/2,0,0,0,sp.pi/6)}.get(name)
    if preferred is not None:
        mapping={p:value for p,value in zip(parameters,preferred) if isinstance(p,sp.Symbol)}
        if set(mapping)==target.free_symbols:
            yield mapping
    yield from _specializations(target)


CASES={name:parameters for name,parameters,scope in symbolic_cases()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds-per-profile',type=float,default=30)
    args=parser.parse_args()
    archive=load(args.archive)
    cache={};records=[];statistics=Counter();timeouts=[]
    def expired(*unused):
        raise TimeoutError('Exchange proof budget expired')
    signal.signal(signal.SIGALRM,expired)
    for row in archive['rows']:
        native=BASIS_SETS[row['basis']].entangler
        if native not in {'SiSWAP','SSWAP'}:
            continue
        if not any(result and result.get('status')=='verified_symbolic'
                   and result['cost']['entanglers']>=3 for result in (row.get('phase_exact'),row.get('projective'))):
            continue
        name=row['gate']
        target=make_gate(name,*CASES[name])
        if target.shape!=(4,4):
            continue
        key=name,native
        if key not in cache:
            cached=None
            signal.setitimer(signal.ITIMER_REAL,args.seconds_per_profile)
            try:
                for substitution in witnesses(name,target):
                    specialized=sp.ImmutableMatrix(target.subs(substitution))
                    proof=exchange_three_pulse_lower_bound(specialized,native)
                    if proof:
                        # Replay the actual stored instance. The construction is
                        # unchanged; this proves only its independent lower bound.
                        assert exchange_three_pulse_lower_bound(specialized,native)==proof
                        cached=(substitution,proof)
                        break
            except TimeoutError:
                timeouts.append(dict(gate=name,native=native))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
            cache[key]=cached
        if cache[key] is None:
            continue
        substitution,proof=cache[key]
        for mode in ('phase_exact','projective'):
            result=row.get(mode)
            if mode=='projective' and row.get('projective_omitted_as_no_better'):
                result=row['phase_exact']
            if not result or result['status']!='verified_symbolic':
                continue
            upper=result['cost']['entanglers']
            if upper<proof['lower_bound']:
                raise ArithmeticError('Exchange bound contradicts construction: '+name+'/'+row['basis'])
            matched=upper==proof['lower_bound']
            scope=('minimum_uniform_family_budget' if substitution else 'fixed_target_minimum') if matched else 'lower_bound_only'
            certificate=dict(lower_bound=proof['lower_bound'],construction_entanglers=upper,
                matched=matched,scope=scope,substitutions={str(p):str(v) for p,v in substitution.items()},
                witness=proof,local_cost_optimality=False,
                specialization_caveat='Other members can need fewer pulses' if substitution else None)
            row.setdefault('exchange_entangler_lower_bound_certificates',{})[mode]=certificate
            if matched:
                # Use the same independent minimum display policy as the cut
                # audit. This does not change original construction flags.
                certificate['matched_construction_entanglers']=upper
                row.setdefault('independent_entangler_certificates',{})[mode]=certificate
            statistics[mode+'_'+scope]+=1
            records.append(dict(gate=name,basis=row['basis'],mode=mode,certificate=certificate))
        if len(cache)%20==0:
            print('Exchange profiles',len(cache),'certified phase modes',len(records),flush=True)
    paths=[Path(__file__),*sorted((ROOT/'squander/decomposition').glob('symbolic_*.py')),
           *sorted((ROOT/'squander/decomposition').glob('symbolic_*.json'))]
    report=dict(input_archive=str(args.archive),input_archive_sha256=digest(args.archive),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in paths},
        summary=dict(statistics),records=records,all_certificates_replayed=True,
        proof_timeouts=timeouts,
        proof_scope='Exact lower-bound witnesses and algebraic native identities; no new constructions',
        local_cost_claim=False)
    write(args.report,report)
    archive['exchange_entangler_proof_summary']=dict(statistics)
    archive['exchange_entangler_proof_supplement']=dict(path=str(args.report),sha256=digest(args.report),
        input_archive_sha256=report['input_archive_sha256'],git_revision=report['git_revision'],
        source_sha256=report['source_sha256'],all_certificates_replayed=True,
        modifies_original_constructions=False)
    write(args.output,archive)
    print(json.dumps(dict(statistics)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
