#!/usr/bin/env python3
"""Replay the new retained global-proof/native-chart records exactly."""

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import sympy as sp
from symbolic_archive import symbolic_cases
from symbolic_bases import get_basis
from symbolic_gates import make_gate
from symbolic_b_cartan import verify_b_cartan_certificate
from symbolic_basis_transfer import verify_basis_transfer_certificate
from symbolic_local_polar import verify_local_polar_certificate
from symbolic_clifford_t_global import _encode,_step,_multiply,solve_global_clifford_t
from symbolic_clifford_search import exact_state,group_data
from symbolic_cost_proof import negative_identity_certificate
from symbolic_feasibility import feasibility_obstruction
from symbolic_contract import validate_operations,circuit_cost,verify_circuit


FILES=['synthesis_global_b_charts_20261107.json.gz',
       'synthesis_global_clifford_t_20261107.json.gz',
       'synthesis_native_basis_transfer_20261107.json.gz',
       'synthesis_local_polar_iswap_20261107.json.gz',
       'synthesis_local_polar_peres_20261107.json.gz',
       'synthesis_native_euler_frames_20261107.json.gz']


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def projective_scalar_certificate():
    """Eight phases exhaust projectively Clifford outputs of H/S/T/Tdg.

    Entries lie in K=Q(sqrt(2),i), of degree four. Determinants of words
    are eighth roots, so a scalar times a Clifford has scalar squared an
    eighth root. An odd sixteenth root has degree eight and is outside K.
    Thus its scalar is an eighth root; no arbitrary phase is missed.
    """
    z=sp.Symbol('phase_root')
    degrees={n:sp.Poly(sp.cyclotomic_poly(n,z),z).degree() for n in (8,16)}
    if degrees!={8:4,16:8}:raise ArithmeticError('Cyclotomic degree proof failed')
    for name in ('H','S','T','Tdg'):
        matrix=make_gate(name);_encode(matrix)
        if sp.simplify(matrix.det()**8-1)!=0:raise ArithmeticError('Primitive determinant order changed')
    analysis,matrices,unused,names,unused=group_data()
    for name in analysis.representatives:
        matrix=matrices[names[name]];_encode(matrix)
        if sp.simplify(matrix.det()**4-1)!=0:raise ArithmeticError('Clifford determinant order changed')
    return dict(coefficient_field='Q(sqrt(2),i)=Q(zeta_8)',coefficient_field_degree=4,
        primitive_determinant_order_divides=8,Clifford_determinant_order_divides=4,
        excluded_odd_sixteenth_root_degree=8,complete_scalar_phase_group='eighth_roots_of_unity',
        implication='Minimizing over the eight exact phases proves the unrestricted projective word minimum')


def available_tradeoff_cost_certificate(archive):
    row=next(row for row in archive['rows'] if (row['gate'],row['basis'])==('C3X','zz_u'))
    current=row.get('projective') or row['phase_exact']
    def history(record):
        for mode in ('phase_exact','projective'):
            result=record.get(mode)
            if result and result.get('status')=='verified_symbolic':yield result
        for earlier in record.get('previous_attempts',[]):yield from history(earlier)
        if record.get('raw_attempt'):yield from history(record['raw_attempt'])
    shorter=[result for result in history(row)
             if result['cost']['entanglers']==current['cost']['entanglers']
             and result['cost']['single_qubit_gates']<current['cost']['single_qubit_gates']]
    if not shorter:return None
    available=min(shorter,key=lambda result:float(sp.sympify(result['cost']['parameter_score'])))
    operations=tuple((op['gate'],*(sp.sympify(p) for p in op['parameters']),tuple(op['wires'])) for op in available['operations'])
    cost=circuit_cost(operations).as_dict()
    if cost!=available['cost']:raise ArithmeticError('Historical shorter alternative changed score')
    if sp.simplify(sp.sympify(cost['parameter_score'])-sp.sympify(current['cost']['parameter_score'])).is_positive is not True:
        raise ArithmeticError('Available shorter circuit does not have the stated higher score')
    return dict(retained=current['cost'],available_shorter=cost,unchanged_by_objective_migration=True,
                claim_scope='These two witnesses; no exclusion of a better unknown circuit')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    archive=load(args.archive)
    cases={name:parameters for name,parameters,scope in symbolic_cases()}
    selected=set();source_checks=[];cache={}
    for name in FILES:
        path=ROOT/'benchmarks/symbolic'/name
        raw=load(path)
        indices=[index for index,run in enumerate(archive['synthesis_runs'])
                 if run.get('generated_at')==raw['generated_at'] and run.get('git_revision')==raw['git_revision']]
        if len(indices)!=1:raise ValueError('Raw source run is missing or ambiguous: '+name)
        selected.add(indices[0])
        for source,digest in raw['source_sha256'].items():
            relative=str(Path(source).relative_to(ROOT)) if Path(source).is_absolute() else source
            key=(raw['git_revision'],relative)
            if key not in cache:
                blob=subprocess.check_output(['git','show',raw['git_revision']+':'+relative],cwd=ROOT)
                cache[key]=hashlib.sha256(blob).hexdigest()
            if cache[key]!=digest:raise ValueError('Source hash mismatch: '+name+'/'+relative)
        source_checks.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            revision=raw['git_revision'],source_blobs=len(raw['source_sha256']),run_index=indices[0]))
    group=solve_global_clifford_t()
    counts=Counter();records=[]
    for row in archive['rows']:
        for mode in ('phase_exact','projective'):
            run=row.get('mode_source_run_indices',{}).get(mode,row.get('synthesis_run_index',0))
            if run not in selected:continue
            if len(records)<3 or len(records)%50==0:
                print('Replaying',len(records),row['gate'],row['basis'],mode,flush=True)
            target=make_gate(row['gate'],*cases[row['gate']]);basis=get_basis(row['basis'])
            result=row.get(mode)
            if mode=='projective' and row.get('projective_omitted_as_no_better'):
                result=copy.deepcopy(row['phase_exact']);result.update(phase_exact=False,virtual_gp=dict(factor='1',physical=False,cost=0))
            if result['status']=='proved_impossible':
                expected=feasibility_obstruction(target,basis,phase_exact=mode=='phase_exact')
                valid=expected is not None and all(result.get(key)==value for key,value in expected.items())
                method='exact_feasibility_obstruction'
            elif basis.name in {'clifford_t_cx','clifford_t_cz'}:
                operations=tuple((op['gate'],tuple(op['wires'])) for op in result['operations'])
                if any(op['parameters'] for op in result['operations']):raise ValueError('Parameterized fixed-alphabet word')
                operations=validate_operations(operations,basis,1)
                actual=_encode(sp.eye(2))
                for operation in operations:actual=_step(actual,operation[0])
                scalar=sp.S.One if mode=='phase_exact' else sp.sympify(result['virtual_gp']['factor'])
                state=exact_state(target)
                expected_length=(len(group['solutions'][state]) if mode=='phase_exact' else
                    min(len(word) for index,word in group['solutions'].items() if index//8==state//8))
                valid=(_multiply(_encode(scalar*sp.eye(2)),actual)==_encode(target)
                    and circuit_cost(operations).as_dict()==result['cost'] and len(operations)==expected_length)
                method='integer_ring_matrix_identity_and_exhaustive_global_word_minimum'
            else:
                method=result['verification']
                verifier={'exact_two_B_Cartan_chart_and_native_lowering':verify_b_cartan_certificate,
                    'exact_native_parent_local_components_and_CZ_commutations':verify_basis_transfer_certificate,
                    'exact_parent_U2_polar_components_and_phase_absorption':verify_local_polar_certificate}.get(method)
                if verifier is None:raise ValueError('Unexpected retained proof: '+method)
                candidate=copy.deepcopy(result)
                certificate=candidate.get('basis_transfer_certificate')
                if certificate is not None and 'native_variant_dp' not in certificate:
                    # The old direct Rz/Rx witness predates these descriptive
                    # fields. Its full parent, components, native word, cost
                    # and scalar are unchanged and are all replayed.
                    if basis.name!='cz_rz_rx':raise ValueError('Unsupported old variant schema')
                    certificate.update(native_variant_dp=False,variant_optimality_scope='constructive finite variant search')
                valid=verifier(target,candidate,basis)
            if not valid:raise ArithmeticError('Proof replay failed: '+row['gate']+'/'+row['basis']+'/'+mode)
            counts[method]+=1
            records.append(dict(gate=row['gate'],basis=row['basis'],mode=mode,method=method,source_run=run))
    # Exact pole regressions also exercise the otherwise absent-radius chart.
    poles=[]
    from symbolic_b_cartan import synthesize_b_cartan
    for sign in (-1,1):
        target=make_gate('Canonical',sp.pi/8,0,sign*sp.pi/4)
        for exact in (False,True):
            print('Replaying exact B pole',sign,exact,flush=True)
            result=synthesize_b_cartan(target,get_basis('berkeley_b'),phase_exact=exact)
            operations=tuple((op['gate'],*(sp.sympify(p) for p in op['parameters']),tuple(op['wires'])) for op in result['operations'])
            factor=None if exact else sp.sympify(result['virtual_gp']['factor'])
            proof=verify_circuit(target,operations,get_basis('berkeley_b'),phase_exact=exact,scalar_factor=factor)
            if proof['status']!='verified_symbolic':raise ArithmeticError('Exact B pole failed')
            poles.append(dict(z=str(sign*sp.pi/4),phase_exact=exact,verification=proof['verification']))
    report=dict(archive=str(args.archive),archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        covered_phase_modes=len(records),methods=dict(counts),failures=[],source_checks=source_checks,
        distinct_source_blobs=len(cache),clifford_exact_word_ball={key:value for key,value in group.items() if key!='solutions'},
        historical_cost_obstruction=negative_identity_certificate(),exact_B_pole_replays=poles,records=records,
        projective_scalar_completeness=projective_scalar_certificate(),
        c3x_available_tradeoff_v2_cost_check=available_tradeoff_cost_certificate(archive),
        policy='Exact arithmetic, component lemmas, and source hashes; no numerical acceptance')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('covered_phase_modes','methods','distinct_source_blobs','failures')}),flush=True)


if __name__=='__main__':main()
