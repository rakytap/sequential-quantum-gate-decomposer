#!/usr/bin/env python3
"""Derive manuscript tables, figures and a portable source bundle from frozen data."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'squander/decomposition'))
from symbolic_bases import BASIS_SETS


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    return str(value).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def effective(row, mode):
    if mode == 'projective' and row.get('projective_omitted_as_no_better'):
        return row['phase_exact']
    return row.get(mode)


def selected_minima(archive):
    counts = Counter()
    for row in archive['rows']:
        for mode in ('phase_exact', 'projective'):
            result = effective(row, mode)
            if not result or result.get('status') != 'verified_symbolic':
                continue
            n = result['cost']['entanglers']
            certificates = [row.get(field, {}).get(mode, {}) for field in
                            ('independent_entangler_certificates', 'exchange_entangler_lower_bound_certificates')]
            scopes = [c.get('scope') for c in certificates
                      if c.get('lower_bound') == n and c.get('matched_construction_entanglers', n) == n]
            uniform = result.get('uniform_entangler_budget_certificate', {})
            fixed = result.get('entangler_optimality_certificate', {})
            if uniform.get('minimum') == n:
                scopes.append('minimum_uniform_family_budget')
            if fixed.get('method') == 'controlled_full_Schmidt_rank_excludes_three_Ising_interactions' and fixed.get('lower_bound') == n:
                scopes.append('fixed_target_minimum')
            if any(s in ('fixed_target_minimum', 'fixed_target_global_entangler_lower_bound') for s in scopes):
                counts[mode + '_fixed'] += 1
            elif any(s in ('minimum_uniform_family_budget', 'uniform_family_budget_minimum') for s in scopes):
                counts[mode + '_uniform'] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz')
    parser.add_argument('--experiments', type=Path, default=ROOT/'benchmarks/symbolic/manuscript_experiments_20261107.json.gz')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'docs/symbolic_manuscript')
    args = parser.parse_args()
    archive, experiments = load(args.archive), load(args.experiments)
    out = args.output_dir
    generated = out/'generated'
    generated.mkdir(parents=True, exist_ok=True)
    notable = json.loads((ROOT/'docs/symbolic_notable_results/notable_results.json').read_text())
    local = notable['comparison_summary']
    numbers = dict(OldSDKWins=archive['matched_reference_summary'].get('sdk_fixed_better', 0),
                   OldSDKTies=archive['matched_reference_summary'].get('sdk_fixed_equal', 0),
                   LocalWins=local.get('equal_E_fewer_locals', 0),
                   LocalTies=local.get('equal_E_equal_locals', 0),
                   LocalLosses=local.get('equal_E_more_locals', 0))
    (generated/'numbers.tex').write_text(''.join('\\newcommand{\\'+key+'}{'+str(value)+'}\n' for key,value in numbers.items()))
    minima = selected_minima(archive)
    lines = [r'\begin{center}\begin{tabular}{lrr}\toprule',
             r'Classification / selected certificate & Literal & Projective\\\midrule']
    for label, status in [('Symbolic construction', 'verified_symbolic'), ('Target/family obstruction', 'proved_impossible')]:
        lines.append(f"{label} & {archive['summary']['phase_exact'][status]:,} & {archive['summary']['projective'][status]:,}"+r'\\')
    lines += [r'\midrule', 'Selected fixed-target entangler minima & '+str(minima.get('phase_exact_fixed',0))+' & '+str(minima.get('projective_fixed',0))+r'\\',
              'Selected uniform-family budget minima & '+str(minima.get('phase_exact_uniform',0))+' & '+str(minima.get('projective_uniform',0))+r'\\',
              r'\bottomrule\end{tabular}\end{center}',
              'The selected-certificate rows take the union of the independent native/exchange audits, the new SQiSW family witnesses, and the four-Ising certificates. They exclude other OSR, Clifford and support-floor certificates and are not additive with those counts.']
    (generated/'coverage.tex').write_text('\n'.join(lines)+'\n')

    baseline = experiments['independent_fixed_targets']
    grouped = defaultdict(list)
    for row in baseline:
        if row.get('status') != 'not_compared':
            grouped[(row['target'],row['sdk'],row['input'],row['entangler'])].append(row)
    lines = [r'\begin{center}\small\begin{tabular}{llllrr}\toprule',
             r'Target & Tool & Input & Basis & $E$ & $L$\\\midrule']
    for key, rows in sorted(grouped.items()):
        assert len({(r['entanglers'],r['locals']) for r in rows}) == 1, 'Do not hide seed variation'
        name,sdk,kind,basis = key
        lines.append(' & '.join([esc(name),sdk,esc(kind),basis,str(rows[0]['entanglers']),str(rows[0]['locals'])])+r'\\')
    for name in ('SYC','fFredkin'):
        row = next(r for r in archive['rows'] if r['gate']==name and r['basis']=='cx_u')
        cost = effective(row,'projective')['cost']
        lines.append(' & '.join([name,'This work','exact identity','CX/U',str(cost['entanglers']),str(cost['single_qubit_gates'])])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
              'Both Qiskit seeds give the same displayed counts. Cirq uses projective replay; Qiskit replay includes its scalar. Raw operations, actual optimization levels and residual errors are archived.']
    (generated/'sdk.tex').write_text('\n'.join(lines)+'\n')

    methods = ('full','without_native_charts','source_lowering')
    labels = ('Full matrix synthesis','Native charts disabled','Independent source lowering')
    aggregate = {}
    lines = [r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',
             r'Method & Verified & Unresolved & Deadline & Errors\\\midrule']
    for method,label in zip(methods,labels):
        c = Counter(r['status'] for r in experiments['ablations'] if r['method']==method)
        aggregate[method]=dict(c)
        lines.append(label+' & '+' & '.join(str(c[s]) for s in ('verified_symbolic','unresolved','budget_exhausted','error'))+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
              'The full method verifies 11 of 12 case--preset pairs within the budget; the gauged affine Cartan/B case exhausts its 20 seconds. Source lowering verifies all 12, generally using more entanglers. This bounded failure is retained.']
    (generated/'ablation.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{center}\scriptsize\begin{tabular}{lllrrl}\toprule',
             r'Case & Preset & Method & $E$ & Seconds & Status\\\midrule']
    short = dict(x_controlled_rotation='X-control', y_controlled_rotation='Y-control',
                 unequal_controlled_su2_blocks='Unequal branches',number_conserving_with_local_phases='Gauged exchange',
                 gauged_affine_cartan='Affine Cartan',unequal_commuting_exchange='Unequal exchange')
    for row in sorted(experiments['ablations'], key=lambda r:(r['case'],r['basis'],r['method'])):
        lines.append(' & '.join([short[row['case']], 'SQiSW' if row['basis']=='sqrt_iswap_u' else 'B',
                                {'full':'Full','without_native_charts':'No chart','source_lowering':'Source'}[row['method']],
                                str(row['cost']['entanglers']) if row.get('cost') else '--',
                                f"{row['wall_seconds']:.3f}",esc(row['status'])])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    (generated/'ablation_detail.tex').write_text('\n'.join(lines)+'\n')
    with (generated/'ablation.csv').open('w') as stream:
        fields=['case','basis','method','status','wall_seconds','budget_seconds','entanglers','locals','parameter_score']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in experiments['ablations']:
            cost=row.get('cost') or {}
            writer.writerow({**{k:row.get(k) for k in fields[:6]},'entanglers':cost.get('entanglers'),
                             'locals':cost.get('single_qubit_gates'),'parameter_score':cost.get('parameter_score')})

    app=experiments['parameter_reuse']
    replay_error=max(v for r in app['numerical_replays'] for v in r.values())
    mantissa,exponent=f'{replay_error:.1e}'.split('e')
    lines=[r'\begin{center}\small\begin{tabular}{lrrrrr}\toprule',
           r'Route & $E$ & $L$ & Depth & SDK compile (s) & 1,000 binds (s)\\\midrule']
    for row in app['rows']:
        lines.append(('SDK source' if row['method']=='sdk_compile_once' else 'Symbolic source')+' & '+
                     ' & '.join([str(row['entanglers']),str(row['locals']),str(row['depth']),
                                 f"{row['compile_seconds']:.4f}",f"{row['binding_seconds']:.4f}"])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
              f"Cold symbolic chart synthesis takes {app['symbolic_chart_seconds']:.3f} seconds in this run, in addition to the symbolic route's SDK compile time. Consequently the cold symbolic workflow is slower here despite its smaller circuit; amortized timing depends on reuse. The largest of the ten full-matrix replay errors is "+
              '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$.']
    (generated/'application.tex').write_text('\n'.join(lines)+'\n')

    lines=[r'\begin{center}\scriptsize\begin{tabular}{lll}\toprule',
           r'Preset & Entangler & One-qubit alphabet\\\midrule']
    for name,basis in BASIS_SETS.items():
        lines.append(esc(name)+' & '+esc(basis.entangler)+' & '+esc(', '.join(basis.single_qubit_gates))+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    (generated/'presets.tex').write_text('\n'.join(lines)+'\n')
    versions=experiments['versions']
    environment=dict(python=platform.python_version(),platform=platform.platform(),**versions)
    lines=[r'\begin{center}\begin{tabular}{lr}\toprule Software & Version\\\midrule']
    lines += [esc(k)+' & '+esc(v)+r'\\' for k,v in [('Python',environment['python']),*versions.items()]]
    lines += [r'\bottomrule\end{tabular}\end{center}',
              'The experiment source revision is '+r'\texttt{'+experiments['git_revision'][:12]+'}. Four fork workers run the bounded jobs with '+r'\texttt{OPENBLAS\_NUM\_THREADS=1}'+'. Timings include concurrent-worker load and are single-run observations.']
    (generated/'environment.tex').write_text('\n'.join(lines)+'\n')

    from symbolic_controlled_rank_bound import four_cnot_source
    from symbolic_contract import encode_sequence
    from symbolic_circuit_atlas import qiskit_latex
    operations=encode_sequence(four_cnot_source())
    drawing=qiskit_latex(3,[(op,1) for op in operations])
    (generated/'ffredkin_circuit.tex').write_text('\n'.join(line.expandtabs().rstrip() for line in drawing.splitlines())+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,2.8),layout='constrained')
    left=np.zeros(3)
    for status,label,color in [('verified_symbolic','Verified','#247c5a'),('unresolved','Unresolved','#b2b2b2'),
                               ('budget_exhausted','Budget exhausted','#d17a18'),('error','Error','#ae3343')]:
        values=np.array([aggregate[m].get(status,0) for m in methods]);ax.barh(labels,values,left=left,label=label,color=color);left+=values
    ax.set_xlim(0,12);ax.set_xlabel('Case–preset pairs (12 per method)');ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=3,frameon=False)
    fig.savefig(generated/'ablation.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,3.0),layout='constrained')
    x=np.arange(3);rows=app['rows']
    for i,row in enumerate(rows):
        values=[row['entanglers'],row['locals'],row['depth']]
        bars=ax.bar(x+(i-.5)*.34,values,.34,label='SDK source' if i==0 else 'Symbolic source',color=['#607c9e','#247c5a'][i]);ax.bar_label(bars)
    ax.set_xticks(x,['CNOT count','Local count','Circuit depth']);ax.set_ylim(0,80);ax.legend(frameon=False)
    fig.savefig(generated/'application.pdf');plt.close(fig)
    manifest=dict(schema='symbolic-manuscript-build-v1',archive=str(args.archive.relative_to(ROOT)),
                  archive_sha256=sha(args.archive),experiments=str(args.experiments.relative_to(ROOT)),
                  experiments_sha256=sha(args.experiments),experiment_source_revision=experiments['git_revision'],
                  generated_numbers=numbers,selected_certificate_coverage=minima,ablations=aggregate,
                  environment=environment,source_generator_sha256=sha(Path(__file__)),
                  limitations=['20 retained grid secondary scores unresolved','One full opaque-matrix ablation exhausts its 20-second budget',
                               'No comparison run with UniversalQCompiler, Quartz, or QUESO',
                               'No unrestricted all-grid local-score optimality or literature priority claim'])
    manifest['generated_files']={str(p.relative_to(out)):sha(p) for p in sorted(generated.iterdir()) if p.is_file()}
    (out/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('generated_numbers','selected_certificate_coverage','ablations')},indent=2))


if __name__=='__main__':
    main()
