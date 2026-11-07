#!/usr/bin/env python3
"""Create a scientific archive view and disjoint, row-resolved review tables."""
import copy,csv,gzip,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'squander/decomposition'))
from symbolic_archive_analysis import merge_synthesis_runs
from symbolic_gates import make_gate
from symbolic_archive import symbolic_cases

DATA=ROOT/'benchmarks/symbolic';DOC=ROOT/'docs/symbolic_manuscript';OUT=DOC/'generated'
def load(name):return json.load(gzip.open(DATA/name,'rt'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def effective(r,m):return r['phase_exact'] if m=='projective' and r.get('projective_omitted_as_no_better') else r.get(m)
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(name,header,rows,columns=None):
    if columns is None:columns='l'+'r'*(len(header)-1)
    text=[r'\begin{center}\small\begin{tabular}{'+columns+r'}\toprule',' & '.join(header)+r'\\\midrule']
    text+=[' & '.join(map(str,row))+r'\\' for row in rows]
    text+=[r'\bottomrule\end{tabular}\end{center}']
    (OUT/(name+'.tex')).write_text('\n'.join(text)+'\n')

def archive_view():
    original=load('synthesis_reference_20261107_manuscript.json.gz')
    new=load('synthesis_ffredkin_referee_20261107.json.gz')
    merged=merge_synthesis_runs([original,new]);index={(r['gate'],r['basis']):r for r in original['rows']}
    for row in merged['rows']:
        old=index[row['gate'],row['basis']]
        for field in ('independent_entangler_certificates','exchange_entangler_lower_bound_certificates','secondary_cost_floor_certificates'):
            if field in old:
                row[field]=copy.deepcopy(old[field])
                for mode,c in list(row[field].items()):
                    result=effective(row,mode)
                    matched=c.get('matched_construction_entanglers',c.get('entangler_count'))
                    if result and result['status']=='verified_symbolic' and matched is not None and result['cost']['entanglers']!=matched:
                        del row[field][mode]
                    elif field=='secondary_cost_floor_certificates' and result and result.get('cost',{}).get('parameter_score')!=c.get('matched_score'):
                        del row[field][mode]
    edits=Counter()
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items() if k not in ('previous_attempts','raw_attempt')}
        if isinstance(value,list):return list(map(clean,value))
        if isinstance(value,str):
            for old,new in [('deferred_by_user','not_run_license_unavailable'),('user declined purchase','kernel unavailable')]:
                if old in value:edits[old]+=1;value=value.replace(old,new)
        return value
    merged=clean(merged)
    base=clean(copy.deepcopy(original))
    base['scientific_view']=dict(original_archive_sha256=sha(DATA/'synthesis_reference_20261107_manuscript.json.gz'),
         policy='Metadata-only scientific view; construction signatures unchanged')
    with gzip.open(DATA/'synthesis_reference_20261107_referee_base.json.gz','wt') as f:json.dump(base,f,separators=(',',':'))
    merged['scientific_view']=dict(original_archive_sha256=sha(DATA/'synthesis_reference_20261107_manuscript.json.gz'),
         supplement_sha256=sha(DATA/'synthesis_ffredkin_referee_20261107.json.gz'),
         editorial_metadata_replacements=dict(edits),omitted_fields=['previous_attempts','raw_attempt'],
         policy='Construction payloads are unchanged except for cost-selected, exactly verified fFredkin replacements; historical attempts remain in immutable source archives')
    path=DATA/'synthesis_reference_20261107_referee.json.gz'
    with gzip.open(path,'wt') as f:json.dump(merged,f,separators=(',',':'))
    return merged,path

def coverage(a):
    widths={n:int(make_gate(n,*p).rows).bit_length()-1 for n,p,_ in symbolic_cases()}
    counts=defaultdict(Counter);obstructions=defaultdict(Counter);ledger=[]
    for r in a['rows']:
        w=widths[r['gate']]
        for mode in ('phase_exact','projective'):
            v=effective(r,mode);c=counts[mode,w];c['cells']+=1
            if v['status']=='proved_impossible':c['obstruction']+=1;obstructions[mode][v['obstruction']]+=1;continue
            E=v['cost']['entanglers'];c['constructed']+=1
            if E==0:c['zero']+=1;continue
            c['positive']+=1;sources=[];scopes=[]
            raw=r.get(mode) or {}
            flag=raw.get('entangler_optimal') is True
            if flag:
                c['raw_flag']+=1
            for field in ('independent_entangler_certificates','exchange_entangler_lower_bound_certificates'):
                certificate=r.get(field,{}).get(mode,{})
                if certificate.get('lower_bound')==E and certificate.get('matched_construction_entanglers',E)==E:
                    scope=certificate.get('scope','')
                    if scope in ('fixed_target_minimum','fixed_target_global_entangler_lower_bound'):
                        sources.append(field);scopes.append('fixed')
                    elif scope in ('minimum_uniform_family_budget','uniform_family_budget_minimum'):
                        sources.append(field);scopes.append('uniform')
            for key in ('entangler_optimality_certificate','uniform_entangler_budget_certificate'):
                cert=v.get(key,{})
                if cert.get('lower_bound',cert.get('minimum'))!=E:continue
                scope=cert.get('scope','')
                if scope=='fixed_target_global_entangler_lower_bound':scopes.append('fixed');sources.append(key)
                elif key=='uniform_entangler_budget_certificate':scopes.append('uniform');sources.append(key)
            osr=v.get('osr_certificate',{})
            if osr.get('lower_bound')==E and osr.get('condition')=='True' or v.get('cnot_invariant_lower_bound')==E:
                scopes.append('uniform' if r['target_scope']=='free_real_family' else 'fixed');sources.append('phase_independent_exact_lower_bound')
            scope='fixed' if 'fixed' in scopes else 'uniform' if 'uniform' in scopes else 'unproved'
            c[scope]+=1
            ledger.append(dict(gate=r['gate'],basis=r['basis'],mode=mode,width=w,entanglers=E,
                     raw_mode_flag=flag,proof_scope=scope,proof_sources=';'.join(sorted(set(sources)))))
    with (OUT/'optimality_ledger.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]),lineterminator='\n');writer.writeheader();writer.writerows(ledger)
    for mode in ('phase_exact','projective'):
        rows=[]
        for w in sorted(widths.values()):
            if any(row[0]==w for row in rows):continue
            c=counts[mode,w];rows.append([w,c['cells'],c['zero'],c['positive'],c['fixed'],c['uniform'],c['unproved'],c['obstruction']])
        total=sum((c for (m,w),c in counts.items() if m==mode),Counter())
        rows.append(['Total',*[total[k] for k in ('cells','zero','positive','fixed','uniform','unproved','obstruction')]])
        table('coverage_'+mode,['Qubits','Cells','$E=0$','$E>0$','Fixed min.','Family min.','Unproved','Obstructed'],rows)
    kinds=sorted(set().union(*map(set,obstructions.values())))
    table('obstructions',['Obstruction','Literal','Projective'],[[esc(k),obstructions['phase_exact'][k],obstructions['projective'][k]] for k in kinds])
    return dict(by_width={m:{str(w):dict(counts[m,w]) for w in sorted(set(widths.values()))} for m in ('phase_exact','projective')},
                obstructions={m:dict(v) for m,v in obstructions.items()},
                row_ledger_sha256=sha(OUT/'optimality_ledger.csv'))

def main():
    OUT.mkdir(exist_ok=True);a,path=archive_view();report=coverage(a)
    sdks=load('referee_strong_sdks_final_20261107.json.gz');rules=load('referee_rules_20261107.json.gz')
    rows=[]
    for name in ('SYC','iSWAP','SSWAP','fFredkin','CCX','CSWAP','C3X','C4X'):
        row=[name]
        for tool in ('qiskit','pytket','bqskit','quartz','queso'):
            choices=[r for r in sdks['rows']+rules['rows'] if r['tool']==tool and r['gate']==name and r['status']=='verified_numeric']
            values=[r.get('normalized_cx_u3',r) for r in choices]
            best=min(values,key=lambda r:(r['entanglers'],r['locals'])) if values else None
            row.append(str(best['entanglers'])+'/'+str(best['locals']) if best else '--')
        result=effective(next(r for r in a['rows'] if r['gate']==name and r['basis']=='cx_u'),'projective')
        row.append(str(result['cost']['entanglers'])+'/'+str(result['cost']['single_qubit_gates']));rows.append(row)
    table('strong_sdks',['Gate','Qiskit','tket','BQSKit','Quartz','QUESO','Exact atlas'],rows,'lrrrrrr')
    app=load('manuscript_experiments_20261107.json.gz')['parameter_reuse'];numeric=app['numeric_recompile_rows']
    rows=[]
    for r in app['rows']:
        rows.append(['Gatewise analytical source' if r['method']=='sdk_compile_once' else 'Three-CX symbolic source',
                    r['entanglers'],r['locals'],r['depth'],f"{r['compile_seconds']*1000:.2f}",f"{r['binding_seconds']*1000:.2f}"])
    rows.append(['Numerical compile per binding',18,40,25,f"{1000*sum(r['wall_seconds'] for r in numeric)/len(numeric):.2f}",'--'])
    table('application_revised',['Route','$E$','$L$','Depth','Compile (ms)','1,000 binds (ms)'],rows)
    component=load('referee_components_20261107.json.gz')
    rows=[]
    for width in (2,3,4):
        for method in ('full','without_OSR_construction'):
            selected=[r for r in component['ablations'] if r['width']==width and r['method']==method];c=Counter(r['status'] for r in selected)
            rows.append([width,'Full' if method=='full' else 'OSR construction off',len(selected),c['verified_symbolic'],c['unresolved'],c['budget_exhausted'],c['error']])
    table('osr_ablation',['Qubits','Method','Cases','Verified','Unresolved','Deadline','Errors'],rows,'llrrrrr')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,2.7),layout='constrained')
    labels=['2q full','2q OSR off','3q full','3q OSR off','4q full','4q OSR off']
    for i,(width,method) in enumerate((w,m) for w in (2,3,4) for m in ('full','without_OSR_construction')):
        counter=Counter(r['status'] for r in component['ablations'] if r['width']==width and r['method']==method)
        left=0
        for status,color in [('verified_symbolic','#247c5a'),('unresolved','#aaaaaa'),('budget_exhausted','#ce861e')]:
            value=counter[status];ax.barh(i,value,left=left,color=color,label=status.replace('_',' ') if i==0 else None);left+=value
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlabel('Case–preset pairs');ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,1.23),frameon=False)
    fig.savefig(OUT/'referee_ablation.pdf');plt.close(fig)
    from symbolic_bases import BASIS_SETS
    table('presets',['Preset','Entangler','Local alphabet'],[[esc(n),esc(b.entangler),esc(', '.join(b.single_qubit_gates))] for n,b in BASIS_SETS.items()],'lll')
    from symbolic_controlled_rank_bound import four_cnot_source
    from symbolic_contract import encode_sequence
    from symbolic_circuit_atlas import qiskit_latex
    drawing=qiskit_latex(3,[(op,1) for op in encode_sequence(four_cnot_source())])
    (OUT/'ffredkin_circuit.tex').write_text('\n'.join(line.expandtabs().rstrip() for line in drawing.splitlines())+'\n')
    peephole=load('referee_peephole_20261107.json.gz');rows=[]
    for r in peephole['rows']:
        rows.append([esc(Path(r['path']).stem),'B' if r['basis']=='berkeley_b' else 'SQiSW',r['width'],*r['before'],*r['after'],len(r['matches'])])
    table('peephole',['Circuit','Basis','Qubits','$E$ before','$L$ before','$E$ after','$L$ after','Rules'],rows,'llrrrrrr')
    report.update(schema='symbolic-referee-revision-v1',archive=str(path.relative_to(ROOT)),archive_sha256=sha(path),
         sdk_statuses={t:dict(Counter(r['status'] for r in sdks['rows']+rules['rows'] if r['tool']==t)) for t in ('qiskit','pytket','bqskit','quartz','queso')},
         dag_binding=component['dag_binding'],peephole_circuits=len(peephole['rows'])//2,
         experiments={p.name:sha(p) for p in DATA.glob('referee_*.json.gz') if 'strong_sdks_20261107' not in p.name})
    (DOC/'referee-revision.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('dag_binding','by_width','experiments')},indent=2))

if __name__=='__main__':main()
