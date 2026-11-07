#!/usr/bin/env python3
"""Exact atlas peepholes on eight bounded, existing QASM benchmarks.

The control is gatewise atlas lowering, not a claim against an optimized SDK.
Numerical fingerprints only propose matches. Exact equality accepts every rule.
"""
import gzip,hashlib,json,re,sys,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import sympy as sp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import symbolic_gates as gates
from symbolic_identity import prove_circuit_identity

ALIASES={'u':'U3','u3':'U3','u2':'U3','u1':'P','h':'H','x':'X','y':'Y','z':'Z','s':'S','sdg':'Sdg','t':'T','tdg':'Tdg',
         'cx':'CX','cz':'CZ','ccx':'CCX','swap':'SWAP','cswap':'CSWAP','rx':'Rx','ry':'Ry','rz':'Rz','cp':'CP'}
def parse(path):
    text=re.sub(r'//[^\n]*','',path.read_text());registers={};width=0;word=[]
    for statement in text.split(';'):
        line=statement.strip()
        if not line or line.startswith(('OPENQASM','include','creg','barrier')):continue
        match=re.fullmatch(r'qreg\s+(\w+)\[(\d+)\]',line)
        if match:registers[match[1]]=width;width+=int(match[2]);continue
        if line.startswith('measure'):continue
        match=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?\s+(.+)',line)
        if not match or match[1] not in ALIASES:raise ValueError('Unsupported QASM: '+line)
        args=tuple(sp.sympify(p,locals={'pi':sp.pi},rational=True) for p in (match[2] or '').split(',') if p)
        qs=tuple(registers[r]+int(i) for r,i in re.findall(r'(\w+)\[(\d+)\]',match[3]))
        if match[1]=='u2':args=(sp.pi/2,)+args
        gates.make_gate(ALIASES[match[1]],*args)
        word.append((ALIASES[match[1]],*args,qs))
    return width,tuple(word)

def effective(row):return row['phase_exact'] if row.get('projective_omitted_as_no_better') else row['projective']
def main():
    archive=json.load(gzip.open(ROOT/'benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz','rt'))
    index={(r['gate'],r['basis']):r for r in archive['rows']}
    presets=('berkeley_b','sqrt_iswap_u');patterns={};setup=time.perf_counter()
    for name in ('CNOT','CZ','SWAP','iSWAP','DCX','CY','CH','SYC','SSWAP'):
        try:matrix=gates.make_gate(name);numeric=np.array(matrix.evalf(18),complex)
        except (ValueError,KeyError):continue
        patterns[name]=(matrix,numeric)
    @lru_cache(None)
    def cost(op,basis):
        if len(op[-1])==1:return (0,1)
        row=index.get(({'CX':'CNOT'}.get(op[0],op[0]),basis))
        if row is None or op[1:-1]:raise ValueError('Uncataloged benchmark primitive '+op[0])
        result=effective(row)
        if not result or result['status']!='verified_symbolic':raise ValueError('No construction '+op[0])
        return result['cost']['entanglers'],result['cost']['single_qubit_gates']
    candidates=[]
    for folder in ('IBMEagle','QASMBenchmarks'):
        for path in sorted((ROOT/'benchmarks/partitioning'/folder).glob('*.qasm')):
            text=path.read_text();width=sum(map(int,re.findall(r'qreg\s+\w+\[(\d+)\]',text)))
            if 2<=width<=8 and text.count(';')<500:candidates.append(path)
    control_word=(('CX',(0,1)),('CX',(1,0)),('CX',(0,1)))
    assert prove_circuit_identity(gates.make_gate('SWAP'),control_word) is True
    controls=[dict(case='three-CX SWAP positive control',basis=b,
             before=[sum(cost(op,b)[k] for op in control_word) for k in (0,1)],
             after=list(cost(('SWAP',(0,1)),b)),exact_identity=True) for b in presets]
    report=dict(schema='atlas-peephole-suite-v1',positive_controls=controls,selection='All IBMEagle and QASMBenchmarks files with 2..8 total declared qubits and <500 semicolons',
       comparison='Gatewise atlas lowering versus contiguous exact two-wire atlas replacement; same lowering control on unmatched gates',
       ancillas=0,fixed_wires=True,maximum_window=8,rows=[],setup_seconds=time.perf_counter()-setup)
    for path in candidates:
        width,word=parse(path);start=time.perf_counter();matches=[];i=0
        while i<len(word):
            support=set();accepted=None
            for end in range(i,min(i+8,len(word))):
                support.update(word[end][-1])
                if len(support)>2:break
                if len(support)!=2 or end==i:continue
                wires=sorted(support);local=tuple((*op[:-1],tuple(wires.index(q) for q in op[-1])) for op in word[i:end+1])
                if sum(len(op[-1])==2 for op in local)<2:continue
                matrix=sp.eye(4)
                for op in local:matrix=gates.apply_to(matrix,2,gates.make_gate(op[0],*op[1:-1]),op[-1])
                actual=np.array(matrix.evalf(16),complex)
                for name,(target,num) in patterns.items():
                    factor=np.vdot(num.flatten(),actual.flatten())/4
                    if abs(abs(factor)-1)>1e-10 or np.max(np.abs(actual-factor*num))>1e-10:continue
                    pivot=next(k for k,v in enumerate(target) if v!=0);exact=sp.simplify(matrix[pivot]/target[pivot])
                    if prove_circuit_identity(exact*target,local) is not True:continue
                    gains={}
                    for basis in presets:
                        before=tuple(sum(cost(op,basis)[k] for op in local) for k in (0,1))
                        after=cost((name,(0,1)),basis)
                        if after<before:gains[basis]=dict(before=before,after=after)
                    if gains:accepted=dict(start=i,end=end+1,gate=name,wires=wires,scalar=str(exact),gains=gains,
                                           exact_identity=True,source_word=[str(op) for op in local])
            if accepted:matches.append(accepted);i=accepted['end']
            else:i+=1
        for basis in presets:
            before=[sum(cost(op,basis)[k] for op in word) for k in (0,1)];after=list(before)
            for m in matches:
                if basis in m['gains']:
                    for k in (0,1):after[k]+=m['gains'][basis]['after'][k]-m['gains'][basis]['before'][k]
            report['rows'].append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 width=width,source_operations=len(word),basis=basis,before=before,after=after,
                 matches=[m for m in matches if basis in m['gains']],match_seconds=time.perf_counter()-start,
                 proof='Composition of exact window identities and archived exact primitive lowerings'))
            print(path.name,basis,before,after,flush=True)
        with gzip.open(ROOT/'benchmarks/symbolic/referee_peephole_20261107.json.gz','wt') as f:json.dump(report,f)

if __name__=='__main__':main()
