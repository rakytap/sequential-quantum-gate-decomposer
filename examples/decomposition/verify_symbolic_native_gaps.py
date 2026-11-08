#!/usr/bin/env python3
"""Replay archived numerical native words using the actual catalog matrices."""
import argparse,gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import sympy as sp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
from symbolic_gates import make_gate
from symbolic_archive import symbolic_cases
from symbolic_bases import get_basis


def verify(path):
    data=json.load(gzip.open(path,'rt'))
    cases={n:make_gate(n,*p) for n,p,_ in symbolic_cases()}
    counts=Counter();maximum=0.
    for row in data['rows']:
        counts[row['status']]+=1
        if row['status']!='verified_numerical_witness':continue
        assert row['numerical_only'] and not row['claims_exact_symbolic_identity'] and not row['claims_minimum']
        target=cases[row['gate']]
        replacements={s:sp.sympify(row['substitutions'][str(s)]) for s in target.free_symbols}
        target=np.array(target.subs(replacements).evalf(20),complex)
        basis=get_basis(row['basis']);actual=np.eye(4,dtype=complex);E=0
        for op in row['operations']:
            name=op['gate'];wires=op['wires']
            assert name!='GP'
            if len(wires)==2:
                assert name==basis.entangler and wires==[0,1];E+=1
            else:assert name in basis.single_qubit_gates and wires in ([0],[1])
            matrix=np.array(make_gate(name,*map(sp.Float,op['parameters'])).evalf(20),complex)
            if len(wires)==1:matrix=np.kron(np.eye(2),matrix) if wires==[0] else np.kron(matrix,np.eye(2))
            actual=matrix@actual
        error=float(np.max(np.abs(actual-target)));maximum=max(maximum,error)
        assert E==row['entanglers'] and error<1e-8,(row['gate'],row['basis'],error)
    return dict(schema='native-column-numerical-replay-v1',archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                statuses=dict(counts),maximum_literal_entry_error=maximum,
                acceptance='All witness operations use their declared alphabet and actual catalog matrices; no GP. Numerical only, not exact or minimum proofs.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'benchmarks/symbolic/native_column_gap_audit_20261107.json.gz')
    p.add_argument('--output',type=Path,default=ROOT/'docs/symbolic_manuscript/native-gap-verification.json')
    a=p.parse_args();report=verify(a.archive);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
