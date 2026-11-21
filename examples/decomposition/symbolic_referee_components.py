#!/usr/bin/env python3
"""Component ablations and full-matrix tests of compiled scalar-DAG binding."""
import argparse,gzip,hashlib,json,multiprocessing as mp,subprocess,sys,time
from pathlib import Path
import numpy as np
import sympy as sp
from symbolic_manuscript_experiments import ROOT,held_out_cases
import symbolic_synthesis as synthesis
import symbolic_gates as gates
from symbolic_archive import wall_time_limit,SymbolicDeadline
from symbolic_identity import prove_circuit_identity
from symbolic_expression_program import compile_numeric_program

def without_osr(target,basis,seconds):
    from symbolic_clifford_two_qubit import clifford_source_candidate
    from symbolic_native_pauli import native_cartan_candidate
    for fn in (clifford_source_candidate,native_cartan_candidate):
        value=fn(target,basis)
        if value is not None:return [value],dict(termination='non_OSR_constructor')
    return [],dict(termination='OSR_factorization_and_residual_search_disabled')

def cases():
    out=[(n,t,'sqrt_iswap_u') for n,t,_ in held_out_cases()]
    out += [(n,t,'berkeley_b') for n,t,_ in held_out_cases()]
    for width in (3,4):
        target=sp.eye(2**width)
        source=(('H',(0,)),('CX',(0,2)),('S',(2,)))
        if width==4:source+=(('CX',(1,3)),)
        for op in source:target=gates.apply_to(target,width,gates.make_gate(op[0],*op[1:-1]),op[-1])
        out.append(('opaque_product_'+str(width)+'q',sp.ImmutableMatrix(target),'cx_u'))
    return out

def ablate(job):
    name,basis,method=job
    target=next(t for n,t,b in cases() if n==name and b==basis)
    row=dict(case=name,basis=basis,method=method,width=int(sp.log(target.rows,2)),budget_seconds=20)
    original=synthesis._matrix_sources;start=time.perf_counter()
    try:
        if method=='without_OSR_construction':synthesis._matrix_sources=without_osr
        with wall_time_limit(20):
            value=synthesis.synthesize_matrix(target,basis,phase_exact=False,seconds=.2)
            row.update(status=value['status'],cost=value.get('cost'),result=value)
    except SymbolicDeadline:row['status']='budget_exhausted'
    except Exception as e:row.update(status='error',error=repr(e))
    finally:synthesis._matrix_sources=original
    row['wall_seconds']=time.perf_counter()-start
    print(name,basis,method,row['status'],flush=True);return row

def binding_test(archive):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    row=next(r for r in archive['rows'] if r['gate']=='Canonical' and r['basis']=='sqrt_iswap_u')
    result=row['projective'];program=result['expression_program']
    parameters=sp.symbols(' '.join(program['inputs']),real=True)
    outputs=[p for op in result['operations'] for p in op['parameters']]+[result['virtual_gp']['factor']]
    start=time.perf_counter();fn=compile_numeric_program(program,parameters,outputs);compile_seconds=time.perf_counter()-start
    points=[(0,0,0),(np.pi/4,0,0),(np.pi/4,np.pi/4,np.pi/4),(.1,.04,.02)]
    rng=np.random.default_rng(219);points+=list(map(tuple,rng.uniform(-np.pi,np.pi,(60,3))))
    rows=[];native=np.array(gates.make_gate('SiSWAP').evalf(18),complex)
    for values in points:
        vals=fn(dict(zip(parameters,values)));c=QuantumCircuit(2);index=0
        for op in result['operations']:
            n=len(op['parameters']);args=vals[index:index+n];index+=n
            if op['gate']=='U3':c.u(*(float(complex(x).real) for x in args),*op['wires'])
            elif op['gate']=='SiSWAP':c.unitary(native,op['wires'])
            else:c.unitary(np.array(gates.make_gate(op['gate'],*args).evalf(18),complex),op['wires'])
        target=np.array(gates.make_gate('Canonical',*values).evalf(18),complex)
        err=float(np.max(np.abs(complex(vals[-1])*Operator(c).data-target)))
        if err>1e-8:raise ArithmeticError('Compiled chart binding replay failed: '+str((values,err)))
        rows.append(dict(parameters=list(map(float,values)),maximum_entry_error=err))
    start=time.perf_counter()
    for i in range(1000):fn(dict(zip(parameters,points[i%len(points)])))
    return dict(binding_nodes=len(program['bindings']),operations=len(result['operations']),
                compile_seconds=compile_seconds,bindings=1000,binding_seconds=time.perf_counter()-start,
                exact_chart_verifier=result['verification'],numerical_replays=rows,
                scope='Numerical evaluation of an already certified symbolic DAG; no new numerical identity acceptance')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=dict(schema='referee-components-v1',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       ablation_contract='Disable OSR factor construction and residual peeling; retain native charts, Clifford/Pauli constructors and posthoc identity/lower-bound checks',ablations=[])
    with mp.get_context('fork').Pool(2) as pool:
        for row in pool.imap_unordered(ablate,[(n,b,m) for n,t,b in cases() for m in ('full','without_OSR_construction')]):
            report['ablations'].append(row)
            with gzip.open(a.output,'wt') as f:json.dump(report,f)
    with gzip.open(ROOT/'benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz','rt') as f:archive=json.load(f)
    report['dag_binding']=binding_test(archive)
    with gzip.open(a.output,'wt') as f:json.dump(report,f)

if __name__=='__main__':main()
