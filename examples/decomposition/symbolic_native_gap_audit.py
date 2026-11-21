#!/usr/bin/env python3
"""Native-pulse numerical upper witnesses; never exact identity certificates.

Fit one representative per numeric Weyl orbit, transport the local frames,
then replay every catalog target independently. Family rows identify one
exactly specified sample and do not claim a free-parameter identity.
"""
import argparse,gzip,hashlib,json,multiprocessing as mp,os,sys,time
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.optimize import least_squares
import cirq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'squander/decomposition'))
import symbolic_gates as gates
from symbolic_archive import symbolic_cases

def u3(p):
    t,f,l=p;c,s=np.cos(t/2),np.sin(t/2)
    return np.array([[c,-np.exp(1j*l)*s],[np.exp(1j*f)*s,np.exp(1j*(f+l))*c]])

def layer(parameters):return np.kron(u3(parameters[3:]),u3(parameters[:3]))
def product(parameters,native,count):
    actual=layer(parameters[:6])
    for i in range(count):actual=layer(parameters[6*(i+1):6*(i+2)])@native@actual
    return actual
def scalar(actual,target):
    value=np.vdot(actual.flatten(),target.flatten())/4
    return value/abs(value) if abs(value)>1e-15 else 1

def fit(job):
    entangler,key,coordinates,attempts=job
    native=np.array(gates.make_gate(entangler).evalf(20),complex)
    target=np.array(gates.make_gate('Canonical',*coordinates).evalf(20),complex)
    rng=np.random.default_rng(219+sum(int(round(abs(t)*1e6)) for t in coordinates));start=time.perf_counter();failed=[]
    own=cirq.kak_decomposition(native).interaction_coefficients
    if np.max(np.abs(np.array(own)-coordinates))<1e-8:
        kak=cirq.kak_decomposition(native)
        before=[m.conj().T for m in kak.single_qubit_operations_before]
        after=[m.conj().T for m in kak.single_qubit_operations_after]
        return dict(entangler=entangler,key=key,count=1,base='inverse_native_KAK',before=before,after=after,wall_seconds=time.perf_counter()-start)
    for count in (2,3):
        best=2.
        for repeat in range(attempts):
            x=rng.uniform(-np.pi,np.pi,6*(count+1))
            def residual(p):
                actual=product(p,native,count);d=scalar(actual,target)*actual-target
                return np.r_[d.real.ravel(),d.imag.ravel()]
            solution=least_squares(residual,x,max_nfev=350,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            actual=product(solution.x,native,count)
            error=float(np.max(np.abs(scalar(actual,target)*actual-target)));best=min(best,error)
            if error<1e-9:
                print(entangler,key,count,'fit',flush=True)
                return dict(entangler=entangler,key=key,count=count,parameters=solution.x.tolist(),fit_error=error,
                            wall_seconds=time.perf_counter()-start,lower_count_searches=failed)
        failed.append(dict(count=count,best_maximum_entry_error=best,starts=attempts))
    print(entangler,key,'unresolved',flush=True)
    return dict(entangler=entangler,key=key,status='no_fit_within_budget',searches=failed,wall_seconds=time.perf_counter()-start)

def local_angles(matrix):
    z0,y,z1=cirq.linalg.deconstruct_single_qubit_matrix_into_angles(matrix)
    return [float(y),float(z1),float(z0)]

def transported_word(record,kak,entangler):
    count=record['count'];word=[]
    def add(matrix,wire):word.append(dict(gate='U3',parameters=local_angles(matrix),wires=[wire]))
    # Cirq's first tensor factor is the most significant, catalog wire 1.
    for index,m in enumerate(kak.single_qubit_operations_before):add(m,1-index)
    if record.get('base')=='inverse_native_KAK':
        for index,m in enumerate(record['before']):add(m,1-index)
        word.append(dict(gate=entangler,parameters=[],wires=[0,1]))
        for index,m in enumerate(record['after']):add(m,1-index)
    else:
        p=record['parameters']
        for i in range(count+1):
            for q in range(2):word.append(dict(gate='U3',parameters=p[6*i+3*q:6*i+3*q+3],wires=[q]))
            if i<count:word.append(dict(gate=entangler,parameters=[],wires=[0,1]))
    for index,m in enumerate(kak.single_qubit_operations_after):add(m,1-index)
    if entangler=='SYC':
        converted=[]
        for op in word:
            if op['gate']!='U3':converted.append(op);continue
            t,p,l=op['parameters']
            converted.append(dict(gate='PhasedXZ',parameters=[t/np.pi,(p+l)/np.pi,.5-l/np.pi],wires=op['wires']))
        word=converted
    return word

def matrix_word(word):
    actual=np.eye(4,dtype=complex)
    for op in word:
        if op['gate']=='U3':local=u3(op['parameters'])
        elif op['gate']=='PhasedXZ':local=cirq.unitary(cirq.PhasedXZGate(x_exponent=op['parameters'][0],z_exponent=op['parameters'][1],axis_phase_exponent=op['parameters'][2]))
        else:actual=np.array(gates.make_gate(op['gate']).evalf(20),complex)@actual;continue
        actual=(np.kron(np.eye(2),local) if op['wires']==[0] else np.kron(local,np.eye(2)))@actual
    return actual

def literal_close(word,target):
    actual=matrix_word(word);gamma=float(np.angle(scalar(actual,target)))
    op=word[0];q=op['wires']
    if op['gate']=='U3':
        t,p,l=op['parameters'];replacement=[dict(gate='U3',parameters=[np.pi-t,0,l-np.pi],wires=q),dict(gate='U3',parameters=[np.pi,gamma+p,np.pi+gamma],wires=q)]
    else:
        x,z,a=op['parameters'];t,p,l=np.pi*x,np.pi*(z+a)-np.pi/2,np.pi/2-np.pi*a
        g=gamma+t-np.pi
        replacement=[]
        for T,P,L in [(np.pi-t,0,l-np.pi),(np.pi,g+p,np.pi+g)]:
            replacement.append(dict(gate='PhasedXZ',parameters=[T/np.pi,(P+L)/np.pi,.5-L/np.pi],wires=q))
    out=replacement+word[1:];error=float(np.max(np.abs(matrix_word(out)-target)))
    if error>1e-8:raise ArithmeticError('Native witness literal replay failed: '+str(error))
    return out,error

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--starts',type=int,default=5);a=p.parse_args()
    entries=[];orbits={}
    for name,parameters,scope in symbolic_cases():
        target=gates.make_gate(name,*parameters)
        if target.shape!=(4,4):continue
        substitutions={s:sp.pi*sp.Rational(i+1,2*i+7) for i,s in enumerate(sorted(target.free_symbols,key=str))}
        sampled=target.subs(substitutions);numeric=np.array(sampled.evalf(20),complex);kak=cirq.kak_decomposition(numeric)
        coords=np.array(kak.interaction_coefficients);key=','.join(f'{v:.8f}' for v in coords)
        if max(abs(coords))<1e-8:continue
        entries.append((name,scope,substitutions,numeric,kak,key))
        orbits.setdefault(key,coords)
    records={}
    jobs=[(entangler,key,coords,a.starts) for entangler in ('SYC','SSWAP') for key,coords in orbits.items()]
    with mp.get_context('fork').Pool(a.workers) as pool:
        for r in pool.imap_unordered(fit,jobs):records[r['entangler'],r['key']]=r
    rows=[]
    for name,scope,substitutions,numeric,kak,key in entries:
        for basis,entangler in [('google_syc','SYC'),('sqrt_swap_u','SSWAP')]:
            record=records[entangler,key]
            row=dict(gate=name,basis=basis,target_scope=scope,substitutions={str(k):str(v) for k,v in substitutions.items()},
                     witness_scope='fixed_target_numerical' if not substitutions else 'one_named_parameter_sample_only',
                     numerical_only=True,claims_exact_symbolic_identity=False,claims_minimum=False)
            if 'count' in record:
                try:
                    word=transported_word(record,kak,entangler);word,error=literal_close(word,numeric)
                    row.update(status='verified_numerical_witness',entanglers=record['count'],operations=word,
                               literal_maximum_entry_error=error,phase_policy='literal replay; allowed local phase absorption; no GP gate')
                except ArithmeticError as error:row.update(status='unverified_numerical_transport',error=str(error))
            else:row.update(status=record['status'])
            rows.append(row)
    # Convert complex matrix frames in the orbit setup to transparent real/imag pairs.
    def encode(value):
        if isinstance(value,np.ndarray):return [[[float(z.real),float(z.imag)] for z in row] for row in value]
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    report=dict(schema='native-column-numerical-gap-audit-v1',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         versions=dict(cirq=cirq.__version__,numpy=np.__version__),seed=219,starts=a.starts,
         contract='Actual catalog SYC and SSWAP matrices; fixed wire order; no ancillas; full 4x4 literal replay <1e-8; numeric proposals never enter exact synthesis',
         rows=rows,orbits=list(records.values()))
    with gzip.open(a.output,'wt') as f:json.dump(report,f,default=encode)
    print('Native-column audit complete:',len(rows),'target/basis rows',flush=True)

if __name__=='__main__':main()
