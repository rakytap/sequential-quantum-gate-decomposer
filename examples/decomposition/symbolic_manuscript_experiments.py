#!/usr/bin/env python3
"""Bounded, source-pinned experiments for the symbolic synthesis manuscript.

Includes opaque held-out family inputs with component ablations, independent
fixed-target SDK runs, parameter reuse against compile-once Qiskit, and an
exact controlled-substitution phase audit. Numerical replay is never an
identity or minimum certificate.
"""
import argparse
from datetime import datetime,timezone
import gzip,hashlib,importlib.metadata,itertools,json,multiprocessing as mp
from pathlib import Path
import subprocess,sys,time
import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import symbolic_gates as gates
from symbolic_archive import wall_time_limit,SymbolicDeadline
from symbolic_bases import get_basis
from symbolic_contract import resource_key
from symbolic_identity import prove_circuit_identity
from symbolic_synthesis import (synthesize,synthesize_matrix,_synthesize_matrix_existing,
                                _native_candidates,_verify_phase_candidate)


def held_out_cases():
    a,b,c,d=sp.symbols('a b c d',real=True)
    H0=gates.apply_to(sp.eye(4),2,gates.make_gate('H'),(0,))
    S0=gates.apply_to(sp.eye(4),2,gates.make_gate('S'),(0,))
    rz0=gates.apply_to(sp.eye(4),2,gates.make_gate('Rz',d),(0,))
    A=gates.make_gate('Rz',a)
    B=gates.make_gate('Rx',b)*gates.make_gate('Ry',c)*A
    return [
      ('x_controlled_rotation',H0*gates.make_gate('CRX',a)*H0,
       (('H',(0,)),('CRX',a,(0,1)),('H',(0,)))),
      ('y_controlled_rotation',S0*H0*gates.make_gate('CRY',a)*H0*S0.H,
       (('Sdg',(0,)),('H',(0,)),('CRY',a,(0,1)),('H',(0,)),('S',(0,)))),
      ('unequal_controlled_su2_blocks',gates.make_controlled(B,1,A),
       (('Rz',a,(1,)),('CRY',c,(0,1)),('CRX',b,(0,1)))),
      ('number_conserving_with_local_phases',
       gates.apply_to(gates.apply_to(gates.make_gate('fSim',a,0),2,gates.make_gate('Rz',b),(0,)),2,gates.make_gate('Rz',c),(1,)),
       (('fSim',a,0,(0,1)),('Rz',b,(0,)),('Rz',c,(1,)))),
      ('gauged_affine_cartan',rz0*gates.make_gate('Canonical',a+b,a-b,c)*rz0.H,
       (('Rz',-d,(0,)),('Canonical',a+b,a-b,c,(0,1)),('Rz',d,(0,)))),
      ('unequal_commuting_exchange',gates.make_gate('Ryy',b)*gates.make_gate('Rxx',a),
       (('Rxx',a,(0,1)),('Ryy',b,(0,1))))]


def ablation_job(job):
    name,basis_name,method,seconds=job
    _,target,source=next(row for row in held_out_cases() if row[0]==name)
    basis=get_basis(basis_name);start=time.perf_counter()
    row=dict(case=name,basis=basis_name,method=method,budget_seconds=seconds,
             input='opaque_exact_matrix' if method!='source_lowering' else 'exact_source_word',
             target_parameters=sorted(map(str,target.free_symbols)),template_name_supplied=False)
    try:
        with wall_time_limit(seconds):
            if prove_circuit_identity(target,source) is not True:raise ArithmeticError('Independent source identity failed')
            if method=='full':result=synthesize_matrix(target,basis,phase_exact=False,seconds=.2)
            elif method=='without_native_charts':result=_synthesize_matrix_existing(target,basis,phase_exact=False,seconds=.2)
            else:
                choices=_native_candidates('Identity',(2,),basis,source_operations=source)
                word,factor=min(choices,key=lambda pair:resource_key(pair[0]))
                result=_verify_phase_candidate(target,word,basis,False,factor)
            row.update(status=result['status'],cost=result.get('cost'),result=result)
    except SymbolicDeadline:row['status']='budget_exhausted'
    except Exception as e:row.update(status='error',error=type(e).__name__+': '+str(e))
    row['wall_seconds']=time.perf_counter()-start
    return row


def _linear_parameter(value,parameters):
    value=sp.sympify(value)
    if not value.free_symbols:return float(value)
    if value in parameters:return parameters[value]
    if value.is_Add:return sum(_linear_parameter(a,parameters) for a in value.args)
    if value.is_Mul:
        out=1
        for a in value.args:out*= _linear_parameter(a,parameters)
        return out
    if value.is_Pow:return _linear_parameter(value.base,parameters)**int(value.exp)
    raise ValueError('Unsupported SDK expression: '+str(value))


def append_symbolic(circuit,operations,parameters,wires):
    for op in operations:
        name=op['gate'];args=[_linear_parameter(sp.sympify(p,locals={str(x):x for x in parameters}),parameters)
                              for p in op['parameters']]
        qs=[wires[q] for q in op['wires']]
        getattr(circuit,{'U3':'u','CX':'cx','CNOT':'cx','Sx':'sx'}.get(name,name.lower()))(*args,*qs)


def counts(circuit):
    return dict(entanglers=sum(len(i.qubits)==2 for i in circuit.data),
                locals=sum(len(i.qubits)==1 for i in circuit.data),depth=circuit.depth(),
                actual_optimization_level=(circuit.metadata or {}).get('actual_optimization_level'))


def transpile_fixed(circuit,**settings):
    from qiskit import transpile
    result=transpile(circuit,**settings);level=settings.get('optimization_level',3)
    if result.layout is not None and result.layout.final_index_layout()!=list(range(result.num_qubits)):
        settings=dict(settings,optimization_level=1)
        result=transpile(circuit,**settings);level=1
    result.metadata=dict(result.metadata or {},actual_optimization_level=level)
    return result


def sdk_operations(circuit):
    return [dict(gate=i.operation.name,parameters=[str(p) for p in i.operation.params],
                 wires=[circuit.find_bit(q).index for q in i.qubits]) for i in circuit.data]


def parameter_reuse(bindings=100):
    from qiskit import QuantumCircuit,transpile
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    t,p=sp.symbols('theta phi',real=True)
    started=time.perf_counter()
    with wall_time_limit(60):result=synthesize('fSim',(t,p),'cx_u',phase_exact=True,osr_seconds=.1)
    symbolic_seconds=time.perf_counter()-started
    if result['status']!='verified_symbolic':raise ArithmeticError('fSim chart unavailable')
    source=QuantumCircuit(4);lowered=QuantumCircuit(4);inputs=[]
    for index,edge in enumerate(((0,1),(2,3),(1,2))*2):
        theta,phi=Parameter('theta_'+str(index)),Parameter('phi_'+str(index));inputs.extend((theta,phi))
        source.rxx(theta,*edge);source.ryy(theta,*edge);source.cp(-phi,*edge)
        append_symbolic(lowered,result['operations'],{t:theta,p:phi},edge)
    settings=dict(basis_gates=['cx','u'],optimization_level=3,seed_transpiler=219,
                  approximation_degree=1.0,qubits_initially_zero=False)
    rows=[];compiled={}
    for label,circuit in (('sdk_compile_once',source),('symbolic_then_sdk_compile_once',lowered)):
        started=time.perf_counter();out=transpile_fixed(circuit,**settings);elapsed=time.perf_counter()-started
        compiled[label]=out;rows.append(dict(method=label,compile_seconds=elapsed,
                                           operations=sdk_operations(out),**counts(out)))
    rng=np.random.default_rng(219)
    values=[dict(zip(inputs,rng.uniform(-2*np.pi,2*np.pi,len(inputs)))) for _ in range(bindings)]
    for row in rows:
        started=time.perf_counter()
        for value in values:compiled[row['method']].assign_parameters(value)
        row['binding_seconds']=time.perf_counter()-started;row['bindings']=bindings
    replay=[]
    recompiles=[]
    for value in values[:min(10,bindings)]:
        target=Operator(source.assign_parameters(value)).data
        errors={label:float(np.max(np.abs(Operator(circuit.assign_parameters(value)).data-target)))
                for label,circuit in compiled.items()}
        if max(errors.values())>2e-10:raise ArithmeticError('Parameter reuse numerical replay failed')
        replay.append(errors)
        started=time.perf_counter();out=transpile_fixed(source.assign_parameters(value),**settings)
        recompiles.append(dict(wall_seconds=time.perf_counter()-started,**counts(out),
                               error=float(np.max(np.abs(Operator(out).data-target)))))
    source_word=(('Rxx',t,(0,1)),('Ryy',t,(0,1)),('CP',-p,(0,1)))
    exact_source=prove_circuit_identity(gates.make_gate('fSim',t,p),source_word)
    if exact_source is not True:raise ArithmeticError('fSim source proof failed')
    return dict(width=4,blocks=6,free_parameters=12,seed=219,source_identity_proved=True,
                exact_chart=result,symbolic_chart_seconds=symbolic_seconds,rows=rows,
                numerical_replays=replay,numeric_recompile_rows=recompiles,
                timing_scope='chart synthesis, SDK compilation, parameter binding, and numeric recompilation reported separately',
                hardware_latency_or_fidelity_claim=False)


def independent_fixed_targets():
    from qiskit import QuantumCircuit,transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    rows=[]
    for name,width in (('SYC',2),('fFredkin',3)):
        target=np.asarray(gates.make_gate(name).evalf(20),complex)
        for input_kind in ('matrix','library_source'):
            source=QuantumCircuit(width)
            if input_kind=='matrix':source.append(UnitaryGate(target),range(width))
            elif name=='SYC':source.rxx(np.pi/2,0,1);source.ryy(np.pi/2,0,1);source.cp(-np.pi/6,0,1)
            else:source.cswap(0,1,2);source.ccz(0,1,2)
            if np.max(np.abs(Operator(source).data-target))>2e-12:raise ArithmeticError('Independent SDK input differs')
            for entangler in ('cx','cz'):
                for seed in (7,219):
                    started=time.perf_counter()
                    out=transpile_fixed(source,basis_gates=[entangler,'u'],optimization_level=3,
                                  approximation_degree=1.0,seed_transpiler=seed,qubits_initially_zero=False)
                    error=float(np.max(np.abs(Operator(out).data-target)))
                    if error>2e-10:raise ArithmeticError('Independent SDK output differs')
                    rows.append(dict(target=name,sdk='qiskit',input=input_kind,entangler=entangler,
                                     seed=seed,requested_optimization_level=3,error=error,
                                     global_phase=float(out.global_phase),operations=sdk_operations(out),
                                     wall_seconds=time.perf_counter()-started,**counts(out)))
        try:
            import cirq
            qs=cirq.LineQubit.range(width)
            source=(cirq.Circuit(cirq.FSimGate(theta=np.pi/2,phi=np.pi/6)(qs[0],qs[1])) if name=='SYC'
                    else cirq.Circuit(cirq.CSWAP(*qs),cirq.CCZ(*qs)))
            wanted_order=qs if name=='SYC' else list(reversed(qs))
            # SYC is wire symmetric; fFredkin has little-endian control q0.
            if np.max(np.abs(source.unitary(qubit_order=wanted_order)-target))>2e-12:
                raise ArithmeticError('Cirq input convention mismatch')
            started=time.perf_counter()
            out=cirq.optimize_for_target_gateset(source,gateset=cirq.CZTargetGateset(allow_partial_czs=False),ignore_failures=False)
            actual=out.unitary(qubit_order=wanted_order)
            phase=np.vdot(actual.flatten(),target.flatten())/(1<<width)
            error=float(np.max(np.abs(phase*actual-target)))
            if error>2e-10:raise ArithmeticError('Cirq projective output differs')
            ops=list(out.all_operations())
            if any(len(op.qubits)>2 for op in ops):raise ArithmeticError('Non-native Cirq output')
            rows.append(dict(target=name,sdk='cirq',input='library_source',entangler='cz',
                             phase_policy='projective',error=error,
                             operations=[dict(gate=str(op.gate),wires=[q.x for q in op.qubits],
                                  matrix=[[float(v.real),float(v.imag)] for v in np.asarray(cirq.unitary(op)).flatten()])
                                  for op in ops],
                             wall_seconds=time.perf_counter()-started,
                             entanglers=sum(len(op.qubits)==2 for op in ops),
                             locals=sum(len(op.qubits)==1 for op in ops)))
        except Exception as e:
            rows.append(dict(target=name,sdk='cirq',status='not_compared',reason=type(e).__name__+': '+str(e)))
    return rows


def phase_embedding():
    p=sp.pi
    word=(('Rz',-p/12,(0,)),('Rz',-p/12,(1,)),('CX',(0,1)),
          ('Rz',-5*p/12,(1,)),('CX',(1,0)),('CX',(0,1)))
    phase=sp.exp(-7*sp.I*p/24)
    target=gates.make_gate('SYC');actual=gates.compile_gates(2,word)
    if prove_circuit_identity(target,word,phase) is not True:raise ArithmeticError('SYC phase chart mismatch')
    wanted=gates.make_controlled(target,2)
    bare=gates.make_controlled(actual,2)
    correction=gates.apply_to(sp.eye(8),3,gates.make_gate('P',-7*p/24),(0,))
    corrected=(correction*bare-wanted).applyfunc(sp.simplify)
    if any(value!=0 for value in corrected):raise ArithmeticError('Controlled phase correction failed')
    witness=next(sp.simplify(value) for value in bare-wanted if sp.simplify(value)!=0)
    return dict(projective_identity_proved=True,controlled_correction_identity_proved=True,
                uncorrected_controlled_substitution_fails=True,nonzero_witness=str(witness),
                required_control_phase='-7*pi/24',existing_SDK_phase_tracking_acknowledged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--bindings',type=int,default=100)
    args=parser.parse_args()
    jobs=[(name,basis,method,args.seconds) for name,_,_ in held_out_cases()
          for basis in ('sqrt_iswap_u','berkeley_b')
          for method in ('full','without_native_charts','source_lowering')]
    report=dict(schema='symbolic-manuscript-experiments-v1',generated_at=datetime.now(timezone.utc).isoformat(),
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__),*sorted((ROOT/'squander/decomposition').glob('symbolic_*.py')),
                                 *sorted((ROOT/'squander/decomposition').glob('symbolic_*.json'))]},
                versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','sympy','qiskit','cirq-core')},
                benchmark_contract=dict(ancillas=0,connectivity='all_to_all',fixed_input_output_wire_order=True,
                    arbitrary_input_states=True,physical_GP=False,seed=219,numerical_acceptance_is_not_symbolic_proof=True),
                ablations=[])
    def checkpoint():
        path=args.output.with_name(args.output.name+'.partial.json.gz')
        with gzip.open(path,'wt') as f:json.dump(report,f,allow_nan=False)
    with mp.get_context('fork').Pool(args.workers) as pool:
        for row in pool.imap_unordered(ablation_job,jobs):
            report['ablations'].append(row);checkpoint()
            print(row['case'],row['basis'],row['method'],row['status'],flush=True)
    report['parameter_reuse']=parameter_reuse(args.bindings);checkpoint()
    report['independent_fixed_targets']=independent_fixed_targets();checkpoint()
    report['phase_embedding']=phase_embedding()
    with gzip.open(args.output,'wt') as f:json.dump(report,f,allow_nan=False,separators=(',',':'))
    print('Completed',args.output,flush=True)


if __name__=='__main__':main()
