#!/usr/bin/env python3
"""Independent Qiskit/tket/BQSKit baselines and a BQSKit fixed-layout search.

Run with the isolated Python 3.12 environment recorded in the output. All
results have complete matrix replay with fixed input/output wires. Failed
fits and process budgets are preserved; numerical fits are not lower bounds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip,hashlib,importlib.metadata,itertools,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'squander/decomposition'))
import symbolic_gates as gates

CASES=('SYC','iSWAP','SSWAP','fFredkin','CCX','CSWAP','C3X','C4X')


def source_case(name):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    target=np.array(gates.make_gate(name).evalf(20),complex)
    width=int(np.log2(len(target)));c=QuantumCircuit(width)
    if name=='SYC':c.rxx(np.pi/2,0,1);c.ryy(np.pi/2,0,1);c.cp(-np.pi/6,0,1)
    elif name=='iSWAP':c.iswap(0,1)
    elif name=='SSWAP':
        from qiskit.circuit.library import SwapGate
        c.append(SwapGate().power(.5),[0,1])
    elif name=='fFredkin':c.cswap(0,1,2);c.ccz(0,1,2)
    elif name=='CCX':c.ccx(0,1,2)
    elif name=='CSWAP':c.cswap(0,1,2)
    else:c.mcx(list(range(width-1)),width-1)
    error=np.max(np.abs(Operator(c).data-target))
    if error>1e-12:raise ArithmeticError('Independent library input convention: '+name)
    return target,c


def reorder(matrix):
    width=int(np.log2(len(matrix)))
    p=[int(f'{i:0{width}b}'[::-1],2) for i in range(len(matrix))]
    return matrix[np.ix_(p,p)]


def replay(actual,target):
    factor=np.vdot(actual.flatten(),target.flatten())/len(target)
    if abs(abs(factor)-1)>1e-7:
        return dict(status='unverified_numeric',error=float(np.max(np.abs(actual-target))))
    error=float(np.max(np.abs(factor*actual-target)))
    return dict(status='verified_numeric' if error<1e-8 else 'unverified_numeric',
                error=error,virtual_scalar=[float(factor.real),float(factor.imag)])


def qiskit_counts(c):
    return dict(entanglers=sum(len(i.qubits)==2 for i in c.data),
                locals=sum(len(i.qubits)==1 for i in c.data),depth=c.depth())


def baseline(tool,name,kind):
    from qiskit import QuantumCircuit,transpile,qasm2
    from qiskit.quantum_info import Operator
    target,source=source_case(name);n=source.num_qubits;start=time.perf_counter()
    if kind=='matrix':
        source=QuantumCircuit(n);source.unitary(target,range(n))
    if tool=='qiskit':
        out=transpile(source,basis_gates=['u3','cx'],optimization_level=3,
                      approximation_degree=1.0,seed_transpiler=219,qubits_initially_zero=False)
        if out.layout is not None and out.layout.final_index_layout()!=list(range(n)):
            raise ArithmeticError('SDK changed output permutation')
        result=dict(**replay(Operator(out).data,target),**qiskit_counts(out),qasm=qasm2.dumps(out),
                    settings=dict(optimization_level=3,seed=219,arbitrary_inputs=True))
    elif tool=='pytket':
        from pytket import Circuit,OpType
        from pytket.circuit import Unitary2qBox,Unitary3qBox
        from pytket.qasm import circuit_from_qasm_str,circuit_to_qasm_str
        from pytket.passes import FullPeepholeOptimise,AutoRebase
        if kind=='matrix':
            c=Circuit(n);box=(Unitary2qBox if n==2 else Unitary3qBox)(reorder(target))
            c.add_gate(box,list(range(n)))
        else:
            c=circuit_from_qasm_str(qasm2.dumps(transpile(source,basis_gates=['u3','cx'],optimization_level=0)))
        if np.max(np.abs(c.get_unitary()-reorder(target)))>1e-11:raise ArithmeticError('tket input ordering')
        FullPeepholeOptimise(allow_swaps=False).apply(c)
        AutoRebase({OpType.U3,OpType.CX},allow_swaps=False).apply(c)
        if any(a!=b for a,b in c.implicit_qubit_permutation().items()):raise ArithmeticError('tket implicit output permutation')
        commands=c.get_commands()
        result=dict(**replay(c.get_unitary(),reorder(target)),entanglers=sum(len(i.qubits)==2 for i in commands),
                    locals=sum(len(i.qubits)==1 for i in commands),depth=c.depth(),qasm=circuit_to_qasm_str(c),
                    settings=dict(pass_name='FullPeepholeOptimise then AutoRebase CX/U3',allow_swaps=False))
    else:
        from bqskit import compile
        from bqskit.ir.circuit import Circuit
        if kind=='matrix':value=reorder(target)
        else:
            value=Circuit.from_qasm(qasm2.dumps(transpile(source,basis_gates=['u3','cx'],optimization_level=0)))
            if np.max(np.abs(np.asarray(value.get_unitary())-reorder(target)))>1e-11:raise ArithmeticError('BQSKit input ordering')
        out=compile(value,optimization_level=3,max_synthesis_size=3,synthesis_epsilon=1e-12,
                    seed=219,num_workers=1,num_blas_threads=1)
        ops=list(out)
        result=dict(**replay(np.array(out.get_unitary()),reorder(target)),entanglers=sum(len(i.location)==2 for i in ops),
                    locals=sum(len(i.location)==1 for i in ops),qasm=out.to('qasm'),
                    settings=dict(optimization_level=3,max_synthesis_size=3,synthesis_epsilon=1e-12,seed=219,num_workers=1))
    result.update(tool=tool,gate=name,input=kind,width=n,wall_seconds=time.perf_counter()-start)
    return result


def layout_search():
    from bqskit.ir.circuit import Circuit
    from bqskit.ir.gates import CNOTGate,U3Gate
    target=reorder(source_case('fFredkin')[0]);rows=[];start=time.perf_counter()
    for depth in (3,4):
        for index,layout in enumerate(itertools.product(((0,1),(0,2),(1,2)),repeat=depth)):
            c=Circuit(3)
            for q in range(3):c.append_gate(U3Gate(),q)
            for edge in layout:
                c.append_gate(CNOTGate(),edge)
                for q in edge:c.append_gate(U3Gate(),q)
            before=time.perf_counter()
            c.instantiate(target,multistarts=3,seed=219+index,method='minimization')
            value=replay(np.array(c.get_unitary()),target)
            value.update(depth=depth,layout=layout,wall_seconds=time.perf_counter()-before)
            if value['status']=='verified_numeric':value['qasm']=c.to('qasm')
            rows.append(value)
    return dict(tool='bqskit_fixed_layout',gate='fFredkin',layouts=rows,wall_seconds=time.perf_counter()-start,
                cost_lower_bound_claim=False,settings=dict(depths=[3,4],all_pair_layouts=True,
                    layouts_per_depth=[27,81],multistarts=3,seed_rule='219+layout_index',method='minimization'))


def run_child(job,budget):
    tool,name,kind=job
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',tool,'--gate',name,'--input-kind',kind]
    try:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,
                                 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        stdout,stderr=process.communicate(timeout=budget)
        if process.returncode:return dict(tool=tool,gate=name,input=kind,status='error',stderr=stderr[-1800:])
        return json.loads(stdout.split('REFEREE_RESULT=')[-1])
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.communicate()
        return dict(tool=tool,gate=name,input=kind,status='budget_exhausted',budget_seconds=budget)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=['qiskit','pytket','bqskit','layout'])
    p.add_argument('--gate');p.add_argument('--input-kind',default='library_source')
    p.add_argument('--output',type=Path);p.add_argument('--tools',nargs='+',default=['qiskit','pytket','bqskit'])
    p.add_argument('--budget',type=float,default=120);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.worker:
        result=layout_search() if args.worker=='layout' else baseline(args.worker,args.gate,args.input_kind)
        print('REFEREE_RESULT='+json.dumps(result,allow_nan=False));return
    jobs=[(t,n,k) for t in args.tools for n in CASES for k in ('library_source','matrix')
          if k=='library_source' or source_case(n)[1].num_qubits<=3]
    report=dict(schema='referee-strong-sdk-baselines-v1',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions={n:importlib.metadata.version(n) for n in ('qiskit','pytket','bqskit','numpy','scipy')},
                contract=dict(phase='projective with explicit scalar replay',ancillas=0,output_permutation='fixed',
                              all_to_all=True,maximum_entry_error=1e-8),budget_seconds=args.budget,rows=[])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_child,j,args.budget) for j in jobs]):
            row=future.result();report['rows'].append(row)
            print(row['tool'],row['gate'],row['input'],row['status'],row.get('entanglers'),flush=True)
            with gzip.open(args.output,'wt') as f:json.dump(report,f,allow_nan=False)


if __name__=='__main__':main()
