#!/usr/bin/env python3
"""Replay bounded Quartz/QUESO runs on independent library-source inputs."""
import argparse,concurrent.futures,gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from qiskit import transpile,qasm2
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from symbolic_referee_sdks import CASES,ROOT,source_case,replay,qiskit_counts

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def job(tool,name,out):
    target,c=source_case(name)
    inp=transpile(c,basis_gates=['h','rz','x','cx'],optimization_level=0)
    source=qasm2.dumps(inp);path=out/(name+'.qasm');path.write_text(source)
    row=dict(tool=tool,gate=name,input='library_source',input_qasm=source,
             input_sha256=digest(path),settings=dict(search_seconds=10,objective='total gate count',
                  ancillas=0,fixed_input_output_wires=True,native_alphabet=['h','rz','x','cx']))
    before=time.perf_counter()
    if tool=='quartz':
        repo=Path('/tmp/symbolic-referee-quartz')
        command=[str(repo/'build/test_optimize'),str(path),name]
    else:
        repo=Path('/tmp/symbolic-referee-queso')
        command=['/usr/lib/jvm/java-17/bin/java','--enable-preview','-Xmx2g','-cp',
          'target/QUESO-1.0-jar-with-dependencies.jar','Optimizer','-c',str(path),'-g','nam',
          '-r','rules_q3_s6_nam.txt','-sr','rules_q3_s6_nam_symb.txt','-t','10',
          '-o',str(out),'-j',name,'-preserveMapping']
    try:
        p=subprocess.run(command,cwd=repo,capture_output=True,text=True,timeout=90)
        row.update(returncode=p.returncode,wall_seconds=time.perf_counter()-before)
        (out/(tool+'_'+name+'.log')).write_text(p.stdout+'\n'+p.stderr)
        if p.returncode:raise RuntimeError(p.stderr[-1200:])
        text=p.stdout.split('Optimized graph:\n')[-1] if tool=='quartz' else (out/('optimized_'+name+'_'+name+'.qasm')).read_text()
        text=text[text.index('OPENQASM'):]
        native=qasm2.loads(text)
        row.update(replay(Operator(native).data,target))
        normalized=PassManager([Optimize1qGatesDecomposition(basis=['u3'])]).run(native)
        row.update(native_counts=dict(native.count_ops()),qasm=text,
                   normalized_cx_u3=qiskit_counts(normalized),normalized_qasm=qasm2.dumps(normalized))
        if replay(Operator(normalized).data,target)['status']!='verified_numeric':raise ArithmeticError('Normalization replay')
    except Exception as e:row.update(status='error',error=repr(e),wall_seconds=time.perf_counter()-before)
    print(tool,name,row['status'],row.get('normalized_cx_u3'),flush=True)
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--normalize-existing',action='store_true');a=p.parse_args()
    if a.normalize_existing:
        with gzip.open(a.output,'rt') as f:report=json.load(f)
        for row in report['rows']:
            if not row.get('qasm'):continue
            native=qasm2.loads(row['qasm']);target=source_case(row['gate'])[0]
            value=replay(Operator(native).data,target)
            if value['status']!='verified_numeric':
                if row.get('error'):row['previous_normalization_error']=row.pop('error')
                row.update(value)
                continue
            normalized=PassManager([Optimize1qGatesDecomposition(basis=['u3'])]).run(native)
            if replay(Operator(normalized).data,target)['status']!='verified_numeric':raise ArithmeticError('Local fusion replay')
            if row.get('error'):row['previous_normalization_error']=row.pop('error')
            row.update(value,normalized_cx_u3=qiskit_counts(normalized),normalized_qasm=qasm2.dumps(normalized))
        report.update(normalization='Local fusion only: Optimize1qGatesDecomposition, U3; entangling skeleton unchanged',
                      normalization_script_sha256=digest(Path(__file__)))
        with gzip.open(a.output,'wt') as f:json.dump(report,f)
        return
    work=Path('/tmp/referee-rule-final');work.mkdir(exist_ok=True)
    report=dict(schema='referee-rule-optimizers-v1',script_sha256=digest(Path(__file__)),
                source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                contract='projective full matrix replay, maximum entry error <1e-8; same independent library input; no ancillas',tools={},rows=[])
    for tool in ('quartz','queso'):
        repo=Path('/tmp/symbolic-referee-'+tool)
        meta=dict(repository=subprocess.check_output(['git','remote','get-url','origin'],cwd=repo,text=True).strip(),
             revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
        if tool=='quartz':
            meta.update(harness_patch=subprocess.check_output(['git','diff'],cwd=repo,text=True),
                        compile_flags='-include cstddef',ecc_sha256=digest(repo/'eccset/Nam_5_3_complete_ECC_set.json'))
        else:
            meta.update(java='OpenJDK 17',rule_generation='Synthesizer -g nam -q 3 -s 6',
                 rules={str(q.name):dict(sha256=digest(q),content=q.read_text()) for q in repo.glob('rules_q3_s6_nam*.txt')})
        report['tools'][tool]=meta
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for f in concurrent.futures.as_completed([pool.submit(job,t,n,work) for t in ('quartz','queso') for n in CASES]):
            report['rows'].append(f.result())
            with gzip.open(a.output,'wt') as h:json.dump(report,h)

if __name__=='__main__':main()
