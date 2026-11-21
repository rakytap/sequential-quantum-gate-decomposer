#!/usr/bin/env python3
"""Audit published no-ancilla C4X routes in the recorded Qiskit environment."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import qiskit
from qiskit import QuantumCircuit,transpile,qasm3
from qiskit.quantum_info import Operator
from qiskit import synthesis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    target=QuantumCircuit(5);target.mcx(list(range(4)),4);target=Operator(target).data
    rows=[]
    for name in ('synth_c4x','synth_mcx_noaux_hp24','synth_mcx_noaux_sp22','synth_mcx_noaux_v24'):
        source=getattr(synthesis,name)() if name=='synth_c4x' else getattr(synthesis,name)(4)
        for level in (0,3):
            result=transpile(source,basis_gates=['cx','u'],optimization_level=level,seed_transpiler=219)
            error=float(np.max(np.abs(Operator(result).data-target)))
            assert result.num_qubits==5 and error<1e-8
            rows.append(dict(method=name,level=level,cnot=result.count_ops().get('cx',0),qubits=5,
                             error=error,output_qasm3=qasm3.dumps(result)))
    report=dict(schema='c4x-noaux-published-route-audit-v1',qiskit=qiskit.__version__,seed=219,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
        contract='Five qubits, no ancillas, arbitrary input states, literal full-matrix replay. Routes are upper bounds, not a best-published claim.')
    with gzip.open(a.output,'wt') as stream:json.dump(report,stream,separators=(',',':'))
    print([(r['method'],r['level'],r['cnot']) for r in rows])
if __name__=='__main__':main()
