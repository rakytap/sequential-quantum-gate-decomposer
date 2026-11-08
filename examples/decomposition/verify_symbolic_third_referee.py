#!/usr/bin/env python3
"""Replay the third report's exact count proofs and numerical CX witnesses.

The polytope verifier uses rational arithmetic and exact Makhlin invariants.
Qiskit's independent matrix implementation checks the numerical witnesses;
these remain numerical evidence, not free-angle identities or lower bounds.
"""
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbolic_monodromy_audit import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(relative, expected):
    path = ROOT / relative
    if path.exists() and sha(path) == expected:
        return dict(path=relative, sha256=expected, resolution='included_current_source')
    history = subprocess.check_output(
        ['git', 'log', '--all', '--format=%H', '--', relative], cwd=ROOT, text=True).splitlines()
    for revision in history:
        result = subprocess.run(['git', 'show', revision+':'+relative], cwd=ROOT, capture_output=True)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
            return dict(path=relative, sha256=expected, resolution='historical_source', revision=revision)
    raise ArithmeticError('No source content matches recorded hash: '+relative)


def main():
    data = ROOT / 'benchmarks/symbolic'
    doc = ROOT / 'docs/symbolic_manuscript'
    counts = {}
    artifacts = {}
    for name in ('native_monodromy_20261108.json.gz',
                 'native_monodromy_deep_20261108.json.gz'):
        path = data / name
        with gzip.open(path, 'rt') as stream:
            report = json.load(stream)
        checked = sum(verify(row) for row in report['rows'])
        artifacts[name] = dict(sha256=sha(path), verified_records=checked,
                              statuses=dict(Counter(row['status'] for row in report['rows'])),
                              sources=[source_record(p, value) for p, value in report['source_sha256'].items()])
        for row in report['rows']:
            key = (row['gate'], row['basis'])
            if row['status'] == 'proved_native_class_minimum':
                counts[key] = row
    fixed = sum(not row['substitutions'] for row in counts.values())
    assert fixed == 64 and len(counts) == 112
    (doc / 'monodromy-verification.json').write_text(json.dumps(dict(
        schema='native-monodromy-verification-v1', artifacts=artifacts,
        unique_minimum_records=len(counts), fixed_target_native_pairs=fixed,
        scope='Exact native-class E minima with arbitrary U(2) locals. Family rows apply only to named substitutions. No short local-frame word or local minimum is implied.'
    ), indent=2) + '\n')

    path = data / 'reflection_seven_numeric_20261108.json.gz'
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    rows = []
    for row in record['rows']:
        circuit = QuantumCircuit(3)
        for op in row['operations']:
            wires = [2-q for q in op['wires']]
            if op['gate'] == 'CX':
                circuit.cx(*wires)
            else:
                assert op['gate'] == 'U3'
                circuit.u(*op['parameters'], wires[0])
        assert circuit.count_ops()['cx'] == 7 and circuit.global_phase == 0
        angle = float(row['angle'])
        vector = np.array([np.cos(angle), 0, 0, np.sin(angle)])
        target = np.eye(8, dtype=complex)
        target[4:, 4:] = np.eye(4) - 2*np.outer(vector, vector)
        error = float(np.max(np.abs(Operator(circuit).data - target)))
        assert error < 1e-12
        rows.append(dict(angle_source=row['angle_source'], literal_maximum_entry_error=error))
    result = dict(schema='reflection-seven-numerical-verification-v1',
        archive_sha256=sha(path), independent_replay='Qiskit QuantumCircuit and Operator',
        source=source_record('examples/decomposition/symbolic_reflection_numeric_audit.py', record['source_sha256']),
        maximum_literal_entry_error=max(row['literal_maximum_entry_error'] for row in rows),
        no_physical_GP=True, rows=rows,
        scope='Ten numerical seven-CX witnesses only. The exact free-angle five-unrestricted-gate identity is separately checked by test_five_unrestricted_reflection_word_with_angle_free. Neither is an interior lower bound.')
    import sympy as sp
    from symbolic_gates import compile_gates, make_controlled
    from symbolic_reflection_involution import reflection
    from symbolic_controlled_chart import _zero
    exact_path = data / 'reflection_seven_exact_20261108.json.gz'
    with gzip.open(exact_path, 'rt') as stream:
        exact = json.load(stream)
    theta = sp.Symbol('theta', real=True)
    word = tuple((o['gate'], *(sp.sympify(v, locals={'theta': theta}) for v in o['parameters']),
                  tuple(o['wires'])) for o in exact['operations'])
    assert sum(o[0] == 'CX' for o in word) == 7 and not any(o[0] == 'GP' for o in word)
    assert _zero(compile_gates(3, word) - make_controlled(reflection(theta), 2))
    assert _zero(compile_gates(3, word + (('Z', (0,)),)) - make_controlled(-reflection(theta), 2))
    result['exact_family'] = dict(archive_sha256=sha(exact_path), entanglers=7, local_gates=18,
        phase_contract='literal; no GP', real_angle_free=True, negative_variant_verified=True,
        source=source_record('squander/decomposition/symbolic_reflection_involution.py', exact['source_sha256']))
    (doc / 'reflection-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(fixed_target_minima=fixed, unique_minimum_records=len(counts),
                         reflection_numeric_replays=len(rows),
                         maximum_reflection_error=result['maximum_literal_entry_error'])))


if __name__ == '__main__':
    main()
