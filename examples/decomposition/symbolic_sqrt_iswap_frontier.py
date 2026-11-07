#!/usr/bin/env python3
"""Archive/replay the 20 three-SQiSW modes without scoring expression aliases.

This is a construction frontier alongside the frozen scored archive. Its
entangler counts are comparable; its unevaluated local scores are not.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'squander/decomposition'))
from symbolic_archive import symbolic_cases, wall_time_limit
from symbolic_bases import get_basis
from symbolic_gates import make_gate
from symbolic_sqrt_iswap_cartan import (synthesize_sqrt_iswap_cartan,
                                       verify_sqrt_iswap_cartan_certificate)

FAMILIES = ('Canonical', 'fSim', 'PhasedFSim', 'SWAPPow', 'TK2')
BASES = ('sqrt_iswap_u', 'google_sqrt_iswap')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(primary_path):
    with gzip.open(primary_path, 'rt') as stream:
        primary = json.load(stream)
    old = {(row['gate'], row['basis']): row for row in primary['rows']}
    cases = {name: (parameters, scope) for name, parameters, scope in symbolic_cases()}
    rows = []
    for name in FAMILIES:
        parameters, scope = cases[name]
        target = make_gate(name, *parameters)
        for basis_name in BASES:
            basis = get_basis(basis_name)
            row = dict(gate=name, parameters=list(map(str, parameters)), target_scope=scope,
                       basis=basis_name)
            for mode in ('phase_exact', 'projective'):
                start = time.monotonic()
                with wall_time_limit(60):
                    result = synthesize_sqrt_iswap_cartan(target, basis, phase_exact=mode == 'phase_exact')
                    if not verify_sqrt_iswap_cartan_certificate(target, result, basis):
                        raise ArithmeticError('Certificate replay failed')
                if result['uniform_entangler_budget_certificate']['minimum'] != 3:
                    raise ArithmeticError('Missing matching family lower bound')
                previous = old[name, basis_name][mode]['cost']['entanglers']
                if previous != 6 or result['cost']['entanglers'] != 3:
                    raise ValueError('Unexpected primary archive or construction count')
                row[mode] = result
                row.setdefault('previous_scored_entangler_counts', {})[mode] = previous
                row.setdefault('construction_and_replay_seconds', {})[mode] = time.monotonic()-start
            rows.append(row)
            print(name+'/'+basis_name+': both phase policies replayed, 6 -> 3 SQiSW', flush=True)
    paths = [Path(__file__), *sorted((ROOT/'squander/decomposition').glob('symbolic_*.py')),
             *sorted((ROOT/'squander/decomposition').glob('symbolic_*.json'))]
    return dict(schema_version=1, kind='exact_SQiSW_construction_frontier',
                generated_at=datetime.now(timezone.utc).isoformat(),
                git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256={str(path.relative_to(ROOT)): digest(path) for path in paths},
                primary_scored_archive=dict(path=str(primary_path.relative_to(ROOT)), sha256=digest(primary_path),
                                           modified=False),
                summary=dict(cells=10, phase_modes=20, all_construction_certificates_replayed=True,
                             matched_minimum_uniform_budgets=20, published_uniform_caps_exceeded=0,
                             local_scores_evaluated=0, global_local_minima_proved=0,
                             unscored_aliases_used_as_scores=False),
                scope='Minimum uniform entangler budgets; not pointwise minima or local-score minima',
                score_policy='Preserve version-2 audited scores; do not assign scores to temporary binding names',
                references=['https://arxiv.org/abs/2105.06074', 'https://arxiv.org/abs/quant-ph/0002045'],
                rows=rows)


def replay(report):
    cases = {name: parameters for name, parameters, _ in symbolic_cases()}
    expected = {(name, basis) for name in FAMILIES for basis in BASES}
    if report.get('kind') != 'exact_SQiSW_construction_frontier':
        raise ValueError('Unexpected archive kind')
    if {(row['gate'], row['basis']) for row in report['rows']} != expected or len(report['rows']) != 10:
        raise ValueError('Incomplete or duplicate frontier grid')
    count = 0
    for row in report['rows']:
        target = make_gate(row['gate'], *cases[row['gate']])
        for mode in ('phase_exact', 'projective'):
            with wall_time_limit(60):
                if not verify_sqrt_iswap_cartan_certificate(target, row[mode], get_basis(row['basis'])):
                    raise ArithmeticError('Certificate replay failed: '+row['gate']+'/'+row['basis']+'/'+mode)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--primary', type=Path, default=ROOT/'benchmarks/symbolic/synthesis_reference_20261106.json.gz')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        with gzip.open(args.verify, 'rt') as stream:
            report = json.load(stream)
        print(str(replay(report))+' exact construction certificates replayed')
        return
    if not args.output:
        parser.error('Supply --output or --verify')
    report = generate(args.primary.resolve())
    temporary = args.output.with_name(args.output.name+'.tmp')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(report, stream, allow_nan=False, separators=(',', ':'))
    temporary.replace(args.output)
    print('Archive SHA256 '+digest(args.output))


if __name__ == '__main__':
    main()
