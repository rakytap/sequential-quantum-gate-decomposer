#!/usr/bin/env python3
"""Certify small native entangler counts independently of archived searches.

Generate, replay, then attach a source-pinned proof supplement. Attaching
does not alter a construction, score, source-run index, or original flag.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'squander/decomposition'
sys.path.insert(0, str(SOURCE))
from symbolic_archive import symbolic_cases, wall_time_limit, SymbolicDeadline
from symbolic_bases import BASIS_SETS
from symbolic_gates import make_gate
from symbolic_entangler_proof import small_count_certificate, verify_small_count_certificate

CASES = {name:parameters for name,parameters,scope in symbolic_cases()}


def load(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(temporary, 'wt') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {str(path.relative_to(ROOT)):digest(path) for path in
            [Path(__file__), *sorted(SOURCE.glob('symbolic_*.py')),
             *sorted(SOURCE.glob('symbolic_*.json'))]}


@lru_cache(maxsize=327)
def target(name):
    return make_gate(name, *CASES[name])


def run_job(job):
    name, basis_name, count, exact, seconds, expected = job
    start = time.monotonic()
    result = dict(gate=name, basis=basis_name, entanglers=count, phase_exact=exact)
    try:
        with wall_time_limit(seconds):
            matrix = target(name)
            if matrix.shape != (4,4):
                result['status'] = 'outside_two_qubit_proof_scope'
            elif expected is None:
                certificate = small_count_certificate(matrix, BASIS_SETS[basis_name], count, phase_exact=exact)
                result['status'] = 'certified' if certificate else 'no_matching_lower_bound'
                if certificate:
                    result['certificate'] = certificate
            else:
                valid = verify_small_count_certificate(matrix, BASIS_SETS[basis_name], expected)
                result['status'] = 'replayed' if valid else 'replay_failed'
    except SymbolicDeadline:
        result['status'] = 'proof_timeout'
    except Exception as error:
        result.update(status='proof_error', error=type(error).__name__+': '+str(error))
    result['elapsed_seconds'] = time.monotonic()-start
    return result


def modes(row):
    for mode in ('phase_exact','projective'):
        result = row.get(mode)
        if mode == 'projective' and row.get('projective_omitted_as_no_better'):
            result = row['phase_exact']
        if result and result.get('status') == 'verified_symbolic':
            yield mode, result


def jobs(archive, seconds):
    selected = []
    for row in archive['rows']:
        # Building the small catalog matrices once in the parent also shares
        # their immutable cached expressions with forked proof workers.
        if target(row['gate']).shape != (4,4):
            continue
        for mode, result in modes(row):
            count = result['cost']['entanglers']
            if 1 <= count <= 4:
                selected.append((row['gate'],row['basis'],count,mode=='phase_exact',seconds,None))
    return selected


def execute(selected, workers, checkpoint=None):
    records = []
    counts = Counter()
    def consume(iterator):
        for result in iterator:
            records.append(result)
            counts[result['status']] += 1
            if len(records) <= 3 or len(records)%50 == 0:
                print(len(records), '/', len(selected), dict(counts), result['gate'],
                      result['basis'], flush=True)
            if checkpoint and len(records)%100 == 0:
                write(checkpoint,dict(records=records,summary=dict(counts)))
    if workers == 1:
        consume(map(run_job, selected))
    else:
        with mp.get_context('fork').Pool(workers) as pool:
            consume(pool.imap_unordered(run_job, selected, chunksize=4))
    records.sort(key=lambda row:(row['gate'],row['basis'],not row['phase_exact']))
    if checkpoint:
        write(checkpoint,dict(records=records,summary=dict(counts)))
    return records, dict(counts)


def check_inputs(archive_path, report):
    if report['input_archive_sha256'] != digest(archive_path):
        raise ValueError('Proof supplement names a different input archive')
    current=sources()
    runner=str(Path(__file__).relative_to(ROOT))
    for path,expected in report['source_sha256'].items():
        blob=subprocess.check_output(['git','show',report['git_revision']+':'+path],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=expected:
            raise ValueError('Pinned proof source differs: '+path)
        # The orchestration runner may fix deadlines/checkpointing. Every
        # mathematical dependency must still match the generating source;
        # the current runner itself is pinned separately in the replay.
        if path!=runner and current.get(path)!=expected:
            raise ValueError('Mathematical proof source differs: '+path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate','replay','attach'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seconds-per-proof', type=float, default=40)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds_per_proof <= 0:
        parser.error('Positive worker count and proof budget required')
    archive = load(args.archive)
    if args.action == 'generate':
        selected = jobs(archive, args.seconds_per_proof)
        print('Selected', len(selected), 'two-qubit phase modes', flush=True)
        records, summary = execute(selected, args.workers,
                                   args.report.with_name(args.report.name+'.partial.gz'))
        report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                      input_archive=str(args.archive), input_archive_sha256=digest(args.archive),
                      git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      source_sha256=sources(), summary=summary, records=records,
                      proof_policy='Exact native cut ranks and spin invariants; no numerical acceptance',
                      construction_policy='Original circuit identities, scores, and provenance unchanged',
                      family_scope='One exact hard specialization proves a minimum uniform family budget; not a pointwise minimum',
                      replay=None)
        write(args.report, report)
        print(json.dumps(summary), flush=True)
        return int(bool(summary.get('proof_error')))
    report = load(args.report)
    check_inputs(args.archive, report)
    if args.action == 'replay':
        selected = [(row['gate'],row['basis'],row['entanglers'],row['phase_exact'],
                     args.seconds_per_proof,row['certificate'])
                    for row in report['records'] if row['status'] == 'certified']
        records, summary = execute(selected, args.workers)
        report['replay'] = dict(summary=summary, records=records,
                                all_certificates_replayed=summary.get('replayed',0)==len(selected),
                                runner_sha256=digest(Path(__file__)),
                                runner_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(args.report, report)
        print(json.dumps(summary), flush=True)
        return int(not report['replay']['all_certificates_replayed'])
    if not args.output:
        parser.error('--output is required when attaching')
    if not (report.get('replay') or {}).get('all_certificates_replayed'):
        raise ValueError('Replay every new certificate before attaching')
    index = {(row['gate'],row['basis']):row for row in archive['rows']}
    statistics = Counter()
    for record in report['records']:
        if record['status'] != 'certified':
            continue
        row = index[record['gate'],record['basis']]
        mode = 'phase_exact' if record['phase_exact'] else 'projective'
        result = dict(modes(row))[mode]
        if result['cost']['entanglers'] != record['entanglers']:
            raise ValueError('Construction budget differs from certificate')
        certificate = record['certificate']
        row.setdefault('independent_entangler_certificates',{})[mode] = certificate
        statistics[mode+'_'+certificate['scope']] += 1
        if not result.get('entangler_optimal') and not result.get('conditional_entangler_optimality'):
            statistics[mode+'_previously_without_minimum_certificate'] += 1
    archive['independent_entangler_proof_summary'] = dict(statistics)
    archive['independent_entangler_proof_supplement'] = dict(
        path=str(args.report), sha256=digest(args.report),
        input_archive_sha256=report['input_archive_sha256'],
        source_sha256=report['source_sha256'], git_revision=report['git_revision'],
        all_certificates_replayed=True, modifies_original_constructions=False)
    write(args.output, archive)
    print(json.dumps(dict(statistics)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
