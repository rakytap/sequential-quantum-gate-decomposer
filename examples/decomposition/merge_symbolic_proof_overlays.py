#!/usr/bin/env python3
"""Merge independent proof metadata while preserving every archived result."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ROW_KEYS={'independent_entangler_certificates','exchange_entangler_lower_bound_certificates'}
ROOT_KEYS={'independent_entangler_proof_summary','independent_entangler_proof_supplement',
           'exchange_entangler_proof_summary','exchange_entangler_proof_supplement'}


def load(path):
    with gzip.open(path,'rt') as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(base,overlays):
    result=copy.deepcopy(base)
    original={(row['gate'],row['basis']):row for row in base['rows']}
    index={(row['gate'],row['basis']):row for row in result['rows']}
    for overlay in overlays:
        if {k:v for k,v in overlay.items() if k not in ROOT_KEYS|{'rows'}} != {
                k:v for k,v in base.items() if k not in ROOT_KEYS|{'rows'}}:
            raise ValueError('An overlay changes original archive metadata')
        rows={(row['gate'],row['basis']):row for row in overlay['rows']}
        if rows.keys()!=original.keys() or len(rows)!=len(overlay['rows']):
            raise ValueError('An overlay changes the target/basis grid')
        for key,row in rows.items():
            if {k:v for k,v in row.items() if k not in ROW_KEYS} != {
                    k:v for k,v in original[key].items() if k not in ROW_KEYS}:
                raise ValueError('An overlay changes a construction, score, or provenance: '+str(key))
            for field in ROW_KEYS:
                if field in row:
                    index[key].setdefault(field,{}).update(copy.deepcopy(row[field]))
        for field in ROOT_KEYS:
            if field in overlay:
                result[field]=copy.deepcopy(overlay[field])
    return result


def check_sources(revision,hashes):
    for relative,expected in hashes.items():
        blob=subprocess.check_output(['git','show',revision+':'+relative],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=expected:
            raise ValueError('Proof source blob differs: '+relative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--overlay',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=load(args.base);overlays=[]
    for path in args.overlay:
        overlay=load(path)
        for field in ('independent_entangler_proof_supplement','exchange_entangler_proof_supplement'):
            metadata=overlay.get(field)
            if not metadata:
                continue
            report_path=Path(metadata['path'])
            if not report_path.is_absolute():
                report_path=ROOT/report_path
            if metadata['input_archive_sha256']!=digest(args.base) or metadata['sha256']!=digest(report_path):
                raise ValueError('Proof input/report hash differs')
            report=load(report_path)
            check_sources(report['git_revision'],report['source_sha256'])
            if field=='independent_entangler_proof_supplement':
                replay=report.get('replay') or {}
                if not replay.get('all_certificates_replayed'):
                    raise ValueError('Native proofs have not all replayed')
                check_sources(replay['runner_revision'],replay['source_sha256'])
            elif not report.get('all_certificates_replayed'):
                raise ValueError('Exchange proofs have not all replayed')
        overlays.append(overlay)
    result=merge(base,overlays)
    with gzip.open(args.output,'wt') as stream:
        json.dump(result,stream,separators=(',',':'),allow_nan=False)
    print(json.dumps(dict(archive_sha256=digest(args.output),
                         construction_payloads_unchanged=True,overlays=len(overlays))))


if __name__=='__main__':
    main()
