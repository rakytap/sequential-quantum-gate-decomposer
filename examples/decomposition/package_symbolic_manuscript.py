#!/usr/bin/env python3
"""Freeze the built referee paper, atlas sources, provenance blobs and validation."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/symbolic_manuscript'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_zip(path,members):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in sorted(members.items()):
            z.writestr(name,value.read_bytes() if isinstance(value,Path) else value)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None


def atlas_package(source,dest):
    build=json.loads((source/'build-report.json').read_text())
    members={}
    for name in ('gates.tex','cliffords.tex','manifest.json.gz','build-report.json'):
        members[name]=source/name
    fragments=set()
    for name in ('gates.tex','cliffords.tex'):
        fragments.update(re.findall(r'\\input\{(circuits/[^}]+\.tex)\}',(source/name).read_text()))
    for name in fragments:members[name]=source/name
    for path in (source/'programs').glob('*.json'):members[str(path.relative_to(source))]=path
    members['build_symbolic_atlas.py']=ROOT/'examples/decomposition/build_symbolic_atlas.py'
    members['BUILD.txt']=('Compile the complete volumes with tectonic gates.tex and tectonic cliffords.tex.\n'
        'For bounded parallel page compilation: python3 build_symbolic_atlas.py --source-dir . --workers 8\n'
        'The helper needs pypdf and tectonic. Retain circuits/ and programs/ relative paths.\n'
        'Shared aliases are exact definitions, not free parameters. Twenty secondary scores are unresolved.\n').encode()
    for name in ('gates.pdf','cliffords.pdf'):shutil.copy2(source/name,dest/name)
    for name in ('manifest.json.gz','build-report.json'):shutil.copy2(source/name,dest/name)
    write_zip(dest/'sources.zip',members)
    return build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path,default=Path('/tmp/symbolic-atlas-full'))
    parser.add_argument('--preview',type=Path,default=Path('/tmp/symbolic-atlas-preview-manuscript'))
    args=parser.parse_args()
    verification=json.loads((DOC/'release-verification.json').read_text())
    assert verification['all_changed_constructions_replayed'] and verification['all_source_checks_passed']
    validation=json.loads((DOC/'test-results.json').read_text())
    assert validation['full_suite']['status']=='passed' and validation['focused_suite']['status']=='passed'
    atlas=ROOT/'docs/symbolic_atlas';build=atlas_package(args.atlas,atlas)
    for name in ('gates','cliffords'):
        shutil.copy2(args.preview/(name+'.pdf'),atlas/('preview-'+name+'.pdf'))
    shutil.copy2(DOC/'release-verification.json',atlas/'source-validation.json')
    summary=dict(schema='symbolic-atlas-manuscript-release-v1',archive='benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz',
                 archive_sha256=verification['archive_sha256'],pages={k:v['pages'] for k,v in build.items()},
                 construction_classification={'phase_exact':{'verified':7702,'obstruction':2762},
                                              'projective':{'verified':10196,'obstruction':268}},
                 exact_identity_replays=verification['changed_construction_modes'],
                 unchanged_payload_comparisons=verification['unchanged_construction_modes'],
                 unresolved_secondary_scores=20,uniform_cap_excesses=0,
                 selected_certificate_coverage=json.loads((DOC/'build-manifest.json').read_text())['selected_certificate_coverage'],
                 validation=validation,limits=['Classification is not unrestricted global optimality',
                   'One full held-out ablation hits its 20-second limit; old proof searches retain timeouts'])
    (atlas/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (atlas/'known_optimization_gaps.csv').open('w') as f:
        csv.writer(f,lineterminator='\n').writerow(['gate','basis','mode','entanglers','published_uniform_cap','source'])
    # Comparison appendix sources always match the regenerated PDF/data.
    notable=ROOT/'docs/symbolic_notable_results'
    write_zip(notable/'sources.zip',{p.name:p for p in notable.iterdir() if p.suffix in ('.tex','.json','.csv')})
    members={p.name:p for p in DOC.iterdir() if p.suffix in ('.tex','.json','.md') and p.name!='release-manifest.json'}
    for p in (DOC/'generated').iterdir():members[str(p.relative_to(DOC))]=p
    archive=ROOT/'benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz'
    data=json.load(gzip.open(archive,'rt'))
    files=['synthesis_reference_20261107_manuscript.json.gz','synthesis_reference_20261107_manuscript_base.json.gz',
           'synthesis_integrated_sqrt_iswap_20261107.json.gz','synthesis_ffredkin_four_20261107.json.gz',
           'manuscript_experiments_20261107.json.gz','native_entangler_proofs_20261107.json.gz',
           'exchange_entangler_proofs_20261107.json.gz','reference_comparison_20261106.json.gz']
    for name in files:members['data/'+name]=ROOT/'benchmarks/symbolic'/name
    members['repository/docs/symbolic_notable_results/notable_results.json']=ROOT/'docs/symbolic_notable_results/notable_results.json'
    exceptions={(r['run'],r['path']):r['resolved_content_revision'] for r in verification['recorded_revision_exceptions']}
    mapping=[]
    for i,run in enumerate(data['synthesis_runs']):
        for name,digest in run['source_sha256'].items():
            path=Path(name);relative=str(path.relative_to(ROOT)) if path.is_absolute() else name
            revision=exceptions.get((i,relative),run['git_revision'])
            key='provenance/source-blobs/'+digest+path.suffix
            if key not in members:
                content=subprocess.check_output(['git','show',revision+':'+relative],cwd=ROOT)
                assert hashlib.sha256(content).hexdigest()==digest
                members[key]=content
            mapping.append(dict(run=i,path=relative,content_sha256=digest,resolved_content_revision=revision,bundle_path=key))
    members['provenance/source-map.json']=(json.dumps(mapping,indent=2)+'\n').encode()
    for path in (ROOT/'squander/decomposition').glob('symbolic_*'):
        if path.suffix in ('.py','.json'):members['repository/'+str(path.relative_to(ROOT))]=path
    for pattern in ('symbolic_*.py','build_symbolic*.py','verify_symbolic_manuscript_release.py','package_symbolic_manuscript.py'):
        for path in (ROOT/'examples/decomposition').glob(pattern):members['repository/'+str(path.relative_to(ROOT))]=path
    for path in (ROOT/'tests/symbolic').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:members['repository/'+str(path.relative_to(ROOT))]=path
    requirements='\n'.join(name+'=='+importlib.metadata.version(name) for name in
                            ('numpy','scipy','sympy','qiskit','cirq-core','matplotlib','pypdf','pytest'))+'\n'
    members['requirements-review.txt']=requirements.encode()
    for name in ('sqrt_iswap_cartan.tex','sqrt_iswap_cartan_circuit.tex','sqrt_iswap_cartan.pdf','proofs.tex','proofs.pdf'):
        members['supporting/'+name]=ROOT/'docs/symbolic_optimality'/name
    for name in ('notable_results.tex','notable_results.pdf'):members['supporting/'+name]=notable/name
    members['BUILD.txt']=('For Prism or a paper-only build, copy manuscript.tex and generated/ together.\n'
       'Run tectonic manuscript.tex. The paper has no external bibliography build step.\n'
       'The repository/ tree contains current pure symbolic Python sources and tests.\n'
       'Install requirements-review.txt for experiments; use the pinned repository for Git-history validation.\n'
       'provenance/ includes all recorded source-content snapshots and the exact run-to-file map.\n'
       'The gate/Clifford atlases and their source bundle are separate downloads.\n').encode()
    write_zip(DOC/'sources.zip',members)
    paths=[DOC/'manuscript.pdf',DOC/'manuscript.tex',DOC/'sources.zip',DOC/'release-verification.json',DOC/'build-manifest.json',
           *[atlas/name for name in ('gates.pdf','cliffords.pdf','preview-gates.pdf','preview-cliffords.pdf','sources.zip')],
           notable/'notable_results.pdf',notable/'sources.zip',archive]
    report=dict(schema='symbolic-referee-package-v1',manuscript_pages=len(PdfReader(DOC/'manuscript.pdf').pages),
                atlas_pages={k:v['pages'] for k,v in build.items()},tests=validation,
                all_changed_constructions_replayed=True,provenance_content_blobs=sum(k.startswith('provenance/source-blobs/') for k in members),
                artifacts={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths})
    assert all(v['bytes']<100_000_000 for v in report['artifacts'].values())
    (DOC/'release-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'},indent=2))


if __name__=='__main__':main()
