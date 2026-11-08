#!/usr/bin/env python3
"""Package the revised paper with complete atlases and scientific records."""
import gzip,hashlib,importlib.metadata,json,re,shutil,sys,zipfile
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/symbolic_manuscript';DATA=ROOT/'benchmarks/symbolic'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_zip(path,members):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,payload in sorted(members.items()):z.writestr(name,payload.read_bytes() if isinstance(payload,Path) else payload)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None

def main():
    archive=DATA/'synthesis_reference_20261107_referee.json.gz'
    revision=json.loads((DOC/'referee-revision.json').read_text())
    verification=json.loads((DOC/'referee-verification.json').read_text())
    assert sha(archive)==revision['archive_sha256']==verification['archive_sha256']
    assert verification['all_changed_constructions_replayed'] and verification['all_source_checks_passed']
    gaps=json.loads((DOC/'native-gap-verification.json').read_text())
    assert gaps['archive_sha256']==sha(DATA/'native_column_gap_audit_20261107.json.gz')
    assert gaps['maximum_literal_entry_error']<1e-8
    shutil.copy2(DOC/'referee-verification.json',DOC/'release-verification.json')
    logs={'full_suite':Path('/tmp/second_referee_full_tests.log'),
          'focused_suite':Path('/tmp/second_referee_focused_tests.log'),
          'new_theory_suite':Path('/tmp/second_referee_new_tests.log'),
          'repaired_suite':Path('/tmp/second_referee_repaired_tests.log')}
    validation={}
    for key,path in logs.items():
        text=path.read_text();matches=re.findall(r'(\d+) passed(?:, (\d+) warning)? in ([\d.]+)s',text)
        assert matches and not re.search(r'\d+ error',text),text[-2000:]
        failures=re.findall(r'^FAILED (tests/\S+)',text,re.MULTILINE)
        if failures:
            assert key=='full_suite' and failures==[
                'tests/symbolic/test_notable_results.py::test_sycamore_local_proof_has_the_restricted_phase_polynomial_scope']
        count,warnings,seconds=matches[-1]
        validation[key]=dict(status='stale_assertion_repaired_and_rerun' if failures else 'passed',
             passed=int(count),initial_failed_tests=failures,warnings=int(warnings or 0),seconds=float(seconds),log_sha256=sha(path))
    assert validation['repaired_suite']['passed']==19
    validation['overall_status']='passed_after_targeted_test_repair'
    validation['covered_tests']=validation['full_suite']['passed']+len(validation['full_suite']['initial_failed_tests'])
    validation['scope']='The full collection ran once. Its sole failure referenced the withdrawn tomography-comparison constant; that stale test was corrected and rerun with all four notable-result tests and fifteen new theorem/native tests. Both unedited logs are included. No synthesis source changed after full-suite collection.'
    (DOC/'test-results.json').write_text(json.dumps(validation,indent=2)+'\n')
    source=Path('/tmp/symbolic-atlas-full');atlas=ROOT/'docs/symbolic_atlas'
    atlas_manifest=json.load(gzip.open(source/'manifest.json.gz','rt'))
    assert atlas_manifest['archive_sha256']==sha(archive)
    build=json.loads((source/'build-report.json').read_text())
    atlas_members={n:source/n for n in ('gates.tex','cliffords.tex','manifest.json.gz','build-report.json')}
    for n in ('gates.tex','cliffords.tex'):
        for fragment in re.findall(r'\\input\{(circuits/[^}]+\.tex)\}',(source/n).read_text()):atlas_members[fragment]=source/fragment
    for p in (source/'programs').glob('*.json'):atlas_members[str(p.relative_to(source))]=p
    atlas_members['build_symbolic_atlas.py']=ROOT/'examples/decomposition/build_symbolic_atlas.py'
    atlas_members['BUILD.txt']=b'Compile gates.tex and cliffords.tex with Tectonic; preserve circuits/ and programs/. For parallel pages use build_symbolic_atlas.py --source-dir . --workers 8. Requires pypdf.\n'
    for n in ('gates.pdf','cliffords.pdf','manifest.json.gz','build-report.json'):shutil.copy2(source/n,atlas/n)
    write_zip(atlas/'sources.zip',atlas_members)
    shutil.copy2(DOC/'referee-verification.json',atlas/'source-validation.json')
    (atlas/'summary.json').write_text(json.dumps(dict(schema='symbolic-atlas-referee-v2',archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),
         pages={k:v['pages'] for k,v in build.items()},coverage=revision['by_width'],tests=validation,
         limits=['Not all-grid optimality','No universal compiler dominance','Twenty secondary scores unresolved']),indent=2)+'\n')
    texts='\n'.join((DOC/n).read_text() for n in ('manuscript.tex','supplement.tex'))
    used=re.findall(r'\\input\{(generated/[^}]+)\}',texts)
    used+=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{(generated/[^}]+)\}',texts)
    members={n:DOC/n for n in ('manuscript.tex','supplement.tex','references.tex','referee-response.tex',
               'first-referee-response.tex','README.md','referee-revision.json','referee-verification.json',
               'native-gap-verification.json','editorial-status.json','test-results.json','claims-review.json')}
    for n in used:members[n]=DOC/n
    members['generated/optimality_ledger.csv']=DOC/'generated/optimality_ledger.csv'
    members['generated/native_column_gaps.csv']=DOC/'generated/native_column_gaps.csv'
    files=['synthesis_reference_20261107_referee.json.gz','synthesis_reference_20261107_referee_base.json.gz',
           'synthesis_ffredkin_referee_20261107.json.gz','referee_strong_sdks_final_20261107.json.gz','referee_rules_20261107.json.gz',
           'referee_bqskit_layouts_20261107.json.gz','referee_components_20261107.json.gz','referee_peephole_20261107.json.gz',
           'manuscript_experiments_20261107.json.gz','synthesis_native_second_referee_20261107.json.gz',
           'synthesis_tk2_second_referee_20261107.json.gz','native_column_gap_audit_20261107.json.gz',
           'c4x_published_routes_20261107.json.gz']
    for n in files:members['data/'+n]=DATA/n
    for pattern in ('symbolic_*.py','symbolic_*.json','sympy_gen.py'):
        for p in (ROOT/'squander/decomposition').glob(pattern):members['repository/'+str(p.relative_to(ROOT))]=p
    for p in (ROOT/'examples/decomposition').glob('*.py'):
        if p.name.startswith(('symbolic_','build_symbolic','verify_symbolic','package_symbolic')):
            members['repository/'+str(p.relative_to(ROOT))]=p
    for p in (ROOT/'tests/symbolic').glob('*.py'):members['repository/'+str(p.relative_to(ROOT))]=p
    requirements='\n'.join(n+'=='+importlib.metadata.version(n) for n in ('numpy','scipy','sympy','qiskit','cirq-core','matplotlib','pypdf','pytest'))+'\n'
    members['requirements-symbolic.txt']=requirements.encode()
    sdk=json.load(gzip.open(DATA/'referee_strong_sdks_final_20261107.json.gz','rt'))
    members['requirements-sdk-python312.txt']=('\n'.join(n+'=='+v for n,v in sdk['versions'].items())+'\n').encode()
    members['BUILD.txt']=b'Paper: tectonic manuscript.tex. Supplement: tectonic supplement.tex. Prism: copy manuscript.tex, supplement.tex, references.tex and generated/. Response: tectonic referee-response.tex. The full review ZIP includes both atlas PDFs and atlas-sources.zip. Current pure-symbolic sources and scientific records are in repository/ and data/. Numeric native witnesses are separate from exact identities; family samples are not uniform bounds. Reconstruct historical implementations using source-content hashes/Git revisions. SDK experiments and the C4X harness use the pinned Python-3.12 environment. Quartz/QUESO need their recorded builds; rules and harness patch are in the raw record.\n'
    for name,path in logs.items():members['validation/'+name+'.log']=path
    write_zip(DOC/'sources.zip',members)
    notable=ROOT/'docs/symbolic_notable_results'
    write_zip(notable/'sources.zip',{p.name:p for p in notable.iterdir() if p.suffix in ('.tex','.json','.csv')})
    package={'manuscript.pdf':DOC/'manuscript.pdf','supplement.pdf':DOC/'supplement.pdf',
             'referee-response.pdf':DOC/'referee-response.pdf','first-referee-response.pdf':DOC/'first-referee-response.pdf',
             'paper-sources.zip':DOC/'sources.zip','gate-atlas.pdf':atlas/'gates.pdf','clifford-atlas.pdf':atlas/'cliffords.pdf',
             'atlas-sources.zip':atlas/'sources.zip','selected-identities.pdf':notable/'notable_results.pdf',
             'selected-identities-sources.zip':notable/'sources.zip','README.md':DOC/'README.md'}
    write_zip(DOC/'review-package.zip',package)
    outputs=[DOC/n for n in ('manuscript.pdf','supplement.pdf','referee-response.pdf','sources.zip','review-package.zip')]+[atlas/n for n in ('gates.pdf','cliffords.pdf','sources.zip')]+[notable/'notable_results.pdf',archive]
    report=dict(schema='symbolic-referee-package-v2',manuscript_pages=len(PdfReader(DOC/'manuscript.pdf').pages),
         response_pages=len(PdfReader(DOC/'referee-response.pdf').pages),atlas_pages={k:v['pages'] for k,v in build.items()},
         supplement_pages=len(PdfReader(DOC/'supplement.pdf').pages),native_numerical_replays=gaps,
         validation=validation,exact_changed_replays=verification['changed_construction_modes'],
         unchanged_payload_comparisons=verification['unchanged_construction_modes'],
         artifacts={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in outputs})
    assert all(p['bytes']<100_000_000 for p in report['artifacts'].values())
    (DOC/'release-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (DOC/'build-manifest.json').write_text(json.dumps(dict(schema='symbolic-referee-build-v2',archive_sha256=sha(archive),
         tables=revision['by_width'],generated_files={n:sha(DOC/n) for n in sorted(set(used))}),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifacts'},indent=2))

if __name__=='__main__':main()
