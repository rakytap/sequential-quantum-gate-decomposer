import copy
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[2]/'examples/decomposition/merge_symbolic_proof_overlays.py'
spec=importlib.util.spec_from_file_location('proof_overlays',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_proof_overlay_preserves_circuit_score_and_source():
    base=dict(rows=[dict(gate='CX',basis='cx_u',phase_exact=dict(
        operations=['CX'],cost={'entanglers':1}),mode_source_run_indices={'phase_exact':2})])
    first=copy.deepcopy(base)
    first['rows'][0]['independent_entangler_certificates']={'phase_exact':{'lower_bound':1}}
    second=copy.deepcopy(base)
    second['rows'][0]['exchange_entangler_lower_bound_certificates']={'projective':{'lower_bound':1}}
    merged=module.merge(base,[first,second])
    assert merged['rows'][0]['phase_exact']==base['rows'][0]['phase_exact']
    assert len(merged['rows'][0])==len(base['rows'][0])+2
    for field,value in [('phase_exact',{'operations':['I'],'cost':{'entanglers':0}}),
                        ('mode_source_run_indices',{'phase_exact':7})]:
        changed=copy.deepcopy(first)
        changed['rows'][0][field]=value
        with pytest.raises(ValueError,match='construction, score, or provenance'):
            module.merge(base,[changed])
