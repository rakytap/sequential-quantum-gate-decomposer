import importlib.util
from pathlib import Path
import time

import sympy as sp


def test_audit_deadline_escapes_broad_algebra_library_exception_handlers(monkeypatch):
    path=Path(__file__).resolve().parents[2]/'examples/decomposition/symbolic_entangler_audit.py'
    spec=importlib.util.spec_from_file_location('native_bound_audit',path)
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    def algebra_library(unused):
        try:
            until=time.monotonic()+0.15
            while time.monotonic()<until:
                pass
        except Exception:
            # Some symbolic routines catch Exception while trying alternate
            # algorithms. The deadline must escape this handler.
            pass
        return sp.eye(4)
    monkeypatch.setattr(audit,'target',algebra_library)
    result=audit.run_job(('CX','cx_u',1,False,0.02,None))
    assert result['status']=='proof_timeout'
