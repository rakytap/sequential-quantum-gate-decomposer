"""Reject silent loss of QASM rotation arguments in benchmark ingestion."""
import importlib.util
from pathlib import Path
import sympy as sp
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('referee_peephole',ROOT/'examples/decomposition/symbolic_referee_peephole.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_spaced_angles_multiple_registers_and_exact_decimals(tmp_path):
    path=tmp_path/'c.qasm'
    path.write_text('OPENQASM 2.0; include "qelib1.inc"; qreg a[2]; qreg b[1]; rx (-0.125) a[0]; rz(pi/4) b[0]; cx a[1],b[0];')
    width,word=module.parse(path)
    assert width==3
    assert word==( ('Rx',-sp.Rational(1,8),(0,)),('Rz',sp.pi/4,(2,)),('CX',(1,2)) )

def test_missing_rotation_angle_is_rejected(tmp_path):
    path=tmp_path/'bad.qasm';path.write_text('qreg q[2]; rz q[0];')
    with pytest.raises(ValueError):module.parse(path)
