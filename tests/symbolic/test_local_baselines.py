import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[2]/"examples/decomposition/symbolic_local_baselines.py"
spec=importlib.util.spec_from_file_location("local_baselines",path)
baseline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_native_cleanup_removes_adapter_identity_gates_before_local_comparison():
    pytest.importorskip("qiskit")
    row=dict(status="verified_numeric",basis="cx_rz_ry",num_qubits=2,global_phase=0,
        entanglers=1,max_error=0,operations=[
            dict(gate="rz",parameters=[0.7],wires=[0]),
            dict(gate="ry",parameters=[0.0],wires=[0]),
            dict(gate="rz",parameters=[0.0],wires=[0]),
            dict(gate="cx",parameters=[],wires=[0,1])])
    result=baseline.normalize_baseline(row)
    assert result["native_local_cleanup"]["raw_local_count"]==3
    assert result["native_local_cleanup"]["retained_local_count"]==1
    assert result["entanglers"]==1
    assert result["max_error"]<2e-10
    assert len(row["operations"])==4


def test_local_cleanup_rejects_inconsistent_entangler_metadata():
    pytest.importorskip("qiskit")
    row=dict(status="verified_numeric",basis="cx_u",num_qubits=2,global_phase=0,
             entanglers=2,max_error=0,operations=[dict(gate="cx",parameters=[],wires=[0,1])])
    with pytest.raises(ValueError,match="entangler count"):
        baseline.normalize_baseline(row)
