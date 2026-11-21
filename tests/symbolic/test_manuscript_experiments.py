"""Protect independent benchmark inputs and phase/resource conventions."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "examples/decomposition/symbolic_manuscript_experiments.py"
spec = importlib.util.spec_from_file_location("manuscript_experiments", PATH)
experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiments)


@pytest.mark.parametrize("case", experiments.held_out_cases(), ids=lambda row: row[0])
def test_held_out_source_is_independent_exact_identity(case):
    _, target, source = case
    assert experiments.prove_circuit_identity(target, source) is True


def test_controlled_substitution_requires_tracked_phase():
    audit = experiments.phase_embedding()
    assert audit["controlled_correction_identity_proved"]
    assert audit["uncorrected_controlled_substitution_fails"]
    assert audit["required_control_phase"] == "-7*pi/24"


def test_sdk_parameter_conversion_preserves_affine_expression():
    pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter
    import sympy as sp
    theta = sp.Symbol("theta", real=True)
    sdk = Parameter("theta")
    expression = experiments._linear_parameter(2*theta-sp.pi/4, {theta: sdk})
    assert float(expression.bind({sdk: .3})) == pytest.approx(.6-float(sp.pi)/4)
