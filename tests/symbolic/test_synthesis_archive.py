import time

import pytest
import sympy as sp

import symbolic_archive as archive
import symbolic_gates as gates
from symbolic_bases import BASIS_SETS
from symbolic_clifford_search import group_data
from symbolic_clifford import analyze_single_qubit_cliffords


def test_archive_covers_every_definition_once_with_free_real_parameters():
    cases=list(archive.symbolic_cases())
    assert len(cases)==327 and len(BASIS_SETS)==32
    assert {c[0] for c in cases}==set(gates.GATE_DEFINITIONS)
    for name,parameters,scope in cases:
        # CZPowGate is also a retained one-parameter legacy definition;
        # resolving its modern alias would add a different optional parameter.
        definition=gates.GATE_DEFINITIONS[name]
        if definition.num_qubits is not None:
            assert len(parameters)==len(definition.parameters)
            assert all(isinstance(p,sp.Symbol) and p.is_real for p in parameters)
            assert scope==("free_real_family" if parameters else "fixed_gate")
        else:
            assert scope=="register_fixture"
    assert dict((name,p) for name,p,_ in cases)["GP"][1]==1


def test_deadline_is_not_an_impossibility_certificate(monkeypatch):
    def slow(*args,**kwargs):
        time.sleep(0.1)
        raise AssertionError("The deadline should interrupt this operation")
    monkeypatch.setattr(archive,"synthesize",slow)
    result=archive.bounded_synthesis("H",(),"cx_u",phase_exact=True,seconds=0.01)
    assert result["status"]=="timeout"
    assert "cost" not in result and "operations" not in result
    with pytest.raises(ValueError):
        with archive.wall_time_limit(0):pass


def test_frozen_clifford_group_matches_independent_exact_analysis():
    assert group_data()[0]==analyze_single_qubit_cliffords()
