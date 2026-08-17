# -*- coding: utf-8 -*-
"""
Tests for wide-circuit optimization flow.
"""

from pathlib import Path

import numpy as np

from squander import utils
import squander.decomposition.qgd_Wide_Circuit_Optimization as Wide_Circuit_Optimization
from squander.decomposition.qgd_Wide_Circuit_Optimization import CNOTGateCount
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit


def _load_qasm_as_squander_circuit(qasm_path):
    """Load a QASM file and normalize legacy/new utils return shapes."""
    loaded = utils.qasm_to_squander_circuit(str(qasm_path))
    if len(loaded) == 2:
        circ, parameters = loaded
    else:
        circ, parameters, _ = loaded
    return circ, parameters


def test_wide_circuit_optimization_bv_n14():
    """Run one wide-circuit optimization pass on bv_n14 and validate outputs."""
    qasm_file = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "decomposition"
        / "bv_n14.qasm"
    )
    assert qasm_file.exists(), f"Missing test circuit file: {qasm_file}"

    circ, parameters = _load_qasm_as_squander_circuit(qasm_file)
    config = {
        "strategy": "TreeSearch",
        "test_subcircuits": False,
        "test_final_circuit": False,
        "max_partition_size": 3,
        "beam": None,
        "use_osr": True,
        "use_graph_search": True,
        "pre-opt-strategy": "TreeSearch",
        "routing-strategy": "seqpam-ilp",
        "tolerance": 1e-10,
        "topology": Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
            circ.get_Qbit_Num()
        ),
    }

    optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization({**config})
    opt_circ, opt_params = optimizer.OptimizeWideCircuit(circ, parameters)

    assert opt_circ is not None
    assert opt_params is not None
    assert opt_circ.get_Qbit_Num() == circ.get_Qbit_Num()
    assert CNOTGateCount(opt_circ, 0) >= 0


def test_equal_cnot_structural_rewrite_gets_followup_round(monkeypatch):
    """An equal-count structural rewrite may unlock the following pass."""
    initial = Circuit(2)
    initial.add_CNOT(1, 0)
    plateau = Circuit(2)
    plateau.add_CNOT(0, 1)
    improved = Circuit(2)
    outputs = iter((plateau, improved, improved))
    calls = []

    def optimize_round(self, circ, parameters, **kwargs):
        calls.append(CNOTGateCount(circ, 0))
        return next(outputs), np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(
        Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization,
        "InnerOptimizeWideCircuit",
        optimize_round,
    )
    optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization(
        {
            "strategy": "TreeSearch",
            "pre-opt-strategy": "TreeSearch",
            "topology": None,
            "test_final_circuit": False,
            "max_equal_cnot_optimization_rounds": 3,
        }
    )
    result, _parameters = optimizer.OptimizeWideCircuit(
        initial, np.empty((0,), dtype=np.float64)
    )

    assert CNOTGateCount(result, 0) == 0
    assert calls == [1, 1, 0]


def test_equal_cnot_parameter_drift_does_not_extend_plateau(monkeypatch):
    """Numerical refits alone do not expose a different partition structure."""
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    calls = []

    def optimize_round(self, circ, parameters, **kwargs):
        calls.append(1)
        return circuit, np.asarray([1e-12, 0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(
        Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization,
        "InnerOptimizeWideCircuit",
        optimize_round,
    )
    optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization(
        {
            "strategy": "TreeSearch",
            "pre-opt-strategy": "TreeSearch",
            "topology": None,
            "test_final_circuit": False,
        }
    )
    result, _parameters = optimizer.OptimizeWideCircuit(
        circuit, np.zeros((3,), dtype=np.float64)
    )

    assert CNOTGateCount(result, 0) == 1
    assert calls == [1]
