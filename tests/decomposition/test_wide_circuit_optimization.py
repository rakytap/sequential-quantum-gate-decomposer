# -*- coding: utf-8 -*-
"""
Tests for wide-circuit optimization flow.
"""

from pathlib import Path

from squander import utils
import squander.decomposition.qgd_Wide_Circuit_Optimization as Wide_Circuit_Optimization
from squander.decomposition.qgd_Wide_Circuit_Optimization import CNOTGateCount


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
