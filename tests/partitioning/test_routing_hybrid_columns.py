import importlib

import numpy as np
import pytest

from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    SquanderPartitionSynthesisResult,
)
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing


def _cnot_circuit(count):
    circuit = qgd_Circuit(2)
    for _ in range(count):
        circuit.add_CNOT(1, 0)
    return circuit


@pytest.mark.parametrize(
    "embedded_cnots,topology_cnots,expected_cnots",
    ((3, 1, 1), (1, 3, 1), (1, 1, 1)),
)
def test_best_of_both_keeps_lower_verified_cnot_column(
    monkeypatch, embedded_cnots, topology_cnots, expected_cnots
):
    optimizer_module = importlib.import_module(
        "squander.decomposition.qgd_Wide_Circuit_Optimization"
    )
    target = _cnot_circuit(1).get_Matrix(
        np.empty((0,), dtype=np.float64), is_f32=False
    )
    calls = []

    def fake_synthesize(_target, _config, *, mini_topology):
        calls.append(mini_topology)
        count = embedded_cnots if mini_topology is None else topology_cnots
        return SquanderPartitionSynthesisResult(
            circuit=_cnot_circuit(count),
            parameters=np.empty((0,), dtype=np.float64),
            config={},
            topology=mini_topology,
        )

    monkeypatch.setattr(
        optimizer_module, "synthesize_partition_with_squander", fake_synthesize
    )
    result = routing._synthesize_routing_column(
        target,
        {
            "routing_column_synthesis_mode": "best-of-both",
            "synthesis_acceptance_tolerance": 1e-10,
        },
        ((0, 1),),
    )

    assert calls == [None, ((0, 1),)]
    assert result.circuit.get_Gate_Nums().get("CNOT", 0) == expected_cnots


def test_best_of_both_rejects_cheaper_incorrect_column(monkeypatch):
    optimizer_module = importlib.import_module(
        "squander.decomposition.qgd_Wide_Circuit_Optimization"
    )
    target = _cnot_circuit(1).get_Matrix(
        np.empty((0,), dtype=np.float64), is_f32=False
    )

    def fake_synthesize(_target, _config, *, mini_topology):
        count = 3 if mini_topology is None else 0
        return SquanderPartitionSynthesisResult(
            circuit=_cnot_circuit(count),
            parameters=np.empty((0,), dtype=np.float64),
            config={},
            topology=mini_topology,
        )

    monkeypatch.setattr(
        optimizer_module, "synthesize_partition_with_squander", fake_synthesize
    )
    result = routing._synthesize_routing_column(
        target,
        {
            "routing_column_synthesis_mode": "best-of-both",
            "synthesis_acceptance_tolerance": 1e-10,
        },
        ((0, 1),),
    )

    assert result.circuit.get_Gate_Nums().get("CNOT", 0) == 3
