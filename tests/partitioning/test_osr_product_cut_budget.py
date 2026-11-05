"""A product cut reuses the complete smaller OSR solve, not its lower bound."""

import re

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    qgd_Wide_Circuit_Optimization as WCO,
)


@pytest.mark.parametrize("spectator", [0, 2, 3])
def test_product_cut_inherits_parent_budget_above_osr_floor(spectator, capfd):
    active = [wire for wire in range(4) if wire != spectator]
    source = QuantumCircuit(4)
    source.ccx(*active)
    source.u(0.31, -0.19, 0.47, spectator)
    target = Operator(source).data
    graph = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    optimizer = WCO({
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "max_partition_size": 4,
        "tree_level_max": 6,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 2,
        "random_seed": 20260926,
        "use_float": False,
        "osr_use_hilbert_schmidt_guidance": True,
        "osr_certified_terminal_completion": True,
        "osr_geometric_completion_motifs": True,
        "osr_physical_cover_motifs": True,
        "osr_certified_cnot_deflation": True,
        "max_inner_iterations_bfgs2": 1000,
        "max_iteration_loops": 8,
        "osr_trace_candidate_placements": True,
    })
    # Only the matrix and device edges reach synthesis, never source angles.
    results = optimizer.DecomposePartition(target, optimizer.config, mini_topology=graph)
    captured = capfd.readouterr()
    completions = re.findall(
        r"OSR product-cut completion: spectator=(\d+), cnots=(\d+), "
        r"floor=(\d+), hs=([^\s]+), certified=1",
        captured.out + captured.err,
    )
    assert completions, "The exact product cut did not use its 3q solver"
    assert any(int(wire) == spectator and int(floor) < int(count) <= 6
               for wire, count, floor, _ in completions)
    assert results
    allowed = {frozenset(edge) for edge in graph}
    for circuit, parameters in results:
        angles = np.asarray(parameters, dtype=np.float64)
        assert np.all(np.isfinite(angles))
        exported = get_Qiskit_Circuit(circuit, parameters)
        native = circuit.get_Matrix(angles)
        matrix = Operator(exported).data
        assert exported.count_ops().get("cx", 0) <= 6
        for left, right in ((target, native), (target, matrix), (native, matrix)):
            phase = np.exp(-1j * np.angle(np.vdot(left, right)))
            squared = np.linalg.norm(left - phase * right) ** 2 / 16
            assert squared - squared ** 2 / 4 <= 1e-10
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                edge = frozenset(exported.find_bit(wire).index for wire in instruction.qubits)
                assert edge in allowed
