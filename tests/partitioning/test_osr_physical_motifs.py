"""Native minimum-cover motifs must preserve topology and strict precision."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    qgd_Wide_Circuit_Optimization,
)


@pytest.mark.parametrize("width,inverse,minimum", [(3, False, 4), (4, False, 7), (4, True, 7)])
def test_tree_minimum_cover_uses_noncommuting_block_order(width, inverse, minimum):
    target = QuantumCircuit(width)
    target.cx(1, 0)
    if width == 4:
        target.swap(2, 3)
    target.swap(1, 2)
    if inverse:
        target = target.inverse()
    unitary = np.asarray(Operator(target).data)
    topology = [(qubit, qubit + 1) for qubit in range(width - 1)]
    optimizer = qgd_Wide_Circuit_Optimization(
        {
            "strategy": "TreeSearch",
            "use_osr": True,
            "use_graph_search": True,
            "max_partition_size": width,
            "parallel": 0,
            "verbosity": 0,
            "random_seed": 20260926,
            "synthesis_acceptance_tolerance": 1e-10,
            "osr_physical_cover_motifs": True,
        }
    )
    results = optimizer.DecomposePartition(
        unitary, optimizer.config, mini_topology=topology
    )
    assert results
    for circuit, parameters in results:
        exported = get_Qiskit_Circuit(circuit, parameters)
        assert exported.count_ops().get("cx", 0) == minimum
        matrix = np.asarray(Operator(exported).data)
        native = np.asarray(circuit.get_Matrix(parameters))
        assert np.isfinite(matrix).all() and np.isfinite(native).all()
        dimension = unitary.shape[0]
        assert 1.0 - abs(np.vdot(unitary, matrix) / dimension) ** 2 <= 1e-10
        assert 1.0 - abs(np.vdot(unitary, native) / dimension) ** 2 <= 1e-10
        assert 1.0 - abs(np.vdot(matrix, native) / dimension) ** 2 <= 1e-10
        for instruction in exported.data:
            if instruction.operation.name == "cx":
                endpoints = tuple(
                    sorted(exported.find_bit(qubit).index for qubit in instruction.qubits)
                )
                assert endpoints in topology
