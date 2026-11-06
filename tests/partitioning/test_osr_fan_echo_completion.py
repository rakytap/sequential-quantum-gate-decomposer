"""Strict matrix-only regression controls for shared OSR motif completion."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization
from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit


@pytest.mark.parametrize('width', [3, 4])
@pytest.mark.parametrize('tolerance', [1e-10, 1e-12])
def test_fan_echo_completion_preserves_unitary_and_input_cnot_bound(width, tolerance):
    # Deterministic targets, not angles supplied to the synthesis optimizer.
    rng = np.random.default_rng(202610100 + 10 * width)
    fan = list(range(2, width))
    word = [(0, fan[0]), (0, 1)] + [(1, wire) for wire in fan] + [(0, 1)]
    source = QuantumCircuit(width)
    for wire in range(width):
        source.u(*rng.uniform(-np.pi, np.pi, 3), wire)
    for first, second in word:
        source.cx(first, second)
        for wire in (first, second):
            source.u(*rng.uniform(-np.pi, np.pi, 3), wire)
    target = np.ascontiguousarray(Operator(source).data)
    config = {
        'strategy': 'TreeSearch', 'optimizer': 'BFGS2', 'use_osr': True,
        'use_graph_search': True, 'use_float': False, 'parallel': 0,
        'random_seed': 20260926, 'osr_path_deterministic_rng': True,
        'synthesis_acceptance_tolerance': tolerance,
        'osr_optimization_tolerance': 1e-6,
        'osr_certified_terminal_completion': True,
        'osr_geometric_completion_motifs': True,
        'osr_geometric_completion_basin_hops': 2,
        'osr_cut_directed_completion': False,
        'max_iteration_loops': 8, 'max_inner_iterations_bfgs2': 1000,
        'verbosity': 0,
    }
    optimizer = qgd_Wide_Circuit_Optimization(config=config)
    topology = list(itertools.combinations(range(width), 2))
    proposals = optimizer.DecomposePartition(target, optimizer.config, mini_topology=topology)
    verified_counts = []
    for circuit, parameters in proposals:
        actual_circuit = get_Qiskit_Circuit(circuit, parameters)
        actual = np.asarray(Operator(actual_circuit).data)
        actual *= np.exp(-1j * np.angle(np.vdot(target, actual)))
        chord = np.linalg.norm(target - actual)**2 / (2 * len(target))
        error = 2 * chord - chord**2
        assert actual_circuit.num_qubits == width
        assert np.all(np.isfinite(parameters))
        assert error < tolerance
        assert np.linalg.norm(actual.conj().T @ actual - np.eye(len(actual))) < 1e-9
        for instruction in actual_circuit.data:
            if len(instruction.qubits) > 1:
                assert instruction.operation.name == 'cx'
        verified_counts.append(actual_circuit.count_ops().get('cx', 0))
    assert verified_counts, 'The matrix-only OSR solve must return a certified circuit.'
    assert min(verified_counts) <= len(word)
