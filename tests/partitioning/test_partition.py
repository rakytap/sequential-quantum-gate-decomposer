import pytest
import numpy as np
from squander import Circuit
from squander import utils

from squander.partitioning.partition import PartitionCircuitQasm
from squander.partitioning.kahn import kahn_partition
from squander.partitioning.ilp import get_all_partitions, routing_partition_weights
from squander.partitioning.tools import get_qubits


"""
CORRECTNESS TESTS
"""


def test_AllPartitionsMatchesExhaustiveConvexContractedSubgraphs():
    """The optimized enumerator must not lose cores when it contracts 1q chains."""
    circuit = Circuit(4)
    circuit.add_CNOT(0, 1)
    circuit.add_U3(1)
    circuit.add_U3(1)
    circuit.add_CNOT(1, 2)
    circuit.add_U3(2)
    circuit.add_CNOT(2, 3)
    circuit.add_CNOT(0, 2)

    enumerated, graph, _, _, chains, gate_qubits, _ = get_all_partitions(
        circuit, 3
    )
    nodes = tuple(graph)

    def descendants(start):
        reached = set()
        pending = list(graph[start])
        while pending:
            gate = pending.pop()
            if gate not in reached:
                reached.add(gate)
                pending.extend(graph[gate] - reached)
        return reached

    reach = {gate: descendants(gate) for gate in nodes}
    exhaustive = set()
    for mask in range(1, 1 << len(nodes)):
        part = frozenset(
            gate for index, gate in enumerate(nodes) if mask & (1 << index)
        )
        qubits = set().union(*(gate_qubits[gate] for gate in part))
        if len(qubits) > 3:
            continue
        convex = True
        for left in part:
            for right in part & reach[left]:
                between = reach[left] & {
                    gate
                    for gate in nodes
                    if gate == right or right in reach[gate]
                }
                if not between <= part:
                    convex = False
                    break
            if not convex:
                break
        if convex:
            exhaustive.add(part)

    assert set(enumerated) == exhaustive
    assert chains == {(1, 2), (4,)}


def test_RoutingWeightsPreferBalancedEntanglerDepth():
    """Equal-count covers should avoid one unnecessarily deep routing block."""
    allparts = [
        frozenset({0}),
        frozenset({1, 2, 3}),
        frozenset({0, 1}),
        frozenset({2, 3}),
    ]
    dependencies = {0: {1}, 1: {2}, 2: {3}, 3: set()}
    gate_to_qubit = {gate: {0, 1} for gate in dependencies}
    weights = routing_partition_weights(
        allparts, dependencies, gate_to_qubit
    )

    unbalanced_cover = weights[0] + weights[1]
    balanced_cover = weights[2] + weights[3]
    assert balanced_cover < unbalanced_cover


def test_RoutingWeightsPreferCompactGateSpans():
    """Equal-count covers should avoid interleaving distant circuit regions."""
    allparts = [
        frozenset({0, 2}),
        frozenset({1, 3}),
        frozenset({0, 1}),
        frozenset({2, 3}),
    ]
    dependencies = {gate: set() for gate in range(4)}
    gate_to_qubit = {gate: {0, 1} for gate in dependencies}
    weights = routing_partition_weights(
        allparts, dependencies, gate_to_qubit
    )

    interleaved_cover = weights[0] + weights[1]
    compact_cover = weights[2] + weights[3]
    assert compact_cover < interleaved_cover


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_PartitionEmptyCircuit(max_qubits):
    """
    Test partitioning an empty circuit
    """
    empty_c = Circuit(5)
    top_c, param_order, _ = kahn_partition(empty_c, max_qubits)
    assert len(top_c.get_Gates()) == 1  # NOTE: should be 0
    assert len(param_order) == 0


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_PartitionSingleGate(max_qubits):
    """
    Test partitioning a circuit with a single gate
    """
    single_c = Circuit(5)
    single_c.add_CNOT(0, 1)
    top_c, param_order, _ = kahn_partition(single_c, max_qubits)
    assert len(top_c.get_Gates()) == 1
    assert len(param_order) == 1


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_PartitionTotalGates(max_qubits):
    """
    Test total gates after partitioning matches original
    """
    c = Circuit(5)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CNOT(2, 3)
    top_c, _, _ = kahn_partition(c, max_qubits)
    total_gates = sum(len(p.get_Gates()) for p in top_c.get_Gates())
    assert total_gates == len(c.get_Gates())


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_PartitionMaxQubitConstraint(max_qubits):
    """
    Test that each partition respects max qubit constraint
    """
    c = Circuit(5)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CNOT(2, 3)
    top_c, _, _ = kahn_partition(c, max_qubits)
    for p in top_c.get_Gates():
        qubits = set.union(*(get_qubits(gate) for gate in p.get_Gates()))
        assert len(qubits) <= max_qubits


@pytest.mark.parametrize("max_qubits", [3, 4, 5])
def test_PartitionMaxQubitsEqualsTotalQubits(max_qubits):
    """
    Test partitioning when max qubits equals total qubits
    """
    c = Circuit(max_qubits)
    c.add_CNOT(0, 1)
    c.add_CNOT(1, 2)
    c.add_CCX(1, [2, 0])
    c.add_CSWAP([1, 2], [0])
    c.add_SWAP([1, 2])

    top_c, _, _ = kahn_partition(c, max_qubits)
    assert len(top_c.get_Gates()) == 1


def test_CorrectnessOfPartitionedCircuit():
    """
    Test correctness of partitioned circuit by comparing output states
    """
    filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"

    initial_circuit, initial_parameters, _ = utils.qasm_to_squander_circuit(filename)

    max_partition_size = 4
    partitined_circuit, partitioned_parameters, _ = PartitionCircuitQasm(
        filename, max_partition_size
    )

    # generate random initial state on which we test the circuits
    qbit_num = initial_circuit.get_Qbit_Num()

    matrix_size = 1 << qbit_num
    initial_state_real = np.random.uniform(-1.0, 1.0, (matrix_size,))
    initial_state_imag = np.random.uniform(-1.0, 1.0, (matrix_size,))
    initial_state = initial_state_real + initial_state_imag * 1j
    initial_state = initial_state / np.linalg.norm(initial_state)

    transformed_state_1 = initial_state.copy()
    transformed_state_2 = initial_state.copy()

    initial_circuit.apply_to(initial_parameters, transformed_state_1)
    partitined_circuit.apply_to(partitioned_parameters, transformed_state_2)

    diff = np.linalg.norm(transformed_state_1 - transformed_state_2)

    assert diff < 1e-10
