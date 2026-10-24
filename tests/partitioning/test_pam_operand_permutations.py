"""PAM operand ordering is not a physical movement of logical states."""
import numpy as np
import pytest
from bqskit.passes import PAMRoutingPass
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing


@pytest.mark.parametrize("local_pre", [(1, 2, 0), (2, 0, 1)])
def test_pam_nonidentity_operands_do_not_insert_physical_swaps(
        local_pre, monkeypatch):
    circuit = qgd_Circuit(3)
    circuit.add_CNOT(2, 0)
    parameters = np.empty(0, dtype=np.float64)
    vertices = (0, 1, 2)
    topology = ((0, 2), (1, 2))
    payload = routing.SynthesizedRoutingPayload(
        circuit=circuit, parameters=parameters, topology=topology,
        input_assignment=vertices, output_assignment=vertices,
        source_circuit=circuit, source_parameters=parameters)
    alternative = routing.RoutingAlternative(
        partition=0, logical_qubits=vertices, input_physical=vertices,
        output_physical=vertices, cnot_count=1, payload=payload)
    choose = PAMRoutingPass._get_best_perm
    observations = []

    def choose_equivalent_nonidentity(self, source, table, graph, frontier,
                                     mapping, distances, extended, qudits):
        default = choose(self, source, table, graph, frontier, mapping,
                         distances, extended, qudits)
        inverse = routing._inverse_permutation(local_pre)
        physical = [mapping[qudits[index]] for index in inverse]
        induced = graph.get_subgraph(physical)
        column = table[induced][(local_pre, local_pre)]
        # Force an equally cheap valid representative, not a new search rule.
        assert column.gate_counts == default[1].gate_counts
        pre = tuple(qudits[index] for index in inverse)
        post = tuple(qudits[index] for index in local_pre)
        observations.append((pre, post))
        return pre, column, post

    monkeypatch.setattr(PAMRoutingPass, "_get_best_perm",
                        choose_equivalent_nonidentity)
    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0,), gate_predecessors={0: ()},
        partitions=(frozenset((0,)),), alternatives={0: (alternative,)},
        logical_qubit_count=3, topology=topology, path=(0, 2, 1),
        layout_passes=1)
    assert observations and all(pre != vertices for pre, _ in observations)
    assert result is not None
    assert result.initial_mapping == result.final_mapping == vertices
    assert result.cnot_count == 1 and result.transition_swaps == ((),)
    assert result.selections[0].alternative is alternative
    fitted, fitted_parameters = routing.construct_exact_routed_circuit(result, 3)
    expected = np.asarray(circuit.get_Matrix(parameters))
    actual = np.asarray(fitted.get_Matrix(fitted_parameters))
    phase = np.exp(-1j * np.angle(np.vdot(expected, actual)))
    assert np.linalg.norm(expected - phase * actual)**2 / 16 < 1e-25
