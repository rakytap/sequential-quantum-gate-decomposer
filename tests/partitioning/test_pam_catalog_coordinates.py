"""PAM tables must obey native assignment and physical topology conventions."""
import itertools
import numpy as np
import pytest
from bqskit.passes import PAMLayoutPass, PAMRoutingPass
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing


def _permutation_matrix(assignment, *, little_endian):
    width = len(assignment)
    result = np.zeros((2**width, 2**width), dtype=np.complex128)
    for column in range(2**width):
        output = [0] * width
        for logical, physical in enumerate(assignment):
            shift = logical if little_endian else width - 1 - logical
            output[physical] = (column >> shift) & 1
        row = sum(bit << (qubit if little_endian else width - 1 - qubit)
                  for qubit, bit in enumerate(output))
        result[row, column] = 1
    return result


def _error(expected, actual):
    phase = np.exp(-1j * np.angle(np.vdot(expected, actual)))
    return np.linalg.norm(expected - phase * actual)**2 / (2 * len(expected))


def _append_swap(circuit, left, right):
    circuit.add_CNOT(right, left)
    circuit.add_CNOT(left, right)
    circuit.add_CNOT(right, left)


@pytest.mark.parametrize("center", [0, 1, 2])
@pytest.mark.parametrize("fanout", [False, True])
@pytest.mark.parametrize("duplicates", [False, "extra_cnots", "parameters"])
def test_pam_topology_frames_and_proxy_cost_match_native_payloads(
        center, fanout, duplicates, monkeypatch):
    vertices = (0, 1, 2)
    leaves = tuple(qubit for qubit in vertices if qubit != center)
    topology = tuple((center, leaf) for leaf in leaves)
    allowed = {tuple(sorted(edge)) for edge in topology}
    path = (leaves[0], center, leaves[1])
    source = qgd_Circuit(3)
    source.add_CNOT(1, 0)
    if fanout:
        source.add_CNOT(2, 0)
    parameters = np.empty(0, dtype=np.float64)
    source_matrix = np.asarray(source.get_Matrix(parameters))
    permutations = tuple(itertools.permutations(vertices))
    alternatives = []
    by_boundary = {}
    for before, after in itertools.product(permutations, repeat=2):
        input_physical = routing._inverse_permutation(before)
        output_physical = routing._inverse_permutation(after)
        control, target = input_physical[:2]
        native = qgd_Circuit(3)
        if fanout:
            if control == center:
                native.add_CNOT(input_physical[1], control)
                native.add_CNOT(input_physical[2], control)
            else:
                other_leaf = next(qubit for qubit in leaves if qubit != control)
                native.add_CNOT(other_leaf, center)
                native.add_CNOT(center, control)
                native.add_CNOT(other_leaf, center)
        elif tuple(sorted((control, target))) in allowed:
            native.add_CNOT(target, control)
        else:
            native.add_CNOT(center, control)
            native.add_CNOT(target, center)
            native.add_CNOT(center, control)
            native.add_CNOT(target, center)
        # Convert the input physical frame to the required output frame.
        token_destination = tuple(output_physical[before[q]] for q in vertices)
        for left, right in routing._path_mapping_swap_sequence(
                vertices, token_destination, path):
            _append_swap(native, left, right)
        pin = _permutation_matrix(before, little_endian=True)
        pout = _permutation_matrix(after, little_endian=True)
        assert _error(pout.conj().T @ source_matrix @ pin,
                      np.asarray(native.get_Matrix(parameters))) < 1e-25
        payload = routing.SynthesizedRoutingPayload(
            circuit=native, parameters=parameters, topology=topology,
            input_assignment=before, output_assignment=after,
            source_circuit=source, source_parameters=parameters)
        alternative = routing.RoutingAlternative(
            partition=0, logical_qubits=vertices,
            input_physical=input_physical, output_physical=output_physical,
            cnot_count=len(native.get_Gates()), payload=payload)
        alternatives.append(alternative)
        by_boundary[(input_physical, output_physical)] = alternative
        if duplicates:
            if duplicates == "extra_cnots":
                worse = native.get_Flat_Circuit()
                worse.add_CNOT(leaves[0], center)
                worse.add_CNOT(leaves[0], center)
                duplicate_parameters = parameters
                duplicate_count = alternative.cnot_count + 2
            else:
                # Same native structure, distinct parameter ownership.
                # Equal-cost columns retain the first exact program on a tie.
                worse = native
                duplicate_parameters = parameters.copy()
                duplicate_count = alternative.cnot_count
            worse_payload = routing.SynthesizedRoutingPayload(
                circuit=worse, parameters=duplicate_parameters, topology=topology,
                input_assignment=before, output_assignment=after,
                source_circuit=source, source_parameters=parameters)
            alternatives.append(routing.RoutingAlternative(
                partition=0, logical_qubits=vertices,
                input_physical=input_physical, output_physical=output_physical,
                cnot_count=duplicate_count, payload=worse_payload))

    observed = dict(entries=0, proxy_choices=0, proper_subgraphs=0)
    checked = set()
    forward = PAMLayoutPass.forward_pass
    choose = PAMRoutingPass._get_best_perm

    def checked_forward(self, circuit, mapping, graph, permutation_data):
        for point, table in permutation_data.items():
            if id(table) in checked:
                continue
            checked.add(id(table))
            original = np.asarray(circuit[point].get_unitary())
            for declared, values in table.items():
                declared_edges = {tuple(sorted(edge)) for edge in declared}
                assert len(declared_edges) == 2
                for (before, after), column in values.items():
                    used = {tuple(sorted(edge)) for edge in column.coupling_graph}
                    assert used <= declared_edges
                    observed["proper_subgraphs"] += used < declared_edges
                    pin = _permutation_matrix(before, little_endian=False)
                    pout = _permutation_matrix(after, little_endian=False)
                    assert _error(pout @ original @ pin.conj().T,
                                  np.asarray(column.get_unitary())) < 1e-25
                    observed["entries"] += 1
        return forward(self, circuit, mapping, graph, permutation_data)

    def checked_choice(self, circuit, block_data, graph, frontier, mapping,
                       distances, extended, qudits):
        result = choose(self, circuit, block_data, graph, frontier, mapping,
                        distances, extended, qudits)
        before = list(mapping)
        input_physical = tuple(before[q] for q in qudits)
        self._apply_perm(result[0], before)
        self._apply_perm(result[2], before)
        output_physical = tuple(before[q] for q in qudits)
        chosen = by_boundary[(input_physical, output_physical)]
        proxy_cnots = sum(count for gate, count in result[1].gate_counts.items()
                          if gate.num_qudits >= 2)
        assert proxy_cnots == chosen.cnot_count
        observed["proxy_choices"] += 1
        return result

    monkeypatch.setattr(PAMLayoutPass, "forward_pass", checked_forward)
    monkeypatch.setattr(PAMRoutingPass, "_get_best_perm", checked_choice)
    count = len(source.get_Gates())
    native_seed = routing._beam_selected_path_warm_start(
        gate_predecessors={gate: (() if gate == 0 else (gate - 1,))
                           for gate in range(count)},
        partitions=(frozenset(range(count)),),
        alternatives={0: tuple(alternatives)}, logical_qubit_count=3,
        path=path, initial_mapping=None, beam_width=1)
    assert native_seed.cnot_count == (2 if fanout else 1)
    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0,),
        gate_predecessors={gate: (() if gate == 0 else (gate - 1,))
                           for gate in range(count)},
        partitions=(frozenset(range(count)),),
        alternatives={0: tuple(alternatives)}, logical_qubit_count=3,
        topology=topology, path=path, layout_passes=3,
        initial_mapping=native_seed.initial_mapping)
    assert result is not None
    selected = result.selections[0].alternative
    assert selected is by_boundary[(selected.input_physical, selected.output_physical)]
    assert result.cnot_count == (2 if fanout else 1)
    assert observed["entries"] > 0 and observed["proxy_choices"] > 0
    if not fanout:
        assert observed["proper_subgraphs"] > 0
    routed, routed_parameters = routing.construct_exact_routed_circuit(result, 3)
    initial = _permutation_matrix(result.initial_mapping, little_endian=True)
    final = _permutation_matrix(result.final_mapping, little_endian=True)
    assert _error(final @ source_matrix @ initial.conj().T,
                  np.asarray(routed.get_Matrix(routed_parameters))) < 1e-25
    assert {tuple(sorted((gate.get_Control_Qbit(), gate.get_Target_Qbit())))
            for gate in routed.get_Gates()} <= allowed
