import numpy as np
import pytest
from qiskit import QuantumCircuit

from squander import Qiskit_IO
from squander.partitioning.routing import (
    RoutingAlternative,
    SynthesizedRoutingPayload,
    load_routing_osr_catalog,
    reconstruct_routing_osr_catalog,
    save_routing_osr_catalog,
)


@pytest.mark.parametrize("compact", [False, True])
def test_catalog_round_trip_with_native_sx(tmp_path, compact):
    qiskit_circuit = QuantumCircuit(2)
    qiskit_circuit.sx(0)
    qiskit_circuit.cx(0, 1)
    circuit, parameters = Qiskit_IO.convert_Qiskit_to_Squander(qiskit_circuit)
    topology = ((0, 1),)
    alternatives = []
    for input_assignment in ((0, 1), (1, 0)):
        for output_assignment in ((0, 1), (1, 0)):
            routed = QuantumCircuit(2)
            if input_assignment == (1, 0):
                routed.cx(0, 1)
                routed.cx(1, 0)
                routed.cx(0, 1)
            routed.compose(qiskit_circuit, inplace=True)
            if output_assignment == (1, 0):
                routed.cx(0, 1)
                routed.cx(1, 0)
                routed.cx(0, 1)
            routed_circuit, routed_parameters = Qiskit_IO.convert_Qiskit_to_Squander(
                routed
            )
            payload = SynthesizedRoutingPayload(
                circuit=routed_circuit,
                parameters=routed_parameters,
                topology=topology,
                input_assignment=input_assignment,
                output_assignment=output_assignment,
                source_circuit=circuit,
                source_parameters=parameters,
            )
            alternatives.append(
                RoutingAlternative(
                    partition=0,
                    logical_qubits=(0, 1),
                    input_physical=input_assignment,
                    output_physical=output_assignment,
                    cnot_count=int(routed_circuit.get_Gate_Nums().get("CNOT", 0)),
                    payload=payload,
                )
            )
    path = tmp_path / "sx.routing-catalog.json.gz"
    save_routing_osr_catalog(
        path,
        circuit=circuit,
        parameters=parameters,
        topology=topology,
        config={"strategy": "TreeSearch"},
        candidate_gate_sets=((0, 1),),
        candidate_gate_orders=((0, 1),),
        feasible_alternatives={0: tuple(alternatives)},
        osr_synthesized_partitions={0},
    )

    archive = load_routing_osr_catalog(path)
    assert "sx " in archive["source_qasm"]
    reconstructed = reconstruct_routing_osr_catalog(path, compact_path=compact)
    if compact:
        from squander.partitioning._routing_catalog_replay import expand_path_cover
        from squander.partitioning.routing import augment_catalog_with_recombined_covers
        reconstructed = augment_catalog_with_recombined_covers(reconstructed, strategies=("kahn",))
        assert reconstructed.implicit_path_frames
        reconstructed = expand_path_cover(reconstructed, (0,))
        full = reconstruct_routing_osr_catalog(path)
        def costs(catalog):
            return {(a.input_physical, a.output_physical): a.cnot_count
                    for a in catalog.alternatives[0]}
        original_costs, replay_costs = costs(full), costs(reconstructed)
        assert set(original_costs) == set(replay_costs)
        # Reflections can expose a cheaper valid realization at a boundary
        # that the old full table represented by a different local program.
        assert all(replay_costs[key] <= value for key, value in original_costs.items())
        from squander.partitioning.routing import permuted_partition_target
        for alternative in reconstructed.alternatives[0]:
            payload = alternative.payload
            target = permuted_partition_target(
                payload.source_circuit.get_Matrix(payload.source_parameters, is_f32=False),
                payload.input_assignment, payload.output_assignment)
            actual = payload.circuit.get_Matrix(payload.parameters, is_f32=False)
            overlap = np.vdot(target, actual)
            np.testing.assert_allclose(actual * np.exp(-1j * np.angle(overlap)), target, atol=1e-12)
            assert alternative.output_physical == tuple(
                alternative.input_physical[payload.input_assignment[payload.output_assignment.index(q)]]
                for q in range(2))
    assert reconstructed.circuit.get_Gate_Nums().get("SX") == 1
    assert np.allclose(
        reconstructed.circuit.get_Matrix(
            reconstructed.parameters, is_f32=False
        ),
        circuit.get_Matrix(parameters, is_f32=False),
    )
