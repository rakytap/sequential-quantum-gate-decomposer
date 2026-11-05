import numpy as np
from qiskit import QuantumCircuit

from squander import Qiskit_IO
from squander.partitioning.routing import (
    RoutingAlternative,
    SynthesizedRoutingPayload,
    load_routing_osr_catalog,
    reconstruct_routing_osr_catalog,
    save_routing_osr_catalog,
)


def test_catalog_round_trip_with_native_sx(tmp_path):
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
    reconstructed = reconstruct_routing_osr_catalog(path)
    assert reconstructed.circuit.get_Gate_Nums().get("SX") == 1
    assert np.allclose(
        reconstructed.circuit.get_Matrix(
            reconstructed.parameters, is_f32=False
        ),
        circuit.get_Matrix(parameters, is_f32=False),
    )
