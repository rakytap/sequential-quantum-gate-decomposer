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
    payload = SynthesizedRoutingPayload(
        circuit=circuit,
        parameters=parameters,
        topology=topology,
        input_assignment=(0, 1),
        output_assignment=(0, 1),
        source_circuit=circuit,
        source_parameters=parameters,
    )
    alternative = RoutingAlternative(
        partition=0,
        logical_qubits=(0, 1),
        input_physical=(0, 1),
        output_physical=(0, 1),
        cnot_count=1,
        payload=payload,
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
        feasible_alternatives={0: (alternative,)},
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
