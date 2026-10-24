"""Exact native audit serialization, including rotation-period endpoints."""
import importlib
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from squander.decomposition._audit_qasm import dump_native_audit_qasm

@pytest.mark.parametrize("angle", [
    np.nextafter(4*np.pi, 0), 4*np.pi, np.nextafter(4*np.pi, np.inf),
    np.nextafter(2*np.pi, 0), 2*np.pi, -2*np.pi, 1e-18, -1e-18, 0.7,
])
def test_native_audit_qasm_preserves_parameter_bits(angle):
    circuit = QuantumCircuit(2)
    circuit.u(angle, -np.pi, 0.19, 0)
    circuit.cx(0, 1)
    circuit.rz(-angle, 1)
    restored = QuantumCircuit.from_qasm_str(dump_native_audit_qasm(circuit))
    assert [list(item.operation.params) for item in restored.data] == [
        list(item.operation.params) for item in circuit.data]
    assert np.array_equal(Operator(restored).data, Operator(circuit).data)

def test_native_basis_audit_replays_angle_just_below_full_period():
    from squander.gates.qgd_Circuit import qgd_Circuit
    from squander.utils import circuit_to_CNOT_basis
    backend = importlib.import_module(
        "squander.decomposition.qgd_Wide_Circuit_Optimization")
    circuit = qgd_Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    parameters = np.array([np.nextafter(2*np.pi, 0), 0.3, 0.7])
    converted, converted_parameters = circuit_to_CNOT_basis(circuit, parameters)
    before = backend._squander_audit_representation(circuit, parameters)
    after = backend._squander_audit_representation(converted, converted_parameters)
    event = dict(input=before, output=after, conversion_applied=True,
                 input_sha256=backend._exact_state_sha256(backend._qasm_exact_state(before)),
                 output_sha256=backend._exact_state_sha256(backend._qasm_exact_state(after)))
    backend._verify_squander_basis_replay(event)

def test_native_audit_qasm_rejects_classical_circuits():
    with pytest.raises(ValueError, match="no classical bits"):
        dump_native_audit_qasm(QuantumCircuit(1, 1))


def test_composition_basis_identity_preserves_near_period_angles():
    from squander.decomposition._audit_composition import NativeAuditComposition
    from squander.gates.qgd_Circuit import qgd_Circuit
    from squander.utils import circuit_to_CNOT_basis
    backend = importlib.import_module(
        "squander.decomposition.qgd_Wide_Circuit_Optimization")
    circuit = qgd_Circuit(2)
    circuit.add_U3(0)
    circuit.add_SWAP(0, 1)
    parameters = np.array([np.nextafter(2*np.pi, 0), 0.3, 0.7])
    converted, converted_parameters = circuit_to_CNOT_basis(circuit, parameters)
    before = backend._squander_audit_representation(circuit, parameters)
    after = backend._squander_audit_representation(converted, converted_parameters)
    event = dict(kind="squander_basis_conversion", event_id="near-period-swap",
                 input=before, output=after, conversion_applied=True,
                 input_sha256=backend._exact_state_sha256(backend._qasm_exact_state(before)),
                 output_sha256=backend._exact_state_sha256(backend._qasm_exact_state(after)))
    with NativeAuditComposition(backend, before, 1e-8, 1e-10) as certificate:
        certificate.consume(event)
        certificate.require_current(after)
        assert certificate.report()["global_process_infidelity_bound"] <= 1e-8
