"""Direct native PAM conversion preserves physical angles and endian order."""
import numpy as np
import pytest
from squander import Qiskit_IO
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing
from squander.partitioning import _pam_circuit_cache as pam_cache


def _phase_aligned_error(expected, actual):
    overlap = np.vdot(expected, actual)
    aligned = actual * np.exp(-1j * np.angle(overlap))
    return np.linalg.norm(expected - aligned) ** 2 / (2 * expected.shape[0])


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("compensate", [False, True])
def test_native_pam_conversion_matches_native_matrix_without_qasm(
        width, compensate, monkeypatch):
    rng = np.random.default_rng(20261009)
    endpoints = [0.0, np.nextafter(4*np.pi, 0.0), 4*np.pi,
                 np.nextafter(4*np.pi, np.inf), -4*np.pi, np.pi/2]

    def forbidden(*args, **kwargs):
        raise AssertionError("Resolved U3/CNOT columns must not use QASM.")

    monkeypatch.setattr(Qiskit_IO, "get_Qiskit_Circuit", forbidden)
    for trial, endpoint in enumerate(endpoints):
        native = qgd_Circuit(width)
        for _ in range(1 + trial % 3):
            for qubit in range(width):
                native.add_U3(qubit)
            for qubit in range(width - 1):
                native.add_CNOT(qubit + 1, qubit)
        parameters = rng.uniform(
            -8*np.pi, 8*np.pi, native.get_Parameter_Num())
        parameters[::3] = endpoint
        unchanged_parameters = parameters.copy()
        matrix = np.asarray(native.get_Matrix(parameters))
        reversal = np.array([
            int(f"{index:0{width}b}"[::-1], 2)
            for index in range(2**width)])
        expected = (matrix if compensate else
                    matrix[np.ix_(reversal, reversal)])
        converted = routing._squander_to_bqskit(
            native, parameters, compensate_endianness=compensate)
        assert _phase_aligned_error(
            expected, np.asarray(converted.get_unitary())) < 1e-25
        assert converted.num_operations == len(native.get_Gates())
        expected_by_wire = [[] for _ in range(width)]
        actual_by_wire = [[] for _ in range(width)]
        for gate in native.get_Gates():
            def wire(qubit):
                return width - 1 - qubit if compensate else qubit
            target = wire(gate.get_Target_Qbit())
            if gate.get_Name() == "CNOT":
                location = (wire(gate.get_Control_Qbit()), target)
                angles = ()
            else:
                location = (target,)
                angles = tuple(gate.Extract_Parameters(parameters))
            for qubit in location:
                expected_by_wire[qubit].append((location, angles))
        for operation in converted:
            location = tuple(operation.location)
            angles = tuple(operation.params)
            for qubit in location:
                actual_by_wire[qubit].append((location, angles))
        # Disjoint operations can change global iteration order when packed
        # into cycles, but each wire must retain its exact causal sequence.
        assert actual_by_wire == expected_by_wire
        converted.set_params(np.asarray(converted.params) + 0.17)
        converted.renumber_qudits(tuple(reversed(range(width))))
        assert np.array_equal(parameters, unchanged_parameters)
        assert np.array_equal(np.asarray(native.get_Matrix(parameters)), matrix)


@pytest.mark.parametrize("compensate", [False, True])
def test_native_pam_conversion_retains_general_gate_fallback(
        compensate, monkeypatch):
    native = qgd_Circuit(2)
    native.add_H(0)
    native.add_CNOT(1, 0)
    parameters = np.empty(0, dtype=np.float64)
    assert pam_cache.native_pam_circuit(
        native, parameters, compensate_endianness=compensate) is None
    decoded = []
    original = pam_cache.decode_pam_qasm

    def decode(qasm):
        decoded.append(qasm)
        return original(qasm)

    monkeypatch.setattr(pam_cache, "decode_pam_qasm", decode)
    converted = routing._squander_to_bqskit(
        native, parameters, compensate_endianness=compensate)
    assert len(decoded) == 1
    matrix = np.asarray(native.get_Matrix(parameters))
    reversal = np.array([0, 2, 1, 3])
    expected = matrix if compensate else matrix[np.ix_(reversal, reversal)]
    assert _phase_aligned_error(
        expected, np.asarray(converted.get_unitary())) < 1e-25
