"""The optional fractional continuation must preserve easy native OSR solves."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    qgd_Wide_Circuit_Optimization as WCO,
)
from squander.IO_interfaces import Qiskit_IO


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("deferred", [False, True])
def test_deferred_profile_preserves_certified_shallow_descent(width, deferred):
    graph = [(qubit, qubit + 1) for qubit in range(width - 1)]
    source = QuantumCircuit(width)
    for qubit in range(width):
        source.u(0.13 * (qubit + 1), -0.17, 0.23, qubit)
    for index, (control, target_qubit) in enumerate(graph):
        source.cx(control, target_qubit)
        source.ry(0.31 * (index + 1), target_qubit)
        source.rz(-0.19, control)
    for qubit in range(width):
        source.u(-0.21, 0.07 * (qubit + 1), -0.11, qubit)
    target = Operator(source).data
    optimizer = WCO({
        "strategy": "TreeSearch", "topology": None,
        "use_osr": True, "use_graph_search": True,
        "max_partition_size": width,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0, "verbosity": 0, "random_seed": 20260926,
        "osr_deferred_profile_plateau_descent": deferred,
    })
    solutions = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    certified_counts = []
    for circuit, parameters in solutions:
        parameters = np.asarray(parameters, dtype=np.float64)
        native = circuit.get_Matrix(parameters)
        exported_circuit = Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)
        exported = Operator(exported_circuit).data
        for left, right in ((target, native), (target, exported),
                            (native, exported)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 4 ** width)
            assert error <= 1e-10
        assert exported_circuit.num_qubits == width
        for instruction in exported_circuit.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported_circuit.find_bit(qubit).index
                    for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}
        certified_counts.append(int(exported_circuit.count_ops().get("cx", 0)))
    assert certified_counts
    assert min(certified_counts) == width - 1


def test_deferred_profile_does_not_preempt_the_three_cnot_class():
    # A matrix-only regression from a local 3q target. The input word is not
    # supplied to the native solver. Committing after the 2-CNOT class gave
    # 5 CNOTs; the original 3-CNOT class constructs a verified 4-CNOT result.
    source = QuantumCircuit.from_qasm_str('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[1],q[0];
u(-0.9007488803034612,-2.3561947376737606,-1.5707962749412403) q[0];
u(-3.14159247037986,-0.14767768562063477,1.3278711996605477) q[1];
cx q[1],q[0];
u(3.1415920641200477,1.3681898770405632,1.2277251015100736) q[0];
u(1.5707952868208228,3.1415920732449276,3.328207887683468) q[1];
u(-6.092344229968252e-07,-2.291564525338528,-0.8982017580621795) q[1];
cx q[1],q[2];
u(-5.434481631987117e-07,5.896662845096309,3.5762871911436895) q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
''')
    target = Operator(source).data
    graph = [(0, 1), (1, 2)]
    optimizer = WCO({
        "strategy": "TreeSearch", "topology": None,
        "use_osr": True, "use_graph_search": True,
        "max_partition_size": 3,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0, "verbosity": 0, "random_seed": 20260926,
        "osr_deferred_profile_plateau_descent": True,
    })
    solutions = optimizer.DecomposePartition(
        target, optimizer.config, mini_topology=graph,
    )
    certified_counts = []
    for circuit, parameters in solutions:
        parameters = np.asarray(parameters, dtype=np.float64)
        native = circuit.get_Matrix(parameters)
        exported_circuit = Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)
        exported = Operator(exported_circuit).data
        for left, right in ((target, native), (target, exported),
                            (native, exported)):
            error = max(0.0, 1.0 - abs(np.vdot(left, right)) ** 2 / 64)
            assert error <= 1e-10
        assert exported_circuit.num_qubits == 3
        for instruction in exported_circuit.data:
            if instruction.operation.name == "cx":
                support = frozenset(
                    exported_circuit.find_bit(qubit).index
                    for qubit in instruction.qubits
                )
                assert support in {frozenset(edge) for edge in graph}
        certified_counts.append(int(exported_circuit.count_ops().get("cx", 0)))
    assert certified_counts
    assert min(certified_counts) <= 4
