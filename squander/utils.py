## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

## \file utils.py
## \brief Generic utility functionalities for SQUANDER


import numpy as np
from squander.IO_interfaces import Qiskit_IO
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from qiskit import QuantumCircuit

import qiskit

qiskit_version = qiskit.version.get_version_info()

if qiskit_version[0] == "0":
    from qiskit import Aer
    from qiskit import execute

    if int(qiskit_version[2]) > 3:
        from qiskit.quantum_info import Operator
else:
    import qiskit_aer as Aer
    from qiskit import transpile
    from qiskit.quantum_info import Operator


##
# @brief Call to retrieve the unitary from QISKIT circuit
def get_unitary_from_qiskit_circuit(circuit: QuantumCircuit):
    """
    Call to extract a unitary from Qiskit circuit

    Args:

        circuit (QuantumCircuit) A Qiskit circuit

    Return:

        Returns with the generated unitary

    """

    if qiskit_version[0] == "0":
        backend = Aer.get_backend("aer_simulator")
        circuit.save_unitary()

        # job execution and getting the result as an object
        job = execute(circuit, backend)

        # the result of the Qiskit job
        result = job.result()

    else:

        circuit.save_unitary()
        backend = Aer.AerSimulator(method="unitary")

        compiled_circuit = transpile(circuit, backend)
        result = backend.run(compiled_circuit).result()

    return np.asarray(result.get_unitary(circuit))


def get_unitary_from_qiskit_circuit_operator(circuit: QuantumCircuit):
    """
    Call to extract a unitary from Qiskit circuit using qiskit.quantum_info.Operator support

    Args:

        circuit (QuantumCircuit) A Qiskit circuit

    Return:

        Returns with the generated unitary

    """

    if qiskit_version[0] == "0" and int(qiskit_version[2]) < 4:

        print(
            "Currently installed version of qiskit does not support extracting the unitary of a circuit via Operator. Using get_unitary_from_qiskit_circuit function instead."
        )
        return get_unitary_from_qiskit_circuit(circuit)

    return Operator(circuit).to_matrix()


def qasm_to_squander_circuit(filename: str, return_transpiled=False):
    """
    Converts a QASM file to a SQUANDER circuit

    Args:

        filename (str) The path to the QASM file

    Return:

        Returns with the SQUANDER circuit, the array of the corresponding parameters, and the transpiled Qiskit circuit
        (None if not transpiled)
    """

    qc = qiskit.QuantumCircuit.from_qasm_file(filename)
    from squander.gates import gates_Wrapper as gate

    SUPPORTED_GATES_NAMES = {
        n.lower().replace("cnot", "cx")
        for n in dir(gate)
        if not n.startswith("_")
        and issubclass(getattr(gate, n), gate.Gate)
        and n not in ("Gate", "CROT", "CR", "SYC","Permutation")
    }
    if any(gate.operation.name not in SUPPORTED_GATES_NAMES for gate in qc.data):
        qc_transpiled = qiskit.transpile(
            qc, basis_gates=SUPPORTED_GATES_NAMES, optimization_level=0
        )
    else:
        qc_transpiled = qc

    circuit_squander, circut_parameters = Qiskit_IO.convert_Qiskit_to_Squander(
        qc_transpiled
    )

    if return_transpiled:
        return circuit_squander, circut_parameters, qc_transpiled

    return circuit_squander, circut_parameters, None


def CompareCircuits(
    circ1: Circuit,
    parameters1: np.ndarray,
    circ2: Circuit,
    parameters2: np.ndarray,
    parallel: int = 1,
    tolerance: float = 1e-5,
    initial_mapping=None,
    final_mapping=None,
):
    """
    Call to test if the two circuits give the same state transformation upon a random input state


    Args:

        circ1 ( Circuit ) A circuit

        parameters1 ( np.ndarray ) A parameter array associated with the input circuit

        circ2 ( Circuit ) A circuit

        parameters2 ( np.ndarray ) A parameter array associated with the input circuit

        parallel (int, optional) Set 0 for sequential evaluation, 1 for using TBB parallelism or 2 for using openMP

        tolerance ( float, optional) The tolerance of the comparision when the inner product of the resulting states is matched to unity.


    Return:

        Returns with True if the two circuits give identical results.
    """

    qbit_num1 = circ1.get_Qbit_Num()
    qbit_num2 = circ2.get_Qbit_Num()

    if qbit_num1 != qbit_num2:
        raise Exception(
            "The two compared circuits should have the same number of qubits."
        )

    if qbit_num1 > 31:
        return  # skip comparison for large qubit numbers, as the current implementation of Gates_block only supports up to 31 qubits. This is a temporary workaround and should be removed once the support for more qubits is implemented in Gates_block.

    matrix_size = 1 << qbit_num1
    initial_state_real = np.random.uniform(-1.0, 1.0, (matrix_size,))
    initial_state_imag = np.random.uniform(-1.0, 1.0, (matrix_size,))
    initial_state = initial_state_real + initial_state_imag * 1j
    norm = np.sum(
        initial_state_real * initial_state_real
        + initial_state_imag * initial_state_imag
    )
    initial_state = initial_state / np.sqrt(norm)

    transformed_state_1 = initial_state.copy()
    transformed_state_2 = initial_state

    circ1.apply_to(parameters1, transformed_state_1, parallel=parallel)
    if initial_mapping is not None:
        from squander.synthesis.qgd_SABRE import qgd_SABRE

        tensor_perm = [
            qbit_num2 - 1 - p
            for p in reversed(qgd_SABRE.get_inverse_pi(initial_mapping))
        ]
        transformed_state_2 = (
            transformed_state_2.reshape([2] * qbit_num2)
            .transpose(tensor_perm)
            .copy()
            .reshape((matrix_size,))
        )
    circ2.apply_to(parameters2, transformed_state_2, parallel=parallel)
    if final_mapping is not None:
        tensor_perm = [qbit_num2 - 1 - p for p in reversed(final_mapping)]
        transformed_state_2 = (
            transformed_state_2.reshape([2] * qbit_num2)
            .transpose(tensor_perm)
            .copy()
            .reshape((matrix_size,))
        )

    overlap = np.sum(transformed_state_1.conj() * transformed_state_2)
    print("Circuit overlap: ", np.abs(overlap))

    assert (1 - np.abs(overlap)) < tolerance, 1 - np.abs(overlap)


def invert_circuit(circ: Circuit, parameters: np.ndarray):
    """
    Return the inverse (adjoint) of a SQUANDER circuit as a new SQUANDER circuit.

    All gate types are handled natively without going through Qiskit, including
    CR, CROT, and SYC which are unsupported by Qiskit export.

    SYC has no parametric inverse in native form; it is first decomposed into the
    CNOT basis and the resulting primitive circuit is then inverted.

    Args:

        circ   ( Circuit )     A SQUANDER circuit to invert.
        parameters ( np.ndarray ) Parameter array associated with the circuit.

    Return:

        Returns with a tuple (inv_circuit, inv_parameters) representing the
        adjoint circuit and its parameter array.
    """
    from squander.gates.gates_Wrapper import (
        H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg,
        CH, CZ, CNOT, SWAP, CCX, CSWAP,
        R, RX, RY, RZ, U1, U2, U3,
        CRY, CRZ, CRX, CP, CU,
        RXX, RYY, RZZ,
        CR, CROT, SYC,
    )

    inv_circuit = Circuit(circ.get_Qbit_Num())
    inv_params = []

    gates = circ.get_Gates()

    for gate in reversed(gates):
        gate_params = parameters[
            gate.get_Parameter_Start_Index():
            gate.get_Parameter_Start_Index() + gate.get_Parameter_Num()
        ]

        # ------------------------------------------------------------------ #
        # Sub-circuit: recurse and inline                                      #
        # ------------------------------------------------------------------ #
        if isinstance(gate, Circuit):
            sub_inv, sub_inv_params = invert_circuit(gate, gate_params)
            for sub_gate in sub_inv.get_Gates():
                sub_g_params = sub_inv_params[
                    sub_gate.get_Parameter_Start_Index():
                    sub_gate.get_Parameter_Start_Index() + sub_gate.get_Parameter_Num()
                ]
                inv_circuit.add_Gate(sub_gate)
                if sub_gate.get_Parameter_Num() > 0:
                    inv_params.append(sub_g_params)

        # ------------------------------------------------------------------ #
        # Self-inverse gates (no parameter change needed)                     #
        # ------------------------------------------------------------------ #
        elif isinstance(gate, CNOT):
            inv_circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
        elif isinstance(gate, CZ):
            inv_circuit.add_CZ(gate.get_Target_Qbit(), gate.get_Control_Qbit())
        elif isinstance(gate, CH):
            inv_circuit.add_CH(gate.get_Target_Qbit(), gate.get_Control_Qbit())
        elif isinstance(gate, H):
            inv_circuit.add_H(gate.get_Target_Qbit())
        elif isinstance(gate, X):
            inv_circuit.add_X(gate.get_Target_Qbit())
        elif isinstance(gate, Y):
            inv_circuit.add_Y(gate.get_Target_Qbit())
        elif isinstance(gate, Z):
            inv_circuit.add_Z(gate.get_Target_Qbit())
        elif isinstance(gate, SWAP):
            inv_circuit.add_SWAP(gate.get_Target_Qbits())
        elif isinstance(gate, CCX):
            inv_circuit.add_CCX(gate.get_Target_Qbit(), gate.get_Control_Qbits())
        elif isinstance(gate, CSWAP):
            inv_circuit.add_CSWAP(gate.get_Target_Qbits(), gate.get_Control_Qbits())

        # ------------------------------------------------------------------ #
        # Gates whose inverse swaps S↔Sdg, T↔Tdg, SX↔SXdg                   #
        # ------------------------------------------------------------------ #
        elif isinstance(gate, S):
            inv_circuit.add_Sdg(gate.get_Target_Qbit())
        elif isinstance(gate, Sdg):
            inv_circuit.add_S(gate.get_Target_Qbit())
        elif isinstance(gate, T):
            inv_circuit.add_Tdg(gate.get_Target_Qbit())
        elif isinstance(gate, Tdg):
            inv_circuit.add_T(gate.get_Target_Qbit())
        elif isinstance(gate, SX):
            inv_circuit.add_SXdg(gate.get_Target_Qbit())
        elif isinstance(gate, SXdg):
            inv_circuit.add_SX(gate.get_Target_Qbit())

        # ------------------------------------------------------------------ #
        # Rotation gates: negate the rotation angle (first parameter)        #
        # ------------------------------------------------------------------ #
        elif isinstance(gate, R):
            inv_circuit.add_R(gate.get_Target_Qbit())
            inv_params.append(np.array([-gate_params[0], gate_params[1]], dtype=gate_params.dtype))
        elif isinstance(gate, RX):
            inv_circuit.add_RX(gate.get_Target_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, RY):
            inv_circuit.add_RY(gate.get_Target_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, RZ):
            inv_circuit.add_RZ(gate.get_Target_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, U1):
            inv_circuit.add_U1(gate.get_Target_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, U2):
            # U2(φ,λ)† = U2(-λ-π, -φ+π) (up to global phase U2 inverse is U(-π/2,-λ,-φ))
            inv_circuit.add_U2(gate.get_Target_Qbit())
            inv_params.append(np.array([-gate_params[1] - np.pi, -gate_params[0] + np.pi], dtype=gate_params.dtype))
        elif isinstance(gate, U3):
            # U3(θ,φ,λ)† = U3(-θ,-λ,-φ)
            inv_circuit.add_U3(gate.get_Target_Qbit())
            inv_params.append(np.array([-gate_params[0], -gate_params[2], -gate_params[1]], dtype=gate_params.dtype))
        elif isinstance(gate, CU):
            # CU(θ,φ,λ,γ)† = CU(-θ,-λ,-φ,-γ)
            inv_circuit.add_CU(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(np.array([-gate_params[0], -gate_params[2], -gate_params[1], -gate_params[3]], dtype=gate_params.dtype))
        elif isinstance(gate, CRY):
            inv_circuit.add_CRY(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, CRZ):
            inv_circuit.add_CRZ(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, CRX):
            inv_circuit.add_CRX(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, CP):
            inv_circuit.add_CP(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(-gate_params)
        elif isinstance(gate, RXX):
            inv_circuit.add_RXX(gate.get_Target_Qbits())
            inv_params.append(-gate_params)
        elif isinstance(gate, RYY):
            inv_circuit.add_RYY(gate.get_Target_Qbits())
            inv_params.append(-gate_params)
        elif isinstance(gate, RZZ):
            inv_circuit.add_RZZ(gate.get_Target_Qbits())
            inv_params.append(-gate_params)

        # ------------------------------------------------------------------ #
        # CR and CROT: inverse is the same gate with negated theta             #
        # CR†(θ,φ)   = CR(-θ,φ)                                              #
        # CROT†(θ,φ) = CROT(-θ,φ)                                            #
        # ------------------------------------------------------------------ #
        elif isinstance(gate, CR):
            inv_circuit.add_CR(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(np.array([-gate_params[0], gate_params[1]], dtype=gate_params.dtype))
        elif isinstance(gate, CROT):
            inv_circuit.add_CROT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            inv_params.append(np.array([-gate_params[0], gate_params[1]], dtype=gate_params.dtype))

        # ------------------------------------------------------------------ #
        # SYC: no parametric form for SYC†; decompose into CNOT basis then   #
        # invert the primitive decomposition, inlining the resulting gates    #
        # ------------------------------------------------------------------ #
        elif isinstance(gate, SYC):
            syc_single = Circuit(circ.get_Qbit_Num())
            syc_single.add_SYC(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            syc_cnot, syc_cnot_params = circuit_to_CNOT_basis(syc_single, np.array([]))
            syc_inv, syc_inv_params = invert_circuit(syc_cnot, syc_cnot_params)
            # Inline all gates from the inverted decomposition
            for sub_gate in syc_inv.get_Gates():
                sub_params = syc_inv_params[
                    sub_gate.get_Parameter_Start_Index():
                    sub_gate.get_Parameter_Start_Index() + sub_gate.get_Parameter_Num()
                ]
                inv_circuit.add_Gate(sub_gate)
                if sub_gate.get_Parameter_Num() > 0:
                    inv_params.append(sub_params)

        else:
            raise ValueError(f"invert_circuit: unsupported gate type {type(gate).__name__}")

    if inv_params:
        inv_parameters = np.concatenate([
            p if isinstance(p, np.ndarray) else np.asarray(p) for p in inv_params
        ])
    else:
        inv_parameters = np.array([], dtype=parameters.dtype if len(parameters) > 0 else np.float64)

    return inv_circuit, inv_parameters


def circuit_to_CNOT_basis(circ: Circuit, parameters: np.ndarray):
    """
    Call to transpile a SQUANDER circuit to CNOT basis


    Args:

        circ ( Circuit ) A circuit

        parameters ( np.ndarray ) A parameter array associated with the input circuit


    Return:

        Returns with the transpiled circuit and the associated parameters
    """
    from squander.gates.gates_Wrapper import (
        CH,
        CZ,
        SYC,
        CRY,
        CU,
        CR,
        CROT,
        CCX,
        CSWAP,
        SWAP,
        CRX,
        CRZ,
        CP,
        RXX,
        RYY,
        RZZ,
    )

    gates = circ.get_Gates()
    circuit = Circuit(circ.get_Qbit_Num())
    params = []
    for gate in gates:
        if isinstance(gate, Circuit):
            subcircuit, subparams = circuit_to_CNOT_basis(
                gate,
                parameters[
                    gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                    + gate.get_Parameter_Num()
                ],
            )
            circuit.add_Gate(subcircuit)
            params.append(subparams)
        elif isinstance(gate, CH):
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            params.append([np.pi / 4 / 2, -np.pi / 4 / 2])
        elif isinstance(gate, CZ):
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            params.append([])
        elif isinstance(gate, SYC):
            circuit.add_U1(gate.get_Target_Qbit())
            circuit.add_U1(gate.get_Control_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_U1(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Control_Qbit(), gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            params.append([-np.pi / 12, -np.pi / 12, -5 * np.pi / 12])
        elif isinstance(gate, CRY):
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            (theta,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([-theta / 2, theta / 2])
        elif isinstance(gate, CU):
            circuit.add_U1(gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            theta, phi, lbda, gamma = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append(
                [
                    (lbda + phi) / 2 + gamma,
                    lbda / 2,
                    theta / 2,
                    -theta / 2,
                    -(phi + lbda) / 2 / 2,
                    (phi - lbda) / 2 / 2,
                ]
            )
        elif isinstance(gate, CR):
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            theta, phi = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append(
                [(-phi + np.pi / 2) / 2, -theta / 2, theta / 2, (phi - np.pi / 2) / 2]
            )
        elif isinstance(gate, CROT):
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RY(gate.get_Target_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            theta, phi = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([-phi / 2, np.pi / 2 / 2, -theta, -np.pi / 2 / 2, phi / 2])
        elif isinstance(gate, CRX):
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_H(gate.get_Target_Qbit())
            (theta,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([-theta / 2, theta / 2])
        elif isinstance(gate, CRZ):
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_RZ(gate.get_Target_Qbit())
            (theta,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([-theta / 2, theta / 2])
        elif isinstance(gate, CP):
            circuit.add_U1(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_U1(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), gate.get_Control_Qbit())
            circuit.add_U1(gate.get_Control_Qbit())
            (phi,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([phi / 2, -phi / 2, phi / 2])
        elif isinstance(gate, CCX):
            c1, c2 = gate.get_Control_Qbits()
            circuit.add_CNOT(c1, c2)
            circuit.add_Tdg(c1)
            circuit.add_T(c2)
            circuit.add_CNOT(c1, c2)
            circuit.add_H(gate.get_Target_Qbit())
            circuit.add_T(gate.get_Target_Qbit())
            circuit.add_T(c1)
            circuit.add_CNOT(gate.get_Target_Qbit(), c2)
            circuit.add_Tdg(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), c1)
            circuit.add_T(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), c2)
            circuit.add_Tdg(gate.get_Target_Qbit())
            circuit.add_CNOT(gate.get_Target_Qbit(), c1)
            circuit.add_H(gate.get_Target_Qbit())
            params.append([])
        elif isinstance(gate, CSWAP):
            t1, t2 = gate.get_Target_Qbits()
            (c,) = gate.get_Control_Qbits()
            circuit.add_CNOT(t2, t1)
            circuit.add_CNOT(t2, c)
            circuit.add_H(t1)
            circuit.add_Tdg(t2)
            circuit.add_T(c)
            circuit.add_T(t1)
            circuit.add_CNOT(t2, c)
            circuit.add_CNOT(t1, c)
            circuit.add_Tdg(t1)
            circuit.add_T(t2)
            circuit.add_CNOT(t1, t2)
            circuit.add_T(t1)
            circuit.add_CNOT(t1, c)
            circuit.add_T(t1)
            circuit.add_SX(t1)
            circuit.add_Sdg(t2)
            circuit.add_CNOT(t2, t1)
            circuit.add_S(t2)
            """
            circuit.add_CNOT(t2, t1)
            circuit.add_CNOT(t2, c)
            circuit.add_Tdg(t2)
            circuit.add_T(c)
            circuit.add_CNOT(t2, c)
            circuit.add_H(t1)
            circuit.add_T(t1)
            circuit.add_T(t2)
            circuit.add_CNOT(t1, c)
            circuit.add_Tdg(t1)
            circuit.add_CNOT(t1, t2)
            circuit.add_T(t1)
            circuit.add_CNOT(t1, c)
            circuit.add_Tdg(t1)
            circuit.add_CNOT(t1, t2)
            circuit.add_H(t1)
            circuit.add_CNOT(t2, t1)
            """
            params.append([])
        elif isinstance(gate, SWAP):
            t1, t2 = gate.get_Target_Qbits()
            circuit.add_CNOT(t1, t2)
            circuit.add_CNOT(t2, t1)
            circuit.add_CNOT(t1, t2)
            params.append([])
        elif isinstance(gate, RXX):
            t1, t2 = gate.get_Target_Qbits()
            circuit.add_CNOT(t1, t2)
            circuit.add_RX(t2)
            circuit.add_CNOT(t1, t2)
            (theta,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([theta])
        elif isinstance(gate, RYY):
            t1, t2 = gate.get_Target_Qbits()
            circuit.add_RX(t1)
            circuit.add_RX(t2)
            circuit.add_CNOT(t1, t2)
            circuit.add_RZ(t1)
            circuit.add_CNOT(t1, t2)
            circuit.add_RX(t1)
            circuit.add_RX(t2)
            (theta,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([np.pi/2/2, np.pi/2/2, theta, -np.pi/2/2, -np.pi/2/2])
        elif isinstance(gate, RZZ):
            t1, t2 = gate.get_Target_Qbits()
            circuit.add_CNOT(t1, t2)
            circuit.add_RZ(t1)
            circuit.add_CNOT(t1, t2)
            (theta,) = parameters[
                gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                + gate.get_Parameter_Num()
            ]
            params.append([theta])
        else:
            circuit.add_Gate(gate)
            params.append(
                parameters[
                    gate.get_Parameter_Start_Index() : gate.get_Parameter_Start_Index()
                    + gate.get_Parameter_Num()
                ]
            )

    return circuit, np.concatenate(params)


def test_circuit_to_CNOT_basis():
    circ1 = Circuit(2)
    circ1.add_CH(0, 1)
    circ1.add_CZ(0, 1)
    circ1.add_CRY(0, 1)
    circ1.add_SYC(0, 1)
    circ1.add_CR(0, 1)
    circ1.add_CROT(0, 1)
    circ1.add_CU(0, 1)
    circ1.add_CP(0, 1)
    circ1.add_CRX(0, 1)
    circ1.add_CRZ(0, 1)
    circ1.add_SWAP([0, 1])
    circ1.add_RXX(0, 1)
    circ1.add_RYY(0, 1)
    circ1.add_RZZ(0, 1)

    paramcount1 = 0 + 0 + 1 + 0 + 2 + 2 + 4 + 1 + 1 + 1 + 0 + 1 + 1 + 1
    circ2 = Circuit(3)
    circ2.add_CCX(0, [1, 2])
    circ2.add_CSWAP([0, 1], 2)
    paramcount2 = 0 + 0
    for circ, paramcount in [(circ1, paramcount1), (circ2, paramcount2)]:
        params = np.random.rand(paramcount) * 2 * np.pi
        newcirc, newparams = circuit_to_CNOT_basis(circ, params)
        Umat = np.eye(1 << circ.get_Qbit_Num(), dtype=np.complex128)
        Umatnew = np.eye(1 << newcirc.get_Qbit_Num(), dtype=np.complex128)
        circ.apply_to(params, Umat)
        newcirc.apply_to(newparams, Umatnew)
        # phase = np.angle(np.linalg.det(Umat @ np.linalg.inv(Umatnew)))
        phase = np.angle((Umatnew @ Umat.conj().T)[0, 0])
        # Normalize one matrix
        Umatnew = Umatnew * np.exp(-1j * phase)
        # Check closeness
        assert np.allclose(Umat, Umatnew), (Umat, Umatnew)


# test_circuit_to_CNOT_basis()
