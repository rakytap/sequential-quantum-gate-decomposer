'''
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
'''

import inspect
import json
import numpy as np
import pytest
import subprocess
import sys

from qiskit import QuantumCircuit

from squander import Qiskit_IO
from squander import utils
from squander.gates import gates_Wrapper as gate
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit


def _discover_gate_names():
    names = []
    for name in dir(gate):
        if name.startswith("_"):
            continue
        obj = getattr(gate, name)
        if not inspect.isclass(obj):
            continue
        try:
            if issubclass(obj, gate.Gate):
                names.append(name)
        except TypeError:
            continue
    return sorted(names)


ALL_GATE_NAMES = _discover_gate_names()
QISKIT_EXCLUDED_GATES = {"SYC", "CR", "CROT"}
QISKIT_MATRIX_UNSUPPORTED = {"Gate"} | QISKIT_EXCLUDED_GATES
NATIVE_UNSAFE_MATRIX_GATES = {"Gate", "CROT"}
NATIVE_UNSAFE_APPLY_GATES = {"Gate", "CROT", "SWAP", "RXX", "RYY", "RZZ"}


def _discover_multi_qubit_gate_names():
    names = []
    for gate_name in ALL_GATE_NAMES:
        if gate_name == "Gate":
            continue
        gate_obj = _instantiate_gate(gate_name)
        if len(gate_obj.get_Involved_Qbits()) >= 2:
            names.append(gate_name)
    return sorted(names)


def _instantiate_gate(gate_name, qbit_num=4):
    gate_cls = getattr(gate, gate_name)

    if gate_name == "Gate":
        return gate_cls(qbit_num)
    if gate_name in {"SWAP", "RXX", "RYY", "RZZ"}:
        return gate_cls(qbit_num, [0, qbit_num - 1])
    if gate_name == "CCX":
        return gate_cls(qbit_num, 0, [qbit_num - 2, qbit_num - 1])
    if gate_name == "CSWAP":
        return gate_cls(qbit_num, [0, 1], [qbit_num - 1])
    if gate_name == "SYC":
        return gate_cls(qbit_num, 0, qbit_num - 1)
    if gate_name.startswith("C"):
        return gate_cls(qbit_num, 0, qbit_num - 1)
    return gate_cls(qbit_num, 0)


def _parameters_for_gate(gate_obj):
    pnum = gate_obj.get_Parameter_Num()
    if pnum == 0:
        return np.asarray([], dtype=np.float64)
    return np.linspace(0.1, 0.1 * pnum, pnum, dtype=np.float64)


def _gate_matrix(gate_obj, parameters):
    if gate_obj.get_Parameter_Num() == 0:
        return np.asarray(gate_obj.get_Matrix())
    return np.asarray(gate_obj.get_Matrix(parameters))


def _assert_matrices_close(a, b, tol=1e-8):
    overlap = np.vdot(a.reshape(-1), b.reshape(-1))
    if np.abs(overlap) > 0:
        b = b * np.exp(-1j * np.angle(overlap))
    err = np.linalg.norm(a - b)
    assert err < tol, f"matrix mismatch: {err}"


def _add_gate_to_qiskit(circuit, gate_name, parameters):
    target = 0
    control = circuit.num_qubits - 1

    qiskit_params = list(parameters)
    if qiskit_params:
        if gate_name not in {"U1", "U2", "CP"}:
            qiskit_params[0] = 2.0 * qiskit_params[0]

    if gate_name == "U1":
        circuit.p(*qiskit_params, target)
    elif gate_name == "U2":
        circuit.u(np.pi / 2.0, qiskit_params[0], qiskit_params[1], target)
    elif gate_name == "U3":
        circuit.u(*qiskit_params, target)
    elif gate_name == "CNOT":
        circuit.cx(control, target)
    elif gate_name == "CU":
        circuit.cu(
            qiskit_params[0],
            qiskit_params[1],
            qiskit_params[2],
            qiskit_params[3],
            control,
            target,
        )
    elif gate_name == "CCX":
        circuit.ccx(circuit.num_qubits - 2, circuit.num_qubits - 1, target)
    elif gate_name == "CSWAP":
        circuit.cswap(circuit.num_qubits - 1, 0, 1)
    elif gate_name in {"SWAP", "RXX", "RYY", "RZZ"}:
        method = getattr(circuit, gate_name.lower())
        if qiskit_params:
            method(*qiskit_params, 0, circuit.num_qubits - 1)
        else:
            method(0, circuit.num_qubits - 1)
    else:
        method = getattr(circuit, gate_name.lower())
        if gate_name.startswith("C"):
            if qiskit_params:
                method(*qiskit_params, control, target)
            else:
                method(control, target)
        else:
            if qiskit_params:
                method(*qiskit_params, target)
            else:
                method(target)


MULTI_QUBIT_GATE_NAMES = _discover_multi_qubit_gate_names()
FORBIDDEN_MULTI_QUBIT_GATES_IN_CNOT_BASIS = {
    gate_name for gate_name in MULTI_QUBIT_GATE_NAMES if gate_name != "CNOT"
}


class TestGates:
    def test_operation_block_creation(self):
        c_circuit = Circuit(3)
        assert c_circuit.get_Qbit_Num() == 3

    def test_operation_block_add_operations(self):
        c_circuit = Circuit(3)
        c_circuit.add_U3(0)
        c_circuit.add_CNOT(0, 1)

        nums = c_circuit.get_Gate_Nums()
        assert nums.get("U3", 0) == 1
        assert nums.get("CNOT", 0) == 1

    @pytest.mark.parametrize("gate_name", ALL_GATE_NAMES)
    def test_gate_constructor_and_base_methods(self, gate_name):
        gate_obj = _instantiate_gate(gate_name)
        base_methods = [
            name
            for name, obj in inspect.getmembers(gate.Gate(4))
            if callable(obj) and not name.startswith("_")
        ]

        for method_name in base_methods:
            assert hasattr(gate_obj, method_name), f"{gate_name} missing method {method_name}"

        p = _parameters_for_gate(gate_obj)
        matrix = None
        if gate_name not in NATIVE_UNSAFE_MATRIX_GATES:
            matrix = _gate_matrix(gate_obj, p)

        assert gate_obj.get_Name()
        assert isinstance(gate_obj.get_Involved_Qbits(), list)
        assert gate_obj.get_Parameter_Num() >= 0
        if matrix is not None:
            assert matrix.shape == (1 << 4, 1 << 4)

        tgt = gate_obj.get_Target_Qbit()
        gate_obj.set_Target_Qbit(tgt)

        tgt_list = list(gate_obj.get_Target_Qbits())
        if tgt_list:
            gate_obj.set_Target_Qbits(tgt_list)

        ctrl_list = list(gate_obj.get_Control_Qbits())
        if ctrl_list:
            gate_obj.set_Control_Qbits(ctrl_list)
            gate_obj.set_Control_Qbit(ctrl_list[0])

        if gate_name not in NATIVE_UNSAFE_APPLY_GATES and matrix is not None:
            state = np.random.uniform(-1.0, 1.0, (1 << 4,)) + 1j * np.random.uniform(-1.0, 1.0, (1 << 4,))
            state = state / np.linalg.norm(state)
            expected = matrix @ state
            state_out = state.copy()

            if gate_obj.get_Parameter_Num() == 0:
                gate_obj.apply_to(state_out)
            else:
                gate_obj.apply_to(state_out, p)

            assert np.linalg.norm(state_out - expected) < 1e-8

    @pytest.mark.parametrize(
        "gate_name",
        [name for name in ALL_GATE_NAMES if name not in QISKIT_MATRIX_UNSUPPORTED],
    )
    def test_gate_matrix_matches_qiskit(self, gate_name):
        gate_obj = _instantiate_gate(gate_name)
        parameters = _parameters_for_gate(gate_obj)
        gate_matrix = _gate_matrix(gate_obj, parameters)

        circuit = QuantumCircuit(4)
        _add_gate_to_qiskit(circuit, gate_name, parameters)
        qiskit_matrix = utils.get_unitary_from_qiskit_circuit_operator(circuit)

        _assert_matrices_close(gate_matrix, np.asarray(qiskit_matrix), tol=1e-7)

    @pytest.mark.parametrize(
        "gate_name",
        [name for name in ALL_GATE_NAMES if name not in (QISKIT_EXCLUDED_GATES | {"Gate"})],
    )
    def test_qiskit_io_roundtrip_per_gate(self, gate_name):
        script = f"""
import json
import numpy as np
from squander import Qiskit_IO
from squander import utils
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    circuit = Circuit(4)
    circuit.add_Gate(gate_obj)
    params = _parameters_for_gate(gate_obj)

    exported = Qiskit_IO.get_Qiskit_Circuit(circuit, params)
    if exported is None:
        print(json.dumps({{"status": "unsupported"}}))
    else:
        exported_inv = Qiskit_IO.get_Qiskit_Circuit_inverse(circuit, params)
        circ_rt, params_rt = Qiskit_IO.convert_Qiskit_to_Squander(exported)

        m1 = np.asarray(circuit.get_Matrix(params))
        m2 = np.asarray(circ_rt.get_Matrix(params_rt))

        overlap = np.vdot(m1.reshape(-1), m2.reshape(-1))
        if np.abs(overlap) > 0:
            m2 = m2 * np.exp(-1j * np.angle(overlap))
        matrix_err = float(np.linalg.norm(m1 - m2))

        forward = np.asarray(utils.get_unitary_from_qiskit_circuit_operator(exported))
        inverse = np.asarray(utils.get_unitary_from_qiskit_circuit_operator(exported_inv))
        ident = forward @ inverse
        ident_raw_err = float(np.linalg.norm(ident - np.eye(ident.shape[0], dtype=np.complex128)))
        phase = np.angle(np.trace(ident))
        ident_phase_aligned = ident * np.exp(-1j * phase)
        ident_phase_err = float(np.linalg.norm(ident_phase_aligned - np.eye(ident.shape[0], dtype=np.complex128)))

        print(json.dumps({{
            "status": "ok",
            "matrix_err": matrix_err,
            "ident_raw_err": ident_raw_err,
            "ident_phase_err": ident_phase_err,
        }}))
except Exception as exc:
    print(json.dumps({{"status": "exception", "message": str(exc)}}))
"""

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            pytest.fail(
                f"Gate {gate_name} crashed in qiskit_IO roundtrip subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])

        assert result["status"] == "ok", result
        assert result["matrix_err"] < 1e-6, result
        assert result["ident_raw_err"] < 1e-6 or result["ident_phase_err"] < 1e-6, result

        if result["ident_raw_err"] >= 1e-6:
            pytest.fail(
                f"{gate_name} is only equivalent up to global phase. "
                f"raw={result['ident_raw_err']}, phase_aligned={result['ident_phase_err']}"
            )

    @pytest.mark.parametrize("gate_name", MULTI_QUBIT_GATE_NAMES)
    def test_circuit_to_cnot_basis_removes_non_cnot_multi_qubit_gates(self, gate_name):
        circuit = Circuit(4)
        circuit.add_Gate(_instantiate_gate(gate_name))

        params = np.linspace(0.2, 0.2 * circuit.get_Parameter_Num(), circuit.get_Parameter_Num(), dtype=np.float64)

        cnot_circuit, cnot_params = utils.circuit_to_CNOT_basis(circuit, params)

        gate_nums = cnot_circuit.get_Gate_Nums()
        assert gate_nums.get("CNOT", 0) > 0

        for forbidden_gate in FORBIDDEN_MULTI_QUBIT_GATES_IN_CNOT_BASIS:
            assert gate_nums.get(forbidden_gate, 0) == 0, (
                f"{forbidden_gate} still present after CNOT-basis conversion "
                f"for source gate {gate_name}"
            )

        original_matrix = np.asarray(circuit.get_Matrix(params))
        transpiled_matrix = np.asarray(cnot_circuit.get_Matrix(cnot_params))
        _assert_matrices_close(original_matrix, transpiled_matrix, tol=1e-6)
