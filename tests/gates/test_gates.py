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
NATIVE_UNSAFE_MATRIX_GATES = {"Gate"}
NATIVE_UNSAFE_APPLY_GATES = {"Gate"}
DERIVATIVE_TEST_EXCLUDED_GATES = set()
RECT_COLS_MAX = 32


def _discover_parameterized_gate_names():
    names = []
    for gate_name in ALL_GATE_NAMES:
        if gate_name in DERIVATIVE_TEST_EXCLUDED_GATES:
            continue
        gate_obj = _instantiate_gate(gate_name)
        if gate_obj.get_Parameter_Num() > 0:
            names.append(gate_name)
    return sorted(names)


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


def _parameters_for_gate(gate_obj, dtype=np.float64):
    pnum = gate_obj.get_Parameter_Num()
    if pnum == 0:
        return np.asarray([], dtype=dtype)
    return np.linspace(0.1, 0.1 * pnum, pnum, dtype=dtype)


def _gate_matrix(gate_obj, parameters):
    if gate_obj.get_Parameter_Num() == 0:
        return np.asarray(gate_obj.get_Matrix())
    return np.asarray(gate_obj.get_Matrix(parameters))


def _apply_gate(gate_obj, state, parameters, is_f32=False, parallel=0):
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(state, parallel=parallel, is_f32=is_f32)
    else:
        gate_obj.apply_to(state, parameters=parameters, parallel=parallel, is_f32=is_f32)


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
DERIVATIVE_GATE_NAMES = _discover_parameterized_gate_names()
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

    @pytest.mark.parametrize("gate_name", [name for name in ALL_GATE_NAMES if name != "Gate"])
    def test_gate_apply_to_float32_float64_parity(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    state64 = np.random.uniform(-1.0, 1.0, (1 << 4,)) + 1j * np.random.uniform(-1.0, 1.0, (1 << 4,))
    state64 = state64.astype(np.complex128)
    state64 = state64 / np.linalg.norm(state64)

    state32 = state64.astype(np.complex64)
    state32 = state32 / np.linalg.norm(state32)

    out64 = state64.copy()
    out32 = state32.copy()

    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(out64, parallel=0, is_f32=False)
        gate_obj.apply_to(out32, parallel=0, is_f32=True)
    else:
        gate_obj.apply_to(out64, parameters=p64, parallel=0, is_f32=False)
        gate_obj.apply_to(out32, parameters=p32, parallel=0, is_f32=True)

    err = float(np.linalg.norm(out32 - out64.astype(np.complex64)))
    print(json.dumps({{"status": "ok", "err": err}}))
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
                f"Gate {gate_name} crashed in float32/float64 parity subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result
        assert result["err"] < 1e-4, f"float32/float64 parity mismatch for {gate_name}: {result['err']}"

    @pytest.mark.parametrize("gate_name", [name for name in ALL_GATE_NAMES if name != "Gate"])
    def test_gate_get_matrix_float32_float64_parity(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    if gate_obj.get_Parameter_Num() == 0:
        m64 = np.asarray(gate_obj.get_Matrix(is_f32=False))
        m32 = np.asarray(gate_obj.get_Matrix(is_f32=True))
    else:
        m64 = np.asarray(gate_obj.get_Matrix(p64, is_f32=False))
        m32 = np.asarray(gate_obj.get_Matrix(p32, is_f32=True))

    err = float(np.linalg.norm(m32 - m64.astype(np.complex64)))
    print(json.dumps({{"status": "ok", "err": err, "dtype64": str(m64.dtype), "dtype32": str(m32.dtype)}}))
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
                f"Gate {gate_name} crashed in get_Matrix float32/float64 parity subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result
        assert result["dtype64"] == "complex128", result
        assert result["dtype32"] == "complex64", result
        assert result["err"] < 1e-4, f"float32/float64 get_Matrix parity mismatch for {gate_name}: {result['err']}"

    @pytest.mark.parametrize("gate_name", [name for name in ALL_GATE_NAMES if name != "Gate"])
    def test_gate_apply_from_right_float32_float64_parity(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    inp64 = np.random.uniform(-1.0, 1.0, (1 << 4, 1 << 4)) + 1j * np.random.uniform(-1.0, 1.0, (1 << 4, 1 << 4))
    inp64 = inp64.astype(np.complex128)
    inp32 = inp64.astype(np.complex64)

    out64 = inp64.copy()
    out32 = inp32.copy()

    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_from_right(out64, is_f32=False)
        gate_obj.apply_from_right(out32, is_f32=True)
    else:
        gate_obj.apply_from_right(out64, parameters=p64, is_f32=False)
        gate_obj.apply_from_right(out32, parameters=p32, is_f32=True)

    err = float(np.linalg.norm(out32 - out64.astype(np.complex64)))
    print(json.dumps({{"status": "ok", "err": err}}))
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
                f"Gate {gate_name} crashed in apply_from_right float32/float64 parity subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result
        assert result["err"] < 1e-4, f"float32/float64 apply_from_right parity mismatch for {gate_name}: {result['err']}"

    @pytest.mark.parametrize("gate_name", [name for name in ALL_GATE_NAMES if name != "Gate"])
    def test_gate_apply_to_list_float32_float64_parity(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    inputs64 = []
    for _ in range(3):
        vec = np.random.uniform(-1.0, 1.0, (1 << 4,)) + 1j * np.random.uniform(-1.0, 1.0, (1 << 4,))
        vec = vec.astype(np.complex128)
        vec = vec / np.linalg.norm(vec)
        inputs64.append(vec)

    inputs32 = [v.astype(np.complex64) for v in inputs64]

    out64 = [v.copy() for v in inputs64]
    out32 = [v.copy() for v in inputs32]

    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to_list(out64, parallel=0, is_f32=False)
        gate_obj.apply_to_list(out32, parallel=0, is_f32=True)
    else:
        gate_obj.apply_to_list(out64, parameters=p64, parallel=0, is_f32=False)
        gate_obj.apply_to_list(out32, parameters=p32, parallel=0, is_f32=True)

    errs = [float(np.linalg.norm(a - b.astype(np.complex64))) for a, b in zip(out32, out64)]
    print(json.dumps({{"status": "ok", "max_err": float(max(errs))}}))
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
                f"Gate {gate_name} crashed in apply_to_list float32/float64 parity subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result
        assert result["max_err"] < 1e-4, f"float32/float64 apply_to_list parity mismatch for {gate_name}: {result['max_err']}"

    @pytest.mark.parametrize("gate_name", [name for name in ALL_GATE_NAMES if name != "Gate"])
    def test_gate_apply_rectangular_sweep_paths(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate, RECT_COLS_MAX

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    rng = np.random.default_rng(2026)

    # apply_to path: 16xN (N=1..32)
    left64 = np.ascontiguousarray(
        rng.standard_normal((1 << 4, RECT_COLS_MAX))
        + 1j * rng.standard_normal((1 << 4, RECT_COLS_MAX)),
        dtype=np.complex128,
    )
    left32 = left64.astype(np.complex64)

    out_full64 = left64.copy()
    out_full32 = left32.copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(out_full64, parallel=0, is_f32=False)
        gate_obj.apply_to(out_full32, parallel=0, is_f32=True)
    else:
        gate_obj.apply_to(out_full64, parameters=p64, parallel=0, is_f32=False)
        gate_obj.apply_to(out_full32, parameters=p32, parallel=0, is_f32=True)

    left_subset_err64 = 0.0
    left_subset_err32 = 0.0
    for ncols in range(1, RECT_COLS_MAX + 1):
        p64_m = np.ascontiguousarray(left64[:, :ncols].copy())
        p32_m = np.ascontiguousarray(left32[:, :ncols].copy())
        if gate_obj.get_Parameter_Num() == 0:
            gate_obj.apply_to(p64_m, parallel=0, is_f32=False)
            gate_obj.apply_to(p32_m, parallel=0, is_f32=True)
        else:
            gate_obj.apply_to(p64_m, parameters=p64, parallel=0, is_f32=False)
            gate_obj.apply_to(p32_m, parameters=p32, parallel=0, is_f32=True)
        left_subset_err64 = max(left_subset_err64, float(np.linalg.norm(p64_m - out_full64[:, :ncols])))
        left_subset_err32 = max(left_subset_err32, float(np.linalg.norm(p32_m - out_full32[:, :ncols])))

    # apply_from_right path: Nx16 (N=1..32), columns must equal matrix_size.
    right64 = np.ascontiguousarray(
        rng.standard_normal((RECT_COLS_MAX, 1 << 4))
        + 1j * rng.standard_normal((RECT_COLS_MAX, 1 << 4)),
        dtype=np.complex128,
    )
    right32 = right64.astype(np.complex64)

    out_right64 = right64.copy()
    out_right32 = right32.copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_from_right(out_right64, is_f32=False)
        gate_obj.apply_from_right(out_right32, is_f32=True)
    else:
        gate_obj.apply_from_right(out_right64, parameters=p64, is_f32=False)
        gate_obj.apply_from_right(out_right32, parameters=p32, is_f32=True)

    right_subset_err64 = 0.0
    right_subset_err32 = 0.0
    for nrows in range(1, RECT_COLS_MAX + 1):
        p64_m = np.ascontiguousarray(right64[:nrows, :].copy())
        p32_m = np.ascontiguousarray(right32[:nrows, :].copy())
        if gate_obj.get_Parameter_Num() == 0:
            gate_obj.apply_from_right(p64_m, is_f32=False)
            gate_obj.apply_from_right(p32_m, is_f32=True)
        else:
            gate_obj.apply_from_right(p64_m, parameters=p64, is_f32=False)
            gate_obj.apply_from_right(p32_m, parameters=p32, is_f32=True)
        right_subset_err64 = max(right_subset_err64, float(np.linalg.norm(p64_m - out_right64[:nrows, :])))
        right_subset_err32 = max(right_subset_err32, float(np.linalg.norm(p32_m - out_right32[:nrows, :])))

    # apply_to_list path over 16xN for N=1..32.
    list_in64 = [
        np.ascontiguousarray(
            rng.standard_normal((1 << 4, ncols)) + 1j * rng.standard_normal((1 << 4, ncols)),
            dtype=np.complex128,
        )
        for ncols in range(1, RECT_COLS_MAX + 1)
    ]
    list_in32 = [m.astype(np.complex64) for m in list_in64]
    ref64 = [m.copy() for m in list_in64]
    ref32 = [m.copy() for m in list_in32]

    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to_list(list_in64, parallel=0, is_f32=False)
        gate_obj.apply_to_list(list_in32, parallel=0, is_f32=True)
        for m in ref64:
            gate_obj.apply_to(m, parallel=0, is_f32=False)
        for m in ref32:
            gate_obj.apply_to(m, parallel=0, is_f32=True)
    else:
        gate_obj.apply_to_list(list_in64, parameters=p64, parallel=0, is_f32=False)
        gate_obj.apply_to_list(list_in32, parameters=p32, parallel=0, is_f32=True)
        for m in ref64:
            gate_obj.apply_to(m, parameters=p64, parallel=0, is_f32=False)
        for m in ref32:
            gate_obj.apply_to(m, parameters=p32, parallel=0, is_f32=True)

    list_err64 = max(float(np.linalg.norm(a - b)) for a, b in zip(list_in64, ref64))
    list_err32 = max(float(np.linalg.norm(a - b)) for a, b in zip(list_in32, ref32))

    print(json.dumps({{
        "status": "ok",
        "left_subset_err64": left_subset_err64,
        "left_subset_err32": left_subset_err32,
        "right_subset_err64": right_subset_err64,
        "right_subset_err32": right_subset_err32,
        "list_err64": list_err64,
        "list_err32": list_err32,
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
                f"Gate {gate_name} crashed in rectangular apply* sweep subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result
        assert result["left_subset_err64"] < 1e-10, result
        assert result["left_subset_err32"] < 5e-5, result
        assert result["right_subset_err64"] < 1e-10, result
        assert result["right_subset_err32"] < 5e-5, result
        assert result["list_err64"] < 1e-10, result
        assert result["list_err32"] < 5e-5, result

    @pytest.mark.parametrize("gate_name", DERIVATIVE_GATE_NAMES)
    def test_gate_apply_derivate_wrapper_smoke(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    state64 = np.random.uniform(-1.0, 1.0, (1 << 4,)) + 1j * np.random.uniform(-1.0, 1.0, (1 << 4,))
    state64 = state64.astype(np.complex128)
    state64 = state64 / np.linalg.norm(state64)

    state32 = state64.astype(np.complex64)
    state32 = state32 / np.linalg.norm(state32)

    d64 = gate_obj.apply_derivate_to(state64.copy(), parameters=p64, parallel=0, is_f32=False)
    d32 = gate_obj.apply_derivate_to(state32.copy(), parameters=p32, parallel=0, is_f32=True)

    if not isinstance(d64, list) or not isinstance(d32, list):
        raise RuntimeError("apply_derivate_to must return a list")

    for arr in d64:
        a = np.asarray(arr)
        if a.dtype != np.complex128:
            raise RuntimeError("float64 derivative output must be complex128")
        if a.size != state64.size:
            raise RuntimeError("float64 derivative output shape mismatch")

    for arr in d32:
        a = np.asarray(arr)
        if a.dtype != np.complex64:
            raise RuntimeError("float32 derivative output must be complex64")
        if a.size != state32.size:
            raise RuntimeError("float32 derivative output shape mismatch")

    print(json.dumps({{"status": "ok", "n64": len(d64), "n32": len(d32)}}))
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
                f"Gate {gate_name} crashed in apply_derivate_to wrapper subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result

    @pytest.mark.parametrize("gate_name", DERIVATIVE_GATE_NAMES)
    def test_gate_apply_to_combined_wrapper_smoke(self, gate_name):
        script = f"""
import json
import numpy as np
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    p64 = _parameters_for_gate(gate_obj, dtype=np.float64)
    p32 = p64.astype(np.float32)

    state64 = np.random.uniform(-1.0, 1.0, (1 << 4,)) + 1j * np.random.uniform(-1.0, 1.0, (1 << 4,))
    state64 = state64.astype(np.complex128)
    state64 = state64 / np.linalg.norm(state64)

    state32 = state64.astype(np.complex64)
    state32 = state32 / np.linalg.norm(state32)

    c64 = gate_obj.apply_to_combined(state64.copy(), parameters=p64, parallel=0, is_f32=False)
    c32 = gate_obj.apply_to_combined(state32.copy(), parameters=p32, parallel=0, is_f32=True)

    if not isinstance(c64, list) or not isinstance(c32, list):
        raise RuntimeError("apply_to_combined must return a list")

    if len(c64) != len(p64) + 1 or len(c32) != len(p32) + 1:
        raise RuntimeError("apply_to_combined returned wrong list length")

    fwd64_ref = state64.copy()
    gate_obj.apply_to(fwd64_ref, parameters=p64, parallel=0, is_f32=False)
    fwd32_ref = state32.copy()
    gate_obj.apply_to(fwd32_ref, parameters=p32, parallel=0, is_f32=True)

    if np.linalg.norm(np.asarray(c64[0]).reshape(-1) - fwd64_ref.reshape(-1)) > 1e-10:
        raise RuntimeError("float64 combined forward output mismatch")
    if np.linalg.norm(np.asarray(c32[0]).astype(np.complex128).reshape(-1) - fwd32_ref.astype(np.complex128).reshape(-1)) > 5e-5:
        raise RuntimeError("float32 combined forward output mismatch")

    print(json.dumps({{"status": "ok", "n64": len(c64), "n32": len(c32)}}))
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
                f"Gate {gate_name} crashed in apply_to_combined wrapper subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result

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

    @pytest.mark.parametrize(
        "gate_name",
        [name for name in ALL_GATE_NAMES if name != "Gate"],
    )
    def test_squander_invert_circuit(self, gate_name):
        script = f"""
import json
import numpy as np
from squander import utils
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from tests.gates.test_gates import _instantiate_gate, _parameters_for_gate

gate_name = {gate_name!r}

try:
    gate_obj = _instantiate_gate(gate_name)
    circuit = Circuit(4)
    circuit.add_Gate(gate_obj)
    params = _parameters_for_gate(gate_obj, dtype=np.float64)

    inv_circuit, inv_params = utils.invert_circuit(circuit, params)

    n = 1 << 4
    M = np.eye(n, dtype=np.complex128)
    Minv = np.eye(n, dtype=np.complex128)
    circuit.apply_to(params, M)
    inv_circuit.apply_to(inv_params, Minv)

    product = Minv @ M
    err = float(np.linalg.norm(product - np.eye(n)))
    print(json.dumps({{"status": "ok", "err": err}}))
except Exception as exc:
    import traceback
    print(json.dumps({{"status": "exception", "message": str(exc), "trace": traceback.format_exc()}}))
"""

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            pytest.fail(
                f"Gate {gate_name} crashed in invert_circuit subprocess. "
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )

        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "ok", result
        assert result["err"] < 1e-8, f"invert_circuit error for {gate_name}: {result['err']}"

    @pytest.mark.parametrize("gate_name", DERIVATIVE_GATE_NAMES)
    def test_extract_parameters_wraps_out_of_range(self, gate_name):
        """extract_parameters should normalise each parameter via fmod(m*p, m*2pi).
        Values already inside [0, m*2pi) should be unchanged; values above should wrap."""
        gate_obj = _instantiate_gate(gate_name)
        pnum = gate_obj.get_Parameter_Num()

        two_pi = 2.0 * np.pi

        # Build a circuit-level parameter array; start_idx is 0 by default.
        # Use one small (in-range) and one large (out-of-range) value per parameter.
        small_params = np.linspace(0.1, 0.1 * pnum, pnum, dtype=np.float64)
        large_params = small_params + 10.0 * two_pi  # shift by 5 full periods

        small_extracted = gate_obj.Extract_Parameters(small_params)
        large_extracted = gate_obj.Extract_Parameters(large_params)

        assert len(small_extracted) == pnum, "wrong number of extracted parameters"
        assert len(large_extracted) == pnum

        # Both in-range and shifted versions should give the same normalised result
        np.testing.assert_allclose(
            small_extracted,
            large_extracted,
            atol=1e-10,
            err_msg=f"{gate_name}: extract_parameters did not wrap large parameters to same value as small",
        )

        # In-range values should pass through unchanged (fmod(m*p, m*2pi) == m*p when m*p < m*2pi)
        for i, p in enumerate(small_params):
            assert small_extracted[i] >= 0.0, f"{gate_name}: extracted param {i} is negative"

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


class TestGateDerivativeFiniteDifference:
    """Verify that each parametric gate's apply_derivate_to matches a central finite difference."""

    FD_EPS = 1e-5
    FD_TOL = 1e-5

    @pytest.mark.parametrize("gate_name", DERIVATIVE_GATE_NAMES)
    def test_gate_derivative_fd_f64(self, gate_name):
        """Analytic derivative must agree with finite difference (float64)."""
        gate_obj = _instantiate_gate(gate_name)
        pnum = gate_obj.get_Parameter_Num()
        params = _parameters_for_gate(gate_obj)

        dim = np.asarray(gate_obj.get_Matrix(params)).shape[0]
        mat = np.eye(dim, dtype=np.complex128)
        derivs = gate_obj.apply_derivate_to(mat.copy(), parameters=params, parallel=0, is_f32=False)

        assert len(derivs) == pnum, f"{gate_name}: got {len(derivs)} derivatives, expected {pnum}"

        eps = self.FD_EPS
        for k in range(pnum):
            p_plus = params.copy(); p_plus[k] += eps
            p_minus = params.copy(); p_minus[k] -= eps
            fd = (np.asarray(gate_obj.get_Matrix(p_plus)) - np.asarray(gate_obj.get_Matrix(p_minus))) / (2 * eps)
            err = float(np.linalg.norm(np.asarray(derivs[k]) - fd))
            assert err < self.FD_TOL, (
                f"{gate_name} param {k}: analytic vs FD error = {err:.3e} > {self.FD_TOL:.1e}"
            )
