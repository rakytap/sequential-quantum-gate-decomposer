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

Tests for qgd_Circuit: covers float32/float64 dispatch, state vector vs unitary,
apply_from_right, gate fusion (set_min_fusion), nested circuits, and boundary qubit
counts that stress AVX kernels (small kernel: qbit_num<14, TBB large kernel: qbit_num>=14).
'''

import numpy as np
import pytest

from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _identity(n, dtype=np.complex128):
    return np.eye(n, dtype=dtype, order='C')


def _random_state(n, dtype=np.complex128, seed=42):
    """Return a normalised random state vector of length n."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = v.astype(dtype)
    return np.ascontiguousarray(v / np.linalg.norm(v))


def _params64(n):
    """Float64 parameter array of length n."""
    if n == 0:
        return np.array([], dtype=np.float64)
    return np.linspace(0.1, 0.1 * n, n, dtype=np.float64)


def _params32(n):
    """Float32 parameter array of length n."""
    if n == 0:
        return np.array([], dtype=np.float32)
    return np.linspace(0.1, 0.1 * n, n, dtype=np.float32)


def _build_rx_rz_cnot_circuit(qbit_num):
    """Build RX(0) → RZ(0) → CNOT(0,1) circuit if qbit_num >= 2, else RX → RZ."""
    c = Circuit(qbit_num)
    c.add_RX(0)
    c.add_RZ(0)
    if qbit_num >= 2:
        c.add_CNOT(0, 1)
        c.add_RY(1)
    return c


def _ref_matrix64(circuit):
    """Compute reference unitary via get_Matrix (float64)."""
    params = _params64(circuit.get_Parameter_Num())
    return np.array(circuit.get_Matrix(params))


def _assert_unitary_close(a, b, rtol=1e-6):
    """Assert two unitaries are the same up to a global phase."""
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    overlap = np.vdot(a.reshape(-1), b.reshape(-1))
    if abs(overlap) > 0:
        b = b * np.exp(-1j * np.angle(overlap))
    err = np.linalg.norm(a - b)
    assert err < rtol, f"unitary mismatch: error={err:.3e}"


def _assert_state_close(a, b, rtol=1e-6):
    """Assert two state vectors are the same up to a global phase."""
    assert a.shape == b.shape
    overlap = np.vdot(a, b)
    if abs(overlap) > 0:
        b = b * np.exp(-1j * np.angle(overlap))
    err = np.linalg.norm(a - b)
    assert err < rtol, f"state mismatch: error={err:.3e}"


def _random_unitary(dim, seed=123):
    """Create a deterministic random unitary matrix using QR decomposition."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    phase = np.where(np.abs(d) > 0, d / np.abs(d), 1.0)
    q = q * phase
    return np.ascontiguousarray(q, dtype=np.complex128)


# ──────────────────────────────────────────────────────────────────────────────
# Qubit counts chosen to stress different AVX dispatch paths.
#   qbit_num = 1,2,3  → small gate kernel (direct index arithmetic)
#   qbit_num = 4,5,6  → medium sizes, apply_large_kernel_to_input_AVX
#   qbit_num = 14     → AVX TBB path (qbit_num >= 14, for fusion)
# ──────────────────────────────────────────────────────────────────────────────
QBIT_NUMS_SMALL = [1, 2, 3]
QBIT_NUMS_MEDIUM = [4, 5, 6]
QBIT_NUMS_ALL = QBIT_NUMS_SMALL + QBIT_NUMS_MEDIUM
RECT_COLS_MAX = 32


class TestApplyToFloat64:
    """Apply_to float64: unitary and state-vector correctness."""

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_apply_to_unitary_matches_get_matrix(self, qbit_num):
        """apply_to on identity should reproduce get_Matrix output."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        ref = np.array(circ.get_Matrix(params))

        dim = 2 ** qbit_num
        identity = _identity(dim)
        circ.apply_to(params, identity)

        _assert_unitary_close(ref, identity)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_apply_to_state_vector(self, qbit_num):
        """apply_to on a state vector should give U @ |state>."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        U = np.array(circ.get_Matrix(params))

        dim = 2 ** qbit_num
        state = _random_state(dim, dtype=np.complex128)
        expected = U @ state

        sv = state.copy()
        circ.apply_to(params, sv.reshape(dim, 1))
        result = sv.reshape(-1)

        _assert_state_close(expected, result)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_apply_to_twice_is_squared_unitary(self, qbit_num):
        """Applying circuit twice equals U^2 applied to identity."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        U = np.array(circ.get_Matrix(params))

        dim = 2 ** qbit_num
        mtx = _identity(dim)
        circ.apply_to(params, mtx)
        circ.apply_to(params, mtx)

        expected = U @ U
        _assert_unitary_close(expected, mtx)


class TestApplyToFloat32:
    """Apply_to float32: complex64 precision correctness against float64 reference."""

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_apply_to_f32_unitary_close_to_f64(self, qbit_num):
        """float32 apply_to result should be close to float64 result."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p64 = _params64(circ.get_Parameter_Num())
        p32 = p64.astype(np.float32)

        dim = 2 ** qbit_num

        # float64 reference
        ref64 = _identity(dim, dtype=np.complex128)
        circ.apply_to(p64, ref64)

        # float32 result
        mat32 = _identity(dim, dtype=np.complex64)
        circ.apply_to(p32, mat32, is_f32=True)

        _assert_unitary_close(ref64, mat32.astype(np.complex128), rtol=1e-4)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_apply_to_f32_state_vector(self, qbit_num):
        """float32 state vector apply matches float64 reference."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p64 = _params64(circ.get_Parameter_Num())
        p32 = p64.astype(np.float32)

        dim = 2 ** qbit_num
        state64 = _random_state(dim, dtype=np.complex128)
        state32 = state64.astype(np.complex64)

        sv64 = state64.reshape(dim, 1).copy()
        circ.apply_to(p64, sv64)
        result64 = sv64.reshape(-1)

        sv32 = state32.reshape(dim, 1).copy()
        circ.apply_to(p32, sv32, is_f32=True)
        result32 = sv32.reshape(-1)

        _assert_state_close(result64, result32.astype(np.complex128), rtol=1e-4)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_apply_to_f32_dtype_preserved(self, qbit_num):
        """Output dtype should remain complex64 after float32 apply_to."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())

        dim = 2 ** qbit_num
        mat = _identity(dim, dtype=np.complex64)
        circ.apply_to(p32, mat, is_f32=True)

        assert mat.dtype == np.complex64, f"expected complex64, got {mat.dtype}"

    def test_apply_to_f32_wrong_dtype_raises(self):
        """Passing complex128 with is_f32=True should raise."""
        circ = Circuit(2)
        circ.add_RX(0)
        params = _params32(1)
        mat = _identity(4, dtype=np.complex128)  # wrong dtype for f32 path
        with pytest.raises(Exception):
            circ.apply_to(params, mat, is_f32=True)

    def test_apply_to_f64_wrong_dtype_raises(self):
        """Passing complex64 with is_f32=False (default) should raise."""
        circ = Circuit(2)
        circ.add_RX(0)
        params = _params64(1)
        mat = _identity(4, dtype=np.complex64)  # wrong dtype for f64 path
        with pytest.raises(Exception):
            circ.apply_to(params, mat, is_f32=False)


class TestApplyFromRight:
    """apply_from_right: both f64 and f32.

    Gates_block::apply_from_right iterates gates in forward order but reads
    the parameter array from the END backward (by design for gradient
    computations).  For a SINGLE-GATE circuit there is no ordering ambiguity,
    so we can verify the result against the matrix; for multi-gate circuits we
    verify f32/f64 consistency and that the matrix is actually mutated.
    """

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_SMALL + [4])
    def test_apply_from_right_single_gate_f64(self, qbit_num):
        """For a single-gate circuit apply_from_right(params, I) == U."""
        circ = Circuit(qbit_num)
        circ.add_RX(0)                              # single parametric gate

        params = _params64(circ.get_Parameter_Num())
        U = np.array(circ.get_Matrix(params))

        dim = 2 ** qbit_num
        mat = _identity(dim)
        circ.apply_from_right(params, mat)          # I @ U = U

        _assert_unitary_close(U, mat)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_SMALL + [4])
    def test_apply_from_right_single_gate_noparams_f64(self, qbit_num):
        """Single parameter-free gate (H): apply_from_right([], A) == A @ H."""
        circ = Circuit(qbit_num)
        circ.add_H(0)

        params = np.array([], dtype=np.float64)
        U = np.array(circ.get_Matrix(params))       # should be H ⊗ I...

        dim = 2 ** qbit_num
        mat = _identity(dim)
        circ.apply_from_right(params, mat)

        _assert_unitary_close(U, mat)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_SMALL + [4])
    def test_apply_from_right_mutates_matrix(self, qbit_num):
        """apply_from_right must modify the input matrix in-place."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())

        dim = 2 ** qbit_num
        mat = _identity(dim)
        circ.apply_from_right(params, mat)

        # As long as the circuit is non-trivial the matrix should differ from I
        assert not np.allclose(mat, np.eye(dim, dtype=np.complex128), atol=1e-6), (
            "apply_from_right did not modify the input matrix"
        )

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_SMALL + [4])
    def test_apply_from_right_f32_close_to_f64(self, qbit_num):
        """float32 apply_from_right result should match float64 reference."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p64 = _params64(circ.get_Parameter_Num())
        p32 = p64.astype(np.float32)

        dim = 2 ** qbit_num
        rng = np.random.default_rng(9)
        A64 = np.ascontiguousarray(
            rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)),
            dtype=np.complex128
        )
        A32 = A64.astype(np.complex64).copy(order='C')

        ref = A64.copy()
        circ.apply_from_right(p64, ref)

        circ.apply_from_right(p32, A32, is_f32=True)

        _assert_unitary_close(ref, A32.astype(np.complex128), rtol=1e-4)


class TestRectangularApplyRoutes:
    """4-qubit rectangular matrix coverage for all apply* methods.

    For apply_to routes, the matrix shape is 16xN (N=1..32).
    For apply_from_right routes, the matrix shape is Nx16 (N=1..32), since
    the right-multiply path requires the number of columns to be 16.
    """

    def _build_circuit_q4(self):
        circ = Circuit(4)
        circ.add_RX(0)
        circ.add_RZ(0)
        circ.add_CNOT(0, 1)
        circ.add_RY(1)
        return circ

    @staticmethod
    def _max_subset_err_apply_to(circ, params, full_input, is_f32=False):
        out_full = np.ascontiguousarray(full_input.copy())
        if is_f32:
            circ.apply_to(params, out_full, is_f32=True)
        else:
            circ.apply_to(params, out_full)

        errs = []
        for ncols in range(1, RECT_COLS_MAX + 1):
            part = np.ascontiguousarray(full_input[:, :ncols].copy())
            if is_f32:
                circ.apply_to(params, part, is_f32=True)
            else:
                circ.apply_to(params, part)
            errs.append(np.linalg.norm(part - out_full[:, :ncols]))

        return float(max(errs))

    @staticmethod
    def _max_subset_err_apply_from_right(circ, params, full_input, is_f32=False):
        out_full = np.ascontiguousarray(full_input.copy())
        if is_f32:
            circ.apply_from_right(params, out_full, is_f32=True)
        else:
            circ.apply_from_right(params, out_full)

        errs = []
        for nrows in range(1, RECT_COLS_MAX + 1):
            part = np.ascontiguousarray(full_input[:nrows, :].copy())
            if is_f32:
                circ.apply_from_right(params, part, is_f32=True)
            else:
                circ.apply_from_right(params, part)
            errs.append(np.linalg.norm(part - out_full[:nrows, :]))

        return float(max(errs))

    def test_apply_to_rectangular_subset_consistency_f64(self):
        circ = self._build_circuit_q4()
        params = _params64(circ.get_Parameter_Num())

        rng = np.random.default_rng(123)
        full = np.ascontiguousarray(
            rng.standard_normal((16, RECT_COLS_MAX))
            + 1j * rng.standard_normal((16, RECT_COLS_MAX)),
            dtype=np.complex128,
        )

        max_err = self._max_subset_err_apply_to(circ, params, full, is_f32=False)
        assert max_err < 1e-10, f"apply_to f64 16xN subset mismatch: {max_err:.3e}"

    def test_apply_to_rectangular_subset_consistency_f32(self):
        circ = self._build_circuit_q4()
        params = _params32(circ.get_Parameter_Num())

        rng = np.random.default_rng(123)
        full = np.ascontiguousarray(
            rng.standard_normal((16, RECT_COLS_MAX))
            + 1j * rng.standard_normal((16, RECT_COLS_MAX)),
            dtype=np.complex64,
        )

        max_err = self._max_subset_err_apply_to(circ, params, full, is_f32=True)
        assert max_err < 5e-5, f"apply_to f32 16xN subset mismatch: {max_err:.3e}"

    def test_apply_from_right_rectangular_subset_consistency_f64(self):
        circ = self._build_circuit_q4()
        params = _params64(circ.get_Parameter_Num())

        rng = np.random.default_rng(124)
        full = np.ascontiguousarray(
            rng.standard_normal((RECT_COLS_MAX, 16))
            + 1j * rng.standard_normal((RECT_COLS_MAX, 16)),
            dtype=np.complex128,
        )

        max_err = self._max_subset_err_apply_from_right(circ, params, full, is_f32=False)
        assert max_err < 1e-10, f"apply_from_right f64 Nx16 subset mismatch: {max_err:.3e}"

    def test_apply_from_right_rectangular_subset_consistency_f32(self):
        circ = self._build_circuit_q4()
        params = _params32(circ.get_Parameter_Num())

        rng = np.random.default_rng(124)
        full = np.ascontiguousarray(
            rng.standard_normal((RECT_COLS_MAX, 16))
            + 1j * rng.standard_normal((RECT_COLS_MAX, 16)),
            dtype=np.complex64,
        )

        max_err = self._max_subset_err_apply_from_right(circ, params, full, is_f32=True)
        assert max_err < 5e-5, f"apply_from_right f32 Nx16 subset mismatch: {max_err:.3e}"


class TestGeneralOperation:
    """Coverage for explicit GENERAL_OPERATION matrix insertion."""

    def _build_circuit_q4(self):
        circ = Circuit(4)
        circ.add_RX(0)
        circ.add_RZ(0)
        circ.add_CNOT(0, 1)
        circ.add_RY(1)
        return circ

    def test_general_operation_all_paths_f64(self):
        qbit_num = 2
        dim = 1 << qbit_num
        U = _random_unitary(dim, seed=91)

        circ = Circuit(qbit_num)
        circ.add_GENERAL(U)

        params = np.array([], dtype=np.float64)

        # get_Matrix
        got = np.asarray(circ.get_Matrix(params), dtype=np.complex128)
        _assert_unitary_close(U, got, rtol=1e-10)

        # apply_to
        rng = np.random.default_rng(92)
        A = np.ascontiguousarray(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)), dtype=np.complex128)
        expected_left = U @ A
        got_left = A.copy(order="C")
        circ.apply_to(params, got_left)
        assert np.allclose(expected_left, got_left, atol=1e-10)

        # apply_from_right
        B = np.ascontiguousarray(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)), dtype=np.complex128)
        expected_right = B @ U
        got_right = B.copy(order="C")
        circ.apply_from_right(params, got_right)
        assert np.allclose(expected_right, got_right, atol=1e-10)

        # apply_to_list
        M0 = np.ascontiguousarray(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)), dtype=np.complex128)
        M1 = np.ascontiguousarray(rng.standard_normal((dim, 1)) + 1j * rng.standard_normal((dim, 1)), dtype=np.complex128)
        inputs = [M0.copy(order="C"), M1.copy(order="C")]
        circ.apply_to_list(inputs, params)
        assert np.allclose(inputs[0], U @ M0, atol=1e-10)
        assert np.allclose(inputs[1], U @ M1, atol=1e-10)

        # derivative is empty for zero-parameter GENERAL_OPERATION
        deriv = circ.apply_derivate_to(params, _identity(dim))
        assert isinstance(deriv, list)
        assert len(deriv) == 0

    def test_general_operation_all_paths_f32(self):
        qbit_num = 2
        dim = 1 << qbit_num
        U64 = _random_unitary(dim, seed=193)
        U32 = np.ascontiguousarray(U64.astype(np.complex64))

        circ = Circuit(qbit_num)
        circ.add_GENERAL(U32, is_f32=True)

        params32 = np.array([], dtype=np.float32)

        # get_Matrix
        got32 = np.asarray(circ.get_Matrix(params32, is_f32=True), dtype=np.complex64)
        assert np.allclose(got32, U32, atol=5e-5)

        rng = np.random.default_rng(194)

        # apply_to
        A32 = np.ascontiguousarray(
            rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)),
            dtype=np.complex64,
        )
        expected_left = U32 @ A32
        got_left = A32.copy(order="C")
        circ.apply_to(params32, got_left, is_f32=True)
        assert np.allclose(expected_left, got_left, atol=5e-5)

        # apply_from_right
        B32 = np.ascontiguousarray(
            rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)),
            dtype=np.complex64,
        )
        expected_right = B32 @ U32
        got_right = B32.copy(order="C")
        circ.apply_from_right(params32, got_right, is_f32=True)
        assert np.allclose(expected_right, got_right, atol=5e-5)

        # apply_to_list
        M0 = np.ascontiguousarray(
            rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)),
            dtype=np.complex64,
        )
        M1 = np.ascontiguousarray(
            rng.standard_normal((dim, 1)) + 1j * rng.standard_normal((dim, 1)),
            dtype=np.complex64,
        )
        inputs = [M0.copy(order="C"), M1.copy(order="C")]
        circ.apply_to_list(inputs, params32, is_f32=True)
        assert np.allclose(inputs[0], U32 @ M0, atol=5e-5)
        assert np.allclose(inputs[1], U32 @ M1, atol=5e-5)

        # derivative is empty for zero-parameter GENERAL_OPERATION
        deriv = circ.apply_derivate_to(params32, _identity(dim, dtype=np.complex64), is_f32=True)
        assert isinstance(deriv, list)
        assert len(deriv) == 0

    def test_apply_to_list_rectangular_sweep_f64(self):
        circ = self._build_circuit_q4()
        params = _params64(circ.get_Parameter_Num())

        rng = np.random.default_rng(125)
        mats = [
            np.ascontiguousarray(
                rng.standard_normal((16, ncols)) + 1j * rng.standard_normal((16, ncols)),
                dtype=np.complex128,
            )
            for ncols in range(1, RECT_COLS_MAX + 1)
        ]
        refs = [m.copy() for m in mats]

        circ.apply_to_list(mats, params)
        for ref in refs:
            circ.apply_to(params, ref)

        errs = [np.linalg.norm(got - expected) for got, expected in zip(mats, refs)]
        assert max(errs) < 1e-10, f"apply_to_list f64 16xN mismatch: {max(errs):.3e}"

    def test_apply_to_list_rectangular_sweep_f32(self):
        circ = self._build_circuit_q4()
        params = _params32(circ.get_Parameter_Num())

        rng = np.random.default_rng(125)
        mats = [
            np.ascontiguousarray(
                rng.standard_normal((16, ncols)) + 1j * rng.standard_normal((16, ncols)),
                dtype=np.complex64,
            )
            for ncols in range(1, RECT_COLS_MAX + 1)
        ]
        refs = [m.copy() for m in mats]

        circ.apply_to_list(mats, params, is_f32=True)
        for ref in refs:
            circ.apply_to(params, ref, is_f32=True)

        errs = [np.linalg.norm(got - expected) for got, expected in zip(mats, refs)]
        assert max(errs) < 5e-5, f"apply_to_list f32 16xN mismatch: {max(errs):.3e}"


class TestGateFusion:
    """Gate fusion via set_min_fusion: fused results must match unfused."""

    def _make_multi_gate_circuit(self, qbit_num):
        """Build a circuit with enough gates to trigger fusion."""
        c = Circuit(qbit_num)
        for q in range(min(qbit_num, 3)):
            c.add_RX(q)
            c.add_RZ(q)
        if qbit_num >= 2:
            c.add_CNOT(0, 1)
            c.add_RY(0)
        if qbit_num >= 3:
            c.add_CNOT(1, 2)
            c.add_RZ(2)
        return c

    @pytest.mark.parametrize("qbit_num,min_fusion", [
        (qn, mf) for qn in [3, 4, 5, 6] for mf in [2, 3, 4, 5, 6] if mf <= qn
    ])
    def test_fusion_unitary_matches_unfused(self, qbit_num, min_fusion):
        """Fused circuit unitary should match unfused for various min_fusion values."""

        circ_ref = self._make_multi_gate_circuit(qbit_num)
        circ_fused = self._make_multi_gate_circuit(qbit_num)
        circ_fused.set_min_fusion(min_fusion)

        params = _params64(circ_ref.get_Parameter_Num())
        dim = 2 ** qbit_num

        ref = _identity(dim)
        circ_ref.apply_to(params, ref)

        fused = _identity(dim)
        circ_fused.apply_to(params, fused)

        _assert_unitary_close(ref, fused, rtol=1e-9)

    @pytest.mark.parametrize("qbit_num", [3, 4, 5, 6])
    @pytest.mark.parametrize("min_fusion", [2, 3])
    def test_fusion_state_vector_matches_unfused(self, qbit_num, min_fusion):
        """Fused circuit applied to a state should match unfused result."""
        if min_fusion > qbit_num:
            pytest.skip("min_fusion > qbit_num — fusion won't activate")

        circ_ref = self._make_multi_gate_circuit(qbit_num)
        circ_fused = self._make_multi_gate_circuit(qbit_num)
        circ_fused.set_min_fusion(min_fusion)

        params = _params64(circ_ref.get_Parameter_Num())
        dim = 2 ** qbit_num
        state = _random_state(dim, dtype=np.complex128)

        sv_ref = state.reshape(dim, 1).copy()
        circ_ref.apply_to(params, sv_ref)

        sv_fused = state.reshape(dim, 1).copy()
        circ_fused.apply_to(params, sv_fused)

        _assert_state_close(sv_ref.reshape(-1), sv_fused.reshape(-1), rtol=1e-9)

    @pytest.mark.parametrize("min_fusion", [2, 3, 5])
    def test_fusion_large_circuit_avx_tbb_path(self, min_fusion):
        """
        With qbit_num=14 the fused path uses apply_large_kernel_to_input_AVX_TBB.
        Compare against an unfused circuit to verify correctness.
        """
        qbit_num = 14
        circ_ref = Circuit(qbit_num)
        circ_fused = Circuit(qbit_num)

        # Only involve qubits 0..1 so fusion kernel sees size==2 involved qubits
        for _ in range(4):
            circ_ref.add_RX(0)
            circ_ref.add_CNOT(0, 1)
            circ_ref.add_RZ(1)
            circ_fused.add_RX(0)
            circ_fused.add_CNOT(0, 1)
            circ_fused.add_RZ(1)

        circ_fused.set_min_fusion(min_fusion)

        params = _params64(circ_ref.get_Parameter_Num())
        dim = 2 ** qbit_num
        state = _random_state(dim, dtype=np.complex128, seed=100)

        sv_ref = state.reshape(dim, 1).copy()
        circ_ref.apply_to(params, sv_ref)

        sv_fused = state.reshape(dim, 1).copy()
        circ_fused.apply_to(params, sv_fused)

        _assert_state_close(sv_ref.reshape(-1), sv_fused.reshape(-1), rtol=1e-9)


class TestNestedCircuit:
    """Nested circuits (add_Circuit): sub-circuit applied inside outer circuit."""

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_nested_matches_flat(self, qbit_num):
        """Outer circuit containing a sub-circuit matches flat equivalent."""
        # Flat reference
        flat = Circuit(qbit_num)
        flat.add_RX(0)
        flat.add_CNOT(0, min(1, qbit_num - 1))
        flat.add_RZ(0)
        flat.add_H(0)

        # Sub-circuit with first two gates
        sub = Circuit(qbit_num)
        sub.add_RX(0)
        sub.add_CNOT(0, min(1, qbit_num - 1))

        # Outer circuit: sub + last two gates
        outer = Circuit(qbit_num)
        outer.add_Circuit(sub)
        outer.add_RZ(0)
        outer.add_H(0)

        params = _params64(flat.get_Parameter_Num())
        assert flat.get_Parameter_Num() == outer.get_Parameter_Num(), (
            "flat and nested circuits should have the same number of parameters"
        )

        dim = 2 ** qbit_num
        mat_flat = _identity(dim)
        flat.apply_to(params, mat_flat)

        mat_nested = _identity(dim)
        outer.apply_to(params, mat_nested)

        _assert_unitary_close(mat_flat, mat_nested)

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_nested_f32_matches_f64(self, qbit_num):
        """float32 apply on nested circuit matches float64 reference."""
        sub = Circuit(qbit_num)
        sub.add_RX(0)
        sub.add_CNOT(0, min(1, qbit_num - 1))

        outer = Circuit(qbit_num)
        outer.add_Circuit(sub)
        outer.add_RZ(0)

        p64 = _params64(outer.get_Parameter_Num())
        p32 = p64.astype(np.float32)

        dim = 2 ** qbit_num
        ref64 = _identity(dim)
        outer.apply_to(p64, ref64)

        mat32 = _identity(dim, dtype=np.complex64)
        outer.apply_to(p32, mat32, is_f32=True)

        _assert_unitary_close(ref64, mat32.astype(np.complex128), rtol=1e-4)

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_doubly_nested_matches_flat(self, qbit_num):
        """Two levels of nesting produce the same result as a flat circuit."""
        flat = Circuit(qbit_num)
        flat.add_H(0)
        flat.add_RX(0)
        flat.add_CNOT(0, min(1, qbit_num - 1))
        flat.add_RZ(0)

        inner = Circuit(qbit_num)
        inner.add_H(0)
        inner.add_RX(0)

        middle = Circuit(qbit_num)
        middle.add_Circuit(inner)
        middle.add_CNOT(0, min(1, qbit_num - 1))

        outer = Circuit(qbit_num)
        outer.add_Circuit(middle)
        outer.add_RZ(0)

        params = _params64(flat.get_Parameter_Num())
        assert flat.get_Parameter_Num() == outer.get_Parameter_Num()

        dim = 2 ** qbit_num
        mat_flat = _identity(dim)
        flat.apply_to(params, mat_flat)

        mat_outer = _identity(dim)
        outer.apply_to(params, mat_outer)

        _assert_unitary_close(mat_flat, mat_outer)


class TestCircuitMisc:
    """Miscellaneous circuit-level tests."""

    def test_parameter_num_consistent(self):
        """get_Parameter_Num matches sum of individual gate parameter counts."""
        circ = Circuit(3)
        circ.add_RX(0)     # 1 param
        circ.add_RZ(1)     # 1 param
        circ.add_U3(2)     # 3 params
        circ.add_CNOT(0, 2)  # 0 params
        assert circ.get_Parameter_Num() == 5

    def test_get_qbit_num(self):
        for n in [1, 2, 4, 8]:
            c = Circuit(n)
            assert c.get_Qbit_Num() == n

    def test_empty_circuit_is_identity_f64(self):
        """An empty circuit should act as identity on any matrix."""
        for qbit_num in [1, 2, 3, 4]:
            circ = Circuit(qbit_num)
            dim = 2 ** qbit_num
            mat = _identity(dim)
            circ.apply_to(np.array([], dtype=np.float64), mat)
            np.testing.assert_allclose(mat, np.eye(dim, dtype=np.complex128), atol=1e-12)

    def test_empty_circuit_is_identity_f32(self):
        """An empty circuit, float32 path, should act as identity."""
        for qbit_num in [1, 2, 3]:
            circ = Circuit(qbit_num)
            dim = 2 ** qbit_num
            mat = _identity(dim, dtype=np.complex64)
            circ.apply_to(np.array([], dtype=np.float32), mat, is_f32=True)
            np.testing.assert_allclose(mat, np.eye(dim, dtype=np.complex64), atol=1e-6)

    @pytest.mark.parametrize("qbit_num", [1, 2, 3, 4, 5, 6])
    def test_get_matrix_matches_apply_to_identity(self, qbit_num):
        """get_Matrix and apply_to on identity give the same result for all qubit counts."""
        circ = Circuit(qbit_num)
        circ.add_RX(0)
        circ.add_RZ(0)
        if qbit_num >= 2:
            circ.add_CNOT(0, 1)

        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        ref = np.array(circ.get_Matrix(params))
        mat = _identity(dim)
        circ.apply_to(params, mat)

        _assert_unitary_close(ref, mat)

    def test_non_contiguous_params_handled(self):
        """Non-contiguous parameter arrays should be handled correctly (no crash)."""
        circ = Circuit(2)
        circ.add_RX(0)
        circ.add_RZ(1)

        params_full = np.array([0.1, 0.0, 0.2], dtype=np.float64)
        params_strided = params_full[::2]   # non-contiguous: values 0.1, 0.2
        assert not params_strided.flags['C_CONTIGUOUS']

        mat = _identity(4)
        circ.apply_to(params_strided, mat)
        assert not np.allclose(mat, np.eye(4, dtype=np.complex128))

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_h_x_cnot_no_params_f64(self, qbit_num):
        """Circuits with only parameter-free gates work with empty param arrays."""
        circ = Circuit(qbit_num)
        circ.add_H(0)
        circ.add_X(0)
        if qbit_num >= 2:
            circ.add_CNOT(0, 1)

        params = np.array([], dtype=np.float64)
        dim = 2 ** qbit_num
        ref = np.array(circ.get_Matrix(params))
        mat = _identity(dim)
        circ.apply_to(params, mat)
        _assert_unitary_close(ref, mat)

    @pytest.mark.parametrize("qbit_num", QBIT_NUMS_ALL)
    def test_h_x_cnot_no_params_f32(self, qbit_num):
        """Parameter-free circuit float32 path produces correct result."""
        circ = Circuit(qbit_num)
        circ.add_H(0)
        circ.add_X(0)
        if qbit_num >= 2:
            circ.add_CNOT(0, 1)

        p64 = np.array([], dtype=np.float64)
        p32 = np.array([], dtype=np.float32)
        dim = 2 ** qbit_num

        ref = np.array(circ.get_Matrix(p64))
        mat32 = _identity(dim, dtype=np.complex64)
        circ.apply_to(p32, mat32, is_f32=True)

        _assert_unitary_close(ref, mat32.astype(np.complex128), rtol=1e-4)


class TestApplyToList:
    """apply_to_list: applies circuit to a list of matrices, all in float64."""

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_apply_to_list_matches_individual_apply_to(self, qbit_num):
        """apply_to_list result must equal calling apply_to on each matrix separately."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        matrices_list = [_identity(dim) for _ in range(5)]
        matrices_individual = [m.copy() for m in matrices_list]

        circ.apply_to_list(matrices_list, params)

        for m in matrices_individual:
            circ.apply_to(params, m)

        for got, expected in zip(matrices_list, matrices_individual):
            _assert_unitary_close(expected, got)

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_apply_to_list_single_element(self, qbit_num):
        """Single-element list should behave like apply_to."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        m_list = [_identity(dim)]
        m_single = _identity(dim)

        circ.apply_to_list(m_list, params)
        circ.apply_to(params, m_single)

        _assert_unitary_close(m_single, m_list[0])

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_apply_to_list_state_vectors(self, qbit_num):
        """apply_to_list works on state vector arrays (dim-column vectors)."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        states = [_random_state(dim, seed=i).reshape(dim, 1).copy()
                  for i in range(4)]
        references = [s.copy() for s in states]

        circ.apply_to_list(states, params)
        for ref in references:
            circ.apply_to(params, ref)

        for got, expected in zip(states, references):
            _assert_state_close(expected.reshape(-1), got.reshape(-1))

    def test_apply_to_list_empty_list_no_crash(self):
        """Empty list should not crash (no-op)."""
        circ = Circuit(2)
        circ.add_RX(0)
        params = _params64(circ.get_Parameter_Num())
        circ.apply_to_list([], params)  # should not raise

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_apply_to_list_wrong_dtype_raises(self, qbit_num):
        """Passing complex64 inputs to apply_to_list should raise a TypeError."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num
        bad = [_identity(dim, dtype=np.complex64)]
        with pytest.raises(Exception):
            circ.apply_to_list(bad, params)


class TestApplyDerivateTo:
    """apply_derivate_to: derivative of circuit w.r.t. each free parameter."""

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_derivative_count_matches_parameter_num(self, qbit_num):
        """Number of returned derivative matrices must equal get_Parameter_Num()."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim)
        derivs = circ.apply_derivate_to(params, mat)

        assert len(derivs) == circ.get_Parameter_Num()

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_derivative_shapes_match_input(self, qbit_num):
        """Each derivative matrix must have the same shape as the input."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim)
        derivs = circ.apply_derivate_to(params, mat)

        for i, d in enumerate(derivs):
            assert d.shape == (dim, dim), f"deriv[{i}] shape {d.shape} != ({dim},{dim})"

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_derivative_dtype_is_complex128(self, qbit_num):
        """Derivative matrices must be complex128."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim)
        derivs = circ.apply_derivate_to(params, mat)

        for i, d in enumerate(derivs):
            assert d.dtype == np.complex128, f"deriv[{i}] dtype {d.dtype} != complex128"

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_derivative_finite_difference_validation(self, qbit_num):
        """Analytic derivative must match finite-difference estimate."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        n_params = circ.get_Parameter_Num()
        params = _params64(n_params)
        dim = 2 ** qbit_num
        eps = 1e-5

        mat = _identity(dim)
        derivs = circ.apply_derivate_to(params, mat)

        for k in range(n_params):
            p_plus = params.copy(); p_plus[k] += eps
            p_minus = params.copy(); p_minus[k] -= eps
            U_plus = np.array(circ.get_Matrix(p_plus))
            U_minus = np.array(circ.get_Matrix(p_minus))
            fd = (U_plus - U_minus) / (2.0 * eps)
            err = np.linalg.norm(derivs[k] - fd)
            assert err < 1e-5, (
                f"Param {k}: analytic vs FD error = {err:.3e}"
            )

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_derivative_nonzero_for_parametric_circuit(self, qbit_num):
        """All derivative matrices must be non-zero for a parametric circuit."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim)
        derivs = circ.apply_derivate_to(params, mat)

        for i, d in enumerate(derivs):
            assert np.linalg.norm(d) > 1e-10, f"deriv[{i}] is unexpectedly zero"

    def test_no_params_returns_empty_list(self):
        """Parameter-free circuit should return an empty list of derivatives."""
        circ = Circuit(2)
        circ.add_H(0)
        circ.add_CNOT(0, 1)
        params = np.array([], dtype=np.float64)
        mat = _identity(4)
        derivs = circ.apply_derivate_to(params, mat)
        assert len(derivs) == 0

    def test_wrong_param_dtype_raises(self):
        """Passing float32 parameters to apply_derivate_to should raise."""
        circ = Circuit(2)
        circ.add_RX(0)
        params32 = np.array([0.3], dtype=np.float32)
        mat = _identity(4)
        with pytest.raises(Exception):
            circ.apply_derivate_to(params32, mat)


class TestApplyToCombined:
    """apply_to_combined: returns [forward_output, derivatives...]."""

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_combined_matches_apply_and_derivative_f64(self, qbit_num):
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params = _params64(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim)
        combined = circ.apply_to_combined(params, mat)

        assert isinstance(combined, list)
        assert len(combined) == circ.get_Parameter_Num() + 1

        expected_forward = _identity(dim)
        circ.apply_to(params, expected_forward)
        _assert_unitary_close(expected_forward, np.asarray(combined[0]))

        expected_derivs = circ.apply_derivate_to(params, mat)
        assert len(expected_derivs) == len(combined) - 1
        for idx, d in enumerate(expected_derivs):
            _assert_unitary_close(d, np.asarray(combined[idx + 1]))

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_combined_matches_apply_and_derivative_f32(self, qbit_num):
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        params32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat32 = _identity(dim, dtype=np.complex64)
        combined = circ.apply_to_combined(params32, mat32, is_f32=True)

        assert isinstance(combined, list)
        assert len(combined) == circ.get_Parameter_Num() + 1
        for arr in combined:
            assert np.asarray(arr).dtype == np.complex64

        expected_forward = _identity(dim, dtype=np.complex64)
        circ.apply_to(params32, expected_forward, is_f32=True)
        _assert_unitary_close(expected_forward.astype(np.complex128), np.asarray(combined[0]).astype(np.complex128), rtol=1e-4)

        expected_derivs = circ.apply_derivate_to(params32, mat32, is_f32=True)
        assert len(expected_derivs) == len(combined) - 1
        for idx, d in enumerate(expected_derivs):
            _assert_unitary_close(d.astype(np.complex128), np.asarray(combined[idx + 1]).astype(np.complex128), rtol=1e-4)

    def test_combined_no_params_returns_forward_only(self):
        circ = Circuit(2)
        circ.add_H(0)
        circ.add_CNOT(0, 1)
        params = np.array([], dtype=np.float64)

        mat = _identity(4)
        combined = circ.apply_to_combined(params, mat)

        assert len(combined) == 1
        expected_forward = _identity(4)
        circ.apply_to(params, expected_forward)
        _assert_unitary_close(expected_forward, np.asarray(combined[0]))


# ---------------------------------------------------------------------------
# Float32 apply_to_list
# ---------------------------------------------------------------------------

class TestApplyToListF32:
    """apply_to_list is_f32=True: float32/complex64 batch path."""

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_f32_matches_f64_reference(self, qbit_num):
        """f32 apply_to_list result must be close to f64 result."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p64 = _params64(circ.get_Parameter_Num())
        p32 = p64.astype(np.float32)
        dim = 2 ** qbit_num

        matrices_f64 = [_identity(dim) for _ in range(4)]
        matrices_f32 = [_identity(dim, dtype=np.complex64) for _ in range(4)]

        circ.apply_to_list(matrices_f64, p64)
        circ.apply_to_list(matrices_f32, p32, is_f32=True)

        for got, expected in zip(matrices_f32, matrices_f64):
            _assert_unitary_close(expected, got.astype(np.complex128), rtol=1e-4)

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_f32_single_element_matches_apply_to_f32(self, qbit_num):
        """Single-element f32 list should match apply_to with is_f32=True."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        m_list = [_identity(dim, dtype=np.complex64)]
        m_single = _identity(dim, dtype=np.complex64)

        circ.apply_to_list(m_list, p32, is_f32=True)
        circ.apply_to(p32, m_single, is_f32=True)

        _assert_unitary_close(m_single.astype(np.complex128),
                              m_list[0].astype(np.complex128), rtol=1e-4)

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_f32_dtype_preserved(self, qbit_num):
        """Output dtype should remain complex64 after f32 apply_to_list."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        matrices = [_identity(dim, dtype=np.complex64) for _ in range(3)]
        circ.apply_to_list(matrices, p32, is_f32=True)

        for m in matrices:
            assert m.dtype == np.complex64

    def test_f32_empty_list_no_crash(self):
        """Empty f32 list should not crash."""
        circ = Circuit(2)
        circ.add_RX(0)
        p32 = _params32(circ.get_Parameter_Num())
        circ.apply_to_list([], p32, is_f32=True)

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_f32_wrong_dtype_raises(self, qbit_num):
        """Passing complex128 inputs with is_f32=True should raise."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num
        bad = [_identity(dim, dtype=np.complex128)]
        with pytest.raises(Exception):
            circ.apply_to_list(bad, p32, is_f32=True)


# ---------------------------------------------------------------------------
# Float32 apply_derivate_to
# ---------------------------------------------------------------------------

class TestApplyDerivateToF32:
    """apply_derivate_to is_f32=True: float32/complex64 derivative path."""

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_f32_derivative_count_matches_parameter_num(self, qbit_num):
        """f32 derivative list length must equal get_Parameter_Num()."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim, dtype=np.complex64)
        derivs = circ.apply_derivate_to(p32, mat, is_f32=True)

        assert len(derivs) == circ.get_Parameter_Num()

    @pytest.mark.parametrize("qbit_num", [2, 3, 4])
    def test_f32_derivative_shapes_match_input(self, qbit_num):
        """Each f32 derivative matrix must have the same shape as the input."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim, dtype=np.complex64)
        derivs = circ.apply_derivate_to(p32, mat, is_f32=True)

        for i, d in enumerate(derivs):
            assert d.shape == (dim, dim), f"deriv[{i}] shape {d.shape} != ({dim},{dim})"

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_f32_derivative_dtype_is_complex64(self, qbit_num):
        """f32 derivative matrices must be complex64."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim, dtype=np.complex64)
        derivs = circ.apply_derivate_to(p32, mat, is_f32=True)

        for i, d in enumerate(derivs):
            assert d.dtype == np.complex64, f"deriv[{i}] dtype {d.dtype} != complex64"

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_f32_derivative_close_to_f64_reference(self, qbit_num):
        """f32 derivatives must be close to f64 reference derivatives."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p64 = _params64(circ.get_Parameter_Num())
        p32 = p64.astype(np.float32)
        dim = 2 ** qbit_num

        mat64 = _identity(dim)
        mat32 = _identity(dim, dtype=np.complex64)

        derivs64 = circ.apply_derivate_to(p64, mat64)
        derivs32 = circ.apply_derivate_to(p32, mat32, is_f32=True)

        assert len(derivs32) == len(derivs64)
        for i, (d32, d64) in enumerate(zip(derivs32, derivs64)):
            err = np.linalg.norm(d32.astype(np.complex128) - d64)
            assert err < 1e-3, f"deriv[{i}]: f32 vs f64 error = {err:.3e}"

    @pytest.mark.parametrize("qbit_num", [2, 3])
    def test_f32_derivative_nonzero_for_parametric_circuit(self, qbit_num):
        """All f32 derivative matrices must be non-zero for a parametric circuit."""
        circ = _build_rx_rz_cnot_circuit(qbit_num)
        p32 = _params32(circ.get_Parameter_Num())
        dim = 2 ** qbit_num

        mat = _identity(dim, dtype=np.complex64)
        derivs = circ.apply_derivate_to(p32, mat, is_f32=True)

        for i, d in enumerate(derivs):
            assert np.linalg.norm(d) > 1e-7, f"f32 deriv[{i}] is unexpectedly zero"

    def test_f32_no_params_returns_empty_list(self):
        """f32 parameter-free circuit should return empty derivative list."""
        circ = Circuit(2)
        circ.add_H(0)
        circ.add_CNOT(0, 1)
        params = np.array([], dtype=np.float32)
        mat = _identity(4, dtype=np.complex64)
        derivs = circ.apply_derivate_to(params, mat, is_f32=True)
        assert len(derivs) == 0

    def test_f32_wrong_param_dtype_raises(self):
        """Passing float64 params with is_f32=True should raise."""
        circ = Circuit(2)
        circ.add_RX(0)
        params64 = np.array([0.3], dtype=np.float64)
        mat = _identity(4, dtype=np.complex64)
        with pytest.raises(Exception):
            circ.apply_derivate_to(params64, mat, is_f32=True)

    def test_f32_wrong_matrix_dtype_raises(self):
        """Passing complex128 matrix with is_f32=True should raise."""
        circ = Circuit(2)
        circ.add_RX(0)
        params32 = np.array([0.3], dtype=np.float32)
        mat64 = _identity(4, dtype=np.complex128)
        with pytest.raises(Exception):
            circ.apply_derivate_to(params32, mat64, is_f32=True)
