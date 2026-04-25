"""
test_scalar_paths.py — Direct in-process tests of all get_matrix and apply_* methods
with parallel=0, exercising the scalar (non-AVX) code paths in both f64 and f32.

These tests run without spawning subprocesses so they catch linker/compile regressions
immediately and also serve as the canonical non-AVX verification suite.

Coverage per gate:
  * get_matrix (f64 and f32)
  * apply_to — zero-param state vector        (f64 and f32)
  * apply_to — parametric state vector        (f64 and f32, parametric gates only)
  * apply_to — zero-param full unitary        (f64 and f32)
  * apply_to — parametric full unitary        (f64 and f32, parametric gates only)
  * apply_from_right — zero-param             (f64 and f32)
  * apply_from_right — parametric             (f64 and f32, parametric gates only)
"""

import numpy as np
import pytest

from tests.gates.test_gates import (
    ALL_GATE_NAMES,
    _instantiate_gate,
    _parameters_for_gate,
)

# Tolerance for float32 vs float64 parity
F32_TOL = 2e-4
# Gates excluded from matrix/apply tests (abstract base)
SKIP_GATES = {"Gate"}


def _f64_state(n=4):
    """Random normalized complex128 state vector of dimension 2^n."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    return (v / np.linalg.norm(v)).astype(np.complex128)


def _f64_unitary(n=4):
    """Random complex128 matrix of shape (2^n, 2^n)."""
    rng = np.random.default_rng(7)
    m = rng.standard_normal((1 << n, 1 << n)) + 1j * rng.standard_normal((1 << n, 1 << n))
    return m.astype(np.complex128)


# Pre-allocate test inputs once
_STATE64 = _f64_state()
_UNITARY64 = _f64_unitary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_matrix_f64(gate_obj, p64):
    if gate_obj.get_Parameter_Num() == 0:
        return np.asarray(gate_obj.get_Matrix(is_f32=False))
    return np.asarray(gate_obj.get_Matrix(p64, is_f32=False))


def _get_matrix_f32(gate_obj, p32):
    if gate_obj.get_Parameter_Num() == 0:
        return np.asarray(gate_obj.get_Matrix(is_f32=True))
    return np.asarray(gate_obj.get_Matrix(p32, is_f32=True))


def _apply_to_state_f64(gate_obj, p64):
    state = _STATE64.copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(state, parallel=0, is_f32=False)
    else:
        gate_obj.apply_to(state, parameters=p64, parallel=0, is_f32=False)
    return state


def _apply_to_state_f32(gate_obj, p32):
    state = _STATE64.astype(np.complex64).copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(state, parallel=0, is_f32=True)
    else:
        gate_obj.apply_to(state, parameters=p32, parallel=0, is_f32=True)
    return state


def _apply_to_unitary_f64(gate_obj, p64):
    mat = _UNITARY64.copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(mat, parallel=0, is_f32=False)
    else:
        gate_obj.apply_to(mat, parameters=p64, parallel=0, is_f32=False)
    return mat


def _apply_to_unitary_f32(gate_obj, p32):
    mat = _UNITARY64.astype(np.complex64).copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_to(mat, parallel=0, is_f32=True)
    else:
        gate_obj.apply_to(mat, parameters=p32, parallel=0, is_f32=True)
    return mat


def _apply_from_right_f64(gate_obj, p64):
    mat = _UNITARY64.copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_from_right(mat, is_f32=False)
    else:
        gate_obj.apply_from_right(mat, parameters=p64, is_f32=False)
    return mat


def _apply_from_right_f32(gate_obj, p32):
    mat = _UNITARY64.astype(np.complex64).copy()
    if gate_obj.get_Parameter_Num() == 0:
        gate_obj.apply_from_right(mat, is_f32=True)
    else:
        gate_obj.apply_from_right(mat, parameters=p32, is_f32=True)
    return mat


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------

TESTABLE_GATES = [g for g in ALL_GATE_NAMES if g not in SKIP_GATES]


# ---------------------------------------------------------------------------
# get_matrix — f64 self-consistency (M is unitary: M† M ≈ I)
# ---------------------------------------------------------------------------

class TestGetMatrixF64:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_get_matrix_f64_is_unitary(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        m = _get_matrix_f64(g, p64)
        assert m.dtype == np.complex128, f"{gate_name}: expected complex128, got {m.dtype}"
        n = m.shape[0]
        product = m.conj().T @ m
        err = np.linalg.norm(product - np.eye(n))
        assert err < 1e-8, f"{gate_name}: get_matrix(f64) not unitary, err={err:.3e}"


# ---------------------------------------------------------------------------
# get_matrix — f32 vs f64 parity
# ---------------------------------------------------------------------------

class TestGetMatrixF32Parity:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_get_matrix_f32_dtype_and_parity(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        p32 = p64.astype(np.float32)
        m64 = _get_matrix_f64(g, p64)
        m32 = _get_matrix_f32(g, p32)
        assert m32.dtype == np.complex64, f"{gate_name}: expected complex64, got {m32.dtype}"
        err = np.linalg.norm(m32 - m64.astype(np.complex64))
        assert err < F32_TOL, f"{gate_name}: get_matrix f32/f64 parity err={err:.3e}"


# ---------------------------------------------------------------------------
# apply_to — state vector — f64 vs get_matrix
# ---------------------------------------------------------------------------

class TestApplyToStateF64:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_apply_to_state_f64_matches_matmul(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        m = _get_matrix_f64(g, p64)
        expected = m @ _STATE64
        result = _apply_to_state_f64(g, p64)
        err = np.linalg.norm(result - expected)
        assert err < 1e-8, f"{gate_name}: apply_to state f64 vs matmul err={err:.3e}"


# ---------------------------------------------------------------------------
# apply_to — state vector — f32 vs f64 parity
# ---------------------------------------------------------------------------

class TestApplyToStateF32Parity:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_apply_to_state_f32_parity(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        p32 = p64.astype(np.float32)
        ref64 = _apply_to_state_f64(g, p64)
        result32 = _apply_to_state_f32(g, p32)
        err = np.linalg.norm(result32 - ref64.astype(np.complex64))
        assert err < F32_TOL, f"{gate_name}: apply_to state f32/f64 parity err={err:.3e}"


# ---------------------------------------------------------------------------
# apply_to — full unitary — f64 vs get_matrix
# ---------------------------------------------------------------------------

class TestApplyToUnitaryF64:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_apply_to_unitary_f64_matches_matmul(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        m = _get_matrix_f64(g, p64)
        expected = m @ _UNITARY64
        result = _apply_to_unitary_f64(g, p64)
        err = np.linalg.norm(result - expected)
        assert err < 1e-8, f"{gate_name}: apply_to unitary f64 vs matmul err={err:.3e}"


# ---------------------------------------------------------------------------
# apply_to — full unitary — f32 vs f64 parity
# ---------------------------------------------------------------------------

class TestApplyToUnitaryF32Parity:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_apply_to_unitary_f32_parity(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        p32 = p64.astype(np.float32)
        ref64 = _apply_to_unitary_f64(g, p64)
        result32 = _apply_to_unitary_f32(g, p32)
        err = np.linalg.norm(result32 - ref64.astype(np.complex64))
        assert err < F32_TOL, f"{gate_name}: apply_to unitary f32/f64 parity err={err:.3e}"


# ---------------------------------------------------------------------------
# apply_from_right — f64 vs get_matrix
# ---------------------------------------------------------------------------

class TestApplyFromRightF64:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_apply_from_right_f64_matches_matmul(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        m = _get_matrix_f64(g, p64)
        expected = _UNITARY64 @ m
        result = _apply_from_right_f64(g, p64)
        err = np.linalg.norm(result - expected)
        assert err < 1e-8, f"{gate_name}: apply_from_right f64 vs matmul err={err:.3e}"


# ---------------------------------------------------------------------------
# apply_from_right — f32 vs f64 parity
# ---------------------------------------------------------------------------

class TestApplyFromRightF32Parity:
    @pytest.mark.parametrize("gate_name", TESTABLE_GATES)
    def test_apply_from_right_f32_parity(self, gate_name):
        g = _instantiate_gate(gate_name)
        p64 = _parameters_for_gate(g, dtype=np.float64)
        p32 = p64.astype(np.float32)
        ref64 = _apply_from_right_f64(g, p64)
        result32 = _apply_from_right_f32(g, p32)
        err = np.linalg.norm(result32 - ref64.astype(np.complex64))
        assert err < F32_TOL, f"{gate_name}: apply_from_right f32/f64 parity err={err:.3e}"
