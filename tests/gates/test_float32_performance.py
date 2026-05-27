import time

import numpy as np
import pytest

from squander import CNOT, U3


QUBIT_NUM = 12
COLS = 256
REPEATS = 40
MIN_FLOAT32_SPEEDUP = 2.0


def _parameters(gate, dtype):
    pnum = gate.get_Parameter_Num()
    if pnum == 0:
        return np.asarray([], dtype=dtype)
    return np.linspace(0.1, 0.1 * pnum, pnum, dtype=dtype)


def _make_inputs():
    rng = np.random.default_rng(20260527)
    data64 = np.ascontiguousarray(
        rng.standard_normal((1 << QUBIT_NUM, COLS))
        + 1j * rng.standard_normal((1 << QUBIT_NUM, COLS)),
        dtype=np.complex128,
    )
    return data64, data64.astype(np.complex64)


def _apply(gate, matrix, params, is_f32):
    if gate.get_Parameter_Num() == 0:
        gate.apply_to(matrix, parallel=0, is_f32=is_f32)
    else:
        gate.apply_to(matrix, parameters=params, parallel=0, is_f32=is_f32)


def _time_apply(gate, dtype):
    matrix64, matrix32 = _make_inputs()
    is_f32 = dtype == np.float32
    matrix = matrix32 if is_f32 else matrix64
    params = _parameters(gate, dtype)

    for _ in range(5):
        _apply(gate, matrix, params, is_f32=is_f32)

    start = time.perf_counter()
    for _ in range(REPEATS):
        _apply(gate, matrix, params, is_f32=is_f32)
    return time.perf_counter() - start


@pytest.mark.parametrize(
    "gate",
    [
        CNOT(QUBIT_NUM, 0, QUBIT_NUM - 1),
        U3(QUBIT_NUM, 0),
    ],
    ids=["CNOT", "U3"],
)
def test_float32_apply_to_hot_path_is_at_least_2x_faster(gate):
    """Float32 should be a real HPC path, not just accepted at the API boundary."""
    timings = []
    for _ in range(3):
        t64 = _time_apply(gate, np.float64)
        t32 = _time_apply(gate, np.float32)
        timings.append(t64 / t32)

    speedup = float(np.median(timings))
    assert speedup >= MIN_FLOAT32_SPEEDUP, (
        f"{gate.get_Name()} float32 speedup {speedup:.2f}x is below "
        f"{MIN_FLOAT32_SPEEDUP:.1f}x; timings={timings}"
    )
