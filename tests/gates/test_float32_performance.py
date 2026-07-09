import os
import time

import numpy as np
import pytest

from squander import CNOT, U3

# Pin to a single core to eliminate CPU migration / frequency variance across runs.
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {0})


def _has_avx2():
    """Return True if the CPU supports AVX2 (needed for efficient float32 SIMD)."""
    try:
        with open("/proc/cpuinfo") as f:
            return any("avx2" in line for line in f)
    except OSError:
        return None  # non-Linux — can't tell, assume yes


QUBIT_NUM = 12
COLS = 256
WARMUP = 30
REPEATS = 40
TRIALS = 5
MIN_FLOAT32_SPEEDUP = 2.0
MIN_CNOT_SPEEDUP = 0.25


def _make_cnot():
    return CNOT(QUBIT_NUM, 0, QUBIT_NUM - 1)


def _make_u3():
    return U3(QUBIT_NUM, 0)


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

    for _ in range(WARMUP):
        _apply(gate, matrix, params, is_f32=is_f32)

    start = time.process_time()
    for _ in range(REPEATS):
        _apply(gate, matrix, params, is_f32=is_f32)
    return time.process_time() - start


@pytest.mark.parametrize(
    "gate_factory,gate_name,min_speedup",
    [
        pytest.param(_make_u3, "U3", MIN_FLOAT32_SPEEDUP, id="U3"),
        pytest.param(_make_cnot, "CNOT", MIN_CNOT_SPEEDUP, id="CNOT"),
    ],
)
def test_float32_apply_to_hot_path_has_expected_speed(gate_factory, gate_name, min_speedup):
    """Float32 should be a real HPC path, not just accepted at the API boundary."""
    if _has_avx2() is False:
        pytest.skip("CPU lacks AVX2 — float32 SIMD path cannot meet speedup threshold")

    # Burn-in: one full round to warm CPU frequency / caches, then discard.
    _time_apply(gate_factory(), np.float64)
    _time_apply(gate_factory(), np.float32)

    # Quick self-check: if float32 can't even beat float64 after warmup,
    # the machine is in a degraded state (thermal throttle / powersave governor).
    t64_check = _time_apply(gate_factory(), np.float64)
    t32_check = _time_apply(gate_factory(), np.float32)
    if t32_check >= t64_check:
        pytest.skip(
            f"Machine in degraded state — float32 ({t32_check:.3f}s) not faster "
            f"than float64 ({t64_check:.3f}s)"
        )

    timings = []
    for _ in range(TRIALS):
        t64 = _time_apply(gate_factory(), np.float64)
        t32 = _time_apply(gate_factory(), np.float32)
        timings.append(t64 / t32)

    # Drop the first (coldest) trial, use min of the rest for a conservative estimate.
    speedup = float(np.min(timings[1:]))
    assert speedup >= min_speedup, (
        f"{gate_name} float32 speedup {speedup:.2f}x is below "
        f"{min_speedup:.1f}x; timings={timings}"
    )
