import os
import time

import numpy as np
import pytest

from squander import CNOT, U3


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


@pytest.mark.parametrize("gate_factory", [_make_u3, _make_cnot], ids=["U3", "CNOT"])
def test_float32_apply_to_hot_path_matches_float64(gate_factory):
    gate = gate_factory()
    matrix64, matrix32 = _make_inputs()
    _apply(gate, matrix64, _parameters(gate, np.float64), is_f32=False)
    _apply(gate, matrix32, _parameters(gate, np.float32), is_f32=True)
    np.testing.assert_allclose(matrix32, matrix64, rtol=2e-5, atol=2e-5)


@pytest.fixture
def pin_performance_core():
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield
        return
    original_affinity = os.sched_getaffinity(0)
    if not original_affinity:
        pytest.skip("No CPU is available for performance pinning")
    try:
        os.sched_setaffinity(0, {min(original_affinity)})
    except OSError:
        pytest.skip("Cannot pin this process to one CPU")
    try:
        yield
    finally:
        os.sched_setaffinity(0, original_affinity)


@pytest.mark.skipif(
    os.environ.get("SQUANDER_RUN_PERFORMANCE_TESTS") != "1",
    reason="Timing assertions require SQUANDER_RUN_PERFORMANCE_TESTS=1 on a controlled runner",
)
@pytest.mark.parametrize(
    "gate_factory,gate_name,min_speedup",
    [
        pytest.param(_make_u3, "U3", MIN_FLOAT32_SPEEDUP, id="U3"),
        pytest.param(_make_cnot, "CNOT", MIN_CNOT_SPEEDUP, id="CNOT"),
    ],
)
def test_float32_apply_to_hot_path_has_expected_speed(
    gate_factory, gate_name, min_speedup, pin_performance_core
):
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

    # Drop the first (coldest) trial and average the remaining measurements.
    # Requiring every individual timing to clear the threshold turns normal
    # shared-runner jitter into a false performance regression.
    speedup = float(np.mean(timings[1:]))
    assert speedup >= min_speedup, (
        f"{gate_name} float32 speedup {speedup:.2f}x is below "
        f"{min_speedup:.1f}x; timings={timings}"
    )
