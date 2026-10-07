#!/usr/bin/env python3
"""Run the M-F5a task-1 E-VQE 4-qubit interop tracer protocol."""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

import squander.VQA.qgd_Variational_Quantum_Eigensolver_Base_Wrapper as vqe_wrapper_ext
from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base as VariationalQuantumEigensolver,
)
from tests.VQE.test_VQE import generate_hamiltonian

COUNTED_PAIRS = 1000
WARMUP_PAIRS = 50
QBIT_NUM = 4
THROUGHPUT_DIVISOR = 3072
Z_95 = 1.644854

DENSITY_NOISE = [
    {
        "channel": "local_depolarizing",
        "target": 0,
        "after_gate_index": 0,
        "error_rate": 0.1,
    },
    {
        "channel": "amplitude_damping",
        "target": 1,
        "after_gate_index": 2,
        "gamma": 0.05,
    },
    {
        "channel": "phase_damping",
        "target": 0,
        "after_gate_index": 4,
        "lambda": 0.07,
    },
]

CONFIG = {
    "max_inner_iterations": 4,
    "max_iterations": 1,
    "convergence_length": 2,
}


def _required_thread_env() -> dict[str, str]:
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {key: os.environ.get(key, "") for key in keys}


def _pin_lowest_allowed_cpu() -> int:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("sched_getaffinity is required for the interop lane")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("process CPU affinity mask is empty")
    cpu_id = allowed[0]
    os.sched_setaffinity(0, {cpu_id})
    return cpu_id


def _assert_perf_counter_clock() -> str:
    implementation = time.get_clock_info("perf_counter").implementation
    if implementation != "clock_gettime(CLOCK_MONOTONIC)":
        raise RuntimeError(
            "perf_counter must use clock_gettime(CLOCK_MONOTONIC); "
            f"got {implementation!r}"
        )
    return implementation


def build_task1_evaluator() -> VariationalQuantumEigensolver:
    topology = [(idx, idx + 1) for idx in range(QBIT_NUM - 1)]
    hamiltonian = generate_hamiltonian(topology, QBIT_NUM)
    vqe = VariationalQuantumEigensolver(
        hamiltonian,
        QBIT_NUM,
        CONFIG,
        backend="density_matrix",
        density_noise=DENSITY_NOISE,
    )
    vqe.set_Ansatz("HEA")
    vqe.Generate_Circuit(1, 1)
    return vqe


def _one_sided_upper_bound(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    bound = mean + Z_95 * std / math.sqrt(arr.size)
    return mean, bound


def _run_pair(
    vqe: VariationalQuantumEigensolver,
    parameters: np.ndarray,
    pair_index: int,
) -> dict[str, Any]:
    public_first = pair_index % 2 == 0

    def public_call() -> int:
        start = time.perf_counter_ns()
        vqe.Optimization_Problem(parameters)
        return time.perf_counter_ns() - start

    def lower_call() -> tuple[int, tuple[int, ...]]:
        t_lower = vqe_wrapper_ext.harness_density_lower_ns(vqe, parameters)
        subtimes = vqe_wrapper_ext.harness_density_subtimes_ns(vqe)
        return t_lower, subtimes

    if public_first:
        t_public = public_call()
        t_lower, subtimes = lower_call()
    else:
        t_lower, subtimes = lower_call()
        t_public = public_call()

    return {
        "t_public_ns": int(t_public),
        "t_lower_ns": int(t_lower),
        "subtimes_ns": [int(value) for value in subtimes],
    }


def run_interop_row(
    *,
    counted_pairs: int = COUNTED_PAIRS,
    warmup_pairs: int = WARMUP_PAIRS,
) -> dict[str, Any]:
    clock_impl = _assert_perf_counter_clock()
    cpu_id = _pin_lowest_allowed_cpu()
    thread_env = _required_thread_env()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if thread_env.get(key, "") != "1":
            raise RuntimeError(f"{key} must be 1 for the interop lane")

    vqe = build_task1_evaluator()
    param_num = vqe.get_Parameter_Num()
    parameters = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
    bridge = vqe.describe_density_bridge()
    operation_count = int(bridge["operation_count"])

    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, True)

    for pair_idx in range(warmup_pairs):
        _run_pair(vqe, parameters, pair_idx)

    samples: list[dict[str, Any]] = []
    o_values: list[float] = []
    throughput_values: list[float] = []

    for pair_idx in range(warmup_pairs, warmup_pairs + counted_pairs):
        sample = _run_pair(vqe, parameters, pair_idx)
        t_public = sample["t_public_ns"]
        t_lower = sample["t_lower_ns"]
        if t_public <= 0:
            raise RuntimeError("non-positive T_public in counted pair")
        o_i = (t_public - t_lower) / t_public
        if not math.isfinite(o_i):
            raise RuntimeError("non-finite overhead sample")
        o_values.append(o_i)
        apply_to_ns = sample["subtimes_ns"][3]
        throughput_values.append(apply_to_ns / THROUGHPUT_DIVISOR)
        samples.append(sample)

    mean_o, bound_o = _one_sided_upper_bound(o_values)
    median_o = float(np.median(np.asarray(o_values, dtype=np.float64)))
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)

    return {
        "suite": "interop_profile_task1_evqe_4q_v1",
        "qbit_num": QBIT_NUM,
        "warmup_pairs": warmup_pairs,
        "counted_pairs": counted_pairs,
        "milestone_counted": False,
        "clean_start": True,
        "claim_boundary": "task-1 tracer row; milestone_counted=false",
        "labels": "E-VQE density_matrix harness tracer",
        "operation_count": operation_count,
        "throughput_divisor": THROUGHPUT_DIVISOR,
        "perf_counter_implementation": clock_impl,
        "affinity_cpu": cpu_id,
        "thread_env": thread_env,
        "overhead": {
            "mean_O": mean_o,
            "median_O": median_o,
            "upper_bound_95_O": bound_o,
        },
        "throughput": {
            "divisor": THROUGHPUT_DIVISOR,
            "mean_ns_per_op": mean_tp,
            "upper_bound_95_ns_per_op": bound_tp,
        },
        "samples": samples,
    }
