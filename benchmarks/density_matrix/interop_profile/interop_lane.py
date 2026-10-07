#!/usr/bin/env python3
"""Run the M-F5a task-1 E-VQE 4-qubit interop tracer protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy

import squander.VQA.qgd_Variational_Quantum_Eigensolver_Base_Wrapper as vqe_wrapper_ext
from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base as VariationalQuantumEigensolver,
)
from tests.VQE.test_VQE import generate_hamiltonian

REPO_ROOT = Path(__file__).resolve().parents[3]

COUNTED_PAIRS = 1000
WARMUP_PAIRS = 50
QBIT_NUM = 4
THROUGHPUT_DIVISOR = 3072
Z_95 = 1.644854
QA008_MEAN_O_ABSOLUTE_MARGIN = 0.02

REGENERATION_COMMAND = (
    "conda run -n qgd --no-capture-output python "
    "benchmarks/density_matrix/interop_profile/validation_pipeline.py"
)

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, str]:
    try:
        display_path = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_path = str(path.resolve())
    return {"path": display_path, "sha256": _sha256_file(path)}


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_cpu_model() -> str:
    try:
        with (Path("/proc/cpuinfo")).open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _compiler_record() -> dict[str, str]:
    cxx = os.environ.get("CXX", "c++")
    try:
        version_line = subprocess.run(
            (cxx, "--version"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        version_line = "unknown"
    return {
        "executable": cxx,
        "version_line": version_line,
        "build_profile": "Release extension via setup.py build_ext (qgd env)",
    }


def _hamiltonian_csr_sha256(hamiltonian: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(hamiltonian.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(hamiltonian.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(hamiltonian.data).tobytes())
    return digest.hexdigest()


def capture_provenance() -> dict[str, Any]:
    """Capture checkout and runtime identity before the counted bundle is written."""
    revision = _git_output("rev-parse", "HEAD")
    status_lines = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty_paths = sorted(line[3:] for line in status_lines if len(line) > 3)

    import squander

    libqgd_path = Path(squander.__file__).resolve().parent / "libqgd.so"
    wrapper_path = Path(vqe_wrapper_ext.__file__).resolve()

    environment = {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "host": platform.node(),
    }
    dependencies = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "squander": metadata.version("squander"),
    }
    extension_identities = []
    if libqgd_path.is_file():
        extension_identities.append(_artifact_identity(libqgd_path))
    if wrapper_path.is_file():
        extension_identities.append(_artifact_identity(wrapper_path))

    clean_start = not dirty_paths
    complete = bool(
        revision
        and environment["conda_default_env"] == "qgd"
        and environment["conda_prefix"]
        and environment["python_executable"]
        and environment["host"]
        and len(extension_identities) == 2
    )
    return {
        "implementation_revision": revision,
        "clean_start": clean_start,
        "dirty_paths": dirty_paths,
        "command": REGENERATION_COMMAND,
        "host": environment["host"],
        "cpu_model": _read_cpu_model(),
        "compiler": _compiler_record(),
        "environment": environment,
        "dependencies": dependencies,
        "extension_identities": extension_identities,
        "provenance_pass": bool(clean_start and complete),
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


def build_task1_evaluator() -> tuple[VariationalQuantumEigensolver, Any]:
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
    return vqe, hamiltonian


def _one_sided_upper_bound(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    bound = mean + Z_95 * std / math.sqrt(arr.size)
    return mean, bound


def _sample_components(sample: Mapping[str, Any]) -> dict[str, int]:
    t_public = int(sample["t_public_ns"])
    t_lower = int(sample["t_lower_ns"])
    subtimes = sample["subtimes_ns"]
    support_outer, construct, lowering, apply_to, contraction, teardown = (
        int(subtimes[0]),
        int(subtimes[1]),
        int(subtimes[2]),
        int(subtimes[3]),
        int(subtimes[4]),
        int(subtimes[5]),
    )
    allocate_build = support_outer + construct + lowering + teardown
    return {
        "wrapper_ns": t_public - t_lower,
        "allocate_build_ns": allocate_build,
        "apply_to_ns": apply_to,
        "contraction_ns": contraction,
    }


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

    sample = {
        "t_public_ns": int(t_public),
        "t_lower_ns": int(t_lower),
        "subtimes_ns": [int(value) for value in subtimes],
    }
    sample["components_ns"] = _sample_components(sample)
    return sample


def run_interop_row(
    *,
    counted_pairs: int = COUNTED_PAIRS,
    warmup_pairs: int = WARMUP_PAIRS,
) -> dict[str, Any]:
    provenance = capture_provenance()

    clock_impl = _assert_perf_counter_clock()
    cpu_id = _pin_lowest_allowed_cpu()
    thread_env = _required_thread_env()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if thread_env.get(key, "") != "1":
            raise RuntimeError(f"{key} must be 1 for the interop lane")

    vqe, hamiltonian = build_task1_evaluator()
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
    wrapper_values: list[float] = []
    allocate_values: list[float] = []
    apply_values: list[float] = []
    contraction_values: list[float] = []

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
        components = sample["components_ns"]
        wrapper_values.append(components["wrapper_ns"])
        allocate_values.append(components["allocate_build_ns"])
        apply_values.append(components["apply_to_ns"])
        contraction_values.append(components["contraction_ns"])
        throughput_values.append(components["apply_to_ns"] / THROUGHPUT_DIVISOR)
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
        "clean_start": provenance["clean_start"],
        "provenance": provenance,
        "claim_boundary": "task-1 tracer row; milestone_counted=false",
        "labels": "E-VQE density_matrix harness tracer",
        "harness_timer_flag": True,
        "operation_count": operation_count,
        "throughput_divisor": THROUGHPUT_DIVISOR,
        "perf_counter_implementation": clock_impl,
        "affinity_cpu": cpu_id,
        "thread_env": thread_env,
        "protocol": {
            "pairing": "paired_not_interleaved",
            "public_first_on_even_index": True,
            "warmup_pairs": warmup_pairs,
            "counted_pairs": counted_pairs,
            "parameter_rule": "linspace(0.05, 0.05 * parameter_count, parameter_count)",
            "parameter_count": int(param_num),
        },
        "estimator": {
            "name": "arithmetic_mean_O_i",
            "formula": "O_i = (T_public_i - T_lower_i) / T_public_i",
            "one_sided_95": "mean + 1.644854 * s / sqrt(n), ddof=1",
            "no_sample_dropped": True,
        },
        "workload": {
            "entry": "Optimization_Problem density_matrix",
            "qbit_num": QBIT_NUM,
            "ansatz": "HEA",
            "layers": 1,
            "inner_blocks": 1,
            "topology": [(idx, idx + 1) for idx in range(QBIT_NUM - 1)],
            "hamiltonian_nnz": int(hamiltonian.nnz),
            "hamiltonian_csr_sha256": _hamiltonian_csr_sha256(hamiltonian),
            "density_noise": DENSITY_NOISE,
            "config": CONFIG,
        },
        "qa008": {
            "categorical_labels_exact": True,
            "mean_O_absolute_margin": QA008_MEAN_O_ABSOLUTE_MARGIN,
        },
        "components": {
            "mean_wrapper_ns": float(np.mean(wrapper_values)),
            "mean_allocate_build_ns": float(np.mean(allocate_values)),
            "mean_apply_to_ns": float(np.mean(apply_values)),
            "mean_contraction_ns": float(np.mean(contraction_values)),
        },
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
