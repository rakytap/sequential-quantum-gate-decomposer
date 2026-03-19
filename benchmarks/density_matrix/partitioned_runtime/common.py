from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import (  # noqa: E402
    build_software_metadata,
)
from squander.density_matrix import DensityMatrix  # noqa: E402
from squander.partitioning.noisy_planner import NoisyPartitionDescriptorSet  # noqa: E402
from squander.partitioning.noisy_runtime import (  # noqa: E402
    NoisyRuntimeExecutionResult,
    execute_partitioned_density,
    execute_sequential_density_reference,
)

PHASE3_RUNTIME_DENSITY_TOL = 1e-10
PHASE3_RUNTIME_ENERGY_TOL = 1e-8


def build_initial_parameters(param_num: int) -> np.ndarray:
    return np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)


def density_energy(hamiltonian, density_matrix: np.ndarray) -> tuple[float, float]:
    energy = np.trace(hamiltonian.dot(density_matrix))
    return float(np.real(energy)), float(np.imag(energy))


def coerce_density_matrix_array(density_matrix: DensityMatrix | np.ndarray) -> np.ndarray:
    if isinstance(density_matrix, DensityMatrix):
        return np.asarray(density_matrix.to_numpy())
    return np.asarray(density_matrix)


def build_density_comparison_metrics(
    partitioned_density: DensityMatrix | np.ndarray,
    reference_density: DensityMatrix | np.ndarray,
) -> dict[str, Any]:
    partitioned = coerce_density_matrix_array(partitioned_density)
    reference = coerce_density_matrix_array(reference_density)
    delta = partitioned - reference
    return {
        "frobenius_norm_diff": float(np.linalg.norm(delta)),
        "max_abs_diff": float(np.max(np.abs(delta))),
    }


def execute_partitioned_with_reference(
    descriptor_set: NoisyPartitionDescriptorSet,
    parameters: np.ndarray,
) -> tuple[NoisyRuntimeExecutionResult, DensityMatrix, dict[str, Any]]:
    runtime_result = execute_partitioned_density(descriptor_set, parameters)
    reference_density = execute_sequential_density_reference(descriptor_set, parameters)
    metrics = build_density_comparison_metrics(
        runtime_result.density_matrix, reference_density
    )
    return runtime_result, reference_density, metrics
