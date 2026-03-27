from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.correctness_evidence.common import (  # noqa: E402
    CORRECTNESS_PACKAGE_SCHEMA_VERSION,
    build_selected_candidate as build_correctness_selected_candidate,
)
from benchmarks.density_matrix.evidence_io import write_artifact_bundle  # noqa: E402
from benchmarks.density_matrix.planner_surface.common import (  # noqa: E402
    build_software_metadata,
)
from benchmarks.density_matrix.planner_surface.workloads import (  # noqa: E402
    DEFAULT_STRUCTURED_SEED,
)
from squander.density_matrix import DensityMatrix  # noqa: E402
from squander.partitioning.noisy_runtime import (  # noqa: E402
    execute_sequential_density_reference,
)

PERFORMANCE_EVIDENCE_CASE_SCHEMA_VERSION = "performance_evidence_record_v1"
PERFORMANCE_EVIDENCE_BENCHMARK_PACKAGE_SCHEMA_VERSION = "performance_evidence_package_v1"
PERFORMANCE_EVIDENCE_SUMMARY_SCHEMA_VERSION = "performance_evidence_summary_v1"

PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY = "continuity_anchor"
PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED = "structured_performance"

PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_INTERNAL = "sequential_density_descriptor_reference"
PERFORMANCE_EVIDENCE_REFERENCE_BACKEND_EXTERNAL = "qiskit_aer_density_matrix"

PERFORMANCE_EVIDENCE_STATUS_COUNTED = "counted_supported"
PERFORMANCE_EVIDENCE_STATUS_DIAGNOSIS_ONLY = "diagnosis_only"
PERFORMANCE_EVIDENCE_STATUS_EXCLUDED = "excluded"
PERFORMANCE_EVIDENCE_STATUS_NOT_REVIEW_CASE = "not_review_case"

PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED = DEFAULT_STRUCTURED_SEED
PERFORMANCE_EVIDENCE_ADDITIONAL_STRUCTURED_SEEDS = (
    DEFAULT_STRUCTURED_SEED + 1,
    DEFAULT_STRUCTURED_SEED + 2,
)
PERFORMANCE_EVIDENCE_REVIEW_NOISE_PATTERN = "sparse"
PERFORMANCE_EVIDENCE_REPETITIONS = 3

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "performance_evidence"
)


def performance_evidence_output_dir(slice_dir_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / slice_dir_name


@lru_cache(maxsize=1)
def build_selected_candidate() -> dict[str, Any]:
    return dict(build_correctness_selected_candidate())


def build_package_software_metadata() -> dict[str, Any]:
    metadata = build_software_metadata()
    metadata["planner_calibration_selected_candidate_id"] = build_selected_candidate()["candidate_id"]
    metadata["correctness_evidence_package_schema_version"] = CORRECTNESS_PACKAGE_SCHEMA_VERSION
    return metadata


@lru_cache(maxsize=1)
def build_correctness_reference_index() -> dict[str, dict[str, Any]]:
    from benchmarks.density_matrix.correctness_evidence.bundle import (
        build_correctness_package_payload,
    )

    payload = build_correctness_package_payload()
    return {case["workload_id"]: dict(case) for case in payload["cases"]}


@lru_cache(maxsize=1)
def build_boundary_evidence() -> tuple[dict[str, Any], ...]:
    from benchmarks.density_matrix.correctness_evidence.bundle import (
        build_correctness_package_payload,
    )

    payload = build_correctness_package_payload()
    return tuple(dict(case) for case in payload["negative_cases"])


def build_performance_evidence_selected_candidate() -> dict[str, Any]:
    return build_selected_candidate()


def build_performance_evidence_software_metadata() -> dict[str, Any]:
    return build_package_software_metadata()


def build_performance_evidence_correctness_evidence_reference_index() -> dict[str, dict[str, Any]]:
    return build_correctness_reference_index()


def build_performance_evidence_boundary_evidence() -> tuple[dict[str, Any], ...]:
    return build_boundary_evidence()


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


@dataclass(frozen=True)
class SequentialReferenceMeasurement:
    density_matrix: DensityMatrix
    runtime_ms: float
    peak_rss_kb: int
    trace_deviation: float
    rho_is_valid: bool


def measure_sequential_density_reference(
    descriptor_set, parameters: Iterable[float]
) -> SequentialReferenceMeasurement:
    start = time.perf_counter()
    density_matrix = execute_sequential_density_reference(descriptor_set, parameters)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    trace = complex(density_matrix.trace())
    return SequentialReferenceMeasurement(
        density_matrix=density_matrix,
        runtime_ms=runtime_ms,
        peak_rss_kb=_peak_rss_kb(),
        trace_deviation=float(abs(trace - 1.0)),
        rho_is_valid=bool(density_matrix.is_valid(tol=1e-10)),
    )
