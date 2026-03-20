from __future__ import annotations

import json
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
    TASK6_CORRECTNESS_PACKAGE_SCHEMA_VERSION,
    build_task6_selected_candidate,
)
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

TASK7_CASE_SCHEMA_VERSION = "phase3_task7_benchmark_record_v1"
TASK7_BENCHMARK_PACKAGE_SCHEMA_VERSION = "phase3_task7_benchmark_package_v1"
TASK7_SUMMARY_SCHEMA_VERSION = "phase3_task7_summary_consistency_v1"

TASK7_BENCHMARK_SLICE_CONTINUITY = "continuity_anchor"
TASK7_BENCHMARK_SLICE_STRUCTURED = "structured_performance"

TASK7_REFERENCE_BACKEND_INTERNAL = "sequential_density_descriptor_reference"
TASK7_REFERENCE_BACKEND_EXTERNAL = "qiskit_aer_density_matrix"

TASK7_STATUS_COUNTED = "counted_supported"
TASK7_STATUS_DIAGNOSIS_ONLY = "diagnosis_only"
TASK7_STATUS_EXCLUDED = "excluded"
TASK7_STATUS_NOT_REVIEW_CASE = "not_review_case"

TASK7_PRIMARY_STRUCTURED_SEED = DEFAULT_STRUCTURED_SEED
TASK7_ADDITIONAL_STRUCTURED_SEEDS = (
    DEFAULT_STRUCTURED_SEED + 1,
    DEFAULT_STRUCTURED_SEED + 2,
)
TASK7_REVIEW_NOISE_PATTERN = "sparse"
TASK7_REPETITIONS = 3

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "phase3_task7"
)


def task7_story_output_dir(story_dir_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / story_dir_name


def write_artifact_bundle(
    bundle: dict[str, Any], output_dir: Path, artifact_filename: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / artifact_filename
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


@lru_cache(maxsize=1)
def build_task7_selected_candidate() -> dict[str, Any]:
    return dict(build_task6_selected_candidate())


def build_task7_software_metadata() -> dict[str, Any]:
    metadata = build_software_metadata()
    metadata["task5_selected_candidate_id"] = build_task7_selected_candidate()[
        "candidate_id"
    ]
    metadata["task6_correctness_package_schema_version"] = (
        TASK6_CORRECTNESS_PACKAGE_SCHEMA_VERSION
    )
    return metadata


@lru_cache(maxsize=1)
def build_task7_task6_reference_index() -> dict[str, dict[str, Any]]:
    from benchmarks.density_matrix.correctness_evidence.bundle import (
        build_task6_correctness_package_payload,
    )

    payload = build_task6_correctness_package_payload()
    return {case["workload_id"]: dict(case) for case in payload["cases"]}


@lru_cache(maxsize=1)
def build_task7_boundary_evidence() -> tuple[dict[str, Any], ...]:
    from benchmarks.density_matrix.correctness_evidence.bundle import (
        build_task6_correctness_package_payload,
    )

    payload = build_task6_correctness_package_payload()
    return tuple(dict(case) for case in payload["negative_cases"])


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
