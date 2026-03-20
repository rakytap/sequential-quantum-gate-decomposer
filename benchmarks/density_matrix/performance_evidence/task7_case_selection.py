from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.density_matrix.performance_evidence.common import (
    TASK7_ADDITIONAL_STRUCTURED_SEEDS,
    TASK7_BENCHMARK_SLICE_CONTINUITY,
    TASK7_BENCHMARK_SLICE_STRUCTURED,
    TASK7_PRIMARY_STRUCTURED_SEED,
    TASK7_REVIEW_NOISE_PATTERN,
    build_task7_selected_candidate,
)
from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import build_phase3_story1_continuity_vqe
from benchmarks.density_matrix.planner_surface.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_story2_structured_descriptor_set,
)
from squander.partitioning.noisy_planner import (
    NoisyPartitionDescriptorSet,
    build_phase3_continuity_partition_descriptor_set,
)

TASK7_CASE_KIND_CONTINUITY = "continuity"
TASK7_CASE_KIND_STRUCTURED = "structured_family"
TASK7_CONTINUITY_QUBITS = (4, 6, 8, 10)
TASK7_EXTERNAL_REFERENCE_CONTINUITY_QUBITS = (4,)


@dataclass(frozen=True)
class Task7CaseContext:
    metadata: dict[str, Any]
    descriptor_set: NoisyPartitionDescriptorSet
    parameters: np.ndarray
    hamiltonian: Any | None = None


def _selected_partition_qubits() -> int:
    return int(build_task7_selected_candidate()["max_partition_qubits"])


def _base_metadata_from_descriptor(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    case_kind: str,
    benchmark_slice: str,
    planning_time_ms: float,
    representative_review_case: bool,
    family_name: str | None = None,
    noise_pattern: str | None = None,
    seed: int | None = None,
    topology: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    selected_candidate = build_task7_selected_candidate()
    external_reference_required = (
        case_kind == TASK7_CASE_KIND_CONTINUITY
        and descriptor_set.qbit_num in TASK7_EXTERNAL_REFERENCE_CONTINUITY_QUBITS
    )
    return {
        "case_name": descriptor_set.workload_id,
        "case_kind": case_kind,
        "benchmark_slice": benchmark_slice,
        "candidate_id": selected_candidate["candidate_id"],
        "planner_family": selected_candidate["planner_family"],
        "planner_variant": selected_candidate["planner_variant"],
        "max_partition_qubits": selected_candidate["max_partition_qubits"],
        "planner_settings": dict(selected_candidate["planner_settings"]),
        "task5_selected_candidate_id": selected_candidate["selected_candidate_id"],
        "task5_claim_selection_schema_version": selected_candidate[
            "claim_selection_schema_version"
        ],
        "task5_claim_selection_rule": selected_candidate["claim_selection_rule"],
        "source_type": descriptor_set.source_type,
        "entry_route": descriptor_set.entry_route,
        "workload_family": descriptor_set.workload_family,
        "workload_id": descriptor_set.workload_id,
        "qbit_num": descriptor_set.qbit_num,
        "parameter_count": descriptor_set.parameter_count,
        "family_name": family_name,
        "noise_pattern": noise_pattern,
        "seed": seed,
        "topology": list(topology) if topology is not None else None,
        "planning_time_ms": planning_time_ms,
        "representative_review_case": representative_review_case,
        "review_group_id": (
            None if family_name is None else f"{family_name}_q{descriptor_set.qbit_num}"
        ),
        "external_reference_required": external_reference_required,
    }


def iter_task7_continuity_cases():
    max_partition_qubits = _selected_partition_qubits()
    for qbit_num in TASK7_CONTINUITY_QUBITS:
        vqe, hamiltonian, topology = build_phase3_story1_continuity_vqe(qbit_num)
        workload_id = f"phase2_xxz_hea_q{qbit_num}_continuity"
        start = perf_counter()
        descriptor_set = build_phase3_continuity_partition_descriptor_set(
            vqe,
            workload_id=workload_id,
            max_partition_qubits=max_partition_qubits,
        )
        planning_time_ms = (perf_counter() - start) * 1000.0
        metadata = _base_metadata_from_descriptor(
            descriptor_set,
            case_kind=TASK7_CASE_KIND_CONTINUITY,
            benchmark_slice=TASK7_BENCHMARK_SLICE_CONTINUITY,
            planning_time_ms=planning_time_ms,
            representative_review_case=False,
            topology=topology,
        )
        yield Task7CaseContext(
            metadata=metadata,
            descriptor_set=descriptor_set,
            parameters=build_initial_parameters(vqe.get_Parameter_Num()),
            hamiltonian=hamiltonian,
        )


def _structured_seed_noise_pairs() -> tuple[tuple[int, str], ...]:
    return (
        (TASK7_PRIMARY_STRUCTURED_SEED, "sparse"),
        (TASK7_PRIMARY_STRUCTURED_SEED, "periodic"),
        (TASK7_PRIMARY_STRUCTURED_SEED, "dense"),
        (TASK7_ADDITIONAL_STRUCTURED_SEEDS[0], TASK7_REVIEW_NOISE_PATTERN),
        (TASK7_ADDITIONAL_STRUCTURED_SEEDS[1], TASK7_REVIEW_NOISE_PATTERN),
    )


@lru_cache(maxsize=1)
def _build_task7_inventory_cases_cached() -> tuple[dict[str, Any], ...]:
    selected_candidate = build_task7_selected_candidate()
    inventory_cases: list[dict[str, Any]] = []
    for qbit_num in TASK7_CONTINUITY_QUBITS:
        inventory_cases.append(
            {
                "case_name": f"phase2_xxz_hea_q{qbit_num}_continuity",
                "case_kind": TASK7_CASE_KIND_CONTINUITY,
                "benchmark_slice": TASK7_BENCHMARK_SLICE_CONTINUITY,
                "candidate_id": selected_candidate["candidate_id"],
                "planner_family": selected_candidate["planner_family"],
                "planner_variant": selected_candidate["planner_variant"],
                "max_partition_qubits": selected_candidate["max_partition_qubits"],
                "planner_settings": dict(selected_candidate["planner_settings"]),
                "task5_selected_candidate_id": selected_candidate["selected_candidate_id"],
                "task5_claim_selection_schema_version": selected_candidate[
                    "claim_selection_schema_version"
                ],
                "task5_claim_selection_rule": selected_candidate[
                    "claim_selection_rule"
                ],
                "qbit_num": qbit_num,
                "family_name": None,
                "noise_pattern": None,
                "seed": None,
                "topology": None,
                "planning_time_ms": None,
                "representative_review_case": False,
                "review_group_id": None,
                "external_reference_required": qbit_num
                in TASK7_EXTERNAL_REFERENCE_CONTINUITY_QUBITS,
            }
        )
    for family_name in STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for seed, noise_pattern in _structured_seed_noise_pairs():
                inventory_cases.append(
                    {
                        "case_name": "{}_q{}_{}_seed{}".format(
                            family_name, qbit_num, noise_pattern, seed
                        ),
                        "case_kind": TASK7_CASE_KIND_STRUCTURED,
                        "benchmark_slice": TASK7_BENCHMARK_SLICE_STRUCTURED,
                        "candidate_id": selected_candidate["candidate_id"],
                        "planner_family": selected_candidate["planner_family"],
                        "planner_variant": selected_candidate["planner_variant"],
                        "max_partition_qubits": selected_candidate[
                            "max_partition_qubits"
                        ],
                        "planner_settings": dict(selected_candidate["planner_settings"]),
                        "task5_selected_candidate_id": selected_candidate[
                            "selected_candidate_id"
                        ],
                        "task5_claim_selection_schema_version": selected_candidate[
                            "claim_selection_schema_version"
                        ],
                        "task5_claim_selection_rule": selected_candidate[
                            "claim_selection_rule"
                        ],
                        "qbit_num": qbit_num,
                        "family_name": family_name,
                        "noise_pattern": noise_pattern,
                        "seed": seed,
                        "topology": None,
                        "planning_time_ms": None,
                        "representative_review_case": (
                            seed == TASK7_PRIMARY_STRUCTURED_SEED
                            and noise_pattern == TASK7_REVIEW_NOISE_PATTERN
                        ),
                        "review_group_id": f"{family_name}_q{qbit_num}",
                        "external_reference_required": False,
                    }
                )
    return tuple(inventory_cases)


def build_task7_inventory_cases() -> list[dict[str, Any]]:
    return deepcopy(list(_build_task7_inventory_cases_cached()))


def iter_task7_structured_cases():
    max_partition_qubits = _selected_partition_qubits()
    for family_name in STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for seed, noise_pattern in _structured_seed_noise_pairs():
                start = perf_counter()
                descriptor_set = build_story2_structured_descriptor_set(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=seed,
                    max_partition_qubits=max_partition_qubits,
                )
                planning_time_ms = (perf_counter() - start) * 1000.0
                metadata = _base_metadata_from_descriptor(
                    descriptor_set,
                    case_kind=TASK7_CASE_KIND_STRUCTURED,
                    benchmark_slice=TASK7_BENCHMARK_SLICE_STRUCTURED,
                    planning_time_ms=planning_time_ms,
                    representative_review_case=(
                        seed == TASK7_PRIMARY_STRUCTURED_SEED
                        and noise_pattern == TASK7_REVIEW_NOISE_PATTERN
                    ),
                    family_name=family_name,
                    noise_pattern=noise_pattern,
                    seed=seed,
                )
                yield Task7CaseContext(
                    metadata=metadata,
                    descriptor_set=descriptor_set,
                    parameters=build_initial_parameters(descriptor_set.parameter_count),
                )


@lru_cache(maxsize=1)
def _build_task7_case_contexts_cached() -> tuple[Task7CaseContext, ...]:
    cases: list[Task7CaseContext] = []
    cases.extend(iter_task7_continuity_cases())
    cases.extend(iter_task7_structured_cases())
    return tuple(cases)


def build_task7_case_contexts() -> list[Task7CaseContext]:
    return deepcopy(list(_build_task7_case_contexts_cached()))
