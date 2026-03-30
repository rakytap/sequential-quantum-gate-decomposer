from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.density_matrix.correctness_evidence.common import build_selected_candidate
from benchmarks.density_matrix.performance_evidence.common import (
    PERFORMANCE_EVIDENCE_ADDITIONAL_STRUCTURED_SEEDS,
    PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY,
    PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED,
    PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED,
    PERFORMANCE_EVIDENCE_REVIEW_NOISE_PATTERN,
)
from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import build_phase2_continuity_vqe
from benchmarks.density_matrix.planner_surface.workloads import (
    DEFAULT_STRUCTURED_SEED,
    MANDATORY_NOISE_PATTERNS,
    PHASE31_CONTROL_NOISE_PATTERNS,
    PHASE31_CONTROL_STRUCTURED_FAMILY_NAMES,
    PHASE31_PRIMARY_NOISE_PATTERNS,
    PHASE31_PRIMARY_SEEDS,
    PHASE31_PRIMARY_STRUCTURED_FAMILY_NAMES,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_phase31_structured_descriptor_set,
    build_structured_descriptor_set,
)
from squander.partitioning.noisy_planner import (
    NoisyPartitionDescriptorSet,
    build_phase3_continuity_partition_descriptor_set,
)

PERFORMANCE_EVIDENCE_CASE_KIND_CONTINUITY = "continuity"
PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED = "structured_family"
PERFORMANCE_EVIDENCE_CONTINUITY_QUBITS = (4, 6, 8, 10)
PERFORMANCE_EVIDENCE_EXTERNAL_REFERENCE_CONTINUITY_QUBITS = (4,)

# Frozen Phase 3.1 hybrid performance pilot (P31-S09-E01); do not rename without updating docs/tests.
# Replaced dense with periodic so the pilot exercises mixed hybrid routing (nonzero channel-native partitions).
PHASE31_HYBRID_PILOT_WORKLOAD_ID = "phase31_pair_repeat_q8_periodic_seed20260318"
BENCHMARK_SLICE_PHASE31_HYBRID_PILOT = "phase31_hybrid_pilot"


@dataclass(frozen=True)
class PerformanceEvidenceCaseContext:
    metadata: dict[str, Any]
    descriptor_set: NoisyPartitionDescriptorSet
    parameters: np.ndarray
    hamiltonian: Any | None = None


def _selected_partition_qubits() -> int:
    return int(build_selected_candidate()["max_partition_qubits"])


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
    selected_candidate = build_selected_candidate()
    external_reference_required = (
        case_kind == PERFORMANCE_EVIDENCE_CASE_KIND_CONTINUITY
        and descriptor_set.qbit_num in PERFORMANCE_EVIDENCE_EXTERNAL_REFERENCE_CONTINUITY_QUBITS
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
        "planner_calibration_selected_candidate_id": selected_candidate["selected_candidate_id"],
        "planner_calibration_claim_selection_schema_version": selected_candidate[
            "claim_selection_schema_version"
        ],
        "planner_calibration_claim_selection_rule": selected_candidate["claim_selection_rule"],
        "source_type": descriptor_set.source_type,
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


def iter_performance_evidence_continuity_cases():
    max_partition_qubits = _selected_partition_qubits()
    for qbit_num in PERFORMANCE_EVIDENCE_CONTINUITY_QUBITS:
        vqe, hamiltonian, topology = build_phase2_continuity_vqe(qbit_num)
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
            case_kind=PERFORMANCE_EVIDENCE_CASE_KIND_CONTINUITY,
            benchmark_slice=PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY,
            planning_time_ms=planning_time_ms,
            representative_review_case=False,
            topology=topology,
        )
        yield PerformanceEvidenceCaseContext(
            metadata=metadata,
            descriptor_set=descriptor_set,
            parameters=build_initial_parameters(vqe.get_Parameter_Num()),
            hamiltonian=hamiltonian,
        )


def _structured_seed_noise_pairs() -> tuple[tuple[int, str], ...]:
    return (
        (PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED, "sparse"),
        (PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED, "periodic"),
        (PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED, "dense"),
        (PERFORMANCE_EVIDENCE_ADDITIONAL_STRUCTURED_SEEDS[0], PERFORMANCE_EVIDENCE_REVIEW_NOISE_PATTERN),
        (PERFORMANCE_EVIDENCE_ADDITIONAL_STRUCTURED_SEEDS[1], PERFORMANCE_EVIDENCE_REVIEW_NOISE_PATTERN),
    )


@lru_cache(maxsize=1)
def _build_performance_evidence_inventory_cases_cached() -> tuple[dict[str, Any], ...]:
    selected_candidate = build_selected_candidate()
    inventory_cases: list[dict[str, Any]] = []
    for qbit_num in PERFORMANCE_EVIDENCE_CONTINUITY_QUBITS:
        inventory_cases.append(
            {
                "case_name": f"phase2_xxz_hea_q{qbit_num}_continuity",
                "case_kind": PERFORMANCE_EVIDENCE_CASE_KIND_CONTINUITY,
                "benchmark_slice": PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_CONTINUITY,
                "candidate_id": selected_candidate["candidate_id"],
                "planner_family": selected_candidate["planner_family"],
                "planner_variant": selected_candidate["planner_variant"],
                "max_partition_qubits": selected_candidate["max_partition_qubits"],
                "planner_settings": dict(selected_candidate["planner_settings"]),
                "planner_calibration_selected_candidate_id": selected_candidate["selected_candidate_id"],
                "planner_calibration_claim_selection_schema_version": selected_candidate[
                    "claim_selection_schema_version"
                ],
                "planner_calibration_claim_selection_rule": selected_candidate[
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
                in PERFORMANCE_EVIDENCE_EXTERNAL_REFERENCE_CONTINUITY_QUBITS,
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
                        "case_kind": PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
                        "benchmark_slice": PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED,
                        "candidate_id": selected_candidate["candidate_id"],
                        "planner_family": selected_candidate["planner_family"],
                        "planner_variant": selected_candidate["planner_variant"],
                        "max_partition_qubits": selected_candidate[
                            "max_partition_qubits"
                        ],
                        "planner_settings": dict(selected_candidate["planner_settings"]),
                        "planner_calibration_selected_candidate_id": selected_candidate[
                            "selected_candidate_id"
                        ],
                        "planner_calibration_claim_selection_schema_version": selected_candidate[
                            "claim_selection_schema_version"
                        ],
                        "planner_calibration_claim_selection_rule": selected_candidate[
                            "claim_selection_rule"
                        ],
                        "qbit_num": qbit_num,
                        "family_name": family_name,
                        "noise_pattern": noise_pattern,
                        "seed": seed,
                        "topology": None,
                        "planning_time_ms": None,
                        "representative_review_case": (
                            seed == PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED
                            and noise_pattern == PERFORMANCE_EVIDENCE_REVIEW_NOISE_PATTERN
                        ),
                        "review_group_id": f"{family_name}_q{qbit_num}",
                        "external_reference_required": False,
                    }
                )
    return tuple(inventory_cases)


def build_performance_evidence_inventory_cases() -> list[dict[str, Any]]:
    return deepcopy(list(_build_performance_evidence_inventory_cases_cached()))


def iter_performance_evidence_structured_cases():
    max_partition_qubits = _selected_partition_qubits()
    for family_name in STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for seed, noise_pattern in _structured_seed_noise_pairs():
                start = perf_counter()
                descriptor_set = build_structured_descriptor_set(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=seed,
                    max_partition_qubits=max_partition_qubits,
                )
                planning_time_ms = (perf_counter() - start) * 1000.0
                metadata = _base_metadata_from_descriptor(
                    descriptor_set,
                    case_kind=PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
                    benchmark_slice=PERFORMANCE_EVIDENCE_BENCHMARK_SLICE_STRUCTURED,
                    planning_time_ms=planning_time_ms,
                    representative_review_case=(
                        seed == PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED
                        and noise_pattern == PERFORMANCE_EVIDENCE_REVIEW_NOISE_PATTERN
                    ),
                    family_name=family_name,
                    noise_pattern=noise_pattern,
                    seed=seed,
                )
                yield PerformanceEvidenceCaseContext(
                    metadata=metadata,
                    descriptor_set=descriptor_set,
                    parameters=build_initial_parameters(descriptor_set.parameter_count),
                )


def build_phase31_performance_inventory_cases() -> list[dict[str, Any]]:
    """Planning helper inventory for the frozen Phase 3.1 counted slice.

    This inventory mirrors the contract closed in the Phase 3.1 docs but is not
    wired into the default Phase 3 performance package builders.
    """

    selected_candidate = build_selected_candidate()
    inventory_cases: list[dict[str, Any]] = []
    for family_name in PHASE31_PRIMARY_STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in PHASE31_PRIMARY_NOISE_PATTERNS:
                for seed in PHASE31_PRIMARY_SEEDS:
                    inventory_cases.append(
                        {
                            "case_name": f"{family_name}_q{qbit_num}_{noise_pattern}_seed{seed}",
                            "case_kind": PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
                            "benchmark_slice": "phase31_structured_performance",
                            "candidate_id": selected_candidate["candidate_id"],
                            "planner_family": selected_candidate["planner_family"],
                            "planner_variant": selected_candidate["planner_variant"],
                            "max_partition_qubits": selected_candidate[
                                "max_partition_qubits"
                            ],
                            "planner_settings": dict(selected_candidate["planner_settings"]),
                            "planner_calibration_selected_candidate_id": selected_candidate[
                                "selected_candidate_id"
                            ],
                            "planner_calibration_claim_selection_schema_version": selected_candidate[
                                "claim_selection_schema_version"
                            ],
                            "planner_calibration_claim_selection_rule": selected_candidate[
                                "claim_selection_rule"
                            ],
                            "qbit_num": qbit_num,
                            "family_name": family_name,
                            "noise_pattern": noise_pattern,
                            "seed": seed,
                            "topology": None,
                            "planning_time_ms": None,
                            "representative_review_case": (
                                seed == PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED
                            ),
                            "review_group_id": f"{family_name}_q{qbit_num}",
                            "external_reference_required": False,
                            "claim_surface_id": "phase31_bounded_mixed_motif_v1",
                            "representation_primary": "kraus_bundle",
                            "contains_noise": True,
                            "counted_phase31_case": True,
                        }
                    )
    for family_name in PHASE31_CONTROL_STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in PHASE31_CONTROL_NOISE_PATTERNS:
                inventory_cases.append(
                    {
                        "case_name": f"{family_name}_q{qbit_num}_{noise_pattern}_seed{DEFAULT_STRUCTURED_SEED}",
                        "case_kind": PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
                        "benchmark_slice": "phase31_control_performance",
                        "candidate_id": selected_candidate["candidate_id"],
                        "planner_family": selected_candidate["planner_family"],
                        "planner_variant": selected_candidate["planner_variant"],
                        "max_partition_qubits": selected_candidate["max_partition_qubits"],
                        "planner_settings": dict(selected_candidate["planner_settings"]),
                        "planner_calibration_selected_candidate_id": selected_candidate[
                            "selected_candidate_id"
                        ],
                        "planner_calibration_claim_selection_schema_version": selected_candidate[
                            "claim_selection_schema_version"
                        ],
                        "planner_calibration_claim_selection_rule": selected_candidate[
                            "claim_selection_rule"
                        ],
                        "qbit_num": qbit_num,
                        "family_name": family_name,
                        "noise_pattern": noise_pattern,
                        "seed": DEFAULT_STRUCTURED_SEED,
                        "topology": None,
                        "planning_time_ms": None,
                        "representative_review_case": True,
                        "review_group_id": f"{family_name}_q{qbit_num}",
                        "external_reference_required": False,
                        "claim_surface_id": "phase31_bounded_mixed_motif_v1",
                        "representation_primary": "kraus_bundle",
                        "contains_noise": True,
                        "counted_phase31_case": True,
                    }
                )
    return inventory_cases


def iter_phase31_performance_cases():
    """Planning helper contexts for the frozen Phase 3.1 counted performance slice."""

    max_partition_qubits = _selected_partition_qubits()
    for family_name in PHASE31_PRIMARY_STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in PHASE31_PRIMARY_NOISE_PATTERNS:
                for seed in PHASE31_PRIMARY_SEEDS:
                    start = perf_counter()
                    descriptor_set = build_phase31_structured_descriptor_set(
                        family_name,
                        qbit_num=qbit_num,
                        noise_pattern=noise_pattern,
                        seed=seed,
                        max_partition_qubits=max_partition_qubits,
                    )
                    planning_time_ms = (perf_counter() - start) * 1000.0
                    metadata = _base_metadata_from_descriptor(
                        descriptor_set,
                        case_kind=PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
                        benchmark_slice="phase31_structured_performance",
                        planning_time_ms=planning_time_ms,
                        representative_review_case=(
                            seed == PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED
                        ),
                        family_name=family_name,
                        noise_pattern=noise_pattern,
                        seed=seed,
                    )
                    metadata.update(
                        {
                            "claim_surface_id": "phase31_bounded_mixed_motif_v1",
                            "representation_primary": "kraus_bundle",
                            "contains_noise": True,
                            "counted_phase31_case": True,
                        }
                    )
                    yield PerformanceEvidenceCaseContext(
                        metadata=metadata,
                        descriptor_set=descriptor_set,
                        parameters=build_initial_parameters(
                            descriptor_set.parameter_count
                        ),
                    )
    for family_name in PHASE31_CONTROL_STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in PHASE31_CONTROL_NOISE_PATTERNS:
                start = perf_counter()
                descriptor_set = build_phase31_structured_descriptor_set(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=DEFAULT_STRUCTURED_SEED,
                    max_partition_qubits=max_partition_qubits,
                )
                planning_time_ms = (perf_counter() - start) * 1000.0
                metadata = _base_metadata_from_descriptor(
                    descriptor_set,
                    case_kind=PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
                    benchmark_slice="phase31_control_performance",
                    planning_time_ms=planning_time_ms,
                    representative_review_case=True,
                    family_name=family_name,
                    noise_pattern=noise_pattern,
                    seed=DEFAULT_STRUCTURED_SEED,
                )
                metadata.update(
                    {
                        "claim_surface_id": "phase31_bounded_mixed_motif_v1",
                        "representation_primary": "kraus_bundle",
                        "contains_noise": True,
                        "counted_phase31_case": True,
                    }
                )
                yield PerformanceEvidenceCaseContext(
                    metadata=metadata,
                    descriptor_set=descriptor_set,
                    parameters=build_initial_parameters(descriptor_set.parameter_count),
                )


def build_phase31_hybrid_pilot_case_context() -> PerformanceEvidenceCaseContext:
    """Single frozen structured workload for the Phase 3.1 hybrid performance pilot row."""
    max_partition_qubits = _selected_partition_qubits()
    family_name = "phase31_pair_repeat"
    qbit_num = 8
    noise_pattern = "periodic"
    seed = PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED
    start = perf_counter()
    descriptor_set = build_phase31_structured_descriptor_set(
        family_name,
        qbit_num=qbit_num,
        noise_pattern=noise_pattern,
        seed=seed,
        max_partition_qubits=max_partition_qubits,
    )
    planning_time_ms = (perf_counter() - start) * 1000.0
    if descriptor_set.workload_id != PHASE31_HYBRID_PILOT_WORKLOAD_ID:
        raise RuntimeError(
            "Phase 3.1 hybrid pilot workload_id mismatch: expected {!r}, got {!r}".format(
                PHASE31_HYBRID_PILOT_WORKLOAD_ID,
                descriptor_set.workload_id,
            )
        )
    metadata = _base_metadata_from_descriptor(
        descriptor_set,
        case_kind=PERFORMANCE_EVIDENCE_CASE_KIND_STRUCTURED,
        benchmark_slice=BENCHMARK_SLICE_PHASE31_HYBRID_PILOT,
        planning_time_ms=planning_time_ms,
        representative_review_case=True,
        family_name=family_name,
        noise_pattern=noise_pattern,
        seed=seed,
    )
    metadata.update(
        {
            "claim_surface_id": "phase31_bounded_mixed_motif_v1",
            "representation_primary": "kraus_bundle",
            "contains_noise": True,
            "counted_phase31_case": True,
            "phase31_hybrid_pilot_case": True,
        }
    )
    return PerformanceEvidenceCaseContext(
        metadata=metadata,
        descriptor_set=descriptor_set,
        parameters=build_initial_parameters(descriptor_set.parameter_count),
    )


@lru_cache(maxsize=1)
def _build_performance_evidence_case_contexts_cached() -> tuple[PerformanceEvidenceCaseContext, ...]:
    cases: list[PerformanceEvidenceCaseContext] = []
    cases.extend(iter_performance_evidence_continuity_cases())
    cases.extend(iter_performance_evidence_structured_cases())
    return tuple(cases)


def build_performance_evidence_case_contexts() -> list[PerformanceEvidenceCaseContext]:
    return deepcopy(list(_build_performance_evidence_case_contexts_cached()))
