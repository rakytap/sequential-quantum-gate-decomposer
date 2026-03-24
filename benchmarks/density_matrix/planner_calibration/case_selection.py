from __future__ import annotations

from time import perf_counter

from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_calibration.common import (
    PlannerCandidate,
    build_planner_candidates,
)
from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    DEFAULT_STRUCTURED_SEED,
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_microcase_descriptor_set,
    build_structured_descriptor_set,
    mandatory_microcase_definitions,
)
from squander.partitioning.noisy_planner import (
    build_phase3_continuity_partition_descriptor_set,
    phase3_workload_family_for_source_type,
)

PLANNER_CALIBRATION_CASE_KIND_CONTINUITY = "continuity"
PLANNER_CALIBRATION_CASE_KIND_MICROCASE = "microcase"
PLANNER_CALIBRATION_CASE_KIND_STRUCTURED = "structured_family"
PLANNER_CALIBRATION_CONTINUITY_QUBITS = (4, 6, 8, 10)
DENSITY_SIGNAL_CONTINUITY_QUBITS = (4,)
DENSITY_SIGNAL_MICROCASE_IDS = ("microcase_4q_partition_boundary_triplet",)
DENSITY_SIGNAL_STRUCTURED_FILTER = {
    "family_name": "layered_nearest_neighbor",
    "qbit_num": 8,
    "noise_patterns": ("sparse", "dense"),
}


def _metadata_from_descriptor(
    candidate: PlannerCandidate,
    descriptor_set,
    *,
    case_kind: str,
    planning_time_ms: float,
    family_name: str | None = None,
    noise_pattern: str | None = None,
    seed: int | None = None,
) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "planner_family": candidate.planner_family,
        "planner_variant": candidate.planner_variant,
        "max_partition_qubits": candidate.max_partition_qubits,
        "case_kind": case_kind,
        "source_type": descriptor_set.source_type,
        "workload_family": phase3_workload_family_for_source_type(
            descriptor_set.source_type
        ),
        "workload_id": descriptor_set.workload_id,
        "qbit_num": descriptor_set.qbit_num,
        "planning_time_ms": planning_time_ms,
        "family_name": family_name,
        "noise_pattern": noise_pattern,
        "seed": seed,
    }


def iter_planner_calibration_continuity_cases(
    candidates: tuple[PlannerCandidate, ...] | None = None,
):
    active_candidates = (
        build_planner_candidates() if candidates is None else tuple(candidates)
    )
    for candidate in active_candidates:
        for qbit_num in PLANNER_CALIBRATION_CONTINUITY_QUBITS:
            vqe, hamiltonian, topology = build_phase2_continuity_vqe(qbit_num)
            start = perf_counter()
            descriptor_set = build_phase3_continuity_partition_descriptor_set(
                vqe,
                workload_id=f"phase2_xxz_hea_q{qbit_num}_continuity",
                max_partition_qubits=candidate.max_partition_qubits,
            )
            planning_time_ms = (perf_counter() - start) * 1000.0
            parameters = build_initial_parameters(vqe.get_Parameter_Num())
            metadata = _metadata_from_descriptor(
                candidate,
                descriptor_set,
                case_kind=PLANNER_CALIBRATION_CASE_KIND_CONTINUITY,
                planning_time_ms=planning_time_ms,
            )
            metadata["topology"] = topology
            yield metadata, descriptor_set, parameters, hamiltonian


def iter_planner_calibration_microcase_cases(
    candidates: tuple[PlannerCandidate, ...] | None = None,
):
    active_candidates = (
        build_planner_candidates() if candidates is None else tuple(candidates)
    )
    for candidate in active_candidates:
        for definition in mandatory_microcase_definitions():
            start = perf_counter()
            descriptor_set = build_microcase_descriptor_set(
                definition["case_name"],
                max_partition_qubits=candidate.max_partition_qubits,
            )
            planning_time_ms = (perf_counter() - start) * 1000.0
            metadata = _metadata_from_descriptor(
                candidate,
                descriptor_set,
                case_kind=PLANNER_CALIBRATION_CASE_KIND_MICROCASE,
                planning_time_ms=planning_time_ms,
                noise_pattern=definition["noise_pattern"],
            )
            parameters = build_initial_parameters(descriptor_set.parameter_count)
            yield metadata, descriptor_set, parameters, None


def iter_planner_calibration_structured_cases(
    candidates: tuple[PlannerCandidate, ...] | None = None,
):
    active_candidates = (
        build_planner_candidates() if candidates is None else tuple(candidates)
    )
    for candidate in active_candidates:
        for family_name in STRUCTURED_FAMILY_NAMES:
            for qbit_num in STRUCTURED_QUBITS:
                for noise_pattern in MANDATORY_NOISE_PATTERNS:
                    start = perf_counter()
                    descriptor_set = build_structured_descriptor_set(
                        family_name,
                        qbit_num=qbit_num,
                        noise_pattern=noise_pattern,
                        seed=DEFAULT_STRUCTURED_SEED,
                        max_partition_qubits=candidate.max_partition_qubits,
                    )
                    planning_time_ms = (perf_counter() - start) * 1000.0
                    metadata = _metadata_from_descriptor(
                        candidate,
                        descriptor_set,
                        case_kind=PLANNER_CALIBRATION_CASE_KIND_STRUCTURED,
                        planning_time_ms=planning_time_ms,
                        family_name=family_name,
                        noise_pattern=noise_pattern,
                        seed=DEFAULT_STRUCTURED_SEED,
                    )
                    parameters = build_initial_parameters(descriptor_set.parameter_count)
                    yield metadata, descriptor_set, parameters, None


def iter_planner_calibration_candidate_cases(
    candidates: tuple[PlannerCandidate, ...] | None = None,
):
    active_candidates = (
        build_planner_candidates() if candidates is None else tuple(candidates)
    )
    yield from iter_planner_calibration_continuity_cases(active_candidates)
    yield from iter_planner_calibration_microcase_cases(active_candidates)
    yield from iter_planner_calibration_structured_cases(active_candidates)


def iter_density_signal_cases(
    candidates: tuple[PlannerCandidate, ...] | None = None,
):
    active_candidates = (
        build_planner_candidates() if candidates is None else tuple(candidates)
    )
    continuity_qbits = set(DENSITY_SIGNAL_CONTINUITY_QUBITS)
    microcase_ids = set(DENSITY_SIGNAL_MICROCASE_IDS)
    structured_noise_patterns = set(DENSITY_SIGNAL_STRUCTURED_FILTER["noise_patterns"])

    for metadata, descriptor_set, parameters, hamiltonian in iter_planner_calibration_continuity_cases(
        active_candidates
    ):
        if metadata["qbit_num"] in continuity_qbits:
            yield metadata, descriptor_set, parameters, hamiltonian

    for metadata, descriptor_set, parameters, hamiltonian in iter_planner_calibration_microcase_cases(
        active_candidates
    ):
        if metadata["workload_id"] in microcase_ids:
            yield metadata, descriptor_set, parameters, hamiltonian

    for metadata, descriptor_set, parameters, hamiltonian in iter_planner_calibration_structured_cases(
        active_candidates
    ):
        if (
            metadata["family_name"] == DENSITY_SIGNAL_STRUCTURED_FILTER["family_name"]
            and metadata["qbit_num"] == DENSITY_SIGNAL_STRUCTURED_FILTER["qbit_num"]
            and metadata["noise_pattern"] in structured_noise_patterns
        ):
            yield metadata, descriptor_set, parameters, hamiltonian
