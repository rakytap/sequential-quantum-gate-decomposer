from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from benchmarks.density_matrix.correctness_evidence.common import (
    CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_EXTERNAL,
    CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
    build_correctness_evidence_selected_candidate,
    build_validation_slice,
)
from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import build_phase2_continuity_vqe
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
    NoisyPartitionDescriptorSet,
    build_phase3_continuity_partition_descriptor_set,
)

CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY = "continuity"
CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE = "microcase"
CORRECTNESS_EVIDENCE_CASE_KIND_STRUCTURED = "structured_family"
CORRECTNESS_EVIDENCE_CONTINUITY_QUBITS = (4, 6, 8, 10)
CORRECTNESS_EVIDENCE_EXTERNAL_REFERENCE_CONTINUITY_QUBITS = (4,)


@dataclass(frozen=True)
class CorrectnessEvidenceCaseContext:
    metadata: dict[str, Any]
    descriptor_set: NoisyPartitionDescriptorSet
    parameters: np.ndarray
    hamiltonian: Any | None = None


def _selected_partition_qubits() -> int:
    return int(build_correctness_evidence_selected_candidate()["max_partition_qubits"])


def _base_metadata_from_descriptor(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    case_kind: str,
    family_name: str | None = None,
    noise_pattern: str | None = None,
    seed: int | None = None,
    topology: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    external_reference_required = case_kind == CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE or (
        case_kind == CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY
        and descriptor_set.qbit_num in CORRECTNESS_EVIDENCE_EXTERNAL_REFERENCE_CONTINUITY_QUBITS
    )
    selected_candidate = build_correctness_evidence_selected_candidate()
    return {
        "case_name": descriptor_set.workload_id,
        "case_kind": case_kind,
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
        "entry_route": descriptor_set.entry_route,
        "workload_family": descriptor_set.workload_family,
        "workload_id": descriptor_set.workload_id,
        "qbit_num": descriptor_set.qbit_num,
        "parameter_count": descriptor_set.parameter_count,
        "family_name": family_name,
        "noise_pattern": noise_pattern,
        "seed": seed,
        "topology": list(topology) if topology is not None else None,
        "validation_slice": build_validation_slice(
            external_reference_required=external_reference_required
        ),
        "external_reference_required": external_reference_required,
        "reference_backend_internal": CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_INTERNAL,
        "reference_backend": (
            CORRECTNESS_EVIDENCE_REFERENCE_BACKEND_EXTERNAL if external_reference_required else None
        ),
    }


def iter_correctness_evidence_continuity_cases():
    max_partition_qubits = _selected_partition_qubits()
    for qbit_num in CORRECTNESS_EVIDENCE_CONTINUITY_QUBITS:
        vqe, hamiltonian, topology = build_phase2_continuity_vqe(qbit_num)
        workload_id = f"phase2_xxz_hea_q{qbit_num}_continuity"
        descriptor_set = build_phase3_continuity_partition_descriptor_set(
            vqe,
            workload_id=workload_id,
            max_partition_qubits=max_partition_qubits,
        )
        metadata = _base_metadata_from_descriptor(
            descriptor_set,
            case_kind=CORRECTNESS_EVIDENCE_CASE_KIND_CONTINUITY,
            topology=topology,
        )
        yield CorrectnessEvidenceCaseContext(
            metadata=metadata,
            descriptor_set=descriptor_set,
            parameters=build_initial_parameters(vqe.get_Parameter_Num()),
            hamiltonian=hamiltonian,
        )


def iter_correctness_evidence_microcase_cases():
    max_partition_qubits = _selected_partition_qubits()
    for definition in mandatory_microcase_definitions():
        descriptor_set = build_microcase_descriptor_set(
            definition["case_name"],
            max_partition_qubits=max_partition_qubits,
        )
        metadata = _base_metadata_from_descriptor(
            descriptor_set,
            case_kind=CORRECTNESS_EVIDENCE_CASE_KIND_MICROCASE,
            noise_pattern=definition["noise_pattern"],
        )
        yield CorrectnessEvidenceCaseContext(
            metadata=metadata,
            descriptor_set=descriptor_set,
            parameters=build_initial_parameters(descriptor_set.parameter_count),
        )


def iter_correctness_evidence_structured_cases():
    max_partition_qubits = _selected_partition_qubits()
    for family_name in STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in MANDATORY_NOISE_PATTERNS:
                descriptor_set = build_structured_descriptor_set(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=DEFAULT_STRUCTURED_SEED,
                    max_partition_qubits=max_partition_qubits,
                )
                metadata = _base_metadata_from_descriptor(
                    descriptor_set,
                    case_kind=CORRECTNESS_EVIDENCE_CASE_KIND_STRUCTURED,
                    family_name=family_name,
                    noise_pattern=noise_pattern,
                    seed=DEFAULT_STRUCTURED_SEED,
                )
                yield CorrectnessEvidenceCaseContext(
                    metadata=metadata,
                    descriptor_set=descriptor_set,
                    parameters=build_initial_parameters(descriptor_set.parameter_count),
                )


@lru_cache(maxsize=1)
def _build_correctness_evidence_case_contexts_cached() -> tuple[CorrectnessEvidenceCaseContext, ...]:
    cases: list[CorrectnessEvidenceCaseContext] = []
    cases.extend(iter_correctness_evidence_continuity_cases())
    cases.extend(iter_correctness_evidence_microcase_cases())
    cases.extend(iter_correctness_evidence_structured_cases())
    return tuple(cases)


def build_correctness_evidence_case_contexts() -> list[CorrectnessEvidenceCaseContext]:
    return deepcopy(list(_build_correctness_evidence_case_contexts_cached()))
