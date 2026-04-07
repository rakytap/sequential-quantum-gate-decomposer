from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PLANNER_CANDIDATE_SCHEMA_VERSION = "phase3_planner_calibration_candidate_v1"
PLANNER_FAMILY_SPAN_BUDGET = "phase3_span_budget"
SUPPORTED_CANDIDATE_PARTITION_QUBITS = (2, 3, 4)


@dataclass(frozen=True)
class PlannerCandidate:
    schema_version: str
    candidate_id: str
    planner_family: str
    planner_variant: str
    max_partition_qubits: int

    @property
    def planner_settings(self) -> dict[str, int]:
        return {"max_partition_qubits": self.max_partition_qubits}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "planner_family": self.planner_family,
            "planner_variant": self.planner_variant,
            "planner_settings": dict(self.planner_settings),
            "max_partition_qubits": self.max_partition_qubits,
        }


def build_planner_candidates(
    *,
    max_partition_qubits_values: Iterable[int] = SUPPORTED_CANDIDATE_PARTITION_QUBITS,
) -> tuple[PlannerCandidate, ...]:
    normalized_values = tuple(sorted({int(value) for value in max_partition_qubits_values}))
    if not normalized_values:
        raise ValueError("Planner-candidate surface requires at least one candidate")
    if normalized_values[0] <= 0:
        raise ValueError(
            "Planner-candidate surface requires max_partition_qubits > 0 for every candidate"
        )

    return tuple(
        PlannerCandidate(
            schema_version=PLANNER_CANDIDATE_SCHEMA_VERSION,
            candidate_id=f"span_budget_q{max_partition_qubits}",
            planner_family=PLANNER_FAMILY_SPAN_BUDGET,
            planner_variant=f"max_partition_qubits_{max_partition_qubits}",
            max_partition_qubits=max_partition_qubits,
        )
        for max_partition_qubits in normalized_values
    )


# Compatibility aliases for existing semantic imports.
PLANNER_CALIBRATION_CANDIDATE_SCHEMA_VERSION = PLANNER_CANDIDATE_SCHEMA_VERSION
PLANNER_CALIBRATION_PLANNER_FAMILY_SPAN_BUDGET = PLANNER_FAMILY_SPAN_BUDGET
PLANNER_CALIBRATION_SUPPORTED_CANDIDATE_PARTITION_QUBITS = SUPPORTED_CANDIDATE_PARTITION_QUBITS
PlannerCalibrationPlannerCandidate = PlannerCandidate


def build_planner_calibration_planner_candidates(
    *,
    max_partition_qubits_values: Iterable[int] = SUPPORTED_CANDIDATE_PARTITION_QUBITS,
) -> tuple[PlannerCandidate, ...]:
    return build_planner_candidates(max_partition_qubits_values=max_partition_qubits_values)
