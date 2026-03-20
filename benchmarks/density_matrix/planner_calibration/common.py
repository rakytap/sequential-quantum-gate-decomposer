from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

TASK5_CANDIDATE_SCHEMA_VERSION = "phase3_task5_planner_candidate_v1"
TASK5_PLANNER_FAMILY_SPAN_BUDGET = "phase3_span_budget"
TASK5_SUPPORTED_CANDIDATE_PARTITION_QUBITS = (2, 3, 4)


@dataclass(frozen=True)
class Task5PlannerCandidate:
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


def build_task5_planner_candidates(
    *,
    max_partition_qubits_values: Iterable[int] = TASK5_SUPPORTED_CANDIDATE_PARTITION_QUBITS,
) -> tuple[Task5PlannerCandidate, ...]:
    normalized_values = tuple(sorted({int(value) for value in max_partition_qubits_values}))
    if not normalized_values:
        raise ValueError("Task 5 candidate surface requires at least one candidate")
    if normalized_values[0] <= 0:
        raise ValueError(
            "Task 5 candidate surface requires max_partition_qubits > 0 for every candidate"
        )

    return tuple(
        Task5PlannerCandidate(
            schema_version=TASK5_CANDIDATE_SCHEMA_VERSION,
            candidate_id=f"span_budget_q{max_partition_qubits}",
            planner_family=TASK5_PLANNER_FAMILY_SPAN_BUDGET,
            planner_variant=f"max_partition_qubits_{max_partition_qubits}",
            max_partition_qubits=max_partition_qubits,
        )
        for max_partition_qubits in normalized_values
    )
