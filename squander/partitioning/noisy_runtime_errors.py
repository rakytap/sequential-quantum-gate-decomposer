from __future__ import annotations

from typing import Any

from squander.partitioning.noisy_types import NoisyPartitionDescriptorSet


class NoisyRuntimeValidationError(ValueError):
    """Structured runtime validation error for the Phase 3 partitioned runtime."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        workload_id: str,
        runtime_path: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.first_unsupported_condition = first_unsupported_condition
        self.failure_stage = failure_stage
        self.source_type = source_type
        self.requested_mode = requested_mode
        self.workload_id = workload_id
        self.runtime_path = runtime_path
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "unsupported_category": self.category,
            "first_unsupported_condition": self.first_unsupported_condition,
            "failure_stage": self.failure_stage,
            "source_type": self.source_type,
            "requested_mode": self.requested_mode,
            "workload_id": self.workload_id,
            "runtime_path": self.runtime_path,
            "reason": self.reason,
        }


def runtime_validation_error(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    category: str,
    first_unsupported_condition: str,
    failure_stage: str,
    runtime_path: str,
    reason: str,
) -> NoisyRuntimeValidationError:
    return NoisyRuntimeValidationError(
        category=category,
        first_unsupported_condition=first_unsupported_condition,
        failure_stage=failure_stage,
        source_type=descriptor_set.source_type,
        requested_mode=descriptor_set.requested_mode,
        workload_id=descriptor_set.workload_id,
        runtime_path=runtime_path,
        reason=reason,
    )
