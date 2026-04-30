from __future__ import annotations

from typing import Any

_MISSING = object()


class NoisyPhase3ValidationErrorBase(ValueError):
    """Shared structured payload and serialization for Phase 3 planner/descriptor/runtime errors."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        reason: str,
        workload_id: str | object = _MISSING,
        runtime_path: str | object = _MISSING,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.first_unsupported_condition = first_unsupported_condition
        self.failure_stage = failure_stage
        self.source_type = source_type
        self.requested_mode = requested_mode
        self.reason = reason
        if workload_id is not _MISSING:
            self.workload_id = workload_id  # type: ignore[assignment]
        if runtime_path is not _MISSING:
            self.runtime_path = runtime_path  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "unsupported_category": self.category,
            "first_unsupported_condition": self.first_unsupported_condition,
            "failure_stage": self.failure_stage,
            "source_type": self.source_type,
            "requested_mode": self.requested_mode,
            "reason": self.reason,
        }
        if hasattr(self, "workload_id"):
            payload["workload_id"] = self.workload_id
        if hasattr(self, "runtime_path"):
            payload["runtime_path"] = self.runtime_path
        return payload


class NoisyPlannerValidationError(NoisyPhase3ValidationErrorBase):
    """Structured planner-entry validation error for the Phase 3 noisy planner surface."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        reason: str,
    ) -> None:
        super().__init__(
            category=category,
            first_unsupported_condition=first_unsupported_condition,
            failure_stage=failure_stage,
            source_type=source_type,
            requested_mode=requested_mode,
            reason=reason,
        )


class NoisyDescriptorValidationError(NoisyPhase3ValidationErrorBase):
    """Structured descriptor-generation validation error for the Phase 3 partition descriptor surface."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        workload_id: str,
        reason: str,
    ) -> None:
        super().__init__(
            category=category,
            first_unsupported_condition=first_unsupported_condition,
            failure_stage=failure_stage,
            source_type=source_type,
            requested_mode=requested_mode,
            reason=reason,
            workload_id=workload_id,
        )


class NoisyRuntimeValidationError(NoisyPhase3ValidationErrorBase):
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
        super().__init__(
            category=category,
            first_unsupported_condition=first_unsupported_condition,
            failure_stage=failure_stage,
            source_type=source_type,
            requested_mode=requested_mode,
            reason=reason,
            workload_id=workload_id,
            runtime_path=runtime_path,
        )
