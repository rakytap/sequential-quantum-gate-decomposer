from __future__ import annotations

from squander.partitioning.noisy_types import NoisyPartitionDescriptorSet
from squander.partitioning.noisy_validation_errors import NoisyRuntimeValidationError


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
