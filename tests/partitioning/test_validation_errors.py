"""Phase 3 structured validation errors share a common base and stable to_dict() schemas."""

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander.partitioning.noisy_validation_errors import (
    NoisyDescriptorValidationError,
    NoisyPhase3ValidationErrorBase,
    NoisyPlannerValidationError,
    NoisyRuntimeValidationError,
)


def test_planner_validation_error_to_dict_has_no_workload_or_runtime_path():
    exc = NoisyPlannerValidationError(
        category="mode",
        first_unsupported_condition="x",
        failure_stage="s",
        source_type="generated_hea",
        requested_mode="other",
        reason="r",
    )
    assert not hasattr(exc, "workload_id")
    assert not hasattr(exc, "runtime_path")
    assert exc.to_dict() == {
        "unsupported_category": "mode",
        "first_unsupported_condition": "x",
        "failure_stage": "s",
        "source_type": "generated_hea",
        "requested_mode": "other",
        "reason": "r",
    }


def test_descriptor_validation_error_to_dict_includes_workload_id_only():
    exc = NoisyDescriptorValidationError(
        category="c",
        first_unsupported_condition="f",
        failure_stage="st",
        source_type="generated_hea",
        requested_mode="partitioned_density",
        workload_id="w1",
        reason="msg",
    )
    assert exc.to_dict() == {
        "unsupported_category": "c",
        "first_unsupported_condition": "f",
        "failure_stage": "st",
        "source_type": "generated_hea",
        "requested_mode": "partitioned_density",
        "workload_id": "w1",
        "reason": "msg",
    }
    assert not hasattr(exc, "runtime_path")


def test_runtime_validation_error_to_dict_includes_workload_and_runtime_path():
    exc = NoisyRuntimeValidationError(
        category="c",
        first_unsupported_condition="f",
        failure_stage="st",
        source_type="generated_hea",
        requested_mode="partitioned_density",
        workload_id="w1",
        runtime_path="partitioned_density_descriptor_baseline",
        reason="msg",
    )
    assert exc.to_dict() == {
        "unsupported_category": "c",
        "first_unsupported_condition": "f",
        "failure_stage": "st",
        "source_type": "generated_hea",
        "requested_mode": "partitioned_density",
        "workload_id": "w1",
        "runtime_path": "partitioned_density_descriptor_baseline",
        "reason": "msg",
    }


@pytest.mark.parametrize(
    "exc",
    [
        NoisyPlannerValidationError(
            category="a",
            first_unsupported_condition="b",
            failure_stage="c",
            source_type="d",
            requested_mode="e",
            reason="f",
        ),
        NoisyDescriptorValidationError(
            category="a",
            first_unsupported_condition="b",
            failure_stage="c",
            source_type="d",
            requested_mode="e",
            workload_id="w",
            reason="f",
        ),
        NoisyRuntimeValidationError(
            category="a",
            first_unsupported_condition="b",
            failure_stage="c",
            source_type="d",
            requested_mode="e",
            workload_id="w",
            runtime_path="p",
            reason="f",
        ),
    ],
)
def test_all_concrete_errors_are_instances_of_shared_base(exc):
    assert isinstance(exc, NoisyPhase3ValidationErrorBase)
    assert isinstance(exc, ValueError)
