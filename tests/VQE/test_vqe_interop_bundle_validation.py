"""Negative and schema checks for interop_profile bundle validation."""

from __future__ import annotations

import copy

import pytest

from benchmarks.density_matrix.interop_profile.interop_bundle_validation import (
    COUNTED_PAIRS_REQUIRED,
    validate_interop_bundle,
)


def _minimal_valid_sample() -> dict:
    subtimes = [100, 200, 300, 4000, 500, 100]
    t_lower = subtimes[0] + subtimes[1] + subtimes[2] + subtimes[5] + subtimes[3] + subtimes[4]
    t_public = t_lower + 5000
    return {
        "t_public_ns": t_public,
        "t_lower_ns": t_lower,
        "subtimes_ns": subtimes,
    }


def _minimal_valid_bundle() -> dict:
    sample = _minimal_valid_sample()
    return {
        "qbit_num": 4,
        "warmup_pairs": 50,
        "counted_pairs": COUNTED_PAIRS_REQUIRED,
        "milestone_counted": False,
        "claim_boundary": "task-1 tracer",
        "labels": "E-VQE",
        "overhead": {"mean_O": 0.1, "upper_bound_95_O": 0.12},
        "throughput": {
            "divisor": 3072,
            "mean_ns_per_op": 10.0,
            "upper_bound_95_ns_per_op": 12.0,
        },
        "samples": [copy.deepcopy(sample) for _ in range(COUNTED_PAIRS_REQUIRED)],
    }


def test_validate_accepts_minimal_fixture():
    validate_interop_bundle(_minimal_valid_bundle())


def test_validate_rejects_fewer_than_1000_pairs():
    bundle = _minimal_valid_bundle()
    bundle["samples"] = bundle["samples"][:10]
    with pytest.raises(ValueError, match="counted_pairs"):
        validate_interop_bundle(bundle)


def test_validate_rejects_qa007_met_label():
    bundle = _minimal_valid_bundle()
    bundle["claim_boundary"] = "QA-007 met on this row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle(bundle)


def test_validate_rejects_wrong_width():
    bundle = _minimal_valid_bundle()
    bundle["qbit_num"] = 6
    with pytest.raises(ValueError, match="width"):
        validate_interop_bundle(bundle)


def test_validate_rejects_broken_partition():
    bundle = _minimal_valid_bundle()
    bundle["samples"][0]["subtimes_ns"] = [1, 1, 1, 1, 1, 1]
    bundle["samples"][0]["t_lower_ns"] = 1_000_000
    with pytest.raises(ValueError, match="partition"):
        validate_interop_bundle(bundle)


def test_validate_rejects_wrong_throughput_divisor():
    bundle = _minimal_valid_bundle()
    bundle["throughput"]["divisor"] = 4096
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle(bundle)


def test_validate_rejects_attribution_route_label():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "R-base overhead row"
    with pytest.raises(ValueError, match="R-base"):
        validate_interop_bundle(bundle)


def test_validate_rejects_batch_entry_label():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "Optimization_Problem_Batch timed"
    with pytest.raises(ValueError, match="Optimization_Problem_Batch"):
        validate_interop_bundle(bundle)
