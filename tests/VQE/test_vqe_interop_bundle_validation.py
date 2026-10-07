"""Negative and schema checks for interop_profile bundle validation."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.density_matrix.interop_profile.interop_bundle_validation import (
    COUNTED_PAIRS_REQUIRED,
    INTEROP_BATCH_EXCLUSION_NOTE,
    THROUGHPUT_DIVISOR_W6_REQUIRED,
    THROUGHPUT_DIVISOR_W8_REQUIRED,
    Z_95,
    assert_mean_o_within_margin,
    spike_count_abs_wrapper_ns_above_20000,
    validate_interop_bundle,
    validate_interop_bundle_w6,
    validate_interop_bundle_w8,
    validate_interop_implementation_paths,
)
from benchmarks.density_matrix.interop_profile.interop_lane import (
    COUNTED_REGENERATION_COMMAND_W6,
    COUNTED_REGENERATION_COMMAND_W8,
    DENSITY_NOISE,
    REGENERATION_COMMAND,
)
from benchmarks.density_matrix.interop_profile.validation_pipeline import (
    resolve_interop_output_path,
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


def _one_sided_upper_bound(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = variance**0.5
    bound = mean + Z_95 * std / (len(values) ** 0.5)
    return mean, bound


def _minimal_provenance() -> dict:
    return {
        "implementation_revision": "deadbeef",
        "clean_start": True,
        "dirty_paths": [],
        "command": "conda run -n qgd --no-capture-output python validation_pipeline.py",
        "host": "test-host",
        "cpu_model": "Test CPU",
        "compiler": {
            "executable": "c++",
            "version_line": "g++ test",
            "build_profile": "Release",
        },
        "environment": {
            "conda_default_env": "qgd",
            "conda_prefix": "/tmp/qgd",
            "python_executable": "/tmp/qgd/bin/python",
            "python_version": "3.13.0",
            "host": "test-host",
        },
        "dependencies": {
            "numpy": "2.0.0",
            "scipy": "1.14.0",
            "squander": "0.0.0",
        },
        "extension_identities": [
            {"path": "squander/libqgd.so", "sha256": "a" * 64},
            {"path": "squander/VQA/wrapper.so", "sha256": "b" * 64},
        ],
        "provenance_pass": True,
    }


def _minimal_valid_bundle() -> dict:
    sample = _minimal_valid_sample()
    samples = [copy.deepcopy(sample) for _ in range(COUNTED_PAIRS_REQUIRED)]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    throughput_values = [row["subtimes_ns"][3] / 3072 for row in samples]
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    wrapper_values = [row["t_public_ns"] - row["t_lower_ns"] for row in samples]
    allocate_values = [
        row["subtimes_ns"][0]
        + row["subtimes_ns"][1]
        + row["subtimes_ns"][2]
        + row["subtimes_ns"][5]
        for row in samples
    ]
    apply_values = [row["subtimes_ns"][3] for row in samples]
    contraction_values = [row["subtimes_ns"][4] for row in samples]

    return {
        "suite": "interop_profile_task1_evqe_4q_v1",
        "operation_count": 12,
        "qbit_num": 4,
        "warmup_pairs": 50,
        "counted_pairs": COUNTED_PAIRS_REQUIRED,
        "milestone_counted": False,
        "clean_start": True,
        "provenance": _minimal_provenance(),
        "claim_boundary": "task-1 tracer",
        "labels": "E-VQE",
        "harness_timer_flag": True,
        "protocol": {
            "pairing": "paired_not_interleaved",
            "public_first_on_even_index": True,
            "warmup_pairs": 50,
            "counted_pairs": COUNTED_PAIRS_REQUIRED,
            "parameter_rule": "linspace",
            "parameter_count": 18,
        },
        "estimator": {
            "name": "arithmetic_mean_O_i",
            "formula": "O_i",
            "one_sided_95": "mean + z*s/sqrt(n)",
            "no_sample_dropped": True,
        },
        "workload": {
            "entry": "Optimization_Problem density_matrix",
            "qbit_num": 4,
            "ansatz": "HEA",
            "hamiltonian_nnz": 40,
            "hamiltonian_csr_sha256": "c" * 64,
        },
        "qa008": {
            "categorical_labels_exact": True,
            "mean_O_absolute_margin": 0.02,
        },
        "components": {
            "mean_wrapper_ns": sum(wrapper_values) / len(wrapper_values),
            "mean_allocate_build_ns": sum(allocate_values) / len(allocate_values),
            "mean_apply_to_ns": sum(apply_values) / len(apply_values),
            "mean_contraction_ns": sum(contraction_values) / len(contraction_values),
        },
        "overhead": {"mean_O": mean_o, "upper_bound_95_O": bound_o},
        "throughput": {
            "divisor": 3072,
            "mean_ns_per_op": mean_tp,
            "upper_bound_95_ns_per_op": bound_tp,
        },
        "samples": samples,
    }


def test_validate_accepts_minimal_fixture():
    validate_interop_bundle(_minimal_valid_bundle())


def test_validate_rejects_fewer_than_1000_pairs():
    bundle = _minimal_valid_bundle()
    bundle["samples"] = bundle["samples"][:10]
    with pytest.raises(ValueError, match="counted_pairs"):
        validate_interop_bundle(bundle)


def test_validate_rejects_qa007_met_in_claim_boundary():
    bundle = _minimal_valid_bundle()
    bundle["claim_boundary"] = "QA-007 met on this row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle(bundle)


def test_validate_rejects_qa007_met_anywhere_in_bundle():
    bundle = _minimal_valid_bundle()
    bundle["suite"] = "interop QA-007 met tracer"
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


def test_validate_rejects_missing_throughput_mean():
    bundle = _minimal_valid_bundle()
    del bundle["throughput"]["mean_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle(bundle)


def test_validate_rejects_missing_throughput_bound():
    bundle = _minimal_valid_bundle()
    del bundle["throughput"]["upper_bound_95_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle(bundle)


def test_validate_rejects_attribution_route_label():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "R-oracle overhead row"
    with pytest.raises(ValueError, match="R-oracle"):
        validate_interop_bundle(bundle)


def test_validate_rejects_batch_entry_label():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "Optimization_Problem_Batch timed"
    with pytest.raises(ValueError, match="Optimization_Problem_Batch"):
        validate_interop_bundle(bundle)


def test_validate_rejects_second_public_energy_symbol():
    bundle = _minimal_valid_bundle()
    bundle["labels"] = "timed harness_density_lower_energy"
    with pytest.raises(ValueError, match="harness_density_lower_energy"):
        validate_interop_bundle(bundle)


def test_validate_rejects_clean_start_false():
    bundle = _minimal_valid_bundle()
    bundle["clean_start"] = False
    bundle["provenance"]["clean_start"] = False
    with pytest.raises(ValueError, match="clean_start"):
        validate_interop_bundle(bundle)


def test_validate_rejects_incomplete_provenance():
    bundle = _minimal_valid_bundle()
    del bundle["provenance"]["cpu_model"]
    with pytest.raises(ValueError, match="cpu_model"):
        validate_interop_bundle(bundle)


def test_batch_exclusion_note_documents_virtual_dispatch():
    assert "virtual" in INTEROP_BATCH_EXCLUSION_NOTE.lower()
    assert "Optimization_Problem_Batch" in INTEROP_BATCH_EXCLUSION_NOTE


def test_forbidden_partitioning_path_in_change_set():
    with pytest.raises(ValueError, match="partitioning"):
        validate_interop_implementation_paths(
            ["squander/partitioning/noisy_planner.py"]
        )


def test_forbidden_path_check_accepts_allowlisted_only():
    validate_interop_implementation_paths(
        [
            "benchmarks/density_matrix/interop_profile/interop_lane.py",
            "tests/VQE/test_vqe_interop_bundle_validation.py",
        ]
    )


def test_serialized_bundle_scan_catches_qa007_met():
    bundle = _minimal_valid_bundle()
    blob = json.dumps(bundle)
    assert "QA-007 met" not in blob


def _minimal_valid_bundle_w6() -> dict:
    bundle = copy.deepcopy(_minimal_valid_bundle())
    bundle["suite"] = "interop_profile_task2_evqe_6q_v1"
    bundle["qbit_num"] = 6
    bundle["operation_count"] = 18
    bundle["claim_boundary"] = "task-2 tracer row; milestone_counted=false"
    bundle["labels"] = (
        "E-VQE density_matrix harness tracer width 6; no reduction taken on widths 4 and 6"
    )
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W6
    bundle["workload"]["qbit_num"] = 6
    bundle["throughput"]["divisor"] = THROUGHPUT_DIVISOR_W6_REQUIRED
    samples = bundle["samples"]
    throughput_values = [row["subtimes_ns"][3] / THROUGHPUT_DIVISOR_W6_REQUIRED for row in samples]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    bundle["overhead"] = {
        "mean_O": mean_o,
        "median_O": sorted(o_values)[len(o_values) // 2],
        "upper_bound_95_O": bound_o,
        "min_O": min(o_values),
        "max_O": max(o_values),
        "spike_count_abs_wrapper_ns_above_20000": spike_count_abs_wrapper_ns_above_20000(
            samples
        ),
    }
    bundle["throughput"]["mean_ns_per_op"] = mean_tp
    bundle["throughput"]["upper_bound_95_ns_per_op"] = bound_tp
    return bundle


def test_validate_w6_accepts_minimal_fixture():
    validate_interop_bundle_w6(_minimal_valid_bundle_w6())


def test_validate_w6_accepts_no_reduction_wording():
    bundle = _minimal_valid_bundle_w6()
    validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_width4_regeneration_command_in_provenance():
    bundle = _minimal_valid_bundle_w6()
    bundle["provenance"]["command"] = REGENERATION_COMMAND
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_provenance_command_missing_width_flag():
    bundle = _minimal_valid_bundle_w6()
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W6.removesuffix(
        " --width 6"
    )
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_attribution_route_label():
    bundle = _minimal_valid_bundle_w6()
    bundle["labels"] = "R-base overhead row"
    with pytest.raises(ValueError, match="attribution route label"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_wrong_divisor_3072():
    bundle = _minimal_valid_bundle_w6()
    bundle["throughput"]["divisor"] = 3072
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_wrong_divisor_4096():
    bundle = _minimal_valid_bundle_w6()
    bundle["throughput"]["divisor"] = 4096
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_wrong_divisor_65536():
    bundle = _minimal_valid_bundle_w6()
    bundle["throughput"]["divisor"] = 65536
    with pytest.raises(ValueError, match="divisor"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_missing_throughput_mean():
    bundle = _minimal_valid_bundle_w6()
    del bundle["throughput"]["mean_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_missing_throughput_bound():
    bundle = _minimal_valid_bundle_w6()
    del bundle["throughput"]["upper_bound_95_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_spike_count_mismatch():
    bundle = _minimal_valid_bundle_w6()
    bundle["overhead"]["spike_count_abs_wrapper_ns_above_20000"] = 999
    with pytest.raises(ValueError, match="spike_count"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_a4_kill_claim():
    bundle = _minimal_valid_bundle_w6()
    bundle["labels"] = "A4 kill on this row"
    with pytest.raises(ValueError, match="A4 kill"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_hold_the_line_claim():
    bundle = _minimal_valid_bundle_w6()
    bundle["claim_boundary"] = "CAP-004 hold-the-line label"
    with pytest.raises(ValueError, match="hold-the-line"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_reduction_taken_claim():
    bundle = _minimal_valid_bundle_w6()
    bundle["labels"] = "reduction taken on width 6"
    with pytest.raises(ValueError, match="reduction taken"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_width_8():
    bundle = _minimal_valid_bundle_w6()
    bundle["qbit_num"] = 8
    with pytest.raises(ValueError, match="width"):
        validate_interop_bundle_w6(bundle)


def test_validate_w6_rejects_qa007_met_in_suite_field():
    bundle = _minimal_valid_bundle_w6()
    bundle["suite"] = "interop QA-007 met row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle_w6(bundle)


def _w6_sample_with_negative_overhead_ratio() -> dict:
    sample = _minimal_valid_sample()
    t_lower = 20_000
    t_public = 10_000
    factor = t_lower / sample["t_lower_ns"]
    sub = [int(round(x * factor)) for x in sample["subtimes_ns"]]
    inner = sub[0] + sub[1] + sub[2] + sub[5] + sub[3] + sub[4]
    sub[3] += t_lower - inner
    return {"t_public_ns": t_public, "t_lower_ns": t_lower, "subtimes_ns": sub}


def test_validate_w6_accepts_negative_mean_o():
    bundle = _minimal_valid_bundle_w6()
    sample = _w6_sample_with_negative_overhead_ratio()
    bundle["samples"] = [copy.deepcopy(sample) for _ in range(COUNTED_PAIRS_REQUIRED)]
    samples = bundle["samples"]
    throughput_values = [
        row["subtimes_ns"][3] / THROUGHPUT_DIVISOR_W6_REQUIRED for row in samples
    ]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    bundle["overhead"] = {
        "mean_O": mean_o,
        "median_O": sorted(o_values)[len(o_values) // 2],
        "upper_bound_95_O": bound_o,
        "min_O": min(o_values),
        "max_O": max(o_values),
        "spike_count_abs_wrapper_ns_above_20000": spike_count_abs_wrapper_ns_above_20000(
            samples
        ),
    }
    bundle["throughput"]["mean_ns_per_op"] = mean_tp
    bundle["throughput"]["upper_bound_95_ns_per_op"] = bound_tp
    wrapper = [row["t_public_ns"] - row["t_lower_ns"] for row in samples]
    allocate = [
        row["subtimes_ns"][0]
        + row["subtimes_ns"][1]
        + row["subtimes_ns"][2]
        + row["subtimes_ns"][5]
        for row in samples
    ]
    apply_to = [row["subtimes_ns"][3] for row in samples]
    contraction = [row["subtimes_ns"][4] for row in samples]
    bundle["components"] = {
        "mean_wrapper_ns": sum(wrapper) / len(wrapper),
        "mean_allocate_build_ns": sum(allocate) / len(allocate),
        "mean_apply_to_ns": sum(apply_to) / len(apply_to),
        "mean_contraction_ns": sum(contraction) / len(contraction),
    }
    validate_interop_bundle_w6(bundle)


def test_spike_count_pure_function():
    samples = [
        {"t_public_ns": 10_000, "t_lower_ns": 50_000, "subtimes_ns": [1, 1, 1, 1, 1, 1]},
        {"t_public_ns": 10_000, "t_lower_ns": 10_500, "subtimes_ns": [1, 1, 1, 1, 1, 1]},
    ]
    assert spike_count_abs_wrapper_ns_above_20000(samples) == 1


def test_assert_mean_o_within_margin_fixture_pair():
    assert_mean_o_within_margin(0.0, 0.0, margin=0.02)
    assert_mean_o_within_margin(0.02, 0.0, margin=0.02)
    with pytest.raises(ValueError, match="margin"):
        assert_mean_o_within_margin(0.03, 0.0, margin=0.02)


def test_resolve_output_path_refuses_width6_to_task1_name(tmp_path):
    bad = tmp_path / "interop_profile_bundle.json"
    with pytest.raises(ValueError, match="interop_profile_bundle_w6"):
        resolve_interop_output_path(6, bad)


def test_resolve_output_path_refuses_width4_to_w6_name(tmp_path):
    bad = tmp_path / "interop_profile_bundle_w6.json"
    with pytest.raises(ValueError, match="interop_profile_bundle.json"):
        resolve_interop_output_path(4, bad)


def test_resolve_output_path_accepts_width6_tmp_output(tmp_path):
    good = tmp_path / "interop_profile_bundle_w6.json"
    assert resolve_interop_output_path(6, good) == good


def _minimal_valid_bundle_w8() -> dict:
    bundle = copy.deepcopy(_minimal_valid_bundle_w6())
    bundle["suite"] = "interop_profile_task3_evqe_8q_v1"
    bundle["qbit_num"] = 8
    bundle["operation_count"] = 24
    bundle["claim_boundary"] = "task-3 tracer row; milestone_counted=false"
    bundle["labels"] = (
        "E-VQE density_matrix harness tracer width 8; QA-007 withheld; "
        "no reduction taken; milestone not complete"
    )
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W8
    bundle["protocol"]["parameter_count"] = 42
    bundle["workload"]["qbit_num"] = 8
    bundle["workload"]["hamiltonian_nnz"] = 1152
    bundle["workload"]["density_noise"] = copy.deepcopy(DENSITY_NOISE)
    bundle["throughput"]["divisor"] = THROUGHPUT_DIVISOR_W8_REQUIRED
    samples = bundle["samples"]
    throughput_values = [
        row["subtimes_ns"][3] / THROUGHPUT_DIVISOR_W8_REQUIRED for row in samples
    ]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    bundle["overhead"] = {
        "mean_O": mean_o,
        "median_O": sorted(o_values)[len(o_values) // 2],
        "upper_bound_95_O": bound_o,
        "min_O": min(o_values),
        "max_O": max(o_values),
        "spike_count_abs_wrapper_ns_above_20000": spike_count_abs_wrapper_ns_above_20000(
            samples
        ),
    }
    bundle["throughput"]["mean_ns_per_op"] = mean_tp
    bundle["throughput"]["upper_bound_95_ns_per_op"] = bound_tp
    return bundle


def test_validate_w8_accepts_minimal_fixture():
    validate_interop_bundle_w8(_minimal_valid_bundle_w8())


def test_validate_w8_rejects_width6_parameter_count():
    bundle = _minimal_valid_bundle_w8()
    bundle["protocol"]["parameter_count"] = 30
    with pytest.raises(ValueError, match="parameter_count"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_width6_hamiltonian_nnz():
    bundle = _minimal_valid_bundle_w8()
    bundle["workload"]["hamiltonian_nnz"] = 224
    with pytest.raises(ValueError, match="hamiltonian_nnz"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_width4_regeneration_command_in_provenance():
    bundle = _minimal_valid_bundle_w8()
    bundle["provenance"]["command"] = REGENERATION_COMMAND
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_width6_command_in_provenance():
    bundle = _minimal_valid_bundle_w8()
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W6
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_provenance_command_missing_width_flag():
    bundle = _minimal_valid_bundle_w8()
    bundle["provenance"]["command"] = COUNTED_REGENERATION_COMMAND_W8.removesuffix(" --width 8")
    with pytest.raises(ValueError, match="provenance.command"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_attribution_route_label():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = "R-base overhead row"
    with pytest.raises(ValueError, match="attribution route label"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_wrong_divisors():
    for divisor in (3072, 73728, 65536):
        bundle = _minimal_valid_bundle_w8()
        bundle["throughput"]["divisor"] = divisor
        with pytest.raises(ValueError, match="divisor"):
            validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_qbit_num_6():
    bundle = _minimal_valid_bundle_w8()
    bundle["qbit_num"] = 6
    with pytest.raises(ValueError, match="width"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_a4_false_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = "A4 FALSE on this row"
    with pytest.raises(ValueError, match="a4 false"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_reduction_justified_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["claim_boundary"] = "reduction justified here"
    with pytest.raises(ValueError, match="reduction justified"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_mf5a_complete_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = "M-F5a complete"
    with pytest.raises(ValueError, match="m-f5a complete"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_a4_kill_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = "A4 kill on this row"
    with pytest.raises(ValueError, match="a4 kill"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_hold_the_line_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["claim_boundary"] = "CAP-004 hold-the-line label"
    with pytest.raises(ValueError, match="hold-the-line"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_reduction_taken_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = "reduction taken on width 8"
    with pytest.raises(ValueError, match="reduction taken"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_reduction_shipped_claim():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = "reduction shipped on this row"
    with pytest.raises(ValueError, match="reduction shipped"):
        validate_interop_bundle_w8(bundle)


@pytest.mark.parametrize("field", ["claim_boundary", "labels"])
def test_validate_w8_rejects_qa007_met_in_metadata(field: str):
    bundle = _minimal_valid_bundle_w8()
    bundle[field] = "QA-007 met on this row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_qa007_met_in_suite_field():
    bundle = _minimal_valid_bundle_w8()
    bundle["suite"] = "interop QA-007 met row"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_operation_count_18():
    bundle = _minimal_valid_bundle_w8()
    bundle["operation_count"] = 18
    with pytest.raises(ValueError, match="operation_count"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_density_noise_not_three_entries():
    bundle = _minimal_valid_bundle_w8()
    bundle["workload"]["density_noise"] = copy.deepcopy(DENSITY_NOISE)[:2]
    with pytest.raises(ValueError, match="density_noise"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_missing_throughput_mean():
    bundle = _minimal_valid_bundle_w8()
    del bundle["throughput"]["mean_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_missing_throughput_bound():
    bundle = _minimal_valid_bundle_w8()
    del bundle["throughput"]["upper_bound_95_ns_per_op"]
    with pytest.raises(ValueError, match="throughput mean"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_rejects_spike_count_mismatch():
    bundle = _minimal_valid_bundle_w8()
    bundle["overhead"]["spike_count_abs_wrapper_ns_above_20000"] = 999
    with pytest.raises(ValueError, match="spike_count"):
        validate_interop_bundle_w8(bundle)


def test_validate_w8_accepts_lawful_negation_phrases():
    bundle = _minimal_valid_bundle_w8()
    bundle["labels"] = (
        "no reduction taken; milestone not complete; QA-007 withheld; no M-F5a complete"
    )
    validate_interop_bundle_w8(bundle)


def _w8_sample_partition_consistent(t_public: int, t_lower: int) -> dict:
    base = _minimal_valid_sample()
    factor = t_lower / base["t_lower_ns"]
    sub = [int(round(x * factor)) for x in base["subtimes_ns"]]
    inner = sub[0] + sub[1] + sub[2] + sub[5] + sub[3] + sub[4]
    sub[3] += t_lower - inner
    return {"t_public_ns": t_public, "t_lower_ns": t_lower, "subtimes_ns": sub}


def _apply_w8_samples_to_bundle(bundle: dict, samples: list[dict]) -> None:
    bundle["samples"] = samples
    throughput_values = [
        row["subtimes_ns"][3] / THROUGHPUT_DIVISOR_W8_REQUIRED for row in samples
    ]
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    mean_o, bound_o = _one_sided_upper_bound(o_values)
    mean_tp, bound_tp = _one_sided_upper_bound(throughput_values)
    wrapper = [row["t_public_ns"] - row["t_lower_ns"] for row in samples]
    allocate = [
        row["subtimes_ns"][0]
        + row["subtimes_ns"][1]
        + row["subtimes_ns"][2]
        + row["subtimes_ns"][5]
        for row in samples
    ]
    apply_to = [row["subtimes_ns"][3] for row in samples]
    contraction = [row["subtimes_ns"][4] for row in samples]
    bundle["overhead"] = {
        "mean_O": mean_o,
        "median_O": float(np_median_helper(o_values)),
        "upper_bound_95_O": bound_o,
        "min_O": min(o_values),
        "max_O": max(o_values),
        "spike_count_abs_wrapper_ns_above_20000": spike_count_abs_wrapper_ns_above_20000(
            samples
        ),
    }
    bundle["throughput"]["mean_ns_per_op"] = mean_tp
    bundle["throughput"]["upper_bound_95_ns_per_op"] = bound_tp
    bundle["components"] = {
        "mean_wrapper_ns": sum(wrapper) / len(wrapper),
        "mean_allocate_build_ns": sum(allocate) / len(allocate),
        "mean_apply_to_ns": sum(apply_to) / len(apply_to),
        "mean_contraction_ns": sum(contraction) / len(contraction),
    }


def test_validate_w8_accepts_negative_mean_o():
    bundle = _minimal_valid_bundle_w8()
    sample = _w8_sample_partition_consistent(10_000, 20_000)
    samples = [copy.deepcopy(sample) for _ in range(COUNTED_PAIRS_REQUIRED)]
    _apply_w8_samples_to_bundle(bundle, samples)
    assert bundle["overhead"]["mean_O"] < 0
    assert bundle["components"]["mean_wrapper_ns"] < 0
    validate_interop_bundle_w8(bundle)


def test_validate_w8_accepts_heterogeneous_overhead_fixture():
    bundle = _minimal_valid_bundle_w8()
    specs = [(30_000, 10_000), (11_000, 18_000)]
    specs += [(20_000, 20_100)] * 499
    specs += [(20_000, 20_060)] * 499
    assert len(specs) == COUNTED_PAIRS_REQUIRED
    samples = [_w8_sample_partition_consistent(t_public, t_lower) for t_public, t_lower in specs]
    _apply_w8_samples_to_bundle(bundle, samples)
    o_values = [
        (row["t_public_ns"] - row["t_lower_ns"]) / row["t_public_ns"] for row in samples
    ]
    assert min(o_values) < -0.5
    assert bundle["overhead"]["mean_O"] < 0
    assert bundle["components"]["mean_wrapper_ns"] < 0
    assert bundle["overhead"]["min_O"] != bundle["overhead"]["max_O"]
    assert bundle["overhead"]["median_O"] not in (
        bundle["overhead"]["min_O"],
        bundle["overhead"]["max_O"],
    )
    validate_interop_bundle_w8(bundle)


def np_median_helper(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def test_resolve_output_path_refuses_width8_to_committed_names(tmp_path):
    for name in ("interop_profile_bundle.json", "interop_profile_bundle_w6.json"):
        bad = tmp_path / name
        with pytest.raises(ValueError, match="interop_profile_bundle_w8"):
            resolve_interop_output_path(8, bad)


def test_resolve_output_path_refuses_width4_and_6_to_w8_name(tmp_path):
    bad = tmp_path / "interop_profile_bundle_w8.json"
    with pytest.raises(ValueError, match="interop_profile_bundle.json"):
        resolve_interop_output_path(4, bad)
    with pytest.raises(ValueError, match="interop_profile_bundle_w6"):
        resolve_interop_output_path(6, bad)


def test_resolve_output_path_accepts_width8_tmp_output(tmp_path):
    good = tmp_path / "interop_profile_bundle_w8.json"
    assert resolve_interop_output_path(8, good) == good


def test_pipeline_width8_dispatches_qbit_num_8(monkeypatch, tmp_path):
    from benchmarks.density_matrix.interop_profile import validation_pipeline as vp

    captured: dict[str, int] = {}

    def fake_run(*, qbit_num: int = 4, **kwargs):
        captured["qbit_num"] = qbit_num
        return _minimal_valid_bundle_w8()

    monkeypatch.setattr(vp, "run_interop_row", fake_run)
    out = tmp_path / "interop_profile_bundle_w8.json"
    assert vp.main(["--width", "8", "--output", str(out)]) == 0
    assert captured["qbit_num"] == 8
    assert out.is_file()


def test_resolve_output_path_rejects_unsupported_width():
    with pytest.raises(ValueError, match="unsupported"):
        resolve_interop_output_path(10, None)


def _minimal_attribution_route_bundle() -> dict:
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        ROUTE_IDS_REQUIRED,
        SUITE_ID_TASK4_ROUTES,
        WORKLOAD_LABEL_TASK4,
        one_sided_upper_bound_route,
    )

    samples = [
        {"orchestration_ns": 1_000, "apply_component_ns": 5_000},
        {"orchestration_ns": 1_100, "apply_component_ns": 5_100},
    ]
    rows = []
    for route_id in ROUTE_IDS_REQUIRED:
        orch_values = [float(sample["orchestration_ns"]) for sample in samples]
        apply_values = [float(sample["apply_component_ns"]) for sample in samples]
        tp_values = [value / 3072 for value in apply_values]
        mean_orch, bound_orch = one_sided_upper_bound_route(orch_values)
        mean_apply, bound_apply = one_sided_upper_bound_route(apply_values)
        mean_tp, bound_tp = one_sided_upper_bound_route(tp_values)
        hybrid_partition_classes = [
            "phase31_channel_native",
            "phase3_unitary_island_fused",
            "phase31_channel_native",
            "phase3_unitary_island_fused",
            "phase3_unitary_island_fused",
        ]
        apply_label = (
            "2× phase31_channel_native; 3× phase3_unitary_island_fused"
            if route_id == "R-hybrid"
            else "synthetic apply label"
        )
        row_payload = {
                "route_id": route_id,
                "entry_symbol": f"execute_{route_id}",
                "apply_label": apply_label,
                "samples": copy.deepcopy(samples),
                "orchestration": {
                    "mean_ns": mean_orch,
                    "upper_bound_95_ns": bound_orch,
                },
                "apply_component": {
                    "mean_ns": mean_apply,
                    "upper_bound_95_ns": bound_apply,
                },
                "throughput": {
                    "divisor": 3072,
                    "mean_ns_per_op": mean_tp,
                    "upper_bound_95_ns_per_op": bound_tp,
                },
            }
        if route_id == "R-hybrid":
            row_payload["partition_runtime_classes"] = hybrid_partition_classes
        rows.append(row_payload)
    return {
        "suite": SUITE_ID_TASK4_ROUTES,
        "qbit_num": 4,
        "milestone_counted": False,
        "workload_label": WORKLOAD_LABEL_TASK4,
        "claim_boundary": "task-4 synthetic fixture",
        "labels": "attribution-only; QA-007 withheld on routes",
        "bridge": {
            "parameter_count": 18,
            "operation_count": 12,
            "gate_count": 9,
            "noise_count": 3,
            "source_type": "generated_hea",
        },
        "rows": rows,
    }


def test_validate_attribution_route_bundle_accepts_four_routes():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    validate_attribution_route_bundle(_minimal_attribution_route_bundle())


def test_validate_attribution_route_bundle_rejects_mean_o():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["rows"][0]["mean_O"] = 0.1
    with pytest.raises(ValueError, match="overhead ratio"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_qa007_met():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["labels"] = "QA-007 met on routes"
    with pytest.raises(ValueError, match="QA-007 met"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_reduction_claim():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["claim_boundary"] = "reduction shipped on routes"
    with pytest.raises(ValueError, match="reduction shipped"):
        validate_attribution_route_bundle(bundle)


def test_validate_r_oracle_claim_boundary_requires_e1_sentence():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_r_oracle_claim_boundary,
    )

    with pytest.raises(ValueError, match="E1"):
        validate_r_oracle_claim_boundary("R-oracle row without the required sentence")
    validate_r_oracle_claim_boundary(
        "R-oracle row exists only to label C++ apply_to for the E-VQE diagnosis"
    )


def test_resolve_attribution_output_refuses_counted_evqe_bundle(tmp_path):
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        resolve_attribution_output_path,
    )

    bad = tmp_path / "interop_profile_bundle.json"
    with pytest.raises(ValueError, match="counted E-VQE"):
        resolve_attribution_output_path(bad)


def test_validate_attribution_route_bundle_rejects_r_oracle_row():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["rows"].append(
        {
            "route_id": "R-oracle",
            "entry_symbol": "execute_sequential_density_reference",
            "apply_label": "oracle",
            "samples": bundle["rows"][0]["samples"],
            "orchestration": bundle["rows"][0]["orchestration"],
            "apply_component": bundle["rows"][0]["apply_component"],
            "throughput": bundle["rows"][0]["throughput"],
        }
    )
    with pytest.raises(ValueError, match="R-oracle"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_r_oracle_label_without_e1():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["labels"] = "R-oracle overhead row"
    with pytest.raises(ValueError, match="R-oracle"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_r_oracle_label_even_with_e1():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        E1_ORACLE_REQUIRED_PHRASE,
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["claim_boundary"] = f"R-oracle row {E1_ORACLE_REQUIRED_PHRASE}"
    with pytest.raises(ValueError, match="R-oracle"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_t_lower_ns():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["rows"][0]["t_lower_ns"] = 1
    with pytest.raises(ValueError, match="t_lower_ns"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_upper_bound_95_o():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["rows"][0]["upper_bound_95_O"] = 0.5
    with pytest.raises(ValueError, match="upper_bound_95_O"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_wrong_throughput_divisor():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["rows"][0]["throughput"]["divisor"] = 4096
    with pytest.raises(ValueError, match="divisor"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_milestone_counted_true():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["milestone_counted"] = True
    with pytest.raises(ValueError, match="milestone_counted"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_three_row_bundle_without_r_strict():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    bundle["rows"] = [row for row in bundle["rows"] if row["route_id"] != "R-strict"]
    with pytest.raises(ValueError, match="four attribution routes"):
        validate_attribution_route_bundle(bundle)


def test_validate_attribution_route_bundle_rejects_hybrid_constant_apply_label():
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        validate_attribution_route_bundle,
    )

    bundle = _minimal_attribution_route_bundle()
    for row in bundle["rows"]:
        if row["route_id"] == "R-hybrid":
            row["apply_label"] = "the executed class"
    with pytest.raises(ValueError, match="partition_runtime_classes"):
        validate_attribution_route_bundle(bundle)
