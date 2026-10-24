"""Reporting must not print mathematical 1e-10 or round positive times to zero."""

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "benchmarks/partitioning/local_synthesis_head_to_head.py"
SPEC = importlib.util.spec_from_file_location("local_synthesis_tex_reporting", SOURCE)
REPORTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTING)


@pytest.mark.parametrize("tolerance,expected", [
    (1e-10, "$10^{-10}$"),
    (2.5e-9, "$2.5\\times 10^{-9}$"),
])
def test_tex_caption_uses_mathematical_exponents(tmp_path, tolerance, expected):
    output = tmp_path / "comparison.tex"
    REPORTING.write_tex_table({}, output, tolerance)
    assert expected in output.read_text()


def test_tex_preserves_small_positive_times_and_distinct_labels(tmp_path):
    summary = {"3q/check": {
        "status_counts": {"osr": {"verified": 1}, "bqskit": {"verified": 1}},
        "circuit_cluster_mean_delta_95pct_ci": None,
        "mean_cnot_delta_osr_minus_bqskit": None,
        "median_seconds_on_paired_verified": {"osr": 2.5e-5, "bqskit": 1.02},
        "osr_wins": 1, "ties": 0, "bqskit_wins": 0,
        "cases": 1, "paired_verified": 1, "bqskit_only_verified": 0,
    }}
    output = tmp_path / "heldout.tex"
    REPORTING.write_tex_table(
        summary, output, 1e-10, table_label="tab:local-osr-heldout",
    )
    text = output.read_text()
    assert "$2.5\\times 10^{-5}/1.02$" in text
    assert "\\label{tab:local-osr-heldout}" in text
