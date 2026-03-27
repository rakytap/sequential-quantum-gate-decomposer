"""Shared helpers for density-matrix evidence validation slice modules."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


def require_bundle_fields(
    bundle: dict[str, Any],
    fields: tuple[str, ...],
    label: str,
) -> None:
    missing = [field for field in fields if field not in bundle]
    if missing:
        raise ValueError(
            "{} missing required fields: {}".format(label, ", ".join(missing))
        )


def run_case_slice_cli(
    argv: list[str] | None,
    *,
    build_cases: Callable[[], list[dict[str, Any]]],
    build_artifact_bundle: Callable[[list[dict[str, Any]]], dict[str, Any]],
    artifact_filename: str,
    default_output_dir: Path,
    description: str,
    output_dir_help: str,
    quiet_report: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """CLI for slices that follow build_cases → build_artifact_bundle(cases) → write."""
    from benchmarks.density_matrix.evidence_io import write_artifact_bundle

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=output_dir_help,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case console output.",
    )
    args = parser.parse_args(argv)

    cases = build_cases()
    bundle = build_artifact_bundle(cases)
    output_path = write_artifact_bundle(bundle, args.output_dir, artifact_filename)

    if not args.quiet:
        if quiet_report is not None:
            quiet_report(bundle)
        print("Wrote {}".format(output_path))

    return 0 if bundle["status"] == "pass" else 1
