#!/usr/bin/env python3
"""Run and emit all density-matrix correctness-evidence validation bundles in one process.

Run with:
    python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.correctness_evidence import (
    correctness_bundle_validation as correctness_package,
)
from benchmarks.density_matrix.correctness_evidence import (
    correctness_matrix_validation as correctness_matrix,
)
from benchmarks.density_matrix.correctness_evidence import (
    external_correctness_validation as external_correctness,
)
from benchmarks.density_matrix.correctness_evidence import (
    mf1a_q4_baseline_validation as mf1a_q4_baseline,
)
from benchmarks.density_matrix.correctness_evidence import (
    output_integrity_validation as output_integrity,
)
from benchmarks.density_matrix.correctness_evidence import (
    runtime_classification_validation as runtime_classification,
)
from benchmarks.density_matrix.correctness_evidence import (
    sequential_correctness_validation as sequential_correctness,
)
from benchmarks.density_matrix.correctness_evidence import (
    summary_consistency_validation as summary_consistency,
)
from benchmarks.density_matrix.correctness_evidence import (
    unsupported_boundary_validation as unsupported_boundary,
)
from benchmarks.density_matrix.correctness_evidence.common import DEFAULT_OUTPUT_ROOT
from benchmarks.density_matrix.correctness_evidence.common import (
    write_artifact_bundle,
)


class _CaseSuiteEntry(NamedTuple):
    module: Any
    cases_attr: str
    bundle_attr: str
    mf1a_sibling: bool = False


class _NullarySuiteEntry(NamedTuple):
    module: Any
    bundle_attr: str
    mf1a_sibling: bool = False


def _write_slice_bundle(module: Any, bundle: dict) -> Path:
    return write_artifact_bundle(bundle, module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)


def _artifact_layout_relative(module: Any) -> Path:
    return (
        module.DEFAULT_OUTPUT_DIR.relative_to(DEFAULT_OUTPUT_ROOT) / module.ARTIFACT_FILENAME
    )


def _find_repo_root(start: Path | None = None) -> Path | None:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        git_path = candidate / ".git"
        if git_path.is_file() or git_path.is_dir():
            return candidate
    return None


def _refuse_historical_output_dir(resolved: Path, repo_root: Path) -> None:
    message = (
        "refused: --historical-output-dir resolves inside the repository "
        f"({resolved}; repo {repo_root})"
    )
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _validate_historical_output_dir(raw: str) -> Path:
    repo_root = _find_repo_root()
    if repo_root is None:
        resolved = Path(raw).expanduser().resolve(strict=False)
        _refuse_historical_output_dir(resolved, Path("<unknown>"))
    resolved = Path(raw).expanduser().resolve(strict=False)
    if resolved == repo_root or repo_root in resolved.parents:
        _refuse_historical_output_dir(resolved, repo_root)
    return resolved


_CASE_SLICE_REGISTRY: tuple[_CaseSuiteEntry, ...] = (
    _CaseSuiteEntry(
        mf1a_q4_baseline, "build_cases", "build_artifact_bundle", mf1a_sibling=True
    ),
    _CaseSuiteEntry(correctness_matrix, "build_cases", "build_artifact_bundle"),
    _CaseSuiteEntry(sequential_correctness, "build_cases", "build_artifact_bundle"),
    _CaseSuiteEntry(external_correctness, "build_cases", "build_artifact_bundle"),
    _CaseSuiteEntry(output_integrity, "build_cases", "build_artifact_bundle"),
    _CaseSuiteEntry(runtime_classification, "build_cases", "build_artifact_bundle"),
    _CaseSuiteEntry(unsupported_boundary, "build_cases", "build_artifact_bundle"),
)

_NULLARY_BUNDLE_REGISTRY: tuple[_NullarySuiteEntry, ...] = (
    _NullarySuiteEntry(correctness_package, "build_artifact_bundle"),
    _NullarySuiteEntry(summary_consistency, "build_artifact_bundle"),
)

_G07_EXCLUDED_SUITES = frozenset(
    {
        external_correctness.SUITE_NAME,
        output_integrity.SUITE_NAME,
    }
)


def registered_suite_names() -> tuple[str, ...]:
    return tuple(
        [entry.module.SUITE_NAME for entry in _CASE_SLICE_REGISTRY]
        + [entry.module.SUITE_NAME for entry in _NULLARY_BUNDLE_REGISTRY]
    )


def g07_included_suite_names() -> tuple[str, ...]:
    return tuple(
        name for name in registered_suite_names() if name not in _G07_EXCLUDED_SUITES
    )


def g07_exit_passes(results: list[tuple[str, str, Path | None]]) -> bool:
    statuses = {suite_name: status for suite_name, status, _ in results}
    included = g07_included_suite_names()
    return (
        set(statuses) == set(registered_suite_names())
        and mf1a_q4_baseline.SUITE_NAME in statuses
        and all(statuses.get(name) == "pass" for name in included)
    )


def _repo_relative_path(path: Path) -> Path:
    repo_root = _find_repo_root()
    if repo_root is None:
        return path
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve())
    except ValueError:
        return resolved


def _format_stdout_line(
    suite_name: str,
    status: str,
    *,
    written_path: Path | None,
    historical_output_dir: Path | None,
    outside_copy_path: Path | None,
) -> str:
    if written_path is not None:
        rel = _repo_relative_path(written_path)
        return f"{suite_name}: status={status} | written {rel.as_posix()}"
    if historical_output_dir is not None and outside_copy_path is not None:
        return (
            f"{suite_name}: status={status} | verified, written outside repo "
            f"{outside_copy_path}"
        )
    return f"{suite_name}: status={status} | verified, not written"


def run_pipeline(
    *, historical_output_dir: Path | None = None
) -> list[tuple[str, str, Path | None]]:
    results: list[tuple[str, str, Path | None]] = []

    for entry in _CASE_SLICE_REGISTRY:
        mod = entry.module
        cases = getattr(mod, entry.cases_attr)()
        bundle = getattr(mod, entry.bundle_attr)(cases)
        if entry.mf1a_sibling is True:
            output_path = _write_slice_bundle(mod, bundle)
            results.append((mod.SUITE_NAME, bundle["status"], output_path))
            continue

        outside_copy: Path | None = None
        if historical_output_dir is not None:
            relative = _artifact_layout_relative(mod)
            outside_copy = historical_output_dir / relative
            outside_copy.parent.mkdir(parents=True, exist_ok=True)
            write_artifact_bundle(bundle, outside_copy.parent, mod.ARTIFACT_FILENAME)
        results.append((mod.SUITE_NAME, bundle["status"], outside_copy))

    for entry in _NULLARY_BUNDLE_REGISTRY:
        mod = entry.module
        bundle = getattr(mod, entry.bundle_attr)()
        if entry.mf1a_sibling is True:
            output_path = _write_slice_bundle(mod, bundle)
            results.append((mod.SUITE_NAME, bundle["status"], output_path))
            continue

        outside_copy = None
        if historical_output_dir is not None:
            relative = _artifact_layout_relative(mod)
            outside_copy = historical_output_dir / relative
            outside_copy.parent.mkdir(parents=True, exist_ok=True)
            write_artifact_bundle(bundle, outside_copy.parent, mod.ARTIFACT_FILENAME)
        results.append((mod.SUITE_NAME, bundle["status"], outside_copy))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-bundle console output.",
    )
    parser.add_argument(
        "--historical-output-dir",
        metavar="path",
        help=(
            "Optional directory outside the repository for verify-only bundle output."
        ),
    )
    args = parser.parse_args(argv)

    historical_output_dir: Path | None = None
    if args.historical_output_dir is not None:
        historical_output_dir = _validate_historical_output_dir(args.historical_output_dir)

    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = run_pipeline(historical_output_dir=historical_output_dir)
    if not args.quiet:
        for suite_name, status, output_path in results:
            entry_is_sibling = False
            for entry in _CASE_SLICE_REGISTRY:
                if entry.module.SUITE_NAME == suite_name:
                    entry_is_sibling = entry.mf1a_sibling is True
                    break
            else:
                for entry in _NULLARY_BUNDLE_REGISTRY:
                    if entry.module.SUITE_NAME == suite_name:
                        entry_is_sibling = entry.mf1a_sibling is True
                        break
            written_path = output_path if entry_is_sibling else None
            outside_copy = None if entry_is_sibling else output_path
            print(
                _format_stdout_line(
                    suite_name,
                    status,
                    written_path=written_path,
                    historical_output_dir=historical_output_dir,
                    outside_copy_path=outside_copy,
                )
            )
    return 0 if g07_exit_passes(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
