#!/usr/bin/env python3
"""Emit and validate the M-F5a interop profile bundle (counted run, C2 artifacts)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.interop_profile.interop_bundle_validation import (
    validate_interop_bundle,
    validate_interop_bundle_w6,
    validate_interop_bundle_w8,
)
from benchmarks.density_matrix.interop_profile.interop_lane import run_interop_row

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "interop_profile"
)
DEFAULT_OUTPUT_PATH_W4 = DEFAULT_OUTPUT_DIR / "interop_profile_bundle.json"
DEFAULT_OUTPUT_PATH_W6 = DEFAULT_OUTPUT_DIR / "interop_profile_bundle_w6.json"
DEFAULT_OUTPUT_PATH_W8 = DEFAULT_OUTPUT_DIR / "interop_profile_bundle_w8.json"
ARTIFACT_NAME_W4 = "interop_profile_bundle.json"
ARTIFACT_NAME_W6 = "interop_profile_bundle_w6.json"
ARTIFACT_NAME_W8 = "interop_profile_bundle_w8.json"
DEFAULT_ATTRIBUTION_ROUTES_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "interop_profile_bundle_routes_w4.json"
)


def resolve_interop_output_path(
    width: int | None,
    output: Path | None,
    *,
    attribution_routes: bool = False,
) -> Path:
    """Resolve the bundle path and refuse width/filename mismatches before any pair."""
    if attribution_routes:
        from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
            QBIT_WIDTH_TASK4,
            resolve_attribution_output_path,
        )

        qbit_num = width if width is not None else QBIT_WIDTH_TASK4
        resolved = resolve_attribution_output_path(
            output if output is not None else DEFAULT_ATTRIBUTION_ROUTES_OUTPUT,
            width=qbit_num,
        )
        return Path(resolved)

    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        refuse_evqe_output_with_routes_name,
    )

    refuse_evqe_output_with_routes_name(output)

    if width in (None, 4):
        qbit_num = 4
        default_path = DEFAULT_OUTPUT_PATH_W4
        expected_name = ARTIFACT_NAME_W4
        forbidden_names = {ARTIFACT_NAME_W6, ARTIFACT_NAME_W8}
    elif width == 6:
        qbit_num = 6
        default_path = DEFAULT_OUTPUT_PATH_W6
        expected_name = ARTIFACT_NAME_W6
        forbidden_names = {ARTIFACT_NAME_W4, ARTIFACT_NAME_W8}
    elif width == 8:
        qbit_num = 8
        default_path = DEFAULT_OUTPUT_PATH_W8
        expected_name = ARTIFACT_NAME_W8
        forbidden_names = {ARTIFACT_NAME_W4, ARTIFACT_NAME_W6}
    else:
        raise ValueError(f"unsupported --width {width}; expected 4, 6, or 8")

    resolved = output if output is not None else default_path
    if qbit_num in (6, 8) and resolved.name != expected_name:
        raise ValueError(
            f"width {qbit_num} must write {expected_name!r}; got {resolved.name!r}"
        )
    if resolved.name in forbidden_names:
        raise ValueError(
            f"width {qbit_num} must not write {resolved.name!r}; use {expected_name!r}"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Tracer width (default 4). Use 6 or 8 for task-2/3 rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the interop bundle JSON artifact",
    )
    parser.add_argument(
        "--attribution-routes",
        action="store_true",
        help="Emit the width-4 attribution route counted bundle (R-strict refusal row).",
    )
    args = parser.parse_args(argv)

    if args.attribution_routes:
        from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
            run_counted_attribution_bundle,
        )

        output_path = resolve_interop_output_path(
            args.width, args.output, attribution_routes=True
        )
        bundle = run_counted_attribution_bundle()
        if bundle.get("clean_start") is not True:
            print(
                "refusing to write attribution route bundle: worktree is not clean at capture",
                file=sys.stderr,
            )
            return 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        return 0

    if args.width == 8:
        qbit_num = 8
    elif args.width == 6:
        qbit_num = 6
    else:
        qbit_num = 4
    output_path = resolve_interop_output_path(args.width, args.output)

    bundle = run_interop_row(qbit_num=qbit_num)
    if qbit_num == 8:
        validate_interop_bundle_w8(bundle)
    elif qbit_num == 6:
        validate_interop_bundle_w6(bundle)
    else:
        validate_interop_bundle(bundle)

    if bundle.get("clean_start") is not True:
        print(
            "refusing to write interop bundle: worktree is not clean at capture",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
