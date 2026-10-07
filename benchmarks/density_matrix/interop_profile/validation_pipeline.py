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
)
from benchmarks.density_matrix.interop_profile.interop_lane import run_interop_row

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmarks" / "density_matrix" / "artifacts" / "interop_profile"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "interop_profile_bundle.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the interop bundle JSON artifact",
    )
    args = parser.parse_args(argv)

    bundle = run_interop_row()
    validate_interop_bundle(bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
