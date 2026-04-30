"""Lightweight JSON artifact helpers for density-matrix evidence (no runtime imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_artifact_bundle(
    bundle: dict[str, Any], output_dir: Path, artifact_filename: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / artifact_filename
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path
