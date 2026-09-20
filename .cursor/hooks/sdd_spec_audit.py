#!/usr/bin/env python3
"""Cursor `afterFileEdit` hook: append an audit line when a spec artifact is edited.

Records who changed which SDD artifact during a session, so a milestone's spec history is
reconstructable even when the work spanned several agents. Observational only: it never
blocks an edit.

Contract: read hook input JSON on stdin, print a JSON object on stdout. Always exits 0.
The log is git-ignored session state, not a spec artifact.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_RELATIVE = Path(".cursor") / "sdd-spec-edits.log"
SPEC_PREFIX = "docs/specs/"
MAX_LOG_BYTES = 512_000


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}

    try:
        raw_path = str(payload.get("file_path") or "")
        root = repo_root()
        try:
            relative = str(Path(raw_path).resolve().relative_to(root))
        except (ValueError, OSError):
            relative = raw_path

        if relative.startswith(SPEC_PREFIX):
            log = root / LOG_RELATIVE
            log.parent.mkdir(parents=True, exist_ok=True)
            # Truncate rather than rotate: this is disposable session state.
            if log.is_file() and log.stat().st_size > MAX_LOG_BYTES:
                log.unlink()
            edits = payload.get("edits") or []
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            agent = os.environ.get("CURSOR_AGENT_ID", "-")
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"{stamp}\t{agent}\t{relative}\t{len(edits)} edit(s)\n")
    except Exception:  # noqa: BLE001 - never break an edit over an audit line
        pass

    print("{}")


if __name__ == "__main__":
    main()
