#!/usr/bin/env python3
"""Cursor `stop` hook: refuse to finish with a broken traceability spine.

When an agent session has touched anything under `docs/specs/`, run the two SDD spec
linters. If they report unwaived errors, return a `followup_message` so the agent keeps
working instead of reporting success over a broken spine.

Contract: read hook input JSON on stdin, print a JSON object on stdout. Always exits 0 and
always prints valid JSON -- a hook that crashes must never block a session. Cursor caps
automatic follow-ups (default 5 per conversation), so this cannot loop indefinitely.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LINTERS = (
    ".cursor/skills/spec-driven-development/scripts/check_artifacts.py",
    ".cursor/skills/spec-driven-development/scripts/check_traceability.py",
)
SPEC_PREFIX = "docs/specs/"
MAX_LISTED = 12


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def touched_specs(root: Path) -> bool:
    """True when the working tree has uncommitted changes under docs/specs/."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", SPEC_PREFIX],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(result.stdout.strip())


def errors_from(root: Path, linter: str) -> list[dict]:
    script = root / linter
    if not script.is_file():
        return []
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return [f for f in payload.get("findings", []) if f.get("severity") == "error"]


def main() -> None:
    response: dict[str, str] = {}
    try:
        try:
            json.load(sys.stdin)
        except (json.JSONDecodeError, ValueError):
            pass

        root = repo_root()
        if touched_specs(root):
            findings: list[dict] = []
            for linter in LINTERS:
                findings.extend(errors_from(root, linter))
            if findings:
                listed = "\n".join(
                    f"- {f.get('code')} {f.get('path')}"
                    f"{':' + str(f['line']) if f.get('line') else ''} — {f.get('message')}"
                    for f in findings[:MAX_LISTED]
                )
                extra = (
                    f"\n- …and {len(findings) - MAX_LISTED} more"
                    if len(findings) > MAX_LISTED
                    else ""
                )
                response["followup_message"] = (
                    "This session changed files under docs/specs/, and the SDD spec "
                    f"linters report {len(findings)} error(s). Do not report the work as "
                    "complete yet.\n\n"
                    f"{listed}{extra}\n\n"
                    "Run `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` "
                    "and fix each error. If a finding is a "
                    "deliberate historical deviation, add a waiver with a reason to "
                    "docs/specs/.sdd-lint.json — never waive a finding on an in-flight "
                    "milestone."
                )
    except Exception:  # noqa: BLE001 - a hook must fail open, never break the session
        response = {}

    print(json.dumps(response))


if __name__ == "__main__":
    main()
