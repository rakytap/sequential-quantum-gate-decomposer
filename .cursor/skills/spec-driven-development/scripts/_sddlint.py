"""Shared helpers for the SDD spec linters (`check_artifacts.py`, `check_traceability.py`).

Standard library only: the linters must run from any bare Python >= 3.10 with no
project dependencies installed, because agents call them mid-planning and hooks
call them at agent stop. `scripts/specs_check.sh` picks the `qgd` conda interpreter
when it is available.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SEVERITIES = ("error", "warn", "info")

DEFAULT_CONFIG_NAME = ".sdd-lint.json"

# A milestone directory is any directory holding an INITIAL_REQUIREMENTS.md. This
# discovers relocated trees (e.g. docs/specs/production/<slug>/) without pinning
# a single parent directory.
MILESTONE_MARKER = "INITIAL_REQUIREMENTS.md"

SLICE_DIR_RE = re.compile(r"^task-(\d+)$")


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    path: str
    message: str
    line: int | None = None
    waiver_reason: str | None = None

    @property
    def waived(self) -> bool:
        return self.waiver_reason is not None

    def render(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        prefix = "waived" if self.waived else self.severity
        text = f"{prefix:>6}  {self.code:<32} {loc}\n        {self.message}"
        if self.waiver_reason:
            text += f"\n        waiver: {self.waiver_reason}"
        return text


@dataclass
class Config:
    """Lint configuration, loaded from `docs/specs/.sdd-lint.json` when present."""

    size_budgets: dict[str, int] = field(default_factory=dict)
    waivers: list[dict] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    known_make_targets: list[str] = field(default_factory=list)
    # Maps a spec-root-relative directory to the roadmap milestone-slug it holds,
    # for trees that were relocated after delivery and no longer match their slug.
    milestone_slugs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        if path is None or not path.is_file():
            return cls(size_budgets=dict(DEFAULT_SIZE_BUDGETS))
        data = json.loads(path.read_text(encoding="utf-8"))
        budgets = dict(DEFAULT_SIZE_BUDGETS)
        budgets.update(data.get("size_budgets", {}))
        return cls(
            size_budgets=budgets,
            waivers=data.get("waivers", []),
            exclude=data.get("exclude", []),
            known_make_targets=data.get("known_make_targets", []),
            milestone_slugs=data.get("milestone_slugs", {}),
        )

    def slug_for(self, root: Path, milestone: Path) -> str:
        """Roadmap slug for a milestone directory, honouring relocated trees."""
        declared = self.milestone_slugs.get(rel_to(root, milestone))
        return declared or milestone.name

    def waiver_for(self, code: str, rel_path: str) -> str | None:
        for waiver in self.waivers:
            codes = waiver.get("codes", [])
            if codes and code not in codes:
                continue
            patterns = waiver.get("paths", [])
            if patterns and not any(fnmatch.fnmatch(rel_path, p) for p in patterns):
                continue
            return waiver.get("reason", "waived by .sdd-lint.json")
        return None

    def is_excluded(self, rel_path: str) -> bool:
        return any(fnmatch.fnmatch(rel_path, p) for p in self.exclude)


# Line budgets per artifact type. Oversized specs are the documented failure mode:
# an agent that cannot hold a slice contract in context re-reads it partially and
# drifts. These are the numbers the SDD skill states in its size-budget table.
DEFAULT_SIZE_BUDGETS: dict[str, int] = {
    "INITIAL_REQUIREMENTS.md": 300,
    "DETAILED_PLANNING": 400,
    "ADRS": 400,
    "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md": 250,
    "MINI_SPEC": 250,
    "DELIVERY_STORIES.md": 200,
    "ENGINEERING_TASKS.md": 300,
    "CLOSEOUT.md": 200,
    "MILESTONE_CLOSEOUT": 250,
    "STEP_4A_HANDBACK.md": 200,
    "CHANGE_CONTROL.md": 200,
}


class Reporter:
    """Collects findings, applies waivers, renders output, and computes exit status."""

    def __init__(
        self,
        root: Path,
        config: Config,
        strict: bool = False,
        honor_waivers: bool = True,
    ) -> None:
        self.root = root
        self.config = config
        self.strict = strict
        self.honor_waivers = honor_waivers
        self.findings: list[Finding] = []

    def add(
        self,
        code: str,
        severity: str,
        path: Path | str,
        message: str,
        line: int | None = None,
    ) -> None:
        rel = rel_to(self.root, path)
        if self.config.is_excluded(rel):
            return
        if self.strict and severity == "warn":
            severity = "error"
        reason = self.config.waiver_for(code, rel) if self.honor_waivers else None
        self.findings.append(
            Finding(
                code=code,
                severity=severity,
                path=rel,
                message=message,
                line=line,
                waiver_reason=reason,
            )
        )

    @property
    def active(self) -> list[Finding]:
        return [f for f in self.findings if not f.waived]

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in SEVERITIES}
        out["waived"] = 0
        for f in self.findings:
            if f.waived:
                out["waived"] += 1
            else:
                out[f.severity] += 1
        return out

    def emit(self, fmt: str, title: str) -> int:
        counts = self.counts()
        if fmt == "json":
            payload = {
                "title": title,
                "root": str(self.root),
                "counts": counts,
                "findings": [
                    {
                        "code": f.code,
                        "severity": "waived" if f.waived else f.severity,
                        "path": f.path,
                        "line": f.line,
                        "message": f.message,
                        "waiver_reason": f.waiver_reason,
                    }
                    for f in self.findings
                ],
            }
            print(json.dumps(payload, indent=2))
        else:
            print(f"== {title} ({self.root})")
            order = {"error": 0, "warn": 1, "info": 2}
            shown = sorted(
                self.findings,
                key=lambda f: (f.waived, order.get(f.severity, 3), f.path, f.code),
            )
            for f in shown:
                print(f.render())
            if not shown:
                print("  no findings")
            print(
                "-- {error} error(s), {warn} warning(s), {info} info, "
                "{waived} waived".format(**counts)
            )
        return 1 if counts["error"] else 0


def rel_to(root: Path, path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def repo_root(start: Path | None = None) -> Path:
    """Walk up from this file (or `start`) to the directory holding pyproject.toml."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def find_milestone_dirs(root: Path, explicit: list[str] | None = None) -> list[Path]:
    if explicit:
        dirs = []
        for item in explicit:
            p = Path(item)
            if p.is_file():
                p = p.parent
            # Accept a slice directory and walk up to its milestone.
            if SLICE_DIR_RE.match(p.name):
                p = p.parent
            dirs.append(p)
        return sorted(set(dirs))
    if not root.is_dir():
        return []
    return sorted({p.parent for p in root.rglob(MILESTONE_MARKER)})


def slice_dirs(milestone: Path) -> list[Path]:
    found = [d for d in milestone.iterdir() if d.is_dir() and SLICE_DIR_RE.match(d.name)]
    return sorted(found, key=lambda d: int(SLICE_DIR_RE.match(d.name).group(1)))


def slug_upper(slug: str) -> str:
    return slug.upper().replace("-", "_")


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def budget_for(path: Path, budgets: dict[str, int]) -> int | None:
    name = path.name
    if name in budgets:
        return budgets[name]
    if name.endswith("_CLOSEOUT.md"):
        return budgets.get("MILESTONE_CLOSEOUT")
    if "MINI_SPEC" in name:
        return budgets.get("MINI_SPEC")
    if name.startswith("DETAILED_PLANNING"):
        return budgets.get("DETAILED_PLANNING")
    if name.startswith("ADRS"):
        return budgets.get("ADRS")
    if name.endswith("DELIVERY_STORIES.md"):
        return budgets.get("DELIVERY_STORIES.md")
    if name.endswith("ENGINEERING_TASKS.md"):
        return budgets.get("ENGINEERING_TASKS.md")
    return None


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "paths",
        nargs="*",
        help="milestone directories (or files inside them). Default: discover under --root.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="spec root to scan (default: <repo>/docs/specs)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"lint config (default: <root>/{DEFAULT_CONFIG_NAME} when present)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors",
    )
    parser.add_argument(
        "--no-waivers",
        action="store_true",
        help="ignore the waiver list and report the full historical debt",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    return parser


def resolve_context(args: argparse.Namespace) -> tuple[Path, Config, list[Path]]:
    root = Path(args.root) if args.root else repo_root() / "docs" / "specs"
    config_path = Path(args.config) if args.config else root / DEFAULT_CONFIG_NAME
    config = Config.load(config_path)
    milestones = find_milestone_dirs(root, args.paths)
    if not milestones:
        print(f"no milestone directories found under {root}", file=sys.stderr)
    return root, config, milestones
