#!/usr/bin/env python3
"""Lint SDD artifact structure: required files, naming, size budgets, context headers.

Run from the repo root:

    conda run -n qgd python .cursor/skills/spec-driven-development/scripts/check_artifacts.py
    conda run -n qgd python .cursor/skills/spec-driven-development/scripts/check_artifacts.py docs/specs/milestones/<slug>

or run both linters through `scripts/specs_check.sh`.

Exit status: 0 when there are no unwaived errors, 1 otherwise. Warnings become
errors with --strict. Waivers and size budgets live in docs/specs/.sdd-lint.json.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sddlint import (  # noqa: E402
    Reporter,
    base_parser,
    budget_for,
    read_lines,
    read_text,
    resolve_context,
    slice_dirs,
    slug_upper,
)

# Header lines an artifact may use to declare purpose/status/traceability. An agent
# doing a partial read needs this in the first few lines to decide whether to
# continue reading the file at all.
CONTEXT_HEADER_RE = re.compile(
    r"^\s*(>|\*\*(Status|Milestone|Slice|Purpose|Scope|Work package|From)\b|- \*\*(Status|Milestone)\b)",
    re.IGNORECASE,
)
CONTEXT_HEADER_WINDOW = 12

NO_HANDBACK_CLAIM_RE = re.compile(
    r"No\s+`?STEP_4A_HANDBACK\.md`?\s+(was|were)\s+raised", re.IGNORECASE
)

LEGACY_STORY_RE = re.compile(r"^TASK_\d+_DELIVERY_STORIES\.md$")
LEGACY_TASKS_RE = re.compile(r"^TASK_\d+_ENGINEERING_TASKS\.md$")


def check_context_header(reporter: Reporter, path: Path) -> None:
    lines = read_lines(path)
    window = [ln for ln in lines[:CONTEXT_HEADER_WINDOW] if ln.strip()]
    if not window:
        return
    if not any(CONTEXT_HEADER_RE.match(ln) for ln in window):
        reporter.add(
            "MISSING_CONTEXT_HEADER",
            "warn",
            path,
            "no status/purpose header in the first "
            f"{CONTEXT_HEADER_WINDOW} lines; a partial read cannot tell what this "
            "artifact is or whether it is current",
        )


def check_size(reporter: Reporter, path: Path, budgets: dict[str, int]) -> None:
    budget = budget_for(path, budgets)
    if budget is None:
        return
    count = len(read_lines(path))
    if count > budget:
        reporter.add(
            "SIZE_BUDGET",
            "warn",
            path,
            f"{count} lines exceeds the {budget}-line budget for this artifact type; "
            "split it rather than appending (see the SDD skill size-budget table)",
        )


def check_layer1(
    reporter: Reporter, milestone: Path, budgets: dict[str, int], slug: str
) -> None:
    required = {
        "INITIAL_REQUIREMENTS.md": "requirements baseline (create-initreq-for-sdd)",
        f"DETAILED_PLANNING_{slug}.md": "Layer 1 milestone plan",
        f"ADRS_{slug}.md": "milestone ADRs",
        "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md": "readiness checklist",
    }
    for name, purpose in required.items():
        path = milestone / name
        if not path.is_file():
            reporter.add(
                "L1_MISSING_ARTIFACT",
                "error",
                path,
                f"missing {purpose}; Layer 1 is incomplete for this milestone",
            )

    # Catch a planning/ADR file whose slug does not match its directory: the
    # uppercase-underscore form of the directory name is the naming contract.
    for path in sorted(milestone.glob("DETAILED_PLANNING_*.md")) + sorted(
        milestone.glob("ADRS_*.md")
    ):
        stem = path.stem
        suffix = stem.split("_", 2)[-1] if stem.startswith("DETAILED_PLANNING") else stem[len("ADRS_") :]
        if suffix != slug:
            reporter.add(
                "SLUG_MISMATCH",
                "error",
                path,
                f"filename slug '{suffix}' does not match directory slug '{slug}'",
            )

    for path in sorted(milestone.glob("*.md")):
        check_size(reporter, path, budgets)
        check_context_header(reporter, path)


def milestone_closeouts(milestone: Path) -> list[Path]:
    return [
        p
        for p in sorted(milestone.glob("*_CLOSEOUT.md"))
        if p.name != "CLOSEOUT.md"
    ]


def check_slice(reporter: Reporter, slice_dir: Path, budgets: dict[str, int]) -> None:
    number = slice_dir.name.split("-", 1)[1]
    mini_spec = slice_dir / f"TASK_{number}_MINI_SPEC.md"
    stories = slice_dir / "DELIVERY_STORIES.md"
    tasks = slice_dir / "ENGINEERING_TASKS.md"
    closeout = slice_dir / "CLOSEOUT.md"

    if not mini_spec.is_file():
        alt = [p for p in slice_dir.glob("*MINI_SPEC*.md")]
        if alt:
            reporter.add(
                "LEGACY_FILENAME",
                "warn",
                alt[0],
                f"expected {mini_spec.name}; found {alt[0].name}",
            )
        else:
            reporter.add(
                "SLICE_MISSING_MINI_SPEC",
                "warn",
                mini_spec,
                "no Layer 2 mini-spec; acceptable only when the slice contract lives "
                "entirely in the Layer 1 plan",
            )

    for canonical, legacy_re, code in (
        (stories, LEGACY_STORY_RE, "SLICE_MISSING_STORIES"),
        (tasks, LEGACY_TASKS_RE, "SLICE_MISSING_TASKS"),
    ):
        if canonical.is_file():
            continue
        legacy = [p for p in slice_dir.iterdir() if p.is_file() and legacy_re.match(p.name)]
        if legacy:
            reporter.add(
                "LEGACY_FILENAME",
                "warn",
                legacy[0],
                f"expected {canonical.name}; found {legacy[0].name} (pre-convention naming)",
            )
        else:
            reporter.add(
                code,
                "error",
                canonical,
                f"missing {canonical.name}; a delivered slice must record its "
                "Layer 3 stories and Layer 4 tasks",
            )

    if not closeout.is_file():
        reporter.add(
            "SLICE_MISSING_CLOSEOUT",
            "warn",
            closeout,
            "no slice closeout; a shipped or handed-back slice must record its "
            "verdict and reproduce commands",
        )
    else:
        text = read_text(closeout)
        status = re.search(r"\*\*Status:\*\*\s*([^\n·]+)", text)
        if status:
            value = status.group(1).strip().lower()
            handback = slice_dir / "STEP_4A_HANDBACK.md"
            if "handback" in value and not handback.is_file():
                reporter.add(
                    "CLOSEOUT_HANDBACK_MISSING",
                    "error",
                    closeout,
                    "status is 'implementation handback' but STEP_4A_HANDBACK.md "
                    "does not exist in this slice",
                )
        else:
            reporter.add(
                "CLOSEOUT_NO_STATUS",
                "warn",
                closeout,
                "no '**Status:**' marker; slice closeouts must state shipped or "
                "implementation handback",
            )

    for path in sorted(slice_dir.glob("*.md")):
        check_size(reporter, path, budgets)
        check_context_header(reporter, path)


def check_closeout_claims(reporter: Reporter, milestone: Path) -> None:
    """A milestone closeout must not contradict the files on disk."""
    handbacks = sorted(milestone.glob("task-*/STEP_4A_HANDBACK.md"))
    for closeout in milestone_closeouts(milestone):
        text = read_text(closeout)
        if handbacks and NO_HANDBACK_CLAIM_RE.search(text):
            names = ", ".join(str(p.relative_to(milestone)) for p in handbacks)
            reporter.add(
                "CLOSEOUT_HANDBACK_CONTRADICTION",
                "warn",
                closeout,
                f"claims no Step 4a handback was raised, but the milestone contains: {names}",
            )


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    root, config, milestones = resolve_context(args)
    if not milestones:
        return 0

    reporter = Reporter(
        root=root,
        config=config,
        strict=args.strict,
        honor_waivers=not args.no_waivers,
    )
    for milestone in milestones:
        slug = slug_upper(config.slug_for(root, milestone))
        check_layer1(reporter, milestone, config.size_budgets, slug)
        check_closeout_claims(reporter, milestone)
        slices = slice_dirs(milestone)
        if not slices:
            reporter.add(
                "NO_SLICES",
                "info",
                milestone,
                "no task-<n> slice directories yet; expected before implementation starts",
            )
        for slice_dir in slices:
            check_slice(reporter, slice_dir, config.size_budgets)

    return reporter.emit(args.format, "SDD artifact structure")


if __name__ == "__main__":
    raise SystemExit(main())
