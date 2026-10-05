#!/usr/bin/env python3
"""Lint SDD artifact structure: required files, naming, size budgets, context headers.

Run from the repo root:

    conda run -n qgd python .cursor/skills/spec-driven-development/scripts/check_artifacts.py
    conda run -n qgd python .cursor/skills/spec-driven-development/scripts/check_artifacts.py docs/specs/milestones/<slug>

or run both linters through `scripts/specs_check.sh`.

A milestone holding only INITIAL_REQUIREMENTS.md -- no Layer 1 file, slice, milestone
closeout, or change control yet -- reports L1_NOT_STARTED (info) instead of missing-Layer-1
errors; once any of those exists, every missing Layer 1 file is an error again.

Exit status: 0 when there are no unwaived errors, 1 otherwise. With --strict,
warnings become errors except an absent-closeout SLICE_MISSING_CLOSEOUT whose
stage parse is step-4a. Waivers and size budgets live in docs/specs/.sdd-lint.json.
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
SDD_STAGE_WINDOW = 12

_SDD_STAGE_LINE_RE = re.compile(
    r"^[ \t]*(?:>[ \t]*)?\*\*SDD stage:\*\*[ \t]*(.*)$"
)

PLACEHOLDER_STATUS = frozenset(
    {
        "placeholder",
        "placeholders",
        "tbd",
        "todo",
        "planning",
        "draft",
        "wip",
        "stub",
        "fixme",
    }
)

CLOSEOUT_STATUS_RE = re.compile(r"\*\*Status:\*\*\s*([^\n·]+)")
CLOSEOUT_HEADING_RE = re.compile(r"^##[ \t]+\S")
CLOSEOUT_FENCE_RE = re.compile(r"^[ \t]*```")

SLICE_MISSING_CLOSEOUT_ABSENT_MSG = (
    "no slice closeout; a shipped or handed-back slice must record its "
    "verdict and reproduce commands"
)
SLICE_MISSING_CLOSEOUT_PLACEHOLDER_MSG = (
    "CLOSEOUT.md is not substantive; an empty or placeholder closeout does not "
    "clear SLICE_MISSING_CLOSEOUT"
)

NO_HANDBACK_CLAIM_RE = re.compile(
    r"No\s+`?STEP_4A_HANDBACK\.md`?\s+(was|were)\s+raised", re.IGNORECASE
)

LEGACY_STORY_RE = re.compile(r"^TASK_\d+_DELIVERY_STORIES\.md$")
LEGACY_TASKS_RE = re.compile(r"^TASK_\d+_ENGINEERING_TASKS\.md$")

# Anything matching these, or any task-<n> slice, means work downstream of the
# requirements baseline has begun, so Layer 1 must be complete.
LAYER1_STARTED_GLOBS = (
    "DETAILED_PLANNING_*.md",
    "ADRS_*.md",
    "ADR_AMENDMENTS_*.md",
    "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md",
    "*_CLOSEOUT.md",
    "CHANGE_CONTROL.md",
)

ADR_AMENDMENTS_PREFIX = "ADR_AMENDMENTS_"


def parse_sdd_stage(tasks_path: Path) -> str:
    """Return step-4a, step-4b-authorized, missing, duplicate, or other."""
    lines = read_lines(tasks_path)[:SDD_STAGE_WINDOW]
    values: list[str] = []
    for line in lines:
        match = _SDD_STAGE_LINE_RE.match(line)
        if match:
            values.append(match.group(1).strip(" \t"))
    if not values:
        return "missing"
    if len(values) > 1:
        return "duplicate"
    value = values[0]
    if value == "step-4a":
        return "step-4a"
    if value == "step-4b-authorized":
        return "step-4b-authorized"
    return "other"


def closeout_kind(path: Path) -> str:
    """Return absent, placeholder, or substantive (substantive-closeout rule)."""
    if not path.is_file():
        return "absent"
    text = read_text(path)
    if not text.strip():
        return "placeholder"
    status = CLOSEOUT_STATUS_RE.search(text)
    if not status:
        return "placeholder"
    status_value = status.group(1).strip()
    if not status_value:
        return "placeholder"
    if status_value.casefold() in PLACEHOLDER_STATUS:
        return "placeholder"
    lines = text.splitlines()
    if not any(CLOSEOUT_HEADING_RE.match(ln) for ln in lines):
        return "placeholder"
    if not any(CLOSEOUT_FENCE_RE.match(ln) for ln in lines):
        return "placeholder"
    return "substantive"


def emit_closeout_status_checks(
    reporter: Reporter, closeout: Path, slice_dir: Path, text: str
) -> None:
    status = CLOSEOUT_STATUS_RE.search(text)
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


def layer1_started(milestone: Path) -> bool:
    """True once the milestone holds anything beyond its requirements baseline.

    Before that, the milestone sits between create-initreq-for-sdd and
    spec-driven-development Step 1, where an absent Layer 1 is the next step, not a gap.
    """
    if slice_dirs(milestone):
        return True
    return any(any(milestone.glob(pattern)) for pattern in LAYER1_STARTED_GLOBS)


def check_layer1(
    reporter: Reporter, milestone: Path, budgets: dict[str, int], slug: str
) -> None:
    required = {
        "INITIAL_REQUIREMENTS.md": "requirements baseline (create-initreq-for-sdd)",
        f"DETAILED_PLANNING_{slug}.md": "Layer 1 milestone plan",
        f"ADRS_{slug}.md": "milestone ADRs",
        "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md": "readiness checklist",
    }
    if (milestone / "INITIAL_REQUIREMENTS.md").is_file() and not layer1_started(milestone):
        reporter.add(
            "L1_NOT_STARTED",
            "info",
            milestone,
            "requirements baseline only; Layer 1 is not started (spec-driven-development "
            "Steps 1-3 write the plan, ADRs, and readiness checklist)",
        )
        required = {"INITIAL_REQUIREMENTS.md": required["INITIAL_REQUIREMENTS.md"]}
    for name, purpose in required.items():
        path = milestone / name
        if not path.is_file():
            reporter.add(
                "L1_MISSING_ARTIFACT",
                "error",
                path,
                f"missing {purpose}; Layer 1 is incomplete for this milestone",
            )

    canonical_adrs = f"ADRS_{slug}.md"
    canonical_amendments = f"{ADR_AMENDMENTS_PREFIX}{slug}.md"
    allowed_adr = {canonical_adrs, canonical_amendments}
    for path in sorted(milestone.glob("ADR*.md")):
        if path.name not in allowed_adr:
            reporter.add(
                "ADR_FORBIDDEN_CONTINUATION",
                "error",
                path,
                "only ADRS_<SLUG>.md and at most one ADR_AMENDMENTS_<SLUG>.md are allowed "
                "at milestone root",
            )

    # Catch a planning/ADR file whose slug does not match its directory: the
    # uppercase-underscore form of the directory name is the naming contract.
    slugged = sorted(milestone.glob("DETAILED_PLANNING_*.md")) + sorted(
        milestone.glob("ADRS_*.md")
    ) + sorted(milestone.glob("ADR_AMENDMENTS_*.md"))
    for path in slugged:
        stem = path.stem
        if stem.startswith("DETAILED_PLANNING"):
            suffix = stem.split("_", 2)[-1]
        elif stem.startswith(ADR_AMENDMENTS_PREFIX):
            suffix = stem[len(ADR_AMENDMENTS_PREFIX) :]
        else:
            suffix = stem[len("ADRS_") :]
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

    stage = parse_sdd_stage(tasks)
    kind = closeout_kind(closeout)
    if kind == "absent":
        reporter.add(
            "SLICE_MISSING_CLOSEOUT",
            "warn",
            closeout,
            SLICE_MISSING_CLOSEOUT_ABSENT_MSG,
            strict_keep_warn=(stage == "step-4a"),
        )
    elif kind == "placeholder":
        reporter.add(
            "SLICE_MISSING_CLOSEOUT",
            "warn",
            closeout,
            SLICE_MISSING_CLOSEOUT_PLACEHOLDER_MSG,
            strict_keep_warn=False,
        )
        emit_closeout_status_checks(
            reporter, closeout, slice_dir, read_text(closeout)
        )
    else:
        emit_closeout_status_checks(
            reporter, closeout, slice_dir, read_text(closeout)
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
